//===------- CGObjCGNU.cpp - Emit LLVM Code from ASTs for a Module --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides Objective-C code generation targeting the GNU runtime.  The
// class in this file generates structures used by the GNU Objective-C runtime
// library.  These structures are defined in objc/objc.h and objc/objc-api.h in
// the GNU runtime distribution.
//
//===----------------------------------------------------------------------===//

#include "CGObjCRuntime.h"
#include "CGCleanup.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/StmtObjC.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Compiler.h"
#include <cstdarg>


using namespace clang;
using namespace CodeGen;


namespace {
/// Class that lazily initialises the runtime function.  Avoids inserting the
/// types and the function declaration into a module if they're not used, and
/// avoids constructing the type more than once if it's used more than once.
class LazyRuntimeFunction {
  CodeGenModule *CGM;
  std::vector<llvm::Type*> ArgTys;
  const char *FunctionName;
  llvm::Constant *Function;
  public:
    /// Constructor leaves this class uninitialized, because it is intended to
    /// be used as a field in another class and not all of the types that are
    /// used as arguments will necessarily be available at construction time.
    LazyRuntimeFunction() : CGM(0), FunctionName(0), Function(0) {}

    /// Initialises the lazy function with the name, return type, and the types
    /// of the arguments.
    END_WITH_NULL
    void init(CodeGenModule *Mod, const char *name,
        llvm::Type *RetTy, ...) {
       CGM =Mod;
       FunctionName = name;
       Function = 0;
       ArgTys.clear();
       va_list Args;
       va_start(Args, RetTy);
         while (llvm::Type *ArgTy = va_arg(Args, llvm::Type*))
           ArgTys.push_back(ArgTy);
       va_end(Args);
       // Push the return type on at the end so we can pop it off easily
       ArgTys.push_back(RetTy);
   }
   /// Overloaded cast operator, allows the class to be implicitly cast to an
   /// LLVM constant.
   operator llvm::Constant*() {
     if (!Function) {
       if (0 == FunctionName) return 0;
       // We put the return type on the end of the vector, so pop it back off
       llvm::Type *RetTy = ArgTys.back();
       ArgTys.pop_back();
       llvm::FunctionType *FTy = llvm::FunctionType::get(RetTy, ArgTys, false);
       Function =
         cast<llvm::Constant>(CGM->CreateRuntimeFunction(FTy, FunctionName));
       // We won't need to use the types again, so we may as well clean up the
       // vector now
       ArgTys.resize(0);
     }
     return Function;
   }
   operator llvm::Function*() {
     return cast<llvm::Function>((llvm::Constant*)*this);
   }

};


/// GNU Objective-C runtime code generation.  This class implements the parts of
/// Objective-C support that are specific to the GNU family of runtimes (GCC,
/// GNUstep and ObjFW).
class CGObjCGNU : public CGObjCRuntime {
protected:
  /// The LLVM module into which output is inserted
  llvm::Module &TheModule;
  /// strut objc_super.  Used for sending messages to super.  This structure
  /// contains the receiver (object) and the expected class.
  llvm::StructType *ObjCSuperTy;
  /// struct objc_super*.  The type of the argument to the superclass message
  /// lookup functions.  
  llvm::PointerType *PtrToObjCSuperTy;
  /// LLVM type for selectors.  Opaque pointer (i8*) unless a header declaring
  /// SEL is included in a header somewhere, in which case it will be whatever
  /// type is declared in that header, most likely {i8*, i8*}.
  llvm::PointerType *SelectorTy;
  /// LLVM i8 type.  Cached here to avoid repeatedly getting it in all of the
  /// places where it's used
  llvm::IntegerType *Int8Ty;
  /// Pointer to i8 - LLVM type of char*, for all of the places where the
  /// runtime needs to deal with C strings.
  llvm::PointerType *PtrToInt8Ty;
  /// Instance Method Pointer type.  This is a pointer to a function that takes,
  /// at a minimum, an object and a selector, and is the generic type for
  /// Objective-C methods.  Due to differences between variadic / non-variadic
  /// calling conventions, it must always be cast to the correct type before
  /// actually being used.
  llvm::PointerType *IMPTy;
  /// Type of an untyped Objective-C object.  Clang treats id as a built-in type
  /// when compiling Objective-C code, so this may be an opaque pointer (i8*),
  /// but if the runtime header declaring it is included then it may be a
  /// pointer to a structure.
  llvm::PointerType *IdTy;
  /// Pointer to a pointer to an Objective-C object.  Used in the new ABI
  /// message lookup function and some GC-related functions.
  llvm::PointerType *PtrToIdTy;
  /// The clang type of id.  Used when using the clang CGCall infrastructure to
  /// call Objective-C methods.
  CanQualType ASTIdTy;
  /// LLVM type for C int type.
  llvm::IntegerType *IntTy;
  /// LLVM type for an opaque pointer.  This is identical to PtrToInt8Ty, but is
  /// used in the code to document the difference between i8* meaning a pointer
  /// to a C string and i8* meaning a pointer to some opaque type.
  llvm::PointerType *PtrTy;
  /// LLVM type for C long type.  The runtime uses this in a lot of places where
  /// it should be using intptr_t, but we can't fix this without breaking
  /// compatibility with GCC...
  llvm::IntegerType *LongTy;
  /// LLVM type for C size_t.  Used in various runtime data structures.
  llvm::IntegerType *SizeTy;
  /// LLVM type for C intptr_t.  
  llvm::IntegerType *IntPtrTy;
  /// LLVM type for C ptrdiff_t.  Mainly used in property accessor functions.
  llvm::IntegerType *PtrDiffTy;
  /// LLVM type for C int*.  Used for GCC-ABI-compatible non-fragile instance
  /// variables.
  llvm::PointerType *PtrToIntTy;
  /// LLVM type for Objective-C BOOL type.
  llvm::Type *BoolTy;
  /// 32-bit integer type, to save us needing to look it up every time it's used.
  llvm::IntegerType *Int32Ty;
  /// 64-bit integer type, to save us needing to look it up every time it's used.
  llvm::IntegerType *Int64Ty;
  /// Metadata kind used to tie method lookups to message sends.  The GNUstep
  /// runtime provides some LLVM passes that can use this to do things like
  /// automatic IMP caching and speculative inlining.
  unsigned msgSendMDKind;
  /// Helper function that generates a constant string and returns a pointer to
  /// the start of the string.  The result of this function can be used anywhere
  /// where the C code specifies const char*.  
  llvm::Constant *MakeConstantString(const std::string &Str,
                                     const std::string &Name="") {
    llvm::Constant *ConstStr = CGM.GetAddrOfConstantCString(Str, Name.c_str());
    return llvm::ConstantExpr::getGetElementPtr(ConstStr, Zeros);
  }
  /// Emits a linkonce_odr string, whose name is the prefix followed by the
  /// string value.  This allows the linker to combine the strings between
  /// different modules.  Used for EH typeinfo names, selector strings, and a
  /// few other things.
  llvm::Constant *ExportUniqueString(const std::string &Str,
                                     const std::string prefix) {
    std::string name = prefix + Str;
    llvm::Constant *ConstStr = TheModule.getGlobalVariable(name);
    if (!ConstStr) {
      llvm::Constant *value = llvm::ConstantDataArray::getString(VMContext,Str);
      ConstStr = new llvm::GlobalVariable(TheModule, value->getType(), true,
              llvm::GlobalValue::LinkOnceODRLinkage, value, prefix + Str);
    }
    return llvm::ConstantExpr::getGetElementPtr(ConstStr, Zeros);
  }
  /// Generates a global structure, initialized by the elements in the vector.
  /// The element types must match the types of the structure elements in the
  /// first argument.
  llvm::GlobalVariable *MakeGlobal(llvm::StructType *Ty,
                                   ArrayRef<llvm::Constant *> V,
                                   StringRef Name="",
                                   llvm::GlobalValue::LinkageTypes linkage
                                         =llvm::GlobalValue::InternalLinkage) {
    llvm::Constant *C = llvm::ConstantStruct::get(Ty, V);
    return new llvm::GlobalVariable(TheModule, Ty, false,
        linkage, C, Name);
  }
  /// Generates a global array.  The vector must contain the same number of
  /// elements that the array type declares, of the type specified as the array
  /// element type.
  llvm::GlobalVariable *MakeGlobal(llvm::ArrayType *Ty,
                                   ArrayRef<llvm::Constant *> V,
                                   StringRef Name="",
                                   llvm::GlobalValue::LinkageTypes linkage
                                         =llvm::GlobalValue::InternalLinkage) {
    llvm::Constant *C = llvm::ConstantArray::get(Ty, V);
    return new llvm::GlobalVariable(TheModule, Ty, false,
                                    linkage, C, Name);
  }
  /// Generates a global array, inferring the array type from the specified
  /// element type and the size of the initialiser.  
  llvm::GlobalVariable *MakeGlobalArray(llvm::Type *Ty,
                                        ArrayRef<llvm::Constant *> V,
                                        StringRef Name="",
                                        llvm::GlobalValue::LinkageTypes linkage
                                         =llvm::GlobalValue::InternalLinkage) {
    llvm::ArrayType *ArrayTy = llvm::ArrayType::get(Ty, V.size());
    return MakeGlobal(ArrayTy, V, Name, linkage);
  }
  /// Returns a property name and encoding string.
  llvm::Constant *MakePropertyEncodingString(const ObjCPropertyDecl *PD,
                                             const Decl *Container) {
    const ObjCRuntime &R = CGM.getLangOpts().ObjCRuntime;
    if ((R.getKind() == ObjCRuntime::GNUstep) &&
        (R.getVersion() >= VersionTuple(1, 6))) {
      std::string NameAndAttributes;
      std::string TypeStr;
      CGM.getContext().getObjCEncodingForPropertyDecl(PD, Container, TypeStr);
      NameAndAttributes += '\0';
      NameAndAttributes += TypeStr.length() + 3;
      NameAndAttributes += TypeStr;
      NameAndAttributes += '\0';
      NameAndAttributes += PD->getNameAsString();
      NameAndAttributes += '\0';
      return llvm::ConstantExpr::getGetElementPtr(
          CGM.GetAddrOfConstantString(NameAndAttributes), Zeros);
    }
    return MakeConstantString(PD->getNameAsString());
  }
  /// Push the property attributes into two structure fields. 
  void PushPropertyAttributes(std::vector<llvm::Constant*> &Fields,
      ObjCPropertyDecl *property, bool isSynthesized=true, bool
      isDynamic=true) {
    int attrs = property->getPropertyAttributes();
    // For read-only properties, clear the copy and retain flags
    if (attrs & ObjCPropertyDecl::OBJC_PR_readonly) {
      attrs &= ~ObjCPropertyDecl::OBJC_PR_copy;
      attrs &= ~ObjCPropertyDecl::OBJC_PR_retain;
      attrs &= ~ObjCPropertyDecl::OBJC_PR_weak;
      attrs &= ~ObjCPropertyDecl::OBJC_PR_strong;
    }
    // The first flags field has the same attribute values as clang uses internally
    Fields.push_back(llvm::ConstantInt::get(Int8Ty, attrs & 0xff));
    attrs >>= 8;
    attrs <<= 2;
    // For protocol properties, synthesized and dynamic have no meaning, so we
    // reuse these flags to indicate that this is a protocol property (both set
    // has no meaning, as a property can't be both synthesized and dynamic)
    attrs |= isSynthesized ? (1<<0) : 0;
    attrs |= isDynamic ? (1<<1) : 0;
    // The second field is the next four fields left shifted by two, with the
    // low bit set to indicate whether the field is synthesized or dynamic.
    Fields.push_back(llvm::ConstantInt::get(Int8Ty, attrs & 0xff));
    // Two padding fields
    Fields.push_back(llvm::ConstantInt::get(Int8Ty, 0));
    Fields.push_back(llvm::ConstantInt::get(Int8Ty, 0));
  }
  /// Ensures that the value has the required type, by inserting a bitcast if
  /// required.  This function lets us avoid inserting bitcasts that are
  /// redundant.
  llvm::Value* EnforceType(CGBuilderTy &B, llvm::Value *V, llvm::Type *Ty) {
    if (V->getType() == Ty) return V;
    return B.CreateBitCast(V, Ty);
  }
  // Some zeros used for GEPs in lots of places.
  llvm::Constant *Zeros[2];
  /// Null pointer value.  Mainly used as a terminator in various arrays.
  llvm::Constant *NULLPtr;
  /// LLVM context.
  llvm::LLVMContext &VMContext;
private:
  /// Placeholder for the class.  Lots of things refer to the class before we've
  /// actually emitted it.  We use this alias as a placeholder, and then replace
  /// it with a pointer to the class structure before finally emitting the
  /// module.
  llvm::GlobalAlias *ClassPtrAlias;
  /// Placeholder for the metaclass.  Lots of things refer to the class before
  /// we've / actually emitted it.  We use this alias as a placeholder, and then
  /// replace / it with a pointer to the metaclass structure before finally
  /// emitting the / module.
  llvm::GlobalAlias *MetaClassPtrAlias;
  /// All of the classes that have been generated for this compilation units.
  std::vector<llvm::Constant*> Classes;
  /// All of the categories that have been generated for this compilation units.
  std::vector<llvm::Constant*> Categories;
  /// All of the Objective-C constant strings that have been generated for this
  /// compilation units.
  std::vector<llvm::Constant*> ConstantStrings;
  /// Map from string values to Objective-C constant strings in the output.
  /// Used to prevent emitting Objective-C strings more than once.  This should
  /// not be required at all - CodeGenModule should manage this list.
  llvm::StringMap<llvm::Constant*> ObjCStrings;
  /// All of the protocols that have been declared.
  llvm::StringMap<llvm::Constant*> ExistingProtocols;
  /// For each variant of a selector, we store the type encoding and a
  /// placeholder value.  For an untyped selector, the type will be the empty
  /// string.  Selector references are all done via the module's selector table,
  /// so we create an alias as a placeholder and then replace it with the real
  /// value later.
  typedef std::pair<std::string, llvm::GlobalAlias*> TypedSelector;
  /// Type of the selector map.  This is roughly equivalent to the structure
  /// used in the GNUstep runtime, which maintains a list of all of the valid
  /// types for a selector in a table.
  typedef llvm::DenseMap<Selector, SmallVector<TypedSelector, 2> >
    SelectorMap;
  /// A map from selectors to selector types.  This allows us to emit all
  /// selectors of the same name and type together.
  SelectorMap SelectorTable;

  /// Selectors related to memory management.  When compiling in GC mode, we
  /// omit these.
  Selector RetainSel, ReleaseSel, AutoreleaseSel;
  /// Runtime functions used for memory management in GC mode.  Note that clang
  /// supports code generation for calling these functions, but neither GNU
  /// runtime actually supports this API properly yet.
  LazyRuntimeFunction IvarAssignFn, StrongCastAssignFn, MemMoveFn, WeakReadFn, 
    WeakAssignFn, GlobalAssignFn;

  typedef std::pair<std::string, std::string> ClassAliasPair;
  /// All classes that have aliases set for them.
  std::vector<ClassAliasPair> ClassAliases;

protected:
  /// Function used for throwing Objective-C exceptions.
  LazyRuntimeFunction ExceptionThrowFn;
  /// Function used for rethrowing exceptions, used at the end of \@finally or
  /// \@synchronize blocks.
  LazyRuntimeFunction ExceptionReThrowFn;
  /// Function called when entering a catch function.  This is required for
  /// differentiating Objective-C exceptions and foreign exceptions.
  LazyRuntimeFunction EnterCatchFn;
  /// Function called when exiting from a catch block.  Used to do exception
  /// cleanup.
  LazyRuntimeFunction ExitCatchFn;
  /// Function called when entering an \@synchronize block.  Acquires the lock.
  LazyRuntimeFunction SyncEnterFn;
  /// Function called when exiting an \@synchronize block.  Releases the lock.
  LazyRuntimeFunction SyncExitFn;

private:

  /// Function called if fast enumeration detects that the collection is
  /// modified during the update.
  LazyRuntimeFunction EnumerationMutationFn;
  /// Function for implementing synthesized property getters that return an
  /// object.
  LazyRuntimeFunction GetPropertyFn;
  /// Function for implementing synthesized property setters that return an
  /// object.
  LazyRuntimeFunction SetPropertyFn;
  /// Function used for non-object declared property getters.
  LazyRuntimeFunction GetStructPropertyFn;
  /// Function used for non-object declared property setters.
  LazyRuntimeFunction SetStructPropertyFn;

  /// The version of the runtime that this class targets.  Must match the
  /// version in the runtime.
  int RuntimeVersion;
  /// The version of the protocol class.  Used to differentiate between ObjC1
  /// and ObjC2 protocols.  Objective-C 1 protocols can not contain optional
  /// components and can not contain declared properties.  We always emit
  /// Objective-C 2 property structures, but we have to pretend that they're
  /// Objective-C 1 property structures when targeting the GCC runtime or it
  /// will abort.
  const int ProtocolVersion;
private:
  /// Generates an instance variable list structure.  This is a structure
  /// containing a size and an array of structures containing instance variable
  /// metadata.  This is used purely for introspection in the fragile ABI.  In
  /// the non-fragile ABI, it's used for instance variable fixup.
  llvm::Constant *GenerateIvarList(ArrayRef<llvm::Constant *> IvarNames,
                                   ArrayRef<llvm::Constant *> IvarTypes,
                                   ArrayRef<llvm::Constant *> IvarOffsets);
  /// Generates a method list structure.  This is a structure containing a size
  /// and an array of structures containing method metadata.
  ///
  /// This structure is used by both classes and categories, and contains a next
  /// pointer allowing them to be chained together in a linked list.
  llvm::Constant *GenerateMethodList(const StringRef &ClassName,
      const StringRef &CategoryName,
      ArrayRef<Selector> MethodSels,
      ArrayRef<llvm::Constant *> MethodTypes,
      bool isClassMethodList);
  /// Emits an empty protocol.  This is used for \@protocol() where no protocol
  /// is found.  The runtime will (hopefully) fix up the pointer to refer to the
  /// real protocol.
  llvm::Constant *GenerateEmptyProtocol(const std::string &ProtocolName);
  /// Generates a list of property metadata structures.  This follows the same
  /// pattern as method and instance variable metadata lists.
  llvm::Constant *GeneratePropertyList(const ObjCImplementationDecl *OID,
        SmallVectorImpl<Selector> &InstanceMethodSels,
        SmallVectorImpl<llvm::Constant*> &InstanceMethodTypes);
  /// Generates a list of referenced protocols.  Classes, categories, and
  /// protocols all use this structure.
  llvm::Constant *GenerateProtocolList(ArrayRef<std::string> Protocols);
  /// To ensure that all protocols are seen by the runtime, we add a category on
  /// a class defined in the runtime, declaring no methods, but adopting the
  /// protocols.  This is a horribly ugly hack, but it allows us to collect all
  /// of the protocols without changing the ABI.
  void GenerateProtocolHolderCategory();
  /// Generates a class structure.
  llvm::Constant *GenerateClassStructure(
      llvm::Constant *MetaClass,
      llvm::Constant *SuperClass,
      unsigned info,
      const char *Name,
      llvm::Constant *Version,
      llvm::Constant *InstanceSize,
      llvm::Constant *IVars,
      llvm::Constant *Methods,
      llvm::Constant *Protocols,
      llvm::Constant *IvarOffsets,
      llvm::Constant *Properties,
      llvm::Constant *StrongIvarBitmap,
      llvm::Constant *WeakIvarBitmap,
      bool isMeta=false);
  /// Generates a method list.  This is used by protocols to define the required
  /// and optional methods.
  llvm::Constant *GenerateProtocolMethodList(
      ArrayRef<llvm::Constant *> MethodNames,
      ArrayRef<llvm::Constant *> MethodTypes);
  /// Returns a selector with the specified type encoding.  An empty string is
  /// used to return an untyped selector (with the types field set to NULL).
  llvm::Value *GetSelector(CodeGenFunction &CGF, Selector Sel,
    const std::string &TypeEncoding, bool lval);
  /// Returns the variable used to store the offset of an instance variable.
  llvm::GlobalVariable *ObjCIvarOffsetVariable(const ObjCInterfaceDecl *ID,
      const ObjCIvarDecl *Ivar);
  /// Emits a reference to a class.  This allows the linker to object if there
  /// is no class of the matching name.
protected:
  void EmitClassRef(const std::string &className);
  /// Emits a pointer to the named class
  virtual llvm::Value *GetClassNamed(CodeGenFunction &CGF,
                                     const std::string &Name, bool isWeak);
  /// Looks up the method for sending a message to the specified object.  This
  /// mechanism differs between the GCC and GNU runtimes, so this method must be
  /// overridden in subclasses.
  virtual llvm::Value *LookupIMP(CodeGenFunction &CGF,
                                 llvm::Value *&Receiver,
                                 llvm::Value *cmd,
                                 llvm::MDNode *node,
                                 MessageSendInfo &MSI) = 0;
  /// Looks up the method for sending a message to a superclass.  This
  /// mechanism differs between the GCC and GNU runtimes, so this method must
  /// be overridden in subclasses.
  virtual llvm::Value *LookupIMPSuper(CodeGenFunction &CGF,
                                      llvm::Value *ObjCSuper,
                                      llvm::Value *cmd,
                                      MessageSendInfo &MSI) = 0;
  /// Libobjc2 uses a bitfield representation where small(ish) bitfields are
  /// stored in a 64-bit value with the low bit set to 1 and the remaining 63
  /// bits set to their values, LSB first, while larger ones are stored in a
  /// structure of this / form:
  /// 
  /// struct { int32_t length; int32_t values[length]; };
  ///
  /// The values in the array are stored in host-endian format, with the least
  /// significant bit being assumed to come first in the bitfield.  Therefore,
  /// a bitfield with the 64th bit set will be (int64_t)&{ 2, [0, 1<<31] },
  /// while a bitfield / with the 63rd bit set will be 1<<64.
  llvm::Constant *MakeBitField(ArrayRef<bool> bits);
public:
  CGObjCGNU(CodeGenModule &cgm, unsigned runtimeABIVersion,
      unsigned protocolClassVersion);

  virtual llvm::Constant *GenerateConstantString(const StringLiteral *);

  virtual RValue
  GenerateMessageSend(CodeGenFunction &CGF,
                      ReturnValueSlot Return,
                      QualType ResultType,
                      Selector Sel,
                      llvm::Value *Receiver,
                      const CallArgList &CallArgs,
                      const ObjCInterfaceDecl *Class,
                      const ObjCMethodDecl *Method);
  virtual RValue
  GenerateMessageSendSuper(CodeGenFunction &CGF,
                           ReturnValueSlot Return,
                           QualType ResultType,
                           Selector Sel,
                           const ObjCInterfaceDecl *Class,
                           bool isCategoryImpl,
                           llvm::Value *Receiver,
                           bool IsClassMessage,
                           const CallArgList &CallArgs,
                           const ObjCMethodDecl *Method);
  virtual llvm::Value *GetClass(CodeGenFunction &CGF,
                                const ObjCInterfaceDecl *OID);
  virtual llvm::Value *GetSelector(CodeGenFunction &CGF, Selector Sel,
                                   bool lval = false);
  virtual llvm::Value *GetSelector(CodeGenFunction &CGF, const ObjCMethodDecl
      *Method);
  virtual llvm::Constant *GetEHType(QualType T);

  virtual llvm::Function *GenerateMethod(const ObjCMethodDecl *OMD,
                                         const ObjCContainerDecl *CD);
  virtual void GenerateCategory(const ObjCCategoryImplDecl *CMD);
  virtual void GenerateClass(const ObjCImplementationDecl *ClassDecl);
  virtual void RegisterAlias(const ObjCCompatibleAliasDecl *OAD);
  virtual llvm::Value *GenerateProtocolRef(CodeGenFunction &CGF,
                                           const ObjCProtocolDecl *PD);
  virtual void GenerateProtocol(const ObjCProtocolDecl *PD);
  virtual llvm::Function *ModuleInitFunction();
  virtual llvm::Constant *GetPropertyGetFunction();
  virtual llvm::Constant *GetPropertySetFunction();
  virtual llvm::Constant *GetOptimizedPropertySetFunction(bool atomic, 
                                                          bool copy);
  virtual llvm::Constant *GetSetStructFunction();
  virtual llvm::Constant *GetGetStructFunction();
  virtual llvm::Constant *GetCppAtomicObjectGetFunction();
  virtual llvm::Constant *GetCppAtomicObjectSetFunction();
  virtual llvm::Constant *EnumerationMutationFunction();

  virtual void EmitTryStmt(CodeGenFunction &CGF,
                           const ObjCAtTryStmt &S);
  virtual void EmitSynchronizedStmt(CodeGenFunction &CGF,
                                    const ObjCAtSynchronizedStmt &S);
  virtual void EmitThrowStmt(CodeGenFunction &CGF,
                             const ObjCAtThrowStmt &S,
                             bool ClearInsertionPoint=true);
  virtual llvm::Value * EmitObjCWeakRead(CodeGenFunction &CGF,
                                         llvm::Value *AddrWeakObj);
  virtual void EmitObjCWeakAssign(CodeGenFunction &CGF,
                                  llvm::Value *src, llvm::Value *dst);
  virtual void EmitObjCGlobalAssign(CodeGenFunction &CGF,
                                    llvm::Value *src, llvm::Value *dest,
                                    bool threadlocal=false);
  virtual void EmitObjCIvarAssign(CodeGenFunction &CGF,
                                    llvm::Value *src, llvm::Value *dest,
                                    llvm::Value *ivarOffset);
  virtual void EmitObjCStrongCastAssign(CodeGenFunction &CGF,
                                        llvm::Value *src, llvm::Value *dest);
  virtual void EmitGCMemmoveCollectable(CodeGenFunction &CGF,
                                        llvm::Value *DestPtr,
                                        llvm::Value *SrcPtr,
                                        llvm::Value *Size);
  virtual LValue EmitObjCValueForIvar(CodeGenFunction &CGF,
                                      QualType ObjectTy,
                                      llvm::Value *BaseValue,
                                      const ObjCIvarDecl *Ivar,
                                      unsigned CVRQualifiers);
  virtual llvm::Value *EmitIvarOffset(CodeGenFunction &CGF,
                                      const ObjCInterfaceDecl *Interface,
                                      const ObjCIvarDecl *Ivar);
  virtual llvm::Value *EmitNSAutoreleasePoolClassRef(CodeGenFunction &CGF);
  virtual llvm::Constant *BuildGCBlockLayout(CodeGenModule &CGM,
                                             const CGBlockInfo &blockInfo) {
    return NULLPtr;
  }
  virtual llvm::Constant *BuildRCBlockLayout(CodeGenModule &CGM,
                                             const CGBlockInfo &blockInfo) {
    return NULLPtr;
  }
  
  virtual llvm::Constant *BuildByrefLayout(CodeGenModule &CGM,
                                           QualType T) {
    return NULLPtr;
  }
  
  virtual llvm::GlobalVariable *GetClassGlobal(const std::string &Name) {
    return 0;
  }
};
/// Class representing the legacy GCC Objective-C ABI.  This is the default when
/// -fobjc-nonfragile-abi is not specified.
///
/// The GCC ABI target actually generates code that is approximately compatible
/// with the new GNUstep runtime ABI, but refrains from using any features that
/// would not work with the GCC runtime.  For example, clang always generates
/// the extended form of the class structure, and the extra fields are simply
/// ignored by GCC libobjc.
class CGObjCGCC : public CGObjCGNU {
  /// The GCC ABI message lookup function.  Returns an IMP pointing to the
  /// method implementation for this message.
  LazyRuntimeFunction MsgLookupFn;
  /// The GCC ABI superclass message lookup function.  Takes a pointer to a
  /// structure describing the receiver and the class, and a selector as
  /// arguments.  Returns the IMP for the corresponding method.
  LazyRuntimeFunction MsgLookupSuperFn;
protected:
  virtual llvm::Value *LookupIMP(CodeGenFunction &CGF,
                                 llvm::Value *&Receiver,
                                 llvm::Value *cmd,
                                 llvm::MDNode *node,
                                 MessageSendInfo &MSI) {
    CGBuilderTy &Builder = CGF.Builder;
    llvm::Value *args[] = {
            EnforceType(Builder, Receiver, IdTy),
            EnforceType(Builder, cmd, SelectorTy) };
    llvm::CallSite imp = CGF.EmitRuntimeCallOrInvoke(MsgLookupFn, args);
    imp->setMetadata(msgSendMDKind, node);
    return imp.getInstruction();
  }
  virtual llvm::Value *LookupIMPSuper(CodeGenFunction &CGF,
                                      llvm::Value *ObjCSuper,
                                      llvm::Value *cmd,
                                      MessageSendInfo &MSI) {
      CGBuilderTy &Builder = CGF.Builder;
      llvm::Value *lookupArgs[] = {EnforceType(Builder, ObjCSuper,
          PtrToObjCSuperTy), cmd};
      return CGF.EmitNounwindRuntimeCall(MsgLookupSuperFn, lookupArgs);
    }
  public:
    CGObjCGCC(CodeGenModule &Mod) : CGObjCGNU(Mod, 8, 2) {
      // IMP objc_msg_lookup(id, SEL);
      MsgLookupFn.init(&CGM, "objc_msg_lookup", IMPTy, IdTy, SelectorTy, NULL);
      // IMP objc_msg_lookup_super(struct objc_super*, SEL);
      MsgLookupSuperFn.init(&CGM, "objc_msg_lookup_super", IMPTy,
              PtrToObjCSuperTy, SelectorTy, NULL);
    }
};
/// Class used when targeting the new GNUstep runtime ABI.
class CGObjCGNUstep : public CGObjCGNU {
    /// The slot lookup function.  Returns a pointer to a cacheable structure
    /// that contains (among other things) the IMP.
    LazyRuntimeFunction SlotLookupFn;
    /// The GNUstep ABI superclass message lookup function.  Takes a pointer to
    /// a structure describing the receiver and the class, and a selector as
    /// arguments.  Returns the slot for the corresponding method.  Superclass
    /// message lookup rarely changes, so this is a good caching opportunity.
    LazyRuntimeFunction SlotLookupSuperFn;
    /// Specialised function for setting atomic retain properties
    LazyRuntimeFunction SetPropertyAtomic;
    /// Specialised function for setting atomic copy properties
    LazyRuntimeFunction SetPropertyAtomicCopy;
    /// Specialised function for setting nonatomic retain properties
    LazyRuntimeFunction SetPropertyNonAtomic;
    /// Specialised function for setting nonatomic copy properties
    LazyRuntimeFunction SetPropertyNonAtomicCopy;
    /// Function to perform atomic copies of C++ objects with nontrivial copy
    /// constructors from Objective-C ivars.
    LazyRuntimeFunction CxxAtomicObjectGetFn;
    /// Function to perform atomic copies of C++ objects with nontrivial copy
    /// constructors to Objective-C ivars.
    LazyRuntimeFunction CxxAtomicObjectSetFn;
    /// Type of an slot structure pointer.  This is returned by the various
    /// lookup functions.
    llvm::Type *SlotTy;
  public:
    virtual llvm::Constant *GetEHType(QualType T);
  protected:
    virtual llvm::Value *LookupIMP(CodeGenFunction &CGF,
                                   llvm::Value *&Receiver,
                                   llvm::Value *cmd,
                                   llvm::MDNode *node,
                                   MessageSendInfo &MSI) {
      CGBuilderTy &Builder = CGF.Builder;
      llvm::Function *LookupFn = SlotLookupFn;

      // Store the receiver on the stack so that we can reload it later
      llvm::Value *ReceiverPtr = CGF.CreateTempAlloca(Receiver->getType());
      Builder.CreateStore(Receiver, ReceiverPtr);

      llvm::Value *self;

      if (isa<ObjCMethodDecl>(CGF.CurCodeDecl)) {
        self = CGF.LoadObjCSelf();
      } else {
        self = llvm::ConstantPointerNull::get(IdTy);
      }

      // The lookup function is guaranteed not to capture the receiver pointer.
      LookupFn->setDoesNotCapture(1);

      llvm::Value *args[] = {
              EnforceType(Builder, ReceiverPtr, PtrToIdTy),
              EnforceType(Builder, cmd, SelectorTy),
              EnforceType(Builder, self, IdTy) };
      llvm::CallSite slot = CGF.EmitRuntimeCallOrInvoke(LookupFn, args);
      slot.setOnlyReadsMemory();
      slot->setMetadata(msgSendMDKind, node);

      // Load the imp from the slot
      llvm::Value *imp =
        Builder.CreateLoad(Builder.CreateStructGEP(slot.getInstruction(), 4));

      // The lookup function may have changed the receiver, so make sure we use
      // the new one.
      Receiver = Builder.CreateLoad(ReceiverPtr, true);
      return imp;
    }
    virtual llvm::Value *LookupIMPSuper(CodeGenFunction &CGF,
                                        llvm::Value *ObjCSuper,
                                        llvm::Value *cmd,
                                        MessageSendInfo &MSI) {
      CGBuilderTy &Builder = CGF.Builder;
      llvm::Value *lookupArgs[] = {ObjCSuper, cmd};

      llvm::CallInst *slot =
        CGF.EmitNounwindRuntimeCall(SlotLookupSuperFn, lookupArgs);
      slot->setOnlyReadsMemory();

      return Builder.CreateLoad(Builder.CreateStructGEP(slot, 4));
    }
  public:
    CGObjCGNUstep(CodeGenModule &Mod) : CGObjCGNU(Mod, 9, 3) {
      const ObjCRuntime &R = CGM.getLangOpts().ObjCRuntime;

      llvm::StructType *SlotStructTy = llvm::StructType::get(PtrTy,
          PtrTy, PtrTy, IntTy, IMPTy, NULL);
      SlotTy = llvm::PointerType::getUnqual(SlotStructTy);
      // Slot_t objc_msg_lookup_sender(id *receiver, SEL selector, id sender);
      SlotLookupFn.init(&CGM, "objc_msg_lookup_sender", SlotTy, PtrToIdTy,
          SelectorTy, IdTy, NULL);
      // Slot_t objc_msg_lookup_super(struct objc_super*, SEL);
      SlotLookupSuperFn.init(&CGM, "objc_slot_lookup_super", SlotTy,
              PtrToObjCSuperTy, SelectorTy, NULL);
      // If we're in ObjC++ mode, then we want to make 
      if (CGM.getLangOpts().CPlusPlus) {
        llvm::Type *VoidTy = llvm::Type::getVoidTy(VMContext);
        // void *__cxa_begin_catch(void *e)
        EnterCatchFn.init(&CGM, "__cxa_begin_catch", PtrTy, PtrTy, NULL);
        // void __cxa_end_catch(void)
        ExitCatchFn.init(&CGM, "__cxa_end_catch", VoidTy, NULL);
        // void _Unwind_Resume_or_Rethrow(void*)
        ExceptionReThrowFn.init(&CGM, "_Unwind_Resume_or_Rethrow", VoidTy,
            PtrTy, NULL);
      } else if (R.getVersion() >= VersionTuple(1, 7)) {
        llvm::Type *VoidTy = llvm::Type::getVoidTy(VMContext);
        // id objc_begin_catch(void *e)
        EnterCatchFn.init(&CGM, "objc_begin_catch", IdTy, PtrTy, NULL);
        // void objc_end_catch(void)
        ExitCatchFn.init(&CGM, "objc_end_catch", VoidTy, NULL);
        // void _Unwind_Resume_or_Rethrow(void*)
        ExceptionReThrowFn.init(&CGM, "objc_exception_rethrow", VoidTy,
            PtrTy, NULL);
      }
      llvm::Type *VoidTy = llvm::Type::getVoidTy(VMContext);
      SetPropertyAtomic.init(&CGM, "objc_setProperty_atomic", VoidTy, IdTy,
          SelectorTy, IdTy, PtrDiffTy, NULL);
      SetPropertyAtomicCopy.init(&CGM, "objc_setProperty_atomic_copy", VoidTy,
          IdTy, SelectorTy, IdTy, PtrDiffTy, NULL);
      SetPropertyNonAtomic.init(&CGM, "objc_setProperty_nonatomic", VoidTy,
          IdTy, SelectorTy, IdTy, PtrDiffTy, NULL);
      SetPropertyNonAtomicCopy.init(&CGM, "objc_setProperty_nonatomic_copy",
          VoidTy, IdTy, SelectorTy, IdTy, PtrDiffTy, NULL);
      // void objc_setCppObjectAtomic(void *dest, const void *src, void
      // *helper);
      CxxAtomicObjectSetFn.init(&CGM, "objc_setCppObjectAtomic", VoidTy, PtrTy,
          PtrTy, PtrTy, NULL);
      // void objc_getCppObjectAtomic(void *dest, const void *src, void
      // *helper);
      CxxAtomicObjectGetFn.init(&CGM, "objc_getCppObjectAtomic", VoidTy, PtrTy,
          PtrTy, PtrTy, NULL);
    }
    virtual llvm::Constant *GetCppAtomicObjectGetFunction() {
      // The optimised functions were added in version 1.7 of the GNUstep
      // runtime.
      assert (CGM.getLangOpts().ObjCRuntime.getVersion() >=
          VersionTuple(1, 7));
      return CxxAtomicObjectGetFn;
    }
    virtual llvm::Constant *GetCppAtomicObjectSetFunction() {
      // The optimised functions were added in version 1.7 of the GNUstep
      // runtime.
      assert (CGM.getLangOpts().ObjCRuntime.getVersion() >=
          VersionTuple(1, 7));
      return CxxAtomicObjectSetFn;
    }
    virtual llvm::Constant *GetOptimizedPropertySetFunction(bool atomic,
                                                            bool copy) {
      // The optimised property functions omit the GC check, and so are not
      // safe to use in GC mode.  The standard functions are fast in GC mode,
      // so there is less advantage in using them.
      assert ((CGM.getLangOpts().getGC() == LangOptions::NonGC));
      // The optimised functions were added in version 1.7 of the GNUstep
      // runtime.
      assert (CGM.getLangOpts().ObjCRuntime.getVersion() >=
          VersionTuple(1, 7));

      if (atomic) {
        if (copy) return SetPropertyAtomicCopy;
        return SetPropertyAtomic;
      }
      if (copy) return SetPropertyNonAtomicCopy;
      return SetPropertyNonAtomic;

      return 0;
    }
};

/// Support for the ObjFW runtime.
class CGObjCObjFW: public CGObjCGNU {
protected:
  /// The GCC ABI message lookup function.  Returns an IMP pointing to the
  /// method implementation for this message.
  LazyRuntimeFunction MsgLookupFn;
  /// stret lookup function.  While this does not seem to make sense at the
  /// first look, this is required to call the correct forwarding function.
  LazyRuntimeFunction MsgLookupFnSRet;
  /// The GCC ABI superclass message lookup function.  Takes a pointer to a
  /// structure describing the receiver and the class, and a selector as
  /// arguments.  Returns the IMP for the corresponding method.
  LazyRuntimeFunction MsgLookupSuperFn, MsgLookupSuperFnSRet;

  virtual llvm::Value *LookupIMP(CodeGenFunction &CGF,
                                 llvm::Value *&Receiver,
                                 llvm::Value *cmd,
                                 llvm::MDNode *node,
                                 MessageSendInfo &MSI) {
    CGBuilderTy &Builder = CGF.Builder;
    llvm::Value *args[] = {
            EnforceType(Builder, Receiver, IdTy),
            EnforceType(Builder, cmd, SelectorTy) };

    llvm::CallSite imp;
    if (CGM.ReturnTypeUsesSRet(MSI.CallInfo))
      imp = CGF.EmitRuntimeCallOrInvoke(MsgLookupFnSRet, args);
    else
      imp = CGF.EmitRuntimeCallOrInvoke(MsgLookupFn, args);

    imp->setMetadata(msgSendMDKind, node);
    return imp.getInstruction();
  }

  virtual llvm::Value *LookupIMPSuper(CodeGenFunction &CGF,
                                      llvm::Value *ObjCSuper,
                                      llvm::Value *cmd,
                                      MessageSendInfo &MSI) {
      CGBuilderTy &Builder = CGF.Builder;
      llvm::Value *lookupArgs[] = {EnforceType(Builder, ObjCSuper,
          PtrToObjCSuperTy), cmd};

      if (CGM.ReturnTypeUsesSRet(MSI.CallInfo))
        return CGF.EmitNounwindRuntimeCall(MsgLookupSuperFnSRet, lookupArgs);
      else
        return CGF.EmitNounwindRuntimeCall(MsgLookupSuperFn, lookupArgs);
    }

  virtual llvm::Value *GetClassNamed(CodeGenFunction &CGF,
                                     const std::string &Name, bool isWeak) {
    if (isWeak)
      return CGObjCGNU::GetClassNamed(CGF, Name, isWeak);

    EmitClassRef(Name);

    std::string SymbolName = "_OBJC_CLASS_" + Name;

    llvm::GlobalVariable *ClassSymbol = TheModule.getGlobalVariable(SymbolName);

    if (!ClassSymbol)
      ClassSymbol = new llvm::GlobalVariable(TheModule, LongTy, false,
                                             llvm::GlobalValue::ExternalLinkage,
                                             0, SymbolName);

    return ClassSymbol;
  }

public:
  CGObjCObjFW(CodeGenModule &Mod): CGObjCGNU(Mod, 9, 3) {
    // IMP objc_msg_lookup(id, SEL);
    MsgLookupFn.init(&CGM, "objc_msg_lookup", IMPTy, IdTy, SelectorTy, NULL);
    MsgLookupFnSRet.init(&CGM, "objc_msg_lookup_stret", IMPTy, IdTy,
                         SelectorTy, NULL);
    // IMP objc_msg_lookup_super(struct objc_super*, SEL);
    MsgLookupSuperFn.init(&CGM, "objc_msg_lookup_super", IMPTy,
                          PtrToObjCSuperTy, SelectorTy, NULL);
    MsgLookupSuperFnSRet.init(&CGM, "objc_msg_lookup_super_stret", IMPTy,
                              PtrToObjCSuperTy, SelectorTy, NULL);
  }
};
} // end anonymous namespace


/// Emits a reference to a dummy variable which is emitted with each class.
/// This ensures that a linker error will be generated when trying to link
/// together modules where a referenced class is not defined.
void CGObjCGNU::EmitClassRef(const std::string &className) {
  std::string symbolRef = "__objc_class_ref_" + className;
  // Don't emit two copies of the same symbol
  if (TheModule.getGlobalVariable(symbolRef))
    return;
  std::string symbolName = "__objc_class_name_" + className;
  llvm::GlobalVariable *ClassSymbol = TheModule.getGlobalVariable(symbolName);
  if (!ClassSymbol) {
    ClassSymbol = new llvm::GlobalVariable(TheModule, LongTy, false,
        llvm::GlobalValue::ExternalLinkage, 0, symbolName);
  }
  new llvm::GlobalVariable(TheModule, ClassSymbol->getType(), true,
    llvm::GlobalValue::WeakAnyLinkage, ClassSymbol, symbolRef);
}

static std::string SymbolNameForMethod(const StringRef &ClassName,
    const StringRef &CategoryName, const Selector MethodName,
    bool isClassMethod) {
  std::string MethodNameColonStripped = MethodName.getAsString();
  std::replace(MethodNameColonStripped.begin(), MethodNameColonStripped.end(),
      ':', '_');
  return (Twine(isClassMethod ? "_c_" : "_i_") + ClassName + "_" +
    CategoryName + "_" + MethodNameColonStripped).str();
}

CGObjCGNU::CGObjCGNU(CodeGenModule &cgm, unsigned runtimeABIVersion,
    unsigned protocolClassVersion)
  : CGObjCRuntime(cgm), TheModule(CGM.getModule()),
    VMContext(cgm.getLLVMContext()), ClassPtrAlias(0), MetaClassPtrAlias(0),
    RuntimeVersion(runtimeABIVersion), ProtocolVersion(protocolClassVersion) {

  msgSendMDKind = VMContext.getMDKindID("GNUObjCMessageSend");

  CodeGenTypes &Types = CGM.getTypes();
  IntTy = cast<llvm::IntegerType>(
      Types.ConvertType(CGM.getContext().IntTy));
  LongTy = cast<llvm::IntegerType>(
      Types.ConvertType(CGM.getContext().LongTy));
  SizeTy = cast<llvm::IntegerType>(
      Types.ConvertType(CGM.getContext().getSizeType()));
  PtrDiffTy = cast<llvm::IntegerType>(
      Types.ConvertType(CGM.getContext().getPointerDiffType()));
  BoolTy = CGM.getTypes().ConvertType(CGM.getContext().BoolTy);

  Int8Ty = llvm::Type::getInt8Ty(VMContext);
  // C string type.  Used in lots of places.
  PtrToInt8Ty = llvm::PointerType::getUnqual(Int8Ty);

  Zeros[0] = llvm::ConstantInt::get(LongTy, 0);
  Zeros[1] = Zeros[0];
  NULLPtr = llvm::ConstantPointerNull::get(PtrToInt8Ty);
  // Get the selector Type.
  QualType selTy = CGM.getContext().getObjCSelType();
  if (QualType() == selTy) {
    SelectorTy = PtrToInt8Ty;
  } else {
    SelectorTy = cast<llvm::PointerType>(CGM.getTypes().ConvertType(selTy));
  }

  PtrToIntTy = llvm::PointerType::getUnqual(IntTy);
  PtrTy = PtrToInt8Ty;

  Int32Ty = llvm::Type::getInt32Ty(VMContext);
  Int64Ty = llvm::Type::getInt64Ty(VMContext);

  IntPtrTy =
      TheModule.getPointerSize() == llvm::Module::Pointer32 ? Int32Ty : Int64Ty;

  // Object type
  QualType UnqualIdTy = CGM.getContext().getObjCIdType();
  ASTIdTy = CanQualType();
  if (UnqualIdTy != QualType()) {
    ASTIdTy = CGM.getContext().getCanonicalType(UnqualIdTy);
    IdTy = cast<llvm::PointerType>(CGM.getTypes().ConvertType(ASTIdTy));
  } else {
    IdTy = PtrToInt8Ty;
  }
  PtrToIdTy = llvm::PointerType::getUnqual(IdTy);

  ObjCSuperTy = llvm::StructType::get(IdTy, IdTy, NULL);
  PtrToObjCSuperTy = llvm::PointerType::getUnqual(ObjCSuperTy);

  llvm::Type *VoidTy = llvm::Type::getVoidTy(VMContext);

  // void objc_exception_throw(id);
  ExceptionThrowFn.init(&CGM, "objc_exception_throw", VoidTy, IdTy, NULL);
  ExceptionReThrowFn.init(&CGM, "objc_exception_throw", VoidTy, IdTy, NULL);
  // int objc_sync_enter(id);
  SyncEnterFn.init(&CGM, "objc_sync_enter", IntTy, IdTy, NULL);
  // int objc_sync_exit(id);
  SyncExitFn.init(&CGM, "objc_sync_exit", IntTy, IdTy, NULL);

  // void objc_enumerationMutation (id)
  EnumerationMutationFn.init(&CGM, "objc_enumerationMutation", VoidTy,
      IdTy, NULL);

  // id objc_getProperty(id, SEL, ptrdiff_t, BOOL)
  GetPropertyFn.init(&CGM, "objc_getProperty", IdTy, IdTy, SelectorTy,
      PtrDiffTy, BoolTy, NULL);
  // void objc_setProperty(id, SEL, ptrdiff_t, id, BOOL, BOOL)
  SetPropertyFn.init(&CGM, "objc_setProperty", VoidTy, IdTy, SelectorTy,
      PtrDiffTy, IdTy, BoolTy, BoolTy, NULL);
  // void objc_setPropertyStruct(void*, void*, ptrdiff_t, BOOL, BOOL)
  GetStructPropertyFn.init(&CGM, "objc_getPropertyStruct", VoidTy, PtrTy, PtrTy, 
      PtrDiffTy, BoolTy, BoolTy, NULL);
  // void objc_setPropertyStruct(void*, void*, ptrdiff_t, BOOL, BOOL)
  SetStructPropertyFn.init(&CGM, "objc_setPropertyStruct", VoidTy, PtrTy, PtrTy, 
      PtrDiffTy, BoolTy, BoolTy, NULL);

  // IMP type
  llvm::Type *IMPArgs[] = { IdTy, SelectorTy };
  IMPTy = llvm::PointerType::getUnqual(llvm::FunctionType::get(IdTy, IMPArgs,
              true));

  const LangOptions &Opts = CGM.getLangOpts();
  if ((Opts.getGC() != LangOptions::NonGC) || Opts.ObjCAutoRefCount)
    RuntimeVersion = 10;

  // Don't bother initialising the GC stuff unless we're compiling in GC mode
  if (Opts.getGC() != LangOptions::NonGC) {
    // This is a bit of an hack.  We should sort this out by having a proper
    // CGObjCGNUstep subclass for GC, but we may want to really support the old
    // ABI and GC added in ObjectiveC2.framework, so we fudge it a bit for now
    // Get selectors needed in GC mode
    RetainSel = GetNullarySelector("retain", CGM.getContext());
    ReleaseSel = GetNullarySelector("release", CGM.getContext());
    AutoreleaseSel = GetNullarySelector("autorelease", CGM.getContext());

    // Get functions needed in GC mode

    // id objc_assign_ivar(id, id, ptrdiff_t);
    IvarAssignFn.init(&CGM, "objc_assign_ivar", IdTy, IdTy, IdTy, PtrDiffTy,
        NULL);
    // id objc_assign_strongCast (id, id*)
    StrongCastAssignFn.init(&CGM, "objc_assign_strongCast", IdTy, IdTy,
        PtrToIdTy, NULL);
    // id objc_assign_global(id, id*);
    GlobalAssignFn.init(&CGM, "objc_assign_global", IdTy, IdTy, PtrToIdTy,
        NULL);
    // id objc_assign_weak(id, id*);
    WeakAssignFn.init(&CGM, "objc_assign_weak", IdTy, IdTy, PtrToIdTy, NULL);
    // id objc_read_weak(id*);
    WeakReadFn.init(&CGM, "objc_read_weak", IdTy, PtrToIdTy, NULL);
    // void *objc_memmove_collectable(void*, void *, size_t);
    MemMoveFn.init(&CGM, "objc_memmove_collectable", PtrTy, PtrTy, PtrTy,
        SizeTy, NULL);
  }
}

llvm::Value *CGObjCGNU::GetClassNamed(CodeGenFunction &CGF,
                                      const std::string &Name,
                                      bool isWeak) {
  llvm::Value *ClassName = CGM.GetAddrOfConstantCString(Name);
  // With the incompatible ABI, this will need to be replaced with a direct
  // reference to the class symbol.  For the compatible nonfragile ABI we are
  // still performing this lookup at run time but emitting the symbol for the
  // class externally so that we can make the switch later.
  //
  // Libobjc2 contains an LLVM pass that replaces calls to objc_lookup_class
  // with memoized versions or with static references if it's safe to do so.
  if (!isWeak)
    EmitClassRef(Name);
  ClassName = CGF.Builder.CreateStructGEP(ClassName, 0);

  llvm::Constant *ClassLookupFn =
    CGM.CreateRuntimeFunction(llvm::FunctionType::get(IdTy, PtrToInt8Ty, true),
                              "objc_lookup_class");
  return CGF.EmitNounwindRuntimeCall(ClassLookupFn, ClassName);
}

// This has to perform the lookup every time, since posing and related
// techniques can modify the name -> class mapping.
llvm::Value *CGObjCGNU::GetClass(CodeGenFunction &CGF,
                                 const ObjCInterfaceDecl *OID) {
  return GetClassNamed(CGF, OID->getNameAsString(), OID->isWeakImported());
}
llvm::Value *CGObjCGNU::EmitNSAutoreleasePoolClassRef(CodeGenFunction &CGF) {
  return GetClassNamed(CGF, "NSAutoreleasePool", false);
}

llvm::Value *CGObjCGNU::GetSelector(CodeGenFunction &CGF, Selector Sel,
    const std::string &TypeEncoding, bool lval) {

  SmallVectorImpl<TypedSelector> &Types = SelectorTable[Sel];
  llvm::GlobalAlias *SelValue = 0;


  for (SmallVectorImpl<TypedSelector>::iterator i = Types.begin(),
      e = Types.end() ; i!=e ; i++) {
    if (i->first == TypeEncoding) {
      SelValue = i->second;
      break;
    }
  }
  if (0 == SelValue) {
    SelValue = new llvm::GlobalAlias(SelectorTy,
                                     llvm::GlobalValue::PrivateLinkage,
                                     ".objc_selector_"+Sel.getAsString(), NULL,
                                     &TheModule);
    Types.push_back(TypedSelector(TypeEncoding, SelValue));
  }

  if (lval) {
    llvm::Value *tmp = CGF.CreateTempAlloca(SelValue->getType());
    CGF.Builder.CreateStore(SelValue, tmp);
    return tmp;
  }
  return SelValue;
}

llvm::Value *CGObjCGNU::GetSelector(CodeGenFunction &CGF, Selector Sel,
                                    bool lval) {
  return GetSelector(CGF, Sel, std::string(), lval);
}

llvm::Value *CGObjCGNU::GetSelector(CodeGenFunction &CGF,
                                    const ObjCMethodDecl *Method) {
  std::string SelTypes;
  CGM.getContext().getObjCEncodingForMethodDecl(Method, SelTypes);
  return GetSelector(CGF, Method->getSelector(), SelTypes, false);
}

llvm::Constant *CGObjCGNU::GetEHType(QualType T) {
  if (T->isObjCIdType() || T->isObjCQualifiedIdType()) {
    // With the old ABI, there was only one kind of catchall, which broke
    // foreign exceptions.  With the new ABI, we use __objc_id_typeinfo as
    // a pointer indicating object catchalls, and NULL to indicate real
    // catchalls
    if (CGM.getLangOpts().ObjCRuntime.isNonFragile()) {
      return MakeConstantString("@id");
    } else {
      return 0;
    }
  }

  // All other types should be Objective-C interface pointer types.
  const ObjCObjectPointerType *OPT = T->getAs<ObjCObjectPointerType>();
  assert(OPT && "Invalid @catch type.");
  const ObjCInterfaceDecl *IDecl = OPT->getObjectType()->getInterface();
  assert(IDecl && "Invalid @catch type.");
  return MakeConstantString(IDecl->getIdentifier()->getName());
}

llvm::Constant *CGObjCGNUstep::GetEHType(QualType T) {
  if (!CGM.getLangOpts().CPlusPlus)
    return CGObjCGNU::GetEHType(T);

  // For Objective-C++, we want to provide the ability to catch both C++ and
  // Objective-C objects in the same function.

  // There's a particular fixed type info for 'id'.
  if (T->isObjCIdType() ||
      T->isObjCQualifiedIdType()) {
    llvm::Constant *IDEHType =
      CGM.getModule().getGlobalVariable("__objc_id_type_info");
    if (!IDEHType)
      IDEHType =
        new llvm::GlobalVariable(CGM.getModule(), PtrToInt8Ty,
                                 false,
                                 llvm::GlobalValue::ExternalLinkage,
                                 0, "__objc_id_type_info");
    return llvm::ConstantExpr::getBitCast(IDEHType, PtrToInt8Ty);
  }

  const ObjCObjectPointerType *PT =
    T->getAs<ObjCObjectPointerType>();
  assert(PT && "Invalid @catch type.");
  const ObjCInterfaceType *IT = PT->getInterfaceType();
  assert(IT && "Invalid @catch type.");
  std::string className = IT->getDecl()->getIdentifier()->getName();

  std::string typeinfoName = "__objc_eh_typeinfo_" + className;

  // Return the existing typeinfo if it exists
  llvm::Constant *typeinfo = TheModule.getGlobalVariable(typeinfoName);
  if (typeinfo)
    return llvm::ConstantExpr::getBitCast(typeinfo, PtrToInt8Ty);

  // Otherwise create it.

  // vtable for gnustep::libobjc::__objc_class_type_info
  // It's quite ugly hard-coding this.  Ideally we'd generate it using the host
  // platform's name mangling.
  const char *vtableName = "_ZTVN7gnustep7libobjc22__objc_class_type_infoE";
  llvm::Constant *Vtable = TheModule.getGlobalVariable(vtableName);
  if (!Vtable) {
    Vtable = new llvm::GlobalVariable(TheModule, PtrToInt8Ty, true,
            llvm::GlobalValue::ExternalLinkage, 0, vtableName);
  }
  llvm::Constant *Two = llvm::ConstantInt::get(IntTy, 2);
  Vtable = llvm::ConstantExpr::getGetElementPtr(Vtable, Two);
  Vtable = llvm::ConstantExpr::getBitCast(Vtable, PtrToInt8Ty);

  llvm::Constant *typeName =
    ExportUniqueString(className, "__objc_eh_typename_");

  std::vector<llvm::Constant*> fields;
  fields.push_back(Vtable);
  fields.push_back(typeName);
  llvm::Constant *TI = 
      MakeGlobal(llvm::StructType::get(PtrToInt8Ty, PtrToInt8Ty,
              NULL), fields, "__objc_eh_typeinfo_" + className,
          llvm::GlobalValue::LinkOnceODRLinkage);
  return llvm::ConstantExpr::getBitCast(TI, PtrToInt8Ty);
}

/// Generate an NSConstantString object.
llvm::Constant *CGObjCGNU::GenerateConstantString(const StringLiteral *SL) {

  std::string Str = SL->getString().str();

  // Look for an existing one
  llvm::StringMap<llvm::Constant*>::iterator old = ObjCStrings.find(Str);
  if (old != ObjCStrings.end())
    return old->getValue();

  StringRef StringClass = CGM.getLangOpts().ObjCConstantStringClass;

  if (StringClass.empty()) StringClass = "NXConstantString";

  std::string Sym = "_OBJC_CLASS_";
  Sym += StringClass;

  llvm::Constant *isa = TheModule.getNamedGlobal(Sym);

  if (!isa)
    isa = new llvm::GlobalVariable(TheModule, IdTy, /* isConstant */false,
            llvm::GlobalValue::ExternalWeakLinkage, 0, Sym);
  else if (isa->getType() != PtrToIdTy)
    isa = llvm::ConstantExpr::getBitCast(isa, PtrToIdTy);

  std::vector<llvm::Constant*> Ivars;
  Ivars.push_back(isa);
  Ivars.push_back(MakeConstantString(Str));
  Ivars.push_back(llvm::ConstantInt::get(IntTy, Str.size()));
  llvm::Constant *ObjCStr = MakeGlobal(
    llvm::StructType::get(PtrToIdTy, PtrToInt8Ty, IntTy, NULL),
    Ivars, ".objc_str");
  ObjCStr = llvm::ConstantExpr::getBitCast(ObjCStr, PtrToInt8Ty);
  ObjCStrings[Str] = ObjCStr;
  ConstantStrings.push_back(ObjCStr);
  return ObjCStr;
}

///Generates a message send where the super is the receiver.  This is a message
///send to self with special delivery semantics indicating which class's method
///should be called.
RValue
CGObjCGNU::GenerateMessageSendSuper(CodeGenFunction &CGF,
                                    ReturnValueSlot Return,
                                    QualType ResultType,
                                    Selector Sel,
                                    const ObjCInterfaceDecl *Class,
                                    bool isCategoryImpl,
                                    llvm::Value *Receiver,
                                    bool IsClassMessage,
                                    const CallArgList &CallArgs,
                                    const ObjCMethodDecl *Method) {
  CGBuilderTy &Builder = CGF.Builder;
  if (CGM.getLangOpts().getGC() == LangOptions::GCOnly) {
    if (Sel == RetainSel || Sel == AutoreleaseSel) {
      return RValue::get(EnforceType(Builder, Receiver,
                  CGM.getTypes().ConvertType(ResultType)));
    }
    if (Sel == ReleaseSel) {
      return RValue::get(0);
    }
  }

  llvm::Value *cmd = GetSelector(CGF, Sel);


  CallArgList ActualArgs;

  ActualArgs.add(RValue::get(EnforceType(Builder, Receiver, IdTy)), ASTIdTy);
  ActualArgs.add(RValue::get(cmd), CGF.getContext().getObjCSelType());
  ActualArgs.addFrom(CallArgs);

  MessageSendInfo MSI = getMessageSendInfo(Method, ResultType, ActualArgs);

  llvm::Value *ReceiverClass = 0;
  if (isCategoryImpl) {
    llvm::Constant *classLookupFunction = 0;
    if (IsClassMessage)  {
      classLookupFunction = CGM.CreateRuntimeFunction(llvm::FunctionType::get(
            IdTy, PtrTy, true), "objc_get_meta_class");
    } else {
      classLookupFunction = CGM.CreateRuntimeFunction(llvm::FunctionType::get(
            IdTy, PtrTy, true), "objc_get_class");
    }
    ReceiverClass = Builder.CreateCall(classLookupFunction,
        MakeConstantString(Class->getNameAsString()));
  } else {
    // Set up global aliases for the metaclass or class pointer if they do not
    // already exist.  These will are forward-references which will be set to
    // pointers to the class and metaclass structure created for the runtime
    // load function.  To send a message to super, we look up the value of the
    // super_class pointer from either the class or metaclass structure.
    if (IsClassMessage)  {
      if (!MetaClassPtrAlias) {
        MetaClassPtrAlias = new llvm::GlobalAlias(IdTy,
            llvm::GlobalValue::InternalLinkage, ".objc_metaclass_ref" +
            Class->getNameAsString(), NULL, &TheModule);
      }
      ReceiverClass = MetaClassPtrAlias;
    } else {
      if (!ClassPtrAlias) {
        ClassPtrAlias = new llvm::GlobalAlias(IdTy,
            llvm::GlobalValue::InternalLinkage, ".objc_class_ref" +
            Class->getNameAsString(), NULL, &TheModule);
      }
      ReceiverClass = ClassPtrAlias;
    }
  }
  // Cast the pointer to a simplified version of the class structure
  ReceiverClass = Builder.CreateBitCast(ReceiverClass,
      llvm::PointerType::getUnqual(
        llvm::StructType::get(IdTy, IdTy, NULL)));
  // Get the superclass pointer
  ReceiverClass = Builder.CreateStructGEP(ReceiverClass, 1);
  // Load the superclass pointer
  ReceiverClass = Builder.CreateLoad(ReceiverClass);
  // Construct the structure used to look up the IMP
  llvm::StructType *ObjCSuperTy = llvm::StructType::get(
      Receiver->getType(), IdTy, NULL);
  llvm::Value *ObjCSuper = Builder.CreateAlloca(ObjCSuperTy);

  Builder.CreateStore(Receiver, Builder.CreateStructGEP(ObjCSuper, 0));
  Builder.CreateStore(ReceiverClass, Builder.CreateStructGEP(ObjCSuper, 1));

  ObjCSuper = EnforceType(Builder, ObjCSuper, PtrToObjCSuperTy);

  // Get the IMP
  llvm::Value *imp = LookupIMPSuper(CGF, ObjCSuper, cmd, MSI);
  imp = EnforceType(Builder, imp, MSI.MessengerType);

  llvm::Value *impMD[] = {
      llvm::MDString::get(VMContext, Sel.getAsString()),
      llvm::MDString::get(VMContext, Class->getSuperClass()->getNameAsString()),
      llvm::ConstantInt::get(llvm::Type::getInt1Ty(VMContext), IsClassMessage)
   };
  llvm::MDNode *node = llvm::MDNode::get(VMContext, impMD);

  llvm::Instruction *call;
  RValue msgRet = CGF.EmitCall(MSI.CallInfo, imp, Return, ActualArgs, 0, &call);
  call->setMetadata(msgSendMDKind, node);
  return msgRet;
}

/// Generate code for a message send expression.
RValue
CGObjCGNU::GenerateMessageSend(CodeGenFunction &CGF,
                               ReturnValueSlot Return,
                               QualType ResultType,
                               Selector Sel,
                               llvm::Value *Receiver,
                               const CallArgList &CallArgs,
                               const ObjCInterfaceDecl *Class,
                               const ObjCMethodDecl *Method) {
  CGBuilderTy &Builder = CGF.Builder;

  // Strip out message sends to retain / release in GC mode
  if (CGM.getLangOpts().getGC() == LangOptions::GCOnly) {
    if (Sel == RetainSel || Sel == AutoreleaseSel) {
      return RValue::get(EnforceType(Builder, Receiver,
                  CGM.getTypes().ConvertType(ResultType)));
    }
    if (Sel == ReleaseSel) {
      return RValue::get(0);
    }
  }

  // If the return type is something that goes in an integer register, the
  // runtime will handle 0 returns.  For other cases, we fill in the 0 value
  // ourselves.
  //
  // The language spec says the result of this kind of message send is
  // undefined, but lots of people seem to have forgotten to read that
  // paragraph and insist on sending messages to nil that have structure
  // returns.  With GCC, this generates a random return value (whatever happens
  // to be on the stack / in those registers at the time) on most platforms,
  // and generates an illegal instruction trap on SPARC.  With LLVM it corrupts
  // the stack.  
  bool isPointerSizedReturn = (ResultType->isAnyPointerType() ||
      ResultType->isIntegralOrEnumerationType() || ResultType->isVoidType());

  llvm::BasicBlock *startBB = 0;
  llvm::BasicBlock *messageBB = 0;
  llvm::BasicBlock *continueBB = 0;

  if (!isPointerSizedReturn) {
    startBB = Builder.GetInsertBlock();
    messageBB = CGF.createBasicBlock("msgSend");
    continueBB = CGF.createBasicBlock("continue");

    llvm::Value *isNil = Builder.CreateICmpEQ(Receiver, 
            llvm::Constant::getNullValue(Receiver->getType()));
    Builder.CreateCondBr(isNil, continueBB, messageBB);
    CGF.EmitBlock(messageBB);
  }

  IdTy = cast<llvm::PointerType>(CGM.getTypes().ConvertType(ASTIdTy));
  llvm::Value *cmd;
  if (Method)
    cmd = GetSelector(CGF, Method);
  else
    cmd = GetSelector(CGF, Sel);
  cmd = EnforceType(Builder, cmd, SelectorTy);
  Receiver = EnforceType(Builder, Receiver, IdTy);

  llvm::Value *impMD[] = {
        llvm::MDString::get(VMContext, Sel.getAsString()),
        llvm::MDString::get(VMContext, Class ? Class->getNameAsString() :""),
        llvm::ConstantInt::get(llvm::Type::getInt1Ty(VMContext), Class!=0)
   };
  llvm::MDNode *node = llvm::MDNode::get(VMContext, impMD);

  CallArgList ActualArgs;
  ActualArgs.add(RValue::get(Receiver), ASTIdTy);
  ActualArgs.add(RValue::get(cmd), CGF.getContext().getObjCSelType());
  ActualArgs.addFrom(CallArgs);

  MessageSendInfo MSI = getMessageSendInfo(Method, ResultType, ActualArgs);

  // Get the IMP to call
  llvm::Value *imp;

  // If we have non-legacy dispatch specified, we try using the objc_msgSend()
  // functions.  These are not supported on all platforms (or all runtimes on a
  // given platform), so we 
  switch (CGM.getCodeGenOpts().getObjCDispatchMethod()) {
    case CodeGenOptions::Legacy:
      imp = LookupIMP(CGF, Receiver, cmd, node, MSI);
      break;
    case CodeGenOptions::Mixed:
    case CodeGenOptions::NonLegacy:
      if (CGM.ReturnTypeUsesFPRet(ResultType)) {
        imp = CGM.CreateRuntimeFunction(llvm::FunctionType::get(IdTy, IdTy, true),
                                  "objc_msgSend_fpret");
      } else if (CGM.ReturnTypeUsesSRet(MSI.CallInfo)) {
        // The actual types here don't matter - we're going to bitcast the
        // function anyway
        imp = CGM.CreateRuntimeFunction(llvm::FunctionType::get(IdTy, IdTy, true),
                                  "objc_msgSend_stret");
      } else {
        imp = CGM.CreateRuntimeFunction(llvm::FunctionType::get(IdTy, IdTy, true),
                                  "objc_msgSend");
      }
  }

  // Reset the receiver in case the lookup modified it
  ActualArgs[0] = CallArg(RValue::get(Receiver), ASTIdTy, false);

  imp = EnforceType(Builder, imp, MSI.MessengerType);

  llvm::Instruction *call;
  RValue msgRet = CGF.EmitCall(MSI.CallInfo, imp, Return, ActualArgs, 0, &call);
  call->setMetadata(msgSendMDKind, node);


  if (!isPointerSizedReturn) {
    messageBB = CGF.Builder.GetInsertBlock();
    CGF.Builder.CreateBr(continueBB);
    CGF.EmitBlock(continueBB);
    if (msgRet.isScalar()) {
      llvm::Value *v = msgRet.getScalarVal();
      llvm::PHINode *phi = Builder.CreatePHI(v->getType(), 2);
      phi->addIncoming(v, messageBB);
      phi->addIncoming(llvm::Constant::getNullValue(v->getType()), startBB);
      msgRet = RValue::get(phi);
    } else if (msgRet.isAggregate()) {
      llvm::Value *v = msgRet.getAggregateAddr();
      llvm::PHINode *phi = Builder.CreatePHI(v->getType(), 2);
      llvm::PointerType *RetTy = cast<llvm::PointerType>(v->getType());
      llvm::AllocaInst *NullVal = 
          CGF.CreateTempAlloca(RetTy->getElementType(), "null");
      CGF.InitTempAlloca(NullVal,
          llvm::Constant::getNullValue(RetTy->getElementType()));
      phi->addIncoming(v, messageBB);
      phi->addIncoming(NullVal, startBB);
      msgRet = RValue::getAggregate(phi);
    } else /* isComplex() */ {
      std::pair<llvm::Value*,llvm::Value*> v = msgRet.getComplexVal();
      llvm::PHINode *phi = Builder.CreatePHI(v.first->getType(), 2);
      phi->addIncoming(v.first, messageBB);
      phi->addIncoming(llvm::Constant::getNullValue(v.first->getType()),
          startBB);
      llvm::PHINode *phi2 = Builder.CreatePHI(v.second->getType(), 2);
      phi2->addIncoming(v.second, messageBB);
      phi2->addIncoming(llvm::Constant::getNullValue(v.second->getType()),
          startBB);
      msgRet = RValue::getComplex(phi, phi2);
    }
  }
  return msgRet;
}

/// Generates a MethodList.  Used in construction of a objc_class and
/// objc_category structures.
llvm::Constant *CGObjCGNU::
GenerateMethodList(const StringRef &ClassName,
                   const StringRef &CategoryName,
                   ArrayRef<Selector> MethodSels,
                   ArrayRef<llvm::Constant *> MethodTypes,
                   bool isClassMethodList) {
  if (MethodSels.empty())
    return NULLPtr;
  // Get the method structure type.
  llvm::StructType *ObjCMethodTy = llvm::StructType::get(
    PtrToInt8Ty, // Really a selector, but the runtime creates it us.
    PtrToInt8Ty, // Method types
    IMPTy, //Method pointer
    NULL);
  std::vector<llvm::Constant*> Methods;
  std::vector<llvm::Constant*> Elements;
  for (unsigned int i = 0, e = MethodTypes.size(); i < e; ++i) {
    Elements.clear();
    llvm::Constant *Method =
      TheModule.getFunction(SymbolNameForMethod(ClassName, CategoryName,
                                                MethodSels[i],
                                                isClassMethodList));
    assert(Method && "Can't generate metadata for method that doesn't exist");
    llvm::Constant *C = MakeConstantString(MethodSels[i].getAsString());
    Elements.push_back(C);
    Elements.push_back(MethodTypes[i]);
    Method = llvm::ConstantExpr::getBitCast(Method,
        IMPTy);
    Elements.push_back(Method);
    Methods.push_back(llvm::ConstantStruct::get(ObjCMethodTy, Elements));
  }

  // Array of method structures
  llvm::ArrayType *ObjCMethodArrayTy = llvm::ArrayType::get(ObjCMethodTy,
                                                            Methods.size());
  llvm::Constant *MethodArray = llvm::ConstantArray::get(ObjCMethodArrayTy,
                                                         Methods);

  // Structure containing list pointer, array and array count
  llvm::StructType *ObjCMethodListTy = llvm::StructType::create(VMContext);
  llvm::Type *NextPtrTy = llvm::PointerType::getUnqual(ObjCMethodListTy);
  ObjCMethodListTy->setBody(
      NextPtrTy,
      IntTy,
      ObjCMethodArrayTy,
      NULL);

  Methods.clear();
  Methods.push_back(llvm::ConstantPointerNull::get(
        llvm::PointerType::getUnqual(ObjCMethodListTy)));
  Methods.push_back(llvm::ConstantInt::get(Int32Ty, MethodTypes.size()));
  Methods.push_back(MethodArray);

  // Create an instance of the structure
  return MakeGlobal(ObjCMethodListTy, Methods, ".objc_method_list");
}

/// Generates an IvarList.  Used in construction of a objc_class.
llvm::Constant *CGObjCGNU::
GenerateIvarList(ArrayRef<llvm::Constant *> IvarNames,
                 ArrayRef<llvm::Constant *> IvarTypes,
                 ArrayRef<llvm::Constant *> IvarOffsets) {
  if (IvarNames.size() == 0)
    return NULLPtr;
  // Get the method structure type.
  llvm::StructType *ObjCIvarTy = llvm::StructType::get(
    PtrToInt8Ty,
    PtrToInt8Ty,
    IntTy,
    NULL);
  std::vector<llvm::Constant*> Ivars;
  std::vector<llvm::Constant*> Elements;
  for (unsigned int i = 0, e = IvarNames.size() ; i < e ; i++) {
    Elements.clear();
    Elements.push_back(IvarNames[i]);
    Elements.push_back(IvarTypes[i]);
    Elements.push_back(IvarOffsets[i]);
    Ivars.push_back(llvm::ConstantStruct::get(ObjCIvarTy, Elements));
  }

  // Array of method structures
  llvm::ArrayType *ObjCIvarArrayTy = llvm::ArrayType::get(ObjCIvarTy,
      IvarNames.size());


  Elements.clear();
  Elements.push_back(llvm::ConstantInt::get(IntTy, (int)IvarNames.size()));
  Elements.push_back(llvm::ConstantArray::get(ObjCIvarArrayTy, Ivars));
  // Structure containing array and array count
  llvm::StructType *ObjCIvarListTy = llvm::StructType::get(IntTy,
    ObjCIvarArrayTy,
    NULL);

  // Create an instance of the structure
  return MakeGlobal(ObjCIvarListTy, Elements, ".objc_ivar_list");
}

/// Generate a class structure
llvm::Constant *CGObjCGNU::GenerateClassStructure(
    llvm::Constant *MetaClass,
    llvm::Constant *SuperClass,
    unsigned info,
    const char *Name,
    llvm::Constant *Version,
    llvm::Constant *InstanceSize,
    llvm::Constant *IVars,
    llvm::Constant *Methods,
    llvm::Constant *Protocols,
    llvm::Constant *IvarOffsets,
    llvm::Constant *Properties,
    llvm::Constant *StrongIvarBitmap,
    llvm::Constant *WeakIvarBitmap,
    bool isMeta) {
  // Set up the class structure
  // Note:  Several of these are char*s when they should be ids.  This is
  // because the runtime performs this translation on load.
  //
  // Fields marked New ABI are part of the GNUstep runtime.  We emit them
  // anyway; the classes will still work with the GNU runtime, they will just
  // be ignored.
  llvm::StructType *ClassTy = llvm::StructType::get(
      PtrToInt8Ty,        // isa 
      PtrToInt8Ty,        // super_class
      PtrToInt8Ty,        // name
      LongTy,             // version
      LongTy,             // info
      LongTy,             // instance_size
      IVars->getType(),   // ivars
      Methods->getType(), // methods
      // These are all filled in by the runtime, so we pretend
      PtrTy,              // dtable
      PtrTy,              // subclass_list
      PtrTy,              // sibling_class
      PtrTy,              // protocols
      PtrTy,              // gc_object_type
      // New ABI:
      LongTy,                 // abi_version
      IvarOffsets->getType(), // ivar_offsets
      Properties->getType(),  // properties
      IntPtrTy,               // strong_pointers
      IntPtrTy,               // weak_pointers
      NULL);
  llvm::Constant *Zero = llvm::ConstantInt::get(LongTy, 0);
  // Fill in the structure
  std::vector<llvm::Constant*> Elements;
  Elements.push_back(llvm::ConstantExpr::getBitCast(MetaClass, PtrToInt8Ty));
  Elements.push_back(SuperClass);
  Elements.push_back(MakeConstantString(Name, ".class_name"));
  Elements.push_back(Zero);
  Elements.push_back(llvm::ConstantInt::get(LongTy, info));
  if (isMeta) {
    llvm::DataLayout td(&TheModule);
    Elements.push_back(
        llvm::ConstantInt::get(LongTy,
                               td.getTypeSizeInBits(ClassTy) /
                                 CGM.getContext().getCharWidth()));
  } else
    Elements.push_back(InstanceSize);
  Elements.push_back(IVars);
  Elements.push_back(Methods);
  Elements.push_back(NULLPtr);
  Elements.push_back(NULLPtr);
  Elements.push_back(NULLPtr);
  Elements.push_back(llvm::ConstantExpr::getBitCast(Protocols, PtrTy));
  Elements.push_back(NULLPtr);
  Elements.push_back(llvm::ConstantInt::get(LongTy, 1));
  Elements.push_back(IvarOffsets);
  Elements.push_back(Properties);
  Elements.push_back(StrongIvarBitmap);
  Elements.push_back(WeakIvarBitmap);
  // Create an instance of the structure
  // This is now an externally visible symbol, so that we can speed up class
  // messages in the next ABI.  We may already have some weak references to
  // this, so check and fix them properly.
  std::string ClassSym((isMeta ? "_OBJC_METACLASS_": "_OBJC_CLASS_") +
          std::string(Name));
  llvm::GlobalVariable *ClassRef = TheModule.getNamedGlobal(ClassSym);
  llvm::Constant *Class = MakeGlobal(ClassTy, Elements, ClassSym,
          llvm::GlobalValue::ExternalLinkage);
  if (ClassRef) {
      ClassRef->replaceAllUsesWith(llvm::ConstantExpr::getBitCast(Class,
                  ClassRef->getType()));
      ClassRef->removeFromParent();
      Class->setName(ClassSym);
  }
  return Class;
}

llvm::Constant *CGObjCGNU::
GenerateProtocolMethodList(ArrayRef<llvm::Constant *> MethodNames,
                           ArrayRef<llvm::Constant *> MethodTypes) {
  // Get the method structure type.
  llvm::StructType *ObjCMethodDescTy = llvm::StructType::get(
    PtrToInt8Ty, // Really a selector, but the runtime does the casting for us.
    PtrToInt8Ty,
    NULL);
  std::vector<llvm::Constant*> Methods;
  std::vector<llvm::Constant*> Elements;
  for (unsigned int i = 0, e = MethodTypes.size() ; i < e ; i++) {
    Elements.clear();
    Elements.push_back(MethodNames[i]);
    Elements.push_back(MethodTypes[i]);
    Methods.push_back(llvm::ConstantStruct::get(ObjCMethodDescTy, Elements));
  }
  llvm::ArrayType *ObjCMethodArrayTy = llvm::ArrayType::get(ObjCMethodDescTy,
      MethodNames.size());
  llvm::Constant *Array = llvm::ConstantArray::get(ObjCMethodArrayTy,
                                                   Methods);
  llvm::StructType *ObjCMethodDescListTy = llvm::StructType::get(
      IntTy, ObjCMethodArrayTy, NULL);
  Methods.clear();
  Methods.push_back(llvm::ConstantInt::get(IntTy, MethodNames.size()));
  Methods.push_back(Array);
  return MakeGlobal(ObjCMethodDescListTy, Methods, ".objc_method_list");
}

// Create the protocol list structure used in classes, categories and so on
llvm::Constant *CGObjCGNU::GenerateProtocolList(ArrayRef<std::string>Protocols){
  llvm::ArrayType *ProtocolArrayTy = llvm::ArrayType::get(PtrToInt8Ty,
      Protocols.size());
  llvm::StructType *ProtocolListTy = llvm::StructType::get(
      PtrTy, //Should be a recurisve pointer, but it's always NULL here.
      SizeTy,
      ProtocolArrayTy,
      NULL);
  std::vector<llvm::Constant*> Elements;
  for (const std::string *iter = Protocols.begin(), *endIter = Protocols.end();
      iter != endIter ; iter++) {
    llvm::Constant *protocol = 0;
    llvm::StringMap<llvm::Constant*>::iterator value =
      ExistingProtocols.find(*iter);
    if (value == ExistingProtocols.end()) {
      protocol = GenerateEmptyProtocol(*iter);
    } else {
      protocol = value->getValue();
    }
    llvm::Constant *Ptr = llvm::ConstantExpr::getBitCast(protocol,
                                                           PtrToInt8Ty);
    Elements.push_back(Ptr);
  }
  llvm::Constant * ProtocolArray = llvm::ConstantArray::get(ProtocolArrayTy,
      Elements);
  Elements.clear();
  Elements.push_back(NULLPtr);
  Elements.push_back(llvm::ConstantInt::get(LongTy, Protocols.size()));
  Elements.push_back(ProtocolArray);
  return MakeGlobal(ProtocolListTy, Elements, ".objc_protocol_list");
}

llvm::Value *CGObjCGNU::GenerateProtocolRef(CodeGenFunction &CGF,
                                            const ObjCProtocolDecl *PD) {
  llvm::Value *protocol = ExistingProtocols[PD->getNameAsString()];
  llvm::Type *T =
    CGM.getTypes().ConvertType(CGM.getContext().getObjCProtoType());
  return CGF.Builder.CreateBitCast(protocol, llvm::PointerType::getUnqual(T));
}

llvm::Constant *CGObjCGNU::GenerateEmptyProtocol(
  const std::string &ProtocolName) {
  SmallVector<std::string, 0> EmptyStringVector;
  SmallVector<llvm::Constant*, 0> EmptyConstantVector;

  llvm::Constant *ProtocolList = GenerateProtocolList(EmptyStringVector);
  llvm::Constant *MethodList =
    GenerateProtocolMethodList(EmptyConstantVector, EmptyConstantVector);
  // Protocols are objects containing lists of the methods implemented and
  // protocols adopted.
  llvm::StructType *ProtocolTy = llvm::StructType::get(IdTy,
      PtrToInt8Ty,
      ProtocolList->getType(),
      MethodList->getType(),
      MethodList->getType(),
      MethodList->getType(),
      MethodList->getType(),
      NULL);
  std::vector<llvm::Constant*> Elements;
  // The isa pointer must be set to a magic number so the runtime knows it's
  // the correct layout.
  Elements.push_back(llvm::ConstantExpr::getIntToPtr(
        llvm::ConstantInt::get(Int32Ty, ProtocolVersion), IdTy));
  Elements.push_back(MakeConstantString(ProtocolName, ".objc_protocol_name"));
  Elements.push_back(ProtocolList);
  Elements.push_back(MethodList);
  Elements.push_back(MethodList);
  Elements.push_back(MethodList);
  Elements.push_back(MethodList);
  return MakeGlobal(ProtocolTy, Elements, ".objc_protocol");
}

void CGObjCGNU::GenerateProtocol(const ObjCProtocolDecl *PD) {
  ASTContext &Context = CGM.getContext();
  std::string ProtocolName = PD->getNameAsString();
  
  // Use the protocol definition, if there is one.
  if (const ObjCProtocolDecl *Def = PD->getDefinition())
    PD = Def;

  SmallVector<std::string, 16> Protocols;
  for (ObjCProtocolDecl::protocol_iterator PI = PD->protocol_begin(),
       E = PD->protocol_end(); PI != E; ++PI)
    Protocols.push_back((*PI)->getNameAsString());
  SmallVector<llvm::Constant*, 16> InstanceMethodNames;
  SmallVector<llvm::Constant*, 16> InstanceMethodTypes;
  SmallVector<llvm::Constant*, 16> OptionalInstanceMethodNames;
  SmallVector<llvm::Constant*, 16> OptionalInstanceMethodTypes;
  for (ObjCProtocolDecl::instmeth_iterator iter = PD->instmeth_begin(),
       E = PD->instmeth_end(); iter != E; iter++) {
    std::string TypeStr;
    Context.getObjCEncodingForMethodDecl(*iter, TypeStr);
    if ((*iter)->getImplementationControl() == ObjCMethodDecl::Optional) {
      OptionalInstanceMethodNames.push_back(
          MakeConstantString((*iter)->getSelector().getAsString()));
      OptionalInstanceMethodTypes.push_back(MakeConstantString(TypeStr));
    } else {
      InstanceMethodNames.push_back(
          MakeConstantString((*iter)->getSelector().getAsString()));
      InstanceMethodTypes.push_back(MakeConstantString(TypeStr));
    }
  }
  // Collect information about class methods:
  SmallVector<llvm::Constant*, 16> ClassMethodNames;
  SmallVector<llvm::Constant*, 16> ClassMethodTypes;
  SmallVector<llvm::Constant*, 16> OptionalClassMethodNames;
  SmallVector<llvm::Constant*, 16> OptionalClassMethodTypes;
  for (ObjCProtocolDecl::classmeth_iterator
         iter = PD->classmeth_begin(), endIter = PD->classmeth_end();
       iter != endIter ; iter++) {
    std::string TypeStr;
    Context.getObjCEncodingForMethodDecl((*iter),TypeStr);
    if ((*iter)->getImplementationControl() == ObjCMethodDecl::Optional) {
      OptionalClassMethodNames.push_back(
          MakeConstantString((*iter)->getSelector().getAsString()));
      OptionalClassMethodTypes.push_back(MakeConstantString(TypeStr));
    } else {
      ClassMethodNames.push_back(
          MakeConstantString((*iter)->getSelector().getAsString()));
      ClassMethodTypes.push_back(MakeConstantString(TypeStr));
    }
  }

  llvm::Constant *ProtocolList = GenerateProtocolList(Protocols);
  llvm::Constant *InstanceMethodList =
    GenerateProtocolMethodList(InstanceMethodNames, InstanceMethodTypes);
  llvm::Constant *ClassMethodList =
    GenerateProtocolMethodList(ClassMethodNames, ClassMethodTypes);
  llvm::Constant *OptionalInstanceMethodList =
    GenerateProtocolMethodList(OptionalInstanceMethodNames,
            OptionalInstanceMethodTypes);
  llvm::Constant *OptionalClassMethodList =
    GenerateProtocolMethodList(OptionalClassMethodNames,
            OptionalClassMethodTypes);

  // Property metadata: name, attributes, isSynthesized, setter name, setter
  // types, getter name, getter types.
  // The isSynthesized value is always set to 0 in a protocol.  It exists to
  // simplify the runtime library by allowing it to use the same data
  // structures for protocol metadata everywhere.
  llvm::StructType *PropertyMetadataTy = llvm::StructType::get(
          PtrToInt8Ty, Int8Ty, Int8Ty, Int8Ty, Int8Ty, PtrToInt8Ty,
          PtrToInt8Ty, PtrToInt8Ty, PtrToInt8Ty, NULL);
  std::vector<llvm::Constant*> Properties;
  std::vector<llvm::Constant*> OptionalProperties;

  // Add all of the property methods need adding to the method list and to the
  // property metadata list.
  for (ObjCContainerDecl::prop_iterator
         iter = PD->prop_begin(), endIter = PD->prop_end();
       iter != endIter ; iter++) {
    std::vector<llvm::Constant*> Fields;
    ObjCPropertyDecl *property = *iter;

    Fields.push_back(MakePropertyEncodingString(property, 0));
    PushPropertyAttributes(Fields, property);

    if (ObjCMethodDecl *getter = property->getGetterMethodDecl()) {
      std::string TypeStr;
      Context.getObjCEncodingForMethodDecl(getter,TypeStr);
      llvm::Constant *TypeEncoding = MakeConstantString(TypeStr);
      InstanceMethodTypes.push_back(TypeEncoding);
      Fields.push_back(MakeConstantString(getter->getSelector().getAsString()));
      Fields.push_back(TypeEncoding);
    } else {
      Fields.push_back(NULLPtr);
      Fields.push_back(NULLPtr);
    }
    if (ObjCMethodDecl *setter = property->getSetterMethodDecl()) {
      std::string TypeStr;
      Context.getObjCEncodingForMethodDecl(setter,TypeStr);
      llvm::Constant *TypeEncoding = MakeConstantString(TypeStr);
      InstanceMethodTypes.push_back(TypeEncoding);
      Fields.push_back(MakeConstantString(setter->getSelector().getAsString()));
      Fields.push_back(TypeEncoding);
    } else {
      Fields.push_back(NULLPtr);
      Fields.push_back(NULLPtr);
    }
    if (property->getPropertyImplementation() == ObjCPropertyDecl::Optional) {
      OptionalProperties.push_back(llvm::ConstantStruct::get(PropertyMetadataTy, Fields));
    } else {
      Properties.push_back(llvm::ConstantStruct::get(PropertyMetadataTy, Fields));
    }
  }
  llvm::Constant *PropertyArray = llvm::ConstantArray::get(
      llvm::ArrayType::get(PropertyMetadataTy, Properties.size()), Properties);
  llvm::Constant* PropertyListInitFields[] =
    {llvm::ConstantInt::get(IntTy, Properties.size()), NULLPtr, PropertyArray};

  llvm::Constant *PropertyListInit =
      llvm::ConstantStruct::getAnon(PropertyListInitFields);
  llvm::Constant *PropertyList = new llvm::GlobalVariable(TheModule,
      PropertyListInit->getType(), false, llvm::GlobalValue::InternalLinkage,
      PropertyListInit, ".objc_property_list");

  llvm::Constant *OptionalPropertyArray =
      llvm::ConstantArray::get(llvm::ArrayType::get(PropertyMetadataTy,
          OptionalProperties.size()) , OptionalProperties);
  llvm::Constant* OptionalPropertyListInitFields[] = {
      llvm::ConstantInt::get(IntTy, OptionalProperties.size()), NULLPtr,
      OptionalPropertyArray };

  llvm::Constant *OptionalPropertyListInit =
      llvm::ConstantStruct::getAnon(OptionalPropertyListInitFields);
  llvm::Constant *OptionalPropertyList = new llvm::GlobalVariable(TheModule,
          OptionalPropertyListInit->getType(), false,
          llvm::GlobalValue::InternalLinkage, OptionalPropertyListInit,
          ".objc_property_list");

  // Protocols are objects containing lists of the methods implemented and
  // protocols adopted.
  llvm::StructType *ProtocolTy = llvm::StructType::get(IdTy,
      PtrToInt8Ty,
      ProtocolList->getType(),
      InstanceMethodList->getType(),
      ClassMethodList->getType(),
      OptionalInstanceMethodList->getType(),
      OptionalClassMethodList->getType(),
      PropertyList->getType(),
      OptionalPropertyList->getType(),
      NULL);
  std::vector<llvm::Constant*> Elements;
  // The isa pointer must be set to a magic number so the runtime knows it's
  // the correct layout.
  Elements.push_back(llvm::ConstantExpr::getIntToPtr(
        llvm::ConstantInt::get(Int32Ty, ProtocolVersion), IdTy));
  Elements.push_back(MakeConstantString(ProtocolName, ".objc_protocol_name"));
  Elements.push_back(ProtocolList);
  Elements.push_back(InstanceMethodList);
  Elements.push_back(ClassMethodList);
  Elements.push_back(OptionalInstanceMethodList);
  Elements.push_back(OptionalClassMethodList);
  Elements.push_back(PropertyList);
  Elements.push_back(OptionalPropertyList);
  ExistingProtocols[ProtocolName] =
    llvm::ConstantExpr::getBitCast(MakeGlobal(ProtocolTy, Elements,
          ".objc_protocol"), IdTy);
}
void CGObjCGNU::GenerateProtocolHolderCategory() {
  // Collect information about instance methods
  SmallVector<Selector, 1> MethodSels;
  SmallVector<llvm::Constant*, 1> MethodTypes;

  std::vector<llvm::Constant*> Elements;
  const std::string ClassName = "__ObjC_Protocol_Holder_Ugly_Hack";
  const std::string CategoryName = "AnotherHack";
  Elements.push_back(MakeConstantString(CategoryName));
  Elements.push_back(MakeConstantString(ClassName));
  // Instance method list
  Elements.push_back(llvm::ConstantExpr::getBitCast(GenerateMethodList(
          ClassName, CategoryName, MethodSels, MethodTypes, false), PtrTy));
  // Class method list
  Elements.push_back(llvm::ConstantExpr::getBitCast(GenerateMethodList(
          ClassName, CategoryName, MethodSels, MethodTypes, true), PtrTy));
  // Protocol list
  llvm::ArrayType *ProtocolArrayTy = llvm::ArrayType::get(PtrTy,
      ExistingProtocols.size());
  llvm::StructType *ProtocolListTy = llvm::StructType::get(
      PtrTy, //Should be a recurisve pointer, but it's always NULL here.
      SizeTy,
      ProtocolArrayTy,
      NULL);
  std::vector<llvm::Constant*> ProtocolElements;
  for (llvm::StringMapIterator<llvm::Constant*> iter =
       ExistingProtocols.begin(), endIter = ExistingProtocols.end();
       iter != endIter ; iter++) {
    llvm::Constant *Ptr = llvm::ConstantExpr::getBitCast(iter->getValue(),
            PtrTy);
    ProtocolElements.push_back(Ptr);
  }
  llvm::Constant * ProtocolArray = llvm::ConstantArray::get(ProtocolArrayTy,
      ProtocolElements);
  ProtocolElements.clear();
  ProtocolElements.push_back(NULLPtr);
  ProtocolElements.push_back(llvm::ConstantInt::get(LongTy,
              ExistingProtocols.size()));
  ProtocolElements.push_back(ProtocolArray);
  Elements.push_back(llvm::ConstantExpr::getBitCast(MakeGlobal(ProtocolListTy,
                  ProtocolElements, ".objc_protocol_list"), PtrTy));
  Categories.push_back(llvm::ConstantExpr::getBitCast(
        MakeGlobal(llvm::StructType::get(PtrToInt8Ty, PtrToInt8Ty,
            PtrTy, PtrTy, PtrTy, NULL), Elements), PtrTy));
}

/// Libobjc2 uses a bitfield representation where small(ish) bitfields are
/// stored in a 64-bit value with the low bit set to 1 and the remaining 63
/// bits set to their values, LSB first, while larger ones are stored in a
/// structure of this / form:
/// 
/// struct { int32_t length; int32_t values[length]; };
///
/// The values in the array are stored in host-endian format, with the least
/// significant bit being assumed to come first in the bitfield.  Therefore, a
/// bitfield with the 64th bit set will be (int64_t)&{ 2, [0, 1<<31] }, while a
/// bitfield / with the 63rd bit set will be 1<<64.
llvm::Constant *CGObjCGNU::MakeBitField(ArrayRef<bool> bits) {
  int bitCount = bits.size();
  int ptrBits =
        (TheModule.getPointerSize() == llvm::Module::Pointer32) ? 32 : 64;
  if (bitCount < ptrBits) {
    uint64_t val = 1;
    for (int i=0 ; i<bitCount ; ++i) {
      if (bits[i]) val |= 1ULL<<(i+1);
    }
    return llvm::ConstantInt::get(IntPtrTy, val);
  }
  SmallVector<llvm::Constant *, 8> values;
  int v=0;
  while (v < bitCount) {
    int32_t word = 0;
    for (int i=0 ; (i<32) && (v<bitCount)  ; ++i) {
      if (bits[v]) word |= 1<<i;
      v++;
    }
    values.push_back(llvm::ConstantInt::get(Int32Ty, word));
  }
  llvm::ArrayType *arrayTy = llvm::ArrayType::get(Int32Ty, values.size());
  llvm::Constant *array = llvm::ConstantArray::get(arrayTy, values);
  llvm::Constant *fields[2] = {
      llvm::ConstantInt::get(Int32Ty, values.size()),
      array };
  llvm::Constant *GS = MakeGlobal(llvm::StructType::get(Int32Ty, arrayTy,
        NULL), fields);
  llvm::Constant *ptr = llvm::ConstantExpr::getPtrToInt(GS, IntPtrTy);
  return ptr;
}

void CGObjCGNU::GenerateCategory(const ObjCCategoryImplDecl *OCD) {
  std::string ClassName = OCD->getClassInterface()->getNameAsString();
  std::string CategoryName = OCD->getNameAsString();
  // Collect information about instance methods
  SmallVector<Selector, 16> InstanceMethodSels;
  SmallVector<llvm::Constant*, 16> InstanceMethodTypes;
  for (ObjCCategoryImplDecl::instmeth_iterator
         iter = OCD->instmeth_begin(), endIter = OCD->instmeth_end();
       iter != endIter ; iter++) {
    InstanceMethodSels.push_back((*iter)->getSelector());
    std::string TypeStr;
    CGM.getContext().getObjCEncodingForMethodDecl(*iter,TypeStr);
    InstanceMethodTypes.push_back(MakeConstantString(TypeStr));
  }

  // Collect information about class methods
  SmallVector<Selector, 16> ClassMethodSels;
  SmallVector<llvm::Constant*, 16> ClassMethodTypes;
  for (ObjCCategoryImplDecl::classmeth_iterator
         iter = OCD->classmeth_begin(), endIter = OCD->classmeth_end();
       iter != endIter ; iter++) {
    ClassMethodSels.push_back((*iter)->getSelector());
    std::string TypeStr;
    CGM.getContext().getObjCEncodingForMethodDecl(*iter,TypeStr);
    ClassMethodTypes.push_back(MakeConstantString(TypeStr));
  }

  // Collect the names of referenced protocols
  SmallVector<std::string, 16> Protocols;
  const ObjCCategoryDecl *CatDecl = OCD->getCategoryDecl();
  const ObjCList<ObjCProtocolDecl> &Protos = CatDecl->getReferencedProtocols();
  for (ObjCList<ObjCProtocolDecl>::iterator I = Protos.begin(),
       E = Protos.end(); I != E; ++I)
    Protocols.push_back((*I)->getNameAsString());

  std::vector<llvm::Constant*> Elements;
  Elements.push_back(MakeConstantString(CategoryName));
  Elements.push_back(MakeConstantString(ClassName));
  // Instance method list
  Elements.push_back(llvm::ConstantExpr::getBitCast(GenerateMethodList(
          ClassName, CategoryName, InstanceMethodSels, InstanceMethodTypes,
          false), PtrTy));
  // Class method list
  Elements.push_back(llvm::ConstantExpr::getBitCast(GenerateMethodList(
          ClassName, CategoryName, ClassMethodSels, ClassMethodTypes, true),
        PtrTy));
  // Protocol list
  Elements.push_back(llvm::ConstantExpr::getBitCast(
        GenerateProtocolList(Protocols), PtrTy));
  Categories.push_back(llvm::ConstantExpr::getBitCast(
        MakeGlobal(llvm::StructType::get(PtrToInt8Ty, PtrToInt8Ty,
            PtrTy, PtrTy, PtrTy, NULL), Elements), PtrTy));
}

llvm::Constant *CGObjCGNU::GeneratePropertyList(const ObjCImplementationDecl *OID,
        SmallVectorImpl<Selector> &InstanceMethodSels,
        SmallVectorImpl<llvm::Constant*> &InstanceMethodTypes) {
  ASTContext &Context = CGM.getContext();
  // Property metadata: name, attributes, attributes2, padding1, padding2,
  // setter name, setter types, getter name, getter types.
  llvm::StructType *PropertyMetadataTy = llvm::StructType::get(
          PtrToInt8Ty, Int8Ty, Int8Ty, Int8Ty, Int8Ty, PtrToInt8Ty,
          PtrToInt8Ty, PtrToInt8Ty, PtrToInt8Ty, NULL);
  std::vector<llvm::Constant*> Properties;

  // Add all of the property methods need adding to the method list and to the
  // property metadata list.
  for (ObjCImplDecl::propimpl_iterator
         iter = OID->propimpl_begin(), endIter = OID->propimpl_end();
       iter != endIter ; iter++) {
    std::vector<llvm::Constant*> Fields;
    ObjCPropertyDecl *property = iter->getPropertyDecl();
    ObjCPropertyImplDecl *propertyImpl = *iter;
    bool isSynthesized = (propertyImpl->getPropertyImplementation() == 
        ObjCPropertyImplDecl::Synthesize);
    bool isDynamic = (propertyImpl->getPropertyImplementation() == 
        ObjCPropertyImplDecl::Dynamic);

    Fields.push_back(MakePropertyEncodingString(property, OID));
    PushPropertyAttributes(Fields, property, isSynthesized, isDynamic);
    if (ObjCMethodDecl *getter = property->getGetterMethodDecl()) {
      std::string TypeStr;
      Context.getObjCEncodingForMethodDecl(getter,TypeStr);
      llvm::Constant *TypeEncoding = MakeConstantString(TypeStr);
      if (isSynthesized) {
        InstanceMethodTypes.push_back(TypeEncoding);
        InstanceMethodSels.push_back(getter->getSelector());
      }
      Fields.push_back(MakeConstantString(getter->getSelector().getAsString()));
      Fields.push_back(TypeEncoding);
    } else {
      Fields.push_back(NULLPtr);
      Fields.push_back(NULLPtr);
    }
    if (ObjCMethodDecl *setter = property->getSetterMethodDecl()) {
      std::string TypeStr;
      Context.getObjCEncodingForMethodDecl(setter,TypeStr);
      llvm::Constant *TypeEncoding = MakeConstantString(TypeStr);
      if (isSynthesized) {
        InstanceMethodTypes.push_back(TypeEncoding);
        InstanceMethodSels.push_back(setter->getSelector());
      }
      Fields.push_back(MakeConstantString(setter->getSelector().getAsString()));
      Fields.push_back(TypeEncoding);
    } else {
      Fields.push_back(NULLPtr);
      Fields.push_back(NULLPtr);
    }
    Properties.push_back(llvm::ConstantStruct::get(PropertyMetadataTy, Fields));
  }
  llvm::ArrayType *PropertyArrayTy =
      llvm::ArrayType::get(PropertyMetadataTy, Properties.size());
  llvm::Constant *PropertyArray = llvm::ConstantArray::get(PropertyArrayTy,
          Properties);
  llvm::Constant* PropertyListInitFields[] =
    {llvm::ConstantInt::get(IntTy, Properties.size()), NULLPtr, PropertyArray};

  llvm::Constant *PropertyListInit =
      llvm::ConstantStruct::getAnon(PropertyListInitFields);
  return new llvm::GlobalVariable(TheModule, PropertyListInit->getType(), false,
          llvm::GlobalValue::InternalLinkage, PropertyListInit,
          ".objc_property_list");
}

void CGObjCGNU::RegisterAlias(const ObjCCompatibleAliasDecl *OAD) {
  // Get the class declaration for which the alias is specified.
  ObjCInterfaceDecl *ClassDecl =
    const_cast<ObjCInterfaceDecl *>(OAD->getClassInterface());
  std::string ClassName = ClassDecl->getNameAsString();
  std::string AliasName = OAD->getNameAsString();
  ClassAliases.push_back(ClassAliasPair(ClassName,AliasName));
}

void CGObjCGNU::GenerateClass(const ObjCImplementationDecl *OID) {
  ASTContext &Context = CGM.getContext();

  // Get the superclass name.
  const ObjCInterfaceDecl * SuperClassDecl =
    OID->getClassInterface()->getSuperClass();
  std::string SuperClassName;
  if (SuperClassDecl) {
    SuperClassName = SuperClassDecl->getNameAsString();
    EmitClassRef(SuperClassName);
  }

  // Get the class name
  ObjCInterfaceDecl *ClassDecl =
    const_cast<ObjCInterfaceDecl *>(OID->getClassInterface());
  std::string ClassName = ClassDecl->getNameAsString();
  // Emit the symbol that is used to generate linker errors if this class is
  // referenced in other modules but not declared.
  std::string classSymbolName = "__objc_class_name_" + ClassName;
  if (llvm::GlobalVariable *symbol =
      TheModule.getGlobalVariable(classSymbolName)) {
    symbol->setInitializer(llvm::ConstantInt::get(LongTy, 0));
  } else {
    new llvm::GlobalVariable(TheModule, LongTy, false,
    llvm::GlobalValue::ExternalLinkage, llvm::ConstantInt::get(LongTy, 0),
    classSymbolName);
  }

  // Get the size of instances.
  int instanceSize = 
    Context.getASTObjCImplementationLayout(OID).getSize().getQuantity();

  // Collect information about instance variables.
  SmallVector<llvm::Constant*, 16> IvarNames;
  SmallVector<llvm::Constant*, 16> IvarTypes;
  SmallVector<llvm::Constant*, 16> IvarOffsets;

  std::vector<llvm::Constant*> IvarOffsetValues;
  SmallVector<bool, 16> WeakIvars;
  SmallVector<bool, 16> StrongIvars;

  int superInstanceSize = !SuperClassDecl ? 0 :
    Context.getASTObjCInterfaceLayout(SuperClassDecl).getSize().getQuantity();
  // For non-fragile ivars, set the instance size to 0 - {the size of just this
  // class}.  The runtime will then set this to the correct value on load.
  if (CGM.getLangOpts().ObjCRuntime.isNonFragile()) {
    instanceSize = 0 - (instanceSize - superInstanceSize);
  }

  for (const ObjCIvarDecl *IVD = ClassDecl->all_declared_ivar_begin(); IVD;
       IVD = IVD->getNextIvar()) {
      // Store the name
      IvarNames.push_back(MakeConstantString(IVD->getNameAsString()));
      // Get the type encoding for this ivar
      std::string TypeStr;
      Context.getObjCEncodingForType(IVD->getType(), TypeStr);
      IvarTypes.push_back(MakeConstantString(TypeStr));
      // Get the offset
      uint64_t BaseOffset = ComputeIvarBaseOffset(CGM, OID, IVD);
      uint64_t Offset = BaseOffset;
      if (CGM.getLangOpts().ObjCRuntime.isNonFragile()) {
        Offset = BaseOffset - superInstanceSize;
      }
      llvm::Constant *OffsetValue = llvm::ConstantInt::get(IntTy, Offset);
      // Create the direct offset value
      std::string OffsetName = "__objc_ivar_offset_value_" + ClassName +"." +
          IVD->getNameAsString();
      llvm::GlobalVariable *OffsetVar = TheModule.getGlobalVariable(OffsetName);
      if (OffsetVar) {
        OffsetVar->setInitializer(OffsetValue);
        // If this is the real definition, change its linkage type so that
        // different modules will use this one, rather than their private
        // copy.
        OffsetVar->setLinkage(llvm::GlobalValue::ExternalLinkage);
      } else
        OffsetVar = new llvm::GlobalVariable(TheModule, IntTy,
          false, llvm::GlobalValue::ExternalLinkage,
          OffsetValue,
          "__objc_ivar_offset_value_" + ClassName +"." +
          IVD->getNameAsString());
      IvarOffsets.push_back(OffsetValue);
      IvarOffsetValues.push_back(OffsetVar);
      Qualifiers::ObjCLifetime lt = IVD->getType().getQualifiers().getObjCLifetime();
      switch (lt) {
        case Qualifiers::OCL_Strong:
          StrongIvars.push_back(true);
          WeakIvars.push_back(false);
          break;
        case Qualifiers::OCL_Weak:
          StrongIvars.push_back(false);
          WeakIvars.push_back(true);
          break;
        default:
          StrongIvars.push_back(false);
          WeakIvars.push_back(false);
      }
  }
  llvm::Constant *StrongIvarBitmap = MakeBitField(StrongIvars);
  llvm::Constant *WeakIvarBitmap = MakeBitField(WeakIvars);
  llvm::GlobalVariable *IvarOffsetArray =
    MakeGlobalArray(PtrToIntTy, IvarOffsetValues, ".ivar.offsets");


  // Collect information about instance methods
  SmallVector<Selector, 16> InstanceMethodSels;
  SmallVector<llvm::Constant*, 16> InstanceMethodTypes;
  for (ObjCImplementationDecl::instmeth_iterator
         iter = OID->instmeth_begin(), endIter = OID->instmeth_end();
       iter != endIter ; iter++) {
    InstanceMethodSels.push_back((*iter)->getSelector());
    std::string TypeStr;
    Context.getObjCEncodingForMethodDecl((*iter),TypeStr);
    InstanceMethodTypes.push_back(MakeConstantString(TypeStr));
  }

  llvm::Constant *Properties = GeneratePropertyList(OID, InstanceMethodSels,
          InstanceMethodTypes);


  // Collect information about class methods
  SmallVector<Selector, 16> ClassMethodSels;
  SmallVector<llvm::Constant*, 16> ClassMethodTypes;
  for (ObjCImplementationDecl::classmeth_iterator
         iter = OID->classmeth_begin(), endIter = OID->classmeth_end();
       iter != endIter ; iter++) {
    ClassMethodSels.push_back((*iter)->getSelector());
    std::string TypeStr;
    Context.getObjCEncodingForMethodDecl((*iter),TypeStr);
    ClassMethodTypes.push_back(MakeConstantString(TypeStr));
  }
  // Collect the names of referenced protocols
  SmallVector<std::string, 16> Protocols;
  for (ObjCInterfaceDecl::protocol_iterator
         I = ClassDecl->protocol_begin(),
         E = ClassDecl->protocol_end(); I != E; ++I)
    Protocols.push_back((*I)->getNameAsString());



  // Get the superclass pointer.
  llvm::Constant *SuperClass;
  if (!SuperClassName.empty()) {
    SuperClass = MakeConstantString(SuperClassName, ".super_class_name");
  } else {
    SuperClass = llvm::ConstantPointerNull::get(PtrToInt8Ty);
  }
  // Empty vector used to construct empty method lists
  SmallVector<llvm::Constant*, 1>  empty;
  // Generate the method and instance variable lists
  llvm::Constant *MethodList = GenerateMethodList(ClassName, "",
      InstanceMethodSels, InstanceMethodTypes, false);
  llvm::Constant *ClassMethodList = GenerateMethodList(ClassName, "",
      ClassMethodSels, ClassMethodTypes, true);
  llvm::Constant *IvarList = GenerateIvarList(IvarNames, IvarTypes,
      IvarOffsets);
  // Irrespective of whether we are compiling for a fragile or non-fragile ABI,
  // we emit a symbol containing the offset for each ivar in the class.  This
  // allows code compiled for the non-Fragile ABI to inherit from code compiled
  // for the legacy ABI, without causing problems.  The converse is also
  // possible, but causes all ivar accesses to be fragile.

  // Offset pointer for getting at the correct field in the ivar list when
  // setting up the alias.  These are: The base address for the global, the
  // ivar array (second field), the ivar in this list (set for each ivar), and
  // the offset (third field in ivar structure)
  llvm::Type *IndexTy = Int32Ty;
  llvm::Constant *offsetPointerIndexes[] = {Zeros[0],
      llvm::ConstantInt::get(IndexTy, 1), 0,
      llvm::ConstantInt::get(IndexTy, 2) };

  unsigned ivarIndex = 0;
  for (const ObjCIvarDecl *IVD = ClassDecl->all_declared_ivar_begin(); IVD;
       IVD = IVD->getNextIvar()) {
      const std::string Name = "__objc_ivar_offset_" + ClassName + '.'
          + IVD->getNameAsString();
      offsetPointerIndexes[2] = llvm::ConstantInt::get(IndexTy, ivarIndex);
      // Get the correct ivar field
      llvm::Constant *offsetValue = llvm::ConstantExpr::getGetElementPtr(
              IvarList, offsetPointerIndexes);
      // Get the existing variable, if one exists.
      llvm::GlobalVariable *offset = TheModule.getNamedGlobal(Name);
      if (offset) {
        offset->setInitializer(offsetValue);
        // If this is the real definition, change its linkage type so that
        // different modules will use this one, rather than their private
        // copy.
        offset->setLinkage(llvm::GlobalValue::ExternalLinkage);
      } else {
        // Add a new alias if there isn't one already.
        offset = new llvm::GlobalVariable(TheModule, offsetValue->getType(),
                false, llvm::GlobalValue::ExternalLinkage, offsetValue, Name);
        (void) offset; // Silence dead store warning.
      }
      ++ivarIndex;
  }
  llvm::Constant *ZeroPtr = llvm::ConstantInt::get(IntPtrTy, 0);
  //Generate metaclass for class methods
  llvm::Constant *MetaClassStruct = GenerateClassStructure(NULLPtr,
      NULLPtr, 0x12L, ClassName.c_str(), 0, Zeros[0], GenerateIvarList(
        empty, empty, empty), ClassMethodList, NULLPtr,
      NULLPtr, NULLPtr, ZeroPtr, ZeroPtr, true);

  // Generate the class structure
  llvm::Constant *ClassStruct =
    GenerateClassStructure(MetaClassStruct, SuperClass, 0x11L,
                           ClassName.c_str(), 0,
      llvm::ConstantInt::get(LongTy, instanceSize), IvarList,
      MethodList, GenerateProtocolList(Protocols), IvarOffsetArray,
      Properties, StrongIvarBitmap, WeakIvarBitmap);

  // Resolve the class aliases, if they exist.
  if (ClassPtrAlias) {
    ClassPtrAlias->replaceAllUsesWith(
        llvm::ConstantExpr::getBitCast(ClassStruct, IdTy));
    ClassPtrAlias->eraseFromParent();
    ClassPtrAlias = 0;
  }
  if (MetaClassPtrAlias) {
    MetaClassPtrAlias->replaceAllUsesWith(
        llvm::ConstantExpr::getBitCast(MetaClassStruct, IdTy));
    MetaClassPtrAlias->eraseFromParent();
    MetaClassPtrAlias = 0;
  }

  // Add class structure to list to be added to the symtab later
  ClassStruct = llvm::ConstantExpr::getBitCast(ClassStruct, PtrToInt8Ty);
  Classes.push_back(ClassStruct);
}


llvm::Function *CGObjCGNU::ModuleInitFunction() {
  // Only emit an ObjC load function if no Objective-C stuff has been called
  if (Classes.empty() && Categories.empty() && ConstantStrings.empty() &&
      ExistingProtocols.empty() && SelectorTable.empty())
    return NULL;

  // Add all referenced protocols to a category.
  GenerateProtocolHolderCategory();

  llvm::StructType *SelStructTy = dyn_cast<llvm::StructType>(
          SelectorTy->getElementType());
  llvm::Type *SelStructPtrTy = SelectorTy;
  if (SelStructTy == 0) {
    SelStructTy = llvm::StructType::get(PtrToInt8Ty, PtrToInt8Ty, NULL);
    SelStructPtrTy = llvm::PointerType::getUnqual(SelStructTy);
  }

  std::vector<llvm::Constant*> Elements;
  llvm::Constant *Statics = NULLPtr;
  // Generate statics list:
  if (ConstantStrings.size()) {
    llvm::ArrayType *StaticsArrayTy = llvm::ArrayType::get(PtrToInt8Ty,
        ConstantStrings.size() + 1);
    ConstantStrings.push_back(NULLPtr);

    StringRef StringClass = CGM.getLangOpts().ObjCConstantStringClass;

    if (StringClass.empty()) StringClass = "NXConstantString";

    Elements.push_back(MakeConstantString(StringClass,
                ".objc_static_class_name"));
    Elements.push_back(llvm::ConstantArray::get(StaticsArrayTy,
       ConstantStrings));
    llvm::StructType *StaticsListTy =
      llvm::StructType::get(PtrToInt8Ty, StaticsArrayTy, NULL);
    llvm::Type *StaticsListPtrTy =
      llvm::PointerType::getUnqual(StaticsListTy);
    Statics = MakeGlobal(StaticsListTy, Elements, ".objc_statics");
    llvm::ArrayType *StaticsListArrayTy =
      llvm::ArrayType::get(StaticsListPtrTy, 2);
    Elements.clear();
    Elements.push_back(Statics);
    Elements.push_back(llvm::Constant::getNullValue(StaticsListPtrTy));
    Statics = MakeGlobal(StaticsListArrayTy, Elements, ".objc_statics_ptr");
    Statics = llvm::ConstantExpr::getBitCast(Statics, PtrTy);
  }
  // Array of classes, categories, and constant objects
  llvm::ArrayType *ClassListTy = llvm::ArrayType::get(PtrToInt8Ty,
      Classes.size() + Categories.size()  + 2);
  llvm::StructType *SymTabTy = llvm::StructType::get(LongTy, SelStructPtrTy,
                                                     llvm::Type::getInt16Ty(VMContext),
                                                     llvm::Type::getInt16Ty(VMContext),
                                                     ClassListTy, NULL);

  Elements.clear();
  // Pointer to an array of selectors used in this module.
  std::vector<llvm::Constant*> Selectors;
  std::vector<llvm::GlobalAlias*> SelectorAliases;
  for (SelectorMap::iterator iter = SelectorTable.begin(),
      iterEnd = SelectorTable.end(); iter != iterEnd ; ++iter) {

    std::string SelNameStr = iter->first.getAsString();
    llvm::Constant *SelName = ExportUniqueString(SelNameStr, ".objc_sel_name");

    SmallVectorImpl<TypedSelector> &Types = iter->second;
    for (SmallVectorImpl<TypedSelector>::iterator i = Types.begin(),
        e = Types.end() ; i!=e ; i++) {

      llvm::Constant *SelectorTypeEncoding = NULLPtr;
      if (!i->first.empty())
        SelectorTypeEncoding = MakeConstantString(i->first, ".objc_sel_types");

      Elements.push_back(SelName);
      Elements.push_back(SelectorTypeEncoding);
      Selectors.push_back(llvm::ConstantStruct::get(SelStructTy, Elements));
      Elements.clear();

      // Store the selector alias for later replacement
      SelectorAliases.push_back(i->second);
    }
  }
  unsigned SelectorCount = Selectors.size();
  // NULL-terminate the selector list.  This should not actually be required,
  // because the selector list has a length field.  Unfortunately, the GCC
  // runtime decides to ignore the length field and expects a NULL terminator,
  // and GCC cooperates with this by always setting the length to 0.
  Elements.push_back(NULLPtr);
  Elements.push_back(NULLPtr);
  Selectors.push_back(llvm::ConstantStruct::get(SelStructTy, Elements));
  Elements.clear();

  // Number of static selectors
  Elements.push_back(llvm::ConstantInt::get(LongTy, SelectorCount));
  llvm::Constant *SelectorList = MakeGlobalArray(SelStructTy, Selectors,
          ".objc_selector_list");
  Elements.push_back(llvm::ConstantExpr::getBitCast(SelectorList,
    SelStructPtrTy));

  // Now that all of the static selectors exist, create pointers to them.
  for (unsigned int i=0 ; i<SelectorCount ; i++) {

    llvm::Constant *Idxs[] = {Zeros[0],
      llvm::ConstantInt::get(Int32Ty, i), Zeros[0]};
    // FIXME: We're generating redundant loads and stores here!
    llvm::Constant *SelPtr = llvm::ConstantExpr::getGetElementPtr(SelectorList,
        makeArrayRef(Idxs, 2));
    // If selectors are defined as an opaque type, cast the pointer to this
    // type.
    SelPtr = llvm::ConstantExpr::getBitCast(SelPtr, SelectorTy);
    SelectorAliases[i]->replaceAllUsesWith(SelPtr);
    SelectorAliases[i]->eraseFromParent();
  }

  // Number of classes defined.
  Elements.push_back(llvm::ConstantInt::get(llvm::Type::getInt16Ty(VMContext),
        Classes.size()));
  // Number of categories defined
  Elements.push_back(llvm::ConstantInt::get(llvm::Type::getInt16Ty(VMContext),
        Categories.size()));
  // Create an array of classes, then categories, then static object instances
  Classes.insert(Classes.end(), Categories.begin(), Categories.end());
  //  NULL-terminated list of static object instances (mainly constant strings)
  Classes.push_back(Statics);
  Classes.push_back(NULLPtr);
  llvm::Constant *ClassList = llvm::ConstantArray::get(ClassListTy, Classes);
  Elements.push_back(ClassList);
  // Construct the symbol table
  llvm::Constant *SymTab= MakeGlobal(SymTabTy, Elements);

  // The symbol table is contained in a module which has some version-checking
  // constants
  llvm::StructType * ModuleTy = llvm::StructType::get(LongTy, LongTy,
      PtrToInt8Ty, llvm::PointerType::getUnqual(SymTabTy), 
      (RuntimeVersion >= 10) ? IntTy : NULL, NULL);
  Elements.clear();
  // Runtime version, used for ABI compatibility checking.
  Elements.push_back(llvm::ConstantInt::get(LongTy, RuntimeVersion));
  // sizeof(ModuleTy)
  llvm::DataLayout td(&TheModule);
  Elements.push_back(
    llvm::ConstantInt::get(LongTy,
                           td.getTypeSizeInBits(ModuleTy) /
                             CGM.getContext().getCharWidth()));

  // The path to the source file where this module was declared
  SourceManager &SM = CGM.getContext().getSourceManager();
  const FileEntry *mainFile = SM.getFileEntryForID(SM.getMainFileID());
  std::string path =
    std::string(mainFile->getDir()->getName()) + '/' + mainFile->getName();
  Elements.push_back(MakeConstantString(path, ".objc_source_file_name"));
  Elements.push_back(SymTab);

  if (RuntimeVersion >= 10)
    switch (CGM.getLangOpts().getGC()) {
      case LangOptions::GCOnly:
        Elements.push_back(llvm::ConstantInt::get(IntTy, 2));
        break;
      case LangOptions::NonGC:
        if (CGM.getLangOpts().ObjCAutoRefCount)
          Elements.push_back(llvm::ConstantInt::get(IntTy, 1));
        else
          Elements.push_back(llvm::ConstantInt::get(IntTy, 0));
        break;
      case LangOptions::HybridGC:
          Elements.push_back(llvm::ConstantInt::get(IntTy, 1));
        break;
    }

  llvm::Value *Module = MakeGlobal(ModuleTy, Elements);

  // Create the load function calling the runtime entry point with the module
  // structure
  llvm::Function * LoadFunction = llvm::Function::Create(
      llvm::FunctionType::get(llvm::Type::getVoidTy(VMContext), false),
      llvm::GlobalValue::InternalLinkage, ".objc_load_function",
      &TheModule);
  llvm::BasicBlock *EntryBB =
      llvm::BasicBlock::Create(VMContext, "entry", LoadFunction);
  CGBuilderTy Builder(VMContext);
  Builder.SetInsertPoint(EntryBB);

  llvm::FunctionType *FT =
    llvm::FunctionType::get(Builder.getVoidTy(),
                            llvm::PointerType::getUnqual(ModuleTy), true);
  llvm::Value *Register = CGM.CreateRuntimeFunction(FT, "__objc_exec_class");
  Builder.CreateCall(Register, Module);

  if (!ClassAliases.empty()) {
    llvm::Type *ArgTypes[2] = {PtrTy, PtrToInt8Ty};
    llvm::FunctionType *RegisterAliasTy =
      llvm::FunctionType::get(Builder.getVoidTy(),
                              ArgTypes, false);
    llvm::Function *RegisterAlias = llvm::Function::Create(
      RegisterAliasTy,
      llvm::GlobalValue::ExternalWeakLinkage, "class_registerAlias_np",
      &TheModule);
    llvm::BasicBlock *AliasBB =
      llvm::BasicBlock::Create(VMContext, "alias", LoadFunction);
    llvm::BasicBlock *NoAliasBB =
      llvm::BasicBlock::Create(VMContext, "no_alias", LoadFunction);

    // Branch based on whether the runtime provided class_registerAlias_np()
    llvm::Value *HasRegisterAlias = Builder.CreateICmpNE(RegisterAlias,
            llvm::Constant::getNullValue(RegisterAlias->getType()));
    Builder.CreateCondBr(HasRegisterAlias, AliasBB, NoAliasBB);

    // The true branch (has alias registration fucntion):
    Builder.SetInsertPoint(AliasBB);
    // Emit alias registration calls:
    for (std::vector<ClassAliasPair>::iterator iter = ClassAliases.begin();
       iter != ClassAliases.end(); ++iter) {
       llvm::Constant *TheClass =
         TheModule.getGlobalVariable(("_OBJC_CLASS_" + iter->first).c_str(),
            true);
       if (0 != TheClass) {
         TheClass = llvm::ConstantExpr::getBitCast(TheClass, PtrTy);
         Builder.CreateCall2(RegisterAlias, TheClass,
            MakeConstantString(iter->second));
       }
    }
    // Jump to end:
    Builder.CreateBr(NoAliasBB);

    // Missing alias registration function, just return from the function:
    Builder.SetInsertPoint(NoAliasBB);
  }
  Builder.CreateRetVoid();

  return LoadFunction;
}

llvm::Function *CGObjCGNU::GenerateMethod(const ObjCMethodDecl *OMD,
                                          const ObjCContainerDecl *CD) {
  const ObjCCategoryImplDecl *OCD =
    dyn_cast<ObjCCategoryImplDecl>(OMD->getDeclContext());
  StringRef CategoryName = OCD ? OCD->getName() : "";
  StringRef ClassName = CD->getName();
  Selector MethodName = OMD->getSelector();
  bool isClassMethod = !OMD->isInstanceMethod();

  CodeGenTypes &Types = CGM.getTypes();
  llvm::FunctionType *MethodTy =
    Types.GetFunctionType(Types.arrangeObjCMethodDeclaration(OMD));
  std::string FunctionName = SymbolNameForMethod(ClassName, CategoryName,
      MethodName, isClassMethod);

  llvm::Function *Method
    = llvm::Function::Create(MethodTy,
                             llvm::GlobalValue::InternalLinkage,
                             FunctionName,
                             &TheModule);
  return Method;
}

llvm::Constant *CGObjCGNU::GetPropertyGetFunction() {
  return GetPropertyFn;
}

llvm::Constant *CGObjCGNU::GetPropertySetFunction() {
  return SetPropertyFn;
}

llvm::Constant *CGObjCGNU::GetOptimizedPropertySetFunction(bool atomic,
                                                           bool copy) {
  return 0;
}

llvm::Constant *CGObjCGNU::GetGetStructFunction() {
  return GetStructPropertyFn;
}
llvm::Constant *CGObjCGNU::GetSetStructFunction() {
  return SetStructPropertyFn;
}
llvm::Constant *CGObjCGNU::GetCppAtomicObjectGetFunction() {
  return 0;
}
llvm::Constant *CGObjCGNU::GetCppAtomicObjectSetFunction() {
  return 0;
}

llvm::Constant *CGObjCGNU::EnumerationMutationFunction() {
  return EnumerationMutationFn;
}

void CGObjCGNU::EmitSynchronizedStmt(CodeGenFunction &CGF,
                                     const ObjCAtSynchronizedStmt &S) {
  EmitAtSynchronizedStmt(CGF, S, SyncEnterFn, SyncExitFn);
}


void CGObjCGNU::EmitTryStmt(CodeGenFunction &CGF,
                            const ObjCAtTryStmt &S) {
  // Unlike the Apple non-fragile runtimes, which also uses
  // unwind-based zero cost exceptions, the GNU Objective C runtime's
  // EH support isn't a veneer over C++ EH.  Instead, exception
  // objects are created by objc_exception_throw and destroyed by
  // the personality function; this avoids the need for bracketing
  // catch handlers with calls to __blah_begin_catch/__blah_end_catch
  // (or even _Unwind_DeleteException), but probably doesn't
  // interoperate very well with foreign exceptions.
  //
  // In Objective-C++ mode, we actually emit something equivalent to the C++
  // exception handler. 
  EmitTryCatchStmt(CGF, S, EnterCatchFn, ExitCatchFn, ExceptionReThrowFn);
  return ;
}

void CGObjCGNU::EmitThrowStmt(CodeGenFunction &CGF,
                              const ObjCAtThrowStmt &S,
                              bool ClearInsertionPoint) {
  llvm::Value *ExceptionAsObject;

  if (const Expr *ThrowExpr = S.getThrowExpr()) {
    llvm::Value *Exception = CGF.EmitObjCThrowOperand(ThrowExpr);
    ExceptionAsObject = Exception;
  } else {
    assert((!CGF.ObjCEHValueStack.empty() && CGF.ObjCEHValueStack.back()) &&
           "Unexpected rethrow outside @catch block.");
    ExceptionAsObject = CGF.ObjCEHValueStack.back();
  }
  ExceptionAsObject = CGF.Builder.CreateBitCast(ExceptionAsObject, IdTy);
  llvm::CallSite Throw =
      CGF.EmitRuntimeCallOrInvoke(ExceptionThrowFn, ExceptionAsObject);
  Throw.setDoesNotReturn();
  CGF.Builder.CreateUnreachable();
  if (ClearInsertionPoint)
    CGF.Builder.ClearInsertionPoint();
}

llvm::Value * CGObjCGNU::EmitObjCWeakRead(CodeGenFunction &CGF,
                                          llvm::Value *AddrWeakObj) {
  CGBuilderTy &B = CGF.Builder;
  AddrWeakObj = EnforceType(B, AddrWeakObj, PtrToIdTy);
  return B.CreateCall(WeakReadFn, AddrWeakObj);
}

void CGObjCGNU::EmitObjCWeakAssign(CodeGenFunction &CGF,
                                   llvm::Value *src, llvm::Value *dst) {
  CGBuilderTy &B = CGF.Builder;
  src = EnforceType(B, src, IdTy);
  dst = EnforceType(B, dst, PtrToIdTy);
  B.CreateCall2(WeakAssignFn, src, dst);
}

void CGObjCGNU::EmitObjCGlobalAssign(CodeGenFunction &CGF,
                                     llvm::Value *src, llvm::Value *dst,
                                     bool threadlocal) {
  CGBuilderTy &B = CGF.Builder;
  src = EnforceType(B, src, IdTy);
  dst = EnforceType(B, dst, PtrToIdTy);
  if (!threadlocal)
    B.CreateCall2(GlobalAssignFn, src, dst);
  else
    // FIXME. Add threadloca assign API
    llvm_unreachable("EmitObjCGlobalAssign - Threal Local API NYI");
}

void CGObjCGNU::EmitObjCIvarAssign(CodeGenFunction &CGF,
                                   llvm::Value *src, llvm::Value *dst,
                                   llvm::Value *ivarOffset) {
  CGBuilderTy &B = CGF.Builder;
  src = EnforceType(B, src, IdTy);
  dst = EnforceType(B, dst, IdTy);
  B.CreateCall3(IvarAssignFn, src, dst, ivarOffset);
}

void CGObjCGNU::EmitObjCStrongCastAssign(CodeGenFunction &CGF,
                                         llvm::Value *src, llvm::Value *dst) {
  CGBuilderTy &B = CGF.Builder;
  src = EnforceType(B, src, IdTy);
  dst = EnforceType(B, dst, PtrToIdTy);
  B.CreateCall2(StrongCastAssignFn, src, dst);
}

void CGObjCGNU::EmitGCMemmoveCollectable(CodeGenFunction &CGF,
                                         llvm::Value *DestPtr,
                                         llvm::Value *SrcPtr,
                                         llvm::Value *Size) {
  CGBuilderTy &B = CGF.Builder;
  DestPtr = EnforceType(B, DestPtr, PtrTy);
  SrcPtr = EnforceType(B, SrcPtr, PtrTy);

  B.CreateCall3(MemMoveFn, DestPtr, SrcPtr, Size);
}

llvm::GlobalVariable *CGObjCGNU::ObjCIvarOffsetVariable(
                              const ObjCInterfaceDecl *ID,
                              const ObjCIvarDecl *Ivar) {
  const std::string Name = "__objc_ivar_offset_" + ID->getNameAsString()
    + '.' + Ivar->getNameAsString();
  // Emit the variable and initialize it with what we think the correct value
  // is.  This allows code compiled with non-fragile ivars to work correctly
  // when linked against code which isn't (most of the time).
  llvm::GlobalVariable *IvarOffsetPointer = TheModule.getNamedGlobal(Name);
  if (!IvarOffsetPointer) {
    // This will cause a run-time crash if we accidentally use it.  A value of
    // 0 would seem more sensible, but will silently overwrite the isa pointer
    // causing a great deal of confusion.
    uint64_t Offset = -1;
    // We can't call ComputeIvarBaseOffset() here if we have the
    // implementation, because it will create an invalid ASTRecordLayout object
    // that we are then stuck with forever, so we only initialize the ivar
    // offset variable with a guess if we only have the interface.  The
    // initializer will be reset later anyway, when we are generating the class
    // description.
    if (!CGM.getContext().getObjCImplementation(
              const_cast<ObjCInterfaceDecl *>(ID)))
      Offset = ComputeIvarBaseOffset(CGM, ID, Ivar);

    llvm::ConstantInt *OffsetGuess = llvm::ConstantInt::get(Int32Ty, Offset,
                             /*isSigned*/true);
    // Don't emit the guess in non-PIC code because the linker will not be able
    // to replace it with the real version for a library.  In non-PIC code you
    // must compile with the fragile ABI if you want to use ivars from a
    // GCC-compiled class.
    if (CGM.getLangOpts().PICLevel || CGM.getLangOpts().PIELevel) {
      llvm::GlobalVariable *IvarOffsetGV = new llvm::GlobalVariable(TheModule,
            Int32Ty, false,
            llvm::GlobalValue::PrivateLinkage, OffsetGuess, Name+".guess");
      IvarOffsetPointer = new llvm::GlobalVariable(TheModule,
            IvarOffsetGV->getType(), false, llvm::GlobalValue::LinkOnceAnyLinkage,
            IvarOffsetGV, Name);
    } else {
      IvarOffsetPointer = new llvm::GlobalVariable(TheModule,
              llvm::Type::getInt32PtrTy(VMContext), false,
              llvm::GlobalValue::ExternalLinkage, 0, Name);
    }
  }
  return IvarOffsetPointer;
}

LValue CGObjCGNU::EmitObjCValueForIvar(CodeGenFunction &CGF,
                                       QualType ObjectTy,
                                       llvm::Value *BaseValue,
                                       const ObjCIvarDecl *Ivar,
                                       unsigned CVRQualifiers) {
  const ObjCInterfaceDecl *ID =
    ObjectTy->getAs<ObjCObjectType>()->getInterface();
  return EmitValueForIvarAtOffset(CGF, ID, BaseValue, Ivar, CVRQualifiers,
                                  EmitIvarOffset(CGF, ID, Ivar));
}

static const ObjCInterfaceDecl *FindIvarInterface(ASTContext &Context,
                                                  const ObjCInterfaceDecl *OID,
                                                  const ObjCIvarDecl *OIVD) {
  for (const ObjCIvarDecl *next = OID->all_declared_ivar_begin(); next;
       next = next->getNextIvar()) {
    if (OIVD == next)
      return OID;
  }

  // Otherwise check in the super class.
  if (const ObjCInterfaceDecl *Super = OID->getSuperClass())
    return FindIvarInterface(Context, Super, OIVD);

  return 0;
}

llvm::Value *CGObjCGNU::EmitIvarOffset(CodeGenFunction &CGF,
                         const ObjCInterfaceDecl *Interface,
                         const ObjCIvarDecl *Ivar) {
  if (CGM.getLangOpts().ObjCRuntime.isNonFragile()) {
    Interface = FindIvarInterface(CGM.getContext(), Interface, Ivar);
    if (RuntimeVersion < 10)
      return CGF.Builder.CreateZExtOrBitCast(
          CGF.Builder.CreateLoad(CGF.Builder.CreateLoad(
                  ObjCIvarOffsetVariable(Interface, Ivar), false, "ivar")),
          PtrDiffTy);
    std::string name = "__objc_ivar_offset_value_" +
      Interface->getNameAsString() +"." + Ivar->getNameAsString();
    llvm::Value *Offset = TheModule.getGlobalVariable(name);
    if (!Offset)
      Offset = new llvm::GlobalVariable(TheModule, IntTy,
          false, llvm::GlobalValue::LinkOnceAnyLinkage,
          llvm::Constant::getNullValue(IntTy), name);
    Offset = CGF.Builder.CreateLoad(Offset);
    if (Offset->getType() != PtrDiffTy)
      Offset = CGF.Builder.CreateZExtOrBitCast(Offset, PtrDiffTy);
    return Offset;
  }
  uint64_t Offset = ComputeIvarBaseOffset(CGF.CGM, Interface, Ivar);
  return llvm::ConstantInt::get(PtrDiffTy, Offset, /*isSigned*/true);
}

CGObjCRuntime *
clang::CodeGen::CreateGNUObjCRuntime(CodeGenModule &CGM) {
  switch (CGM.getLangOpts().ObjCRuntime.getKind()) {
  case ObjCRuntime::GNUstep:
    return new CGObjCGNUstep(CGM);

  case ObjCRuntime::GCC:
    return new CGObjCGCC(CGM);

  case ObjCRuntime::ObjFW:
    return new CGObjCObjFW(CGM);

  case ObjCRuntime::FragileMacOSX:
  case ObjCRuntime::MacOSX:
  case ObjCRuntime::iOS:
    llvm_unreachable("these runtimes are not GNU runtimes");
  }
  llvm_unreachable("bad runtime");
}
