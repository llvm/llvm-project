//===--- CodeGenModule.h - Per-Module state for LLVM CodeGen ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the internal per-translation-unit state used for llvm translation.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CODEGENMODULE_H
#define CLANG_CODEGEN_CODEGENMODULE_H

#include "CGVTables.h"
#include "CodeGenTypes.h"
#include "SanitizerBlacklist.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Mangle.h"
#include "clang/Basic/ABI.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/Module.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueHandle.h"

namespace llvm {
class Module;
class Constant;
class ConstantInt;
class Function;
class GlobalValue;
class DataLayout;
class FunctionType;
class LLVMContext;
class IndexedInstrProfReader;
}

namespace clang {
class TargetCodeGenInfo;
class ASTContext;
class AtomicType;
class FunctionDecl;
class IdentifierInfo;
class ObjCMethodDecl;
class ObjCImplementationDecl;
class ObjCCategoryImplDecl;
class ObjCProtocolDecl;
class ObjCEncodeExpr;
class BlockExpr;
class CharUnits;
class Decl;
class Expr;
class Stmt;
class InitListExpr;
class StringLiteral;
class NamedDecl;
class ValueDecl;
class VarDecl;
class LangOptions;
class CodeGenOptions;
class DiagnosticsEngine;
class AnnotateAttr;
class CXXDestructorDecl;
class Module;

namespace CodeGen {

class CallArgList;
class CodeGenFunction;
class CodeGenTBAA;
class CGCXXABI;
class CGDebugInfo;
class CGObjCRuntime;
class CGOpenCLRuntime;
class CGOpenMPRuntime;
class CGCUDARuntime;
class BlockFieldFlags;
class FunctionArgList;

struct OrderGlobalInits {
  unsigned int priority;
  unsigned int lex_order;
  OrderGlobalInits(unsigned int p, unsigned int l)
      : priority(p), lex_order(l) {}

  bool operator==(const OrderGlobalInits &RHS) const {
    return priority == RHS.priority && lex_order == RHS.lex_order;
  }

  bool operator<(const OrderGlobalInits &RHS) const {
    return std::tie(priority, lex_order) <
           std::tie(RHS.priority, RHS.lex_order);
  }
};

struct CodeGenTypeCache {
  /// void
  llvm::Type *VoidTy;

  /// i8, i16, i32, and i64
  llvm::IntegerType *Int8Ty, *Int16Ty, *Int32Ty, *Int64Ty;
  /// float, double
  llvm::Type *FloatTy, *DoubleTy;

  /// int
  llvm::IntegerType *IntTy;

  /// intptr_t, size_t, and ptrdiff_t, which we assume are the same size.
  union {
    llvm::IntegerType *IntPtrTy;
    llvm::IntegerType *SizeTy;
    llvm::IntegerType *PtrDiffTy;
  };

  /// void* in address space 0
  union {
    llvm::PointerType *VoidPtrTy;
    llvm::PointerType *Int8PtrTy;
  };

  /// void** in address space 0
  union {
    llvm::PointerType *VoidPtrPtrTy;
    llvm::PointerType *Int8PtrPtrTy;
  };

  /// The width of a pointer into the generic address space.
  unsigned char PointerWidthInBits;

  /// The size and alignment of a pointer into the generic address
  /// space.
  union {
    unsigned char PointerAlignInBytes;
    unsigned char PointerSizeInBytes;
    unsigned char SizeSizeInBytes; // sizeof(size_t)
  };

  llvm::CallingConv::ID RuntimeCC;
  llvm::CallingConv::ID getRuntimeCC() const { return RuntimeCC; }
};

struct RREntrypoints {
  RREntrypoints() { memset(this, 0, sizeof(*this)); }
  /// void objc_autoreleasePoolPop(void*);
  llvm::Constant *objc_autoreleasePoolPop;

  /// void *objc_autoreleasePoolPush(void);
  llvm::Constant *objc_autoreleasePoolPush;
};

struct ARCEntrypoints {
  ARCEntrypoints() { memset(this, 0, sizeof(*this)); }

  /// id objc_autorelease(id);
  llvm::Constant *objc_autorelease;

  /// id objc_autoreleaseReturnValue(id);
  llvm::Constant *objc_autoreleaseReturnValue;

  /// void objc_copyWeak(id *dest, id *src);
  llvm::Constant *objc_copyWeak;

  /// void objc_destroyWeak(id*);
  llvm::Constant *objc_destroyWeak;

  /// id objc_initWeak(id*, id);
  llvm::Constant *objc_initWeak;

  /// id objc_loadWeak(id*);
  llvm::Constant *objc_loadWeak;

  /// id objc_loadWeakRetained(id*);
  llvm::Constant *objc_loadWeakRetained;

  /// void objc_moveWeak(id *dest, id *src);
  llvm::Constant *objc_moveWeak;

  /// id objc_retain(id);
  llvm::Constant *objc_retain;

  /// id objc_retainAutorelease(id);
  llvm::Constant *objc_retainAutorelease;

  /// id objc_retainAutoreleaseReturnValue(id);
  llvm::Constant *objc_retainAutoreleaseReturnValue;

  /// id objc_retainAutoreleasedReturnValue(id);
  llvm::Constant *objc_retainAutoreleasedReturnValue;

  /// id objc_retainBlock(id);
  llvm::Constant *objc_retainBlock;

  /// void objc_release(id);
  llvm::Constant *objc_release;

  /// id objc_storeStrong(id*, id);
  llvm::Constant *objc_storeStrong;

  /// id objc_storeWeak(id*, id);
  llvm::Constant *objc_storeWeak;

  /// A void(void) inline asm to use to mark that the return value of
  /// a call will be immediately retain.
  llvm::InlineAsm *retainAutoreleasedReturnValueMarker;

  /// void clang.arc.use(...);
  llvm::Constant *clang_arc_use;
};

/// This class records statistics on instrumentation based profiling.
class InstrProfStats {
  uint32_t VisitedInMainFile;
  uint32_t MissingInMainFile;
  uint32_t Visited;
  uint32_t Missing;
  uint32_t Mismatched;

public:
  InstrProfStats()
      : VisitedInMainFile(0), MissingInMainFile(0), Visited(0), Missing(0),
        Mismatched(0) {}
  /// Record that we've visited a function and whether or not that function was
  /// in the main source file.
  void addVisited(bool MainFile) {
    if (MainFile)
      ++VisitedInMainFile;
    ++Visited;
  }
  /// Record that a function we've visited has no profile data.
  void addMissing(bool MainFile) {
    if (MainFile)
      ++MissingInMainFile;
    ++Missing;
  }
  /// Record that a function we've visited has mismatched profile data.
  void addMismatched(bool MainFile) { ++Mismatched; }
  /// Whether or not the stats we've gathered indicate any potential problems.
  bool hasDiagnostics() { return Missing || Mismatched; }
  /// Report potential problems we've found to \c Diags.
  void reportDiagnostics(DiagnosticsEngine &Diags, StringRef MainFile);
};

/// This class organizes the cross-function state that is used while generating
/// LLVM code.
class CodeGenModule : public CodeGenTypeCache {
  CodeGenModule(const CodeGenModule &) LLVM_DELETED_FUNCTION;
  void operator=(const CodeGenModule &) LLVM_DELETED_FUNCTION;

  struct Structor {
    Structor() : Priority(0), Initializer(nullptr), AssociatedData(nullptr) {}
    Structor(int Priority, llvm::Constant *Initializer,
             llvm::Constant *AssociatedData)
        : Priority(Priority), Initializer(Initializer),
          AssociatedData(AssociatedData) {}
    int Priority;
    llvm::Constant *Initializer;
    llvm::Constant *AssociatedData;
  };

  typedef std::vector<Structor> CtorList;

  ASTContext &Context;
  const LangOptions &LangOpts;
  const CodeGenOptions &CodeGenOpts;
  llvm::Module &TheModule;
  DiagnosticsEngine &Diags;
  const llvm::DataLayout &TheDataLayout;
  const TargetInfo &Target;
  std::unique_ptr<CGCXXABI> ABI;
  llvm::LLVMContext &VMContext;

  CodeGenTBAA *TBAA;
  
  mutable const TargetCodeGenInfo *TheTargetCodeGenInfo;
  
  // This should not be moved earlier, since its initialization depends on some
  // of the previous reference members being already initialized and also checks
  // if TheTargetCodeGenInfo is NULL
  CodeGenTypes Types;
 
  /// Holds information about C++ vtables.
  CodeGenVTables VTables;

  CGObjCRuntime* ObjCRuntime;
  CGOpenCLRuntime* OpenCLRuntime;
  CGOpenMPRuntime* OpenMPRuntime;
  CGCUDARuntime* CUDARuntime;
  CGDebugInfo* DebugInfo;
  ARCEntrypoints *ARCData;
  llvm::MDNode *NoObjCARCExceptionsMetadata;
  RREntrypoints *RRData;
  std::unique_ptr<llvm::IndexedInstrProfReader> PGOReader;
  InstrProfStats PGOStats;

  // A set of references that have only been seen via a weakref so far. This is
  // used to remove the weak of the reference if we ever see a direct reference
  // or a definition.
  llvm::SmallPtrSet<llvm::GlobalValue*, 10> WeakRefReferences;

  /// This contains all the decls which have definitions but/ which are deferred
  /// for emission and therefore should only be output if they are actually
  /// used. If a decl is in this, then it is known to have not been referenced
  /// yet.
  std::map<StringRef, GlobalDecl> DeferredDecls;

  /// This is a list of deferred decls which we have seen that *are* actually
  /// referenced. These get code generated when the module is done.
  struct DeferredGlobal {
    DeferredGlobal(llvm::GlobalValue *GV, GlobalDecl GD) : GV(GV), GD(GD) {}
    llvm::AssertingVH<llvm::GlobalValue> GV;
    GlobalDecl GD;
  };
  std::vector<DeferredGlobal> DeferredDeclsToEmit;
  void addDeferredDeclToEmit(llvm::GlobalValue *GV, GlobalDecl GD) {
    DeferredDeclsToEmit.push_back(DeferredGlobal(GV, GD));
  }

  /// List of alias we have emitted. Used to make sure that what they point to
  /// is defined once we get to the end of the of the translation unit.
  std::vector<GlobalDecl> Aliases;

  typedef llvm::StringMap<llvm::TrackingVH<llvm::Constant> > ReplacementsTy;
  ReplacementsTy Replacements;

  /// A queue of (optional) vtables to consider emitting.
  std::vector<const CXXRecordDecl*> DeferredVTables;

  /// List of global values which are required to be present in the object file;
  /// bitcast to i8*. This is used for forcing visibility of symbols which may
  /// otherwise be optimized out.
  std::vector<llvm::WeakVH> LLVMUsed;
  std::vector<llvm::WeakVH> LLVMCompilerUsed;

  /// Store the list of global constructors and their respective priorities to
  /// be emitted when the translation unit is complete.
  CtorList GlobalCtors;

  /// Store the list of global destructors and their respective priorities to be
  /// emitted when the translation unit is complete.
  CtorList GlobalDtors;

  /// An ordered map of canonical GlobalDecls to their mangled names.
  llvm::MapVector<GlobalDecl, StringRef> MangledDeclNames;
  llvm::StringMap<GlobalDecl, llvm::BumpPtrAllocator> Manglings;

  /// Global annotations.
  std::vector<llvm::Constant*> Annotations;

  /// Map used to get unique annotation strings.
  llvm::StringMap<llvm::Constant*> AnnotationStrings;

  llvm::StringMap<llvm::Constant*> CFConstantStringMap;

  llvm::DenseMap<llvm::Constant *, llvm::GlobalVariable *> ConstantStringMap;
  llvm::DenseMap<const Decl*, llvm::Constant *> StaticLocalDeclMap;
  llvm::DenseMap<const Decl*, llvm::GlobalVariable*> StaticLocalDeclGuardMap;
  llvm::DenseMap<const Expr*, llvm::Constant *> MaterializedGlobalTemporaryMap;

  llvm::DenseMap<QualType, llvm::Constant *> AtomicSetterHelperFnMap;
  llvm::DenseMap<QualType, llvm::Constant *> AtomicGetterHelperFnMap;

  /// Map used to get unique type descriptor constants for sanitizers.
  llvm::DenseMap<QualType, llvm::Constant *> TypeDescriptorMap;

  /// Map used to track internal linkage functions declared within
  /// extern "C" regions.
  typedef llvm::MapVector<IdentifierInfo *,
                          llvm::GlobalValue *> StaticExternCMap;
  StaticExternCMap StaticExternCValues;

  /// \brief thread_local variables defined or used in this TU.
  std::vector<std::pair<const VarDecl *, llvm::GlobalVariable *> >
    CXXThreadLocals;

  /// \brief thread_local variables with initializers that need to run
  /// before any thread_local variable in this TU is odr-used.
  std::vector<llvm::Constant*> CXXThreadLocalInits;

  /// Global variables with initializers that need to run before main.
  std::vector<llvm::Constant*> CXXGlobalInits;

  /// When a C++ decl with an initializer is deferred, null is
  /// appended to CXXGlobalInits, and the index of that null is placed
  /// here so that the initializer will be performed in the correct
  /// order.
  llvm::DenseMap<const Decl*, unsigned> DelayedCXXInitPosition;
  
  typedef std::pair<OrderGlobalInits, llvm::Function*> GlobalInitData;

  struct GlobalInitPriorityCmp {
    bool operator()(const GlobalInitData &LHS,
                    const GlobalInitData &RHS) const {
      return LHS.first.priority < RHS.first.priority;
    }
  };

  /// Global variables with initializers whose order of initialization is set by
  /// init_priority attribute.
  SmallVector<GlobalInitData, 8> PrioritizedCXXGlobalInits;

  /// Global destructor functions and arguments that need to run on termination.
  std::vector<std::pair<llvm::WeakVH,llvm::Constant*> > CXXGlobalDtors;

  /// \brief The complete set of modules that has been imported.
  llvm::SetVector<clang::Module *> ImportedModules;

  /// \brief A vector of metadata strings.
  SmallVector<llvm::Value *, 16> LinkerOptionsMetadata;

  /// @name Cache for Objective-C runtime types
  /// @{

  /// Cached reference to the class for constant strings. This value has type
  /// int * but is actually an Obj-C class pointer.
  llvm::WeakVH CFConstantStringClassRef;

  /// Cached reference to the class for constant strings. This value has type
  /// int * but is actually an Obj-C class pointer.
  llvm::WeakVH ConstantStringClassRef;

  /// \brief The LLVM type corresponding to NSConstantString.
  llvm::StructType *NSConstantStringType;
  
  /// \brief The type used to describe the state of a fast enumeration in
  /// Objective-C's for..in loop.
  QualType ObjCFastEnumerationStateType;
  
  /// @}

  /// Lazily create the Objective-C runtime
  void createObjCRuntime();

  void createOpenCLRuntime();
  void createOpenMPRuntime();
  void createCUDARuntime();

  bool isTriviallyRecursive(const FunctionDecl *F);
  bool shouldEmitFunction(GlobalDecl GD);

  /// @name Cache for Blocks Runtime Globals
  /// @{

  llvm::Constant *NSConcreteGlobalBlock;
  llvm::Constant *NSConcreteStackBlock;

  llvm::Constant *BlockObjectAssign;
  llvm::Constant *BlockObjectDispose;

  llvm::Type *BlockDescriptorType;
  llvm::Type *GenericBlockLiteralType;

  struct {
    int GlobalUniqueCount;
  } Block;

  /// void @llvm.lifetime.start(i64 %size, i8* nocapture <ptr>)
  llvm::Constant *LifetimeStartFn;

  /// void @llvm.lifetime.end(i64 %size, i8* nocapture <ptr>)
  llvm::Constant *LifetimeEndFn;

  GlobalDecl initializedGlobalDecl;

  SanitizerBlacklist SanitizerBL;

  /// @}
public:
  CodeGenModule(ASTContext &C, const CodeGenOptions &CodeGenOpts,
                llvm::Module &M, const llvm::DataLayout &TD,
                DiagnosticsEngine &Diags);

  ~CodeGenModule();

  void clear();

  /// Finalize LLVM code generation.
  void Release();

  /// Return a reference to the configured Objective-C runtime.
  CGObjCRuntime &getObjCRuntime() {
    if (!ObjCRuntime) createObjCRuntime();
    return *ObjCRuntime;
  }

  /// Return true iff an Objective-C runtime has been configured.
  bool hasObjCRuntime() { return !!ObjCRuntime; }

  /// Return a reference to the configured OpenCL runtime.
  CGOpenCLRuntime &getOpenCLRuntime() {
    assert(OpenCLRuntime != nullptr);
    return *OpenCLRuntime;
  }

  /// Return a reference to the configured OpenMP runtime.
  CGOpenMPRuntime &getOpenMPRuntime() {
    assert(OpenMPRuntime != nullptr);
    return *OpenMPRuntime;
  }

  /// Return a reference to the configured CUDA runtime.
  CGCUDARuntime &getCUDARuntime() {
    assert(CUDARuntime != nullptr);
    return *CUDARuntime;
  }

  ARCEntrypoints &getARCEntrypoints() const {
    assert(getLangOpts().ObjCAutoRefCount && ARCData != nullptr);
    return *ARCData;
  }

  RREntrypoints &getRREntrypoints() const {
    assert(RRData != nullptr);
    return *RRData;
  }

  InstrProfStats &getPGOStats() { return PGOStats; }
  llvm::IndexedInstrProfReader *getPGOReader() const { return PGOReader.get(); }

  llvm::Constant *getStaticLocalDeclAddress(const VarDecl *D) {
    return StaticLocalDeclMap[D];
  }
  void setStaticLocalDeclAddress(const VarDecl *D, 
                                 llvm::Constant *C) {
    StaticLocalDeclMap[D] = C;
  }

  llvm::GlobalVariable *getStaticLocalDeclGuardAddress(const VarDecl *D) {
    return StaticLocalDeclGuardMap[D];
  }
  void setStaticLocalDeclGuardAddress(const VarDecl *D, 
                                      llvm::GlobalVariable *C) {
    StaticLocalDeclGuardMap[D] = C;
  }

  bool lookupRepresentativeDecl(StringRef MangledName,
                                GlobalDecl &Result) const;

  llvm::Constant *getAtomicSetterHelperFnMap(QualType Ty) {
    return AtomicSetterHelperFnMap[Ty];
  }
  void setAtomicSetterHelperFnMap(QualType Ty,
                            llvm::Constant *Fn) {
    AtomicSetterHelperFnMap[Ty] = Fn;
  }

  llvm::Constant *getAtomicGetterHelperFnMap(QualType Ty) {
    return AtomicGetterHelperFnMap[Ty];
  }
  void setAtomicGetterHelperFnMap(QualType Ty,
                            llvm::Constant *Fn) {
    AtomicGetterHelperFnMap[Ty] = Fn;
  }

  llvm::Constant *getTypeDescriptorFromMap(QualType Ty) {
    return TypeDescriptorMap[Ty];
  }
  void setTypeDescriptorInMap(QualType Ty, llvm::Constant *C) {
    TypeDescriptorMap[Ty] = C;
  }

  CGDebugInfo *getModuleDebugInfo() { return DebugInfo; }

  llvm::MDNode *getNoObjCARCExceptionsMetadata() {
    if (!NoObjCARCExceptionsMetadata)
      NoObjCARCExceptionsMetadata =
        llvm::MDNode::get(getLLVMContext(),
                          SmallVector<llvm::Value*,1>());
    return NoObjCARCExceptionsMetadata;
  }

  ASTContext &getContext() const { return Context; }
  const LangOptions &getLangOpts() const { return LangOpts; }
  const CodeGenOptions &getCodeGenOpts() const { return CodeGenOpts; }
  llvm::Module &getModule() const { return TheModule; }
  DiagnosticsEngine &getDiags() const { return Diags; }
  const llvm::DataLayout &getDataLayout() const { return TheDataLayout; }
  const TargetInfo &getTarget() const { return Target; }
  CGCXXABI &getCXXABI() const { return *ABI; }
  llvm::LLVMContext &getLLVMContext() { return VMContext; }

  bool shouldUseTBAA() const { return TBAA != nullptr; }

  const TargetCodeGenInfo &getTargetCodeGenInfo(); 
  
  CodeGenTypes &getTypes() { return Types; }
 
  CodeGenVTables &getVTables() { return VTables; }

  ItaniumVTableContext &getItaniumVTableContext() {
    return VTables.getItaniumVTableContext();
  }

  MicrosoftVTableContext &getMicrosoftVTableContext() {
    return VTables.getMicrosoftVTableContext();
  }

  llvm::MDNode *getTBAAInfo(QualType QTy);
  llvm::MDNode *getTBAAInfoForVTablePtr();
  llvm::MDNode *getTBAAStructInfo(QualType QTy);
  /// Return the MDNode in the type DAG for the given struct type.
  llvm::MDNode *getTBAAStructTypeInfo(QualType QTy);
  /// Return the path-aware tag for given base type, access node and offset.
  llvm::MDNode *getTBAAStructTagInfo(QualType BaseTy, llvm::MDNode *AccessN,
                                     uint64_t O);

  bool isTypeConstant(QualType QTy, bool ExcludeCtorDtor);

  bool isPaddedAtomicType(QualType type);
  bool isPaddedAtomicType(const AtomicType *type);

  /// Decorate the instruction with a TBAA tag. For scalar TBAA, the tag
  /// is the same as the type. For struct-path aware TBAA, the tag
  /// is different from the type: base type, access type and offset.
  /// When ConvertTypeToTag is true, we create a tag based on the scalar type.
  void DecorateInstruction(llvm::Instruction *Inst,
                           llvm::MDNode *TBAAInfo,
                           bool ConvertTypeToTag = true);

  /// Emit the given number of characters as a value of type size_t.
  llvm::ConstantInt *getSize(CharUnits numChars);

  /// Set the visibility for the given LLVM GlobalValue.
  void setGlobalVisibility(llvm::GlobalValue *GV, const NamedDecl *D) const;

  /// Set the TLS mode for the given LLVM GlobalVariable for the thread-local
  /// variable declaration D.
  void setTLSMode(llvm::GlobalVariable *GV, const VarDecl &D) const;

  static llvm::GlobalValue::VisibilityTypes GetLLVMVisibility(Visibility V) {
    switch (V) {
    case DefaultVisibility:   return llvm::GlobalValue::DefaultVisibility;
    case HiddenVisibility:    return llvm::GlobalValue::HiddenVisibility;
    case ProtectedVisibility: return llvm::GlobalValue::ProtectedVisibility;
    }
    llvm_unreachable("unknown visibility!");
  }

  llvm::Constant *GetAddrOfGlobal(GlobalDecl GD) {
    if (isa<CXXConstructorDecl>(GD.getDecl()))
      return GetAddrOfCXXConstructor(cast<CXXConstructorDecl>(GD.getDecl()),
                                     GD.getCtorType());
    else if (isa<CXXDestructorDecl>(GD.getDecl()))
      return GetAddrOfCXXDestructor(cast<CXXDestructorDecl>(GD.getDecl()),
                                     GD.getDtorType());
    else if (isa<FunctionDecl>(GD.getDecl()))
      return GetAddrOfFunction(GD);
    else
      return GetAddrOfGlobalVar(cast<VarDecl>(GD.getDecl()));
  }

  /// Will return a global variable of the given type. If a variable with a
  /// different type already exists then a new  variable with the right type
  /// will be created and all uses of the old variable will be replaced with a
  /// bitcast to the new variable.
  llvm::GlobalVariable *
  CreateOrReplaceCXXRuntimeVariable(StringRef Name, llvm::Type *Ty,
                                    llvm::GlobalValue::LinkageTypes Linkage);

  /// Return the address space of the underlying global variable for D, as
  /// determined by its declaration. Normally this is the same as the address
  /// space of D's type, but in CUDA, address spaces are associated with
  /// declarations, not types.
  unsigned GetGlobalVarAddressSpace(const VarDecl *D, unsigned AddrSpace);

  /// Return the llvm::Constant for the address of the given global variable.
  /// If Ty is non-null and if the global doesn't exist, then it will be greated
  /// with the specified type instead of whatever the normal requested type
  /// would be.
  llvm::Constant *GetAddrOfGlobalVar(const VarDecl *D,
                                     llvm::Type *Ty = nullptr);

  /// Return the address of the given function. If Ty is non-null, then this
  /// function will use the specified type if it has to create it.
  llvm::Constant *GetAddrOfFunction(GlobalDecl GD, llvm::Type *Ty = 0,
                                    bool ForVTable = false,
                                    bool DontDefer = false);

  /// Get the address of the RTTI descriptor for the given type.
  llvm::Constant *GetAddrOfRTTIDescriptor(QualType Ty, bool ForEH = false);

  /// Get the address of a uuid descriptor .
  llvm::Constant *GetAddrOfUuidDescriptor(const CXXUuidofExpr* E);

  /// Get the address of the thunk for the given global decl.
  llvm::Constant *GetAddrOfThunk(GlobalDecl GD, const ThunkInfo &Thunk);

  /// Get a reference to the target of VD.
  llvm::Constant *GetWeakRefReference(const ValueDecl *VD);

  /// Returns the offset from a derived class to  a class. Returns null if the
  /// offset is 0.
  llvm::Constant *
  GetNonVirtualBaseClassOffset(const CXXRecordDecl *ClassDecl,
                               CastExpr::path_const_iterator PathBegin,
                               CastExpr::path_const_iterator PathEnd);

  /// A pair of helper functions for a __block variable.
  class ByrefHelpers : public llvm::FoldingSetNode {
  public:
    llvm::Constant *CopyHelper;
    llvm::Constant *DisposeHelper;

    /// The alignment of the field.  This is important because
    /// different offsets to the field within the byref struct need to
    /// have different helper functions.
    CharUnits Alignment;

    ByrefHelpers(CharUnits alignment) : Alignment(alignment) {}
    virtual ~ByrefHelpers();

    void Profile(llvm::FoldingSetNodeID &id) const {
      id.AddInteger(Alignment.getQuantity());
      profileImpl(id);
    }
    virtual void profileImpl(llvm::FoldingSetNodeID &id) const = 0;

    virtual bool needsCopy() const { return true; }
    virtual void emitCopy(CodeGenFunction &CGF,
                          llvm::Value *dest, llvm::Value *src) = 0;

    virtual bool needsDispose() const { return true; }
    virtual void emitDispose(CodeGenFunction &CGF, llvm::Value *field) = 0;
  };

  llvm::FoldingSet<ByrefHelpers> ByrefHelpersCache;

  /// Fetches the global unique block count.
  int getUniqueBlockCount() { return ++Block.GlobalUniqueCount; }
  
  /// Fetches the type of a generic block descriptor.
  llvm::Type *getBlockDescriptorType();

  /// The type of a generic block literal.
  llvm::Type *getGenericBlockLiteralType();

  /// Gets the address of a block which requires no captures.
  llvm::Constant *GetAddrOfGlobalBlock(const BlockExpr *BE, const char *);
  
  /// Return a pointer to a constant CFString object for the given string.
  llvm::Constant *GetAddrOfConstantCFString(const StringLiteral *Literal);

  /// Return a pointer to a constant NSString object for the given string. Or a
  /// user defined String object as defined via
  /// -fconstant-string-class=class_name option.
  llvm::Constant *GetAddrOfConstantString(const StringLiteral *Literal);

  /// Return a constant array for the given string.
  llvm::Constant *GetConstantArrayFromStringLiteral(const StringLiteral *E);

  /// Return a pointer to a constant array for the given string literal.
  llvm::GlobalVariable *
  GetAddrOfConstantStringFromLiteral(const StringLiteral *S);

  /// Return a pointer to a constant array for the given ObjCEncodeExpr node.
  llvm::GlobalVariable *
  GetAddrOfConstantStringFromObjCEncode(const ObjCEncodeExpr *);

  /// Returns a pointer to a character array containing the literal and a
  /// terminating '\0' character. The result has pointer to array type.
  ///
  /// \param GlobalName If provided, the name to use for the global (if one is
  /// created).
  llvm::GlobalVariable *
  GetAddrOfConstantCString(const std::string &Str,
                           const char *GlobalName = nullptr,
                           unsigned Alignment = 0);

  /// Returns a pointer to a constant global variable for the given file-scope
  /// compound literal expression.
  llvm::Constant *GetAddrOfConstantCompoundLiteral(const CompoundLiteralExpr*E);

  /// \brief Returns a pointer to a global variable representing a temporary
  /// with static or thread storage duration.
  llvm::Constant *GetAddrOfGlobalTemporary(const MaterializeTemporaryExpr *E,
                                           const Expr *Inner);

  /// \brief Retrieve the record type that describes the state of an
  /// Objective-C fast enumeration loop (for..in).
  QualType getObjCFastEnumerationStateType();
  
  /// Return the address of the constructor of the given type.
  llvm::GlobalValue *
  GetAddrOfCXXConstructor(const CXXConstructorDecl *ctor, CXXCtorType ctorType,
                          const CGFunctionInfo *fnInfo = nullptr,
                          bool DontDefer = false);

  /// Return the address of the constructor of the given type.
  llvm::GlobalValue *
  GetAddrOfCXXDestructor(const CXXDestructorDecl *dtor,
                         CXXDtorType dtorType,
                         const CGFunctionInfo *fnInfo = nullptr,
                         llvm::FunctionType *fnType = nullptr,
                         bool DontDefer = false);

  /// Given a builtin id for a function like "__builtin_fabsf", return a
  /// Function* for "fabsf".
  llvm::Value *getBuiltinLibFunction(const FunctionDecl *FD,
                                     unsigned BuiltinID);

  llvm::Function *getIntrinsic(unsigned IID, ArrayRef<llvm::Type*> Tys = None);

  /// Emit code for a single top level declaration.
  void EmitTopLevelDecl(Decl *D);

  /// Tell the consumer that this variable has been instantiated.
  void HandleCXXStaticMemberVarInstantiation(VarDecl *VD);

  /// \brief If the declaration has internal linkage but is inside an
  /// extern "C" linkage specification, prepare to emit an alias for it
  /// to the expected name.
  template<typename SomeDecl>
  void MaybeHandleStaticInExternC(const SomeDecl *D, llvm::GlobalValue *GV);

  /// Add a global to a list to be added to the llvm.used metadata.
  void addUsedGlobal(llvm::GlobalValue *GV);

  /// Add a global to a list to be added to the llvm.compiler.used metadata.
  void addCompilerUsedGlobal(llvm::GlobalValue *GV);

  /// Add a destructor and object to add to the C++ global destructor function.
  void AddCXXDtorEntry(llvm::Constant *DtorFn, llvm::Constant *Object) {
    CXXGlobalDtors.push_back(std::make_pair(DtorFn, Object));
  }

  /// Create a new runtime function with the specified type and name.
  llvm::Constant *CreateRuntimeFunction(llvm::FunctionType *Ty,
                                        StringRef Name,
                                        llvm::AttributeSet ExtraAttrs =
                                          llvm::AttributeSet());
  /// Create a new runtime global variable with the specified type and name.
  llvm::Constant *CreateRuntimeVariable(llvm::Type *Ty,
                                        StringRef Name);

  ///@name Custom Blocks Runtime Interfaces
  ///@{

  llvm::Constant *getNSConcreteGlobalBlock();
  llvm::Constant *getNSConcreteStackBlock();
  llvm::Constant *getBlockObjectAssign();
  llvm::Constant *getBlockObjectDispose();

  ///@}

  llvm::Constant *getLLVMLifetimeStartFn();
  llvm::Constant *getLLVMLifetimeEndFn();

  // Make sure that this type is translated.
  void UpdateCompletedType(const TagDecl *TD);

  llvm::Constant *getMemberPointerConstant(const UnaryOperator *e);

  /// Try to emit the initializer for the given declaration as a constant;
  /// returns 0 if the expression cannot be emitted as a constant.
  llvm::Constant *EmitConstantInit(const VarDecl &D,
                                   CodeGenFunction *CGF = nullptr);

  /// Try to emit the given expression as a constant; returns 0 if the
  /// expression cannot be emitted as a constant.
  llvm::Constant *EmitConstantExpr(const Expr *E, QualType DestType,
                                   CodeGenFunction *CGF = nullptr);

  /// Emit the given constant value as a constant, in the type's scalar
  /// representation.
  llvm::Constant *EmitConstantValue(const APValue &Value, QualType DestType,
                                    CodeGenFunction *CGF = nullptr);

  /// Emit the given constant value as a constant, in the type's memory
  /// representation.
  llvm::Constant *EmitConstantValueForMemory(const APValue &Value,
                                             QualType DestType,
                                             CodeGenFunction *CGF = nullptr);

  /// Return the result of value-initializing the given type, i.e. a null
  /// expression of the given type.  This is usually, but not always, an LLVM
  /// null constant.
  llvm::Constant *EmitNullConstant(QualType T);

  /// Return a null constant appropriate for zero-initializing a base class with
  /// the given type. This is usually, but not always, an LLVM null constant.
  llvm::Constant *EmitNullConstantForBase(const CXXRecordDecl *Record);

  /// Emit a general error that something can't be done.
  void Error(SourceLocation loc, StringRef error);

  /// Print out an error that codegen doesn't support the specified stmt yet.
  void ErrorUnsupported(const Stmt *S, const char *Type);

  /// Print out an error that codegen doesn't support the specified decl yet.
  void ErrorUnsupported(const Decl *D, const char *Type);

  /// Set the attributes on the LLVM function for the given decl and function
  /// info. This applies attributes necessary for handling the ABI as well as
  /// user specified attributes like section.
  void SetInternalFunctionAttributes(const Decl *D, llvm::Function *F,
                                     const CGFunctionInfo &FI);

  /// Set the LLVM function attributes (sext, zext, etc).
  void SetLLVMFunctionAttributes(const Decl *D,
                                 const CGFunctionInfo &Info,
                                 llvm::Function *F);

  /// Set the LLVM function attributes which only apply to a function
  /// definintion.
  void SetLLVMFunctionAttributesForDefinition(const Decl *D, llvm::Function *F);

  /// Return true iff the given type uses 'sret' when used as a return type.
  bool ReturnTypeUsesSRet(const CGFunctionInfo &FI);

  /// Return true iff the given type uses an argument slot when 'sret' is used
  /// as a return type.
  bool ReturnSlotInterferesWithArgs(const CGFunctionInfo &FI);

  /// Return true iff the given type uses 'fpret' when used as a return type.
  bool ReturnTypeUsesFPRet(QualType ResultType);

  /// Return true iff the given type uses 'fp2ret' when used as a return type.
  bool ReturnTypeUsesFP2Ret(QualType ResultType);

  /// Get the LLVM attributes and calling convention to use for a particular
  /// function type.
  ///
  /// \param Info - The function type information.
  /// \param TargetDecl - The decl these attributes are being constructed
  /// for. If supplied the attributes applied to this decl may contribute to the
  /// function attributes and calling convention.
  /// \param PAL [out] - On return, the attribute list to use.
  /// \param CallingConv [out] - On return, the LLVM calling convention to use.
  void ConstructAttributeList(const CGFunctionInfo &Info,
                              const Decl *TargetDecl,
                              AttributeListType &PAL,
                              unsigned &CallingConv,
                              bool AttrOnCallSite);

  StringRef getMangledName(GlobalDecl GD);
  StringRef getBlockMangledName(GlobalDecl GD, const BlockDecl *BD);

  void EmitTentativeDefinition(const VarDecl *D);

  void EmitVTable(CXXRecordDecl *Class, bool DefinitionRequired);

  /// Emit the RTTI descriptors for the builtin types.
  void EmitFundamentalRTTIDescriptors();

  /// \brief Appends Opts to the "Linker Options" metadata value.
  void AppendLinkerOptions(StringRef Opts);

  /// \brief Appends a detect mismatch command to the linker options.
  void AddDetectMismatch(StringRef Name, StringRef Value);

  /// \brief Appends a dependent lib to the "Linker Options" metadata value.
  void AddDependentLib(StringRef Lib);

  llvm::GlobalVariable::LinkageTypes getFunctionLinkage(GlobalDecl GD);

  void setFunctionLinkage(GlobalDecl GD, llvm::Function *F) {
    F->setLinkage(getFunctionLinkage(GD));
  }

  /// Return the appropriate linkage for the vtable, VTT, and type information
  /// of the given class.
  llvm::GlobalVariable::LinkageTypes getVTableLinkage(const CXXRecordDecl *RD);

  /// Return the store size, in character units, of the given LLVM type.
  CharUnits GetTargetTypeStoreSize(llvm::Type *Ty) const;
  
  /// Returns LLVM linkage for a declarator.
  llvm::GlobalValue::LinkageTypes
  getLLVMLinkageForDeclarator(const DeclaratorDecl *D, GVALinkage Linkage,
                              bool IsConstantVariable);

  /// Returns LLVM linkage for a declarator.
  llvm::GlobalValue::LinkageTypes
  getLLVMLinkageVarDefinition(const VarDecl *VD, bool IsConstant);

  /// Emit all the global annotations.
  void EmitGlobalAnnotations();

  /// Emit an annotation string.
  llvm::Constant *EmitAnnotationString(StringRef Str);

  /// Emit the annotation's translation unit.
  llvm::Constant *EmitAnnotationUnit(SourceLocation Loc);

  /// Emit the annotation line number.
  llvm::Constant *EmitAnnotationLineNo(SourceLocation L);

  /// Generate the llvm::ConstantStruct which contains the annotation
  /// information for a given GlobalValue. The annotation struct is
  /// {i8 *, i8 *, i8 *, i32}. The first field is a constant expression, the
  /// GlobalValue being annotated. The second field is the constant string
  /// created from the AnnotateAttr's annotation. The third field is a constant
  /// string containing the name of the translation unit. The fourth field is
  /// the line number in the file of the annotated value declaration.
  llvm::Constant *EmitAnnotateAttr(llvm::GlobalValue *GV,
                                   const AnnotateAttr *AA,
                                   SourceLocation L);

  /// Add global annotations that are set on D, for the global GV. Those
  /// annotations are emitted during finalization of the LLVM code.
  void AddGlobalAnnotations(const ValueDecl *D, llvm::GlobalValue *GV);

  const SanitizerBlacklist &getSanitizerBlacklist() const {
    return SanitizerBL;
  }

  void reportGlobalToASan(llvm::GlobalVariable *GV, const VarDecl &D,
                          bool IsDynInit = false);
  void reportGlobalToASan(llvm::GlobalVariable *GV, SourceLocation Loc,
                          StringRef Name, bool IsDynInit = false,
                          bool IsBlacklisted = false);

  /// Disable sanitizer instrumentation for this global.
  void disableSanitizerForGlobal(llvm::GlobalVariable *GV);

  void addDeferredVTable(const CXXRecordDecl *RD) {
    DeferredVTables.push_back(RD);
  }

  /// Emit code for a singal global function or var decl. Forward declarations
  /// are emitted lazily.
  void EmitGlobal(GlobalDecl D);

private:
  llvm::GlobalValue *GetGlobalValue(StringRef Ref);

  llvm::Constant *
  GetOrCreateLLVMFunction(StringRef MangledName, llvm::Type *Ty, GlobalDecl D,
                          bool ForVTable, bool DontDefer = false,
                          llvm::AttributeSet ExtraAttrs = llvm::AttributeSet());

  llvm::Constant *GetOrCreateLLVMGlobal(StringRef MangledName,
                                        llvm::PointerType *PTy,
                                        const VarDecl *D);

  /// Set attributes which are common to any form of a global definition (alias,
  /// Objective-C method, function, global variable).
  ///
  /// NOTE: This should only be called for definitions.
  void SetCommonAttributes(const Decl *D, llvm::GlobalValue *GV);

  void setNonAliasAttributes(const Decl *D, llvm::GlobalObject *GO);

  /// Set attributes for a global definition.
  void setFunctionDefinitionAttributes(const FunctionDecl *D,
                                       llvm::Function *F);

  /// Set function attributes for a function declaration.
  void SetFunctionAttributes(GlobalDecl GD,
                             llvm::Function *F,
                             bool IsIncompleteFunction);

  void EmitGlobalDefinition(GlobalDecl D, llvm::GlobalValue *GV = nullptr);

  void EmitGlobalFunctionDefinition(GlobalDecl GD, llvm::GlobalValue *GV);
  void EmitGlobalVarDefinition(const VarDecl *D);
  void EmitAliasDefinition(GlobalDecl GD);
  void EmitObjCPropertyImplementations(const ObjCImplementationDecl *D);
  void EmitObjCIvarInitializations(ObjCImplementationDecl *D);
  
  // C++ related functions.

  bool TryEmitDefinitionAsAlias(GlobalDecl Alias, GlobalDecl Target,
                                bool InEveryTU);
  bool TryEmitBaseDestructorAsAlias(const CXXDestructorDecl *D);

  void EmitNamespace(const NamespaceDecl *D);
  void EmitLinkageSpec(const LinkageSpecDecl *D);
  void CompleteDIClassType(const CXXMethodDecl* D);

  /// Emit a single constructor with the given type from a C++ constructor Decl.
  void EmitCXXConstructor(const CXXConstructorDecl *D, CXXCtorType Type);

  /// Emit a single destructor with the given type from a C++ destructor Decl.
  void EmitCXXDestructor(const CXXDestructorDecl *D, CXXDtorType Type);

  /// \brief Emit the function that initializes C++ thread_local variables.
  void EmitCXXThreadLocalInitFunc();

  /// Emit the function that initializes C++ globals.
  void EmitCXXGlobalInitFunc();

  /// Emit the function that destroys C++ globals.
  void EmitCXXGlobalDtorFunc();

  /// Emit the function that initializes the specified global (if PerformInit is
  /// true) and registers its destructor.
  void EmitCXXGlobalVarDeclInitFunc(const VarDecl *D,
                                    llvm::GlobalVariable *Addr,
                                    bool PerformInit);

  void EmitPointerToInitFunc(const VarDecl *VD, llvm::GlobalVariable *Addr,
                             llvm::Function *InitFunc, InitSegAttr *ISA);

  // FIXME: Hardcoding priority here is gross.
  void AddGlobalCtor(llvm::Function *Ctor, int Priority = 65535,
                     llvm::Constant *AssociatedData = 0);
  void AddGlobalDtor(llvm::Function *Dtor, int Priority = 65535);

  /// Generates a global array of functions and priorities using the given list
  /// and name. This array will have appending linkage and is suitable for use
  /// as a LLVM constructor or destructor array.
  void EmitCtorList(const CtorList &Fns, const char *GlobalName);

  /// Emit the RTTI descriptors for the given type.
  void EmitFundamentalRTTIDescriptor(QualType Type);

  /// Emit any needed decls for which code generation was deferred.
  void EmitDeferred();

  /// Call replaceAllUsesWith on all pairs in Replacements.
  void applyReplacements();

  void checkAliases();

  /// Emit any vtables which we deferred and still have a use for.
  void EmitDeferredVTables();

  /// Emit the llvm.used and llvm.compiler.used metadata.
  void emitLLVMUsed();

  /// \brief Emit the link options introduced by imported modules.
  void EmitModuleLinkOptions();

  /// \brief Emit aliases for internal-linkage declarations inside "C" language
  /// linkage specifications, giving them the "expected" name where possible.
  void EmitStaticExternCAliases();

  void EmitDeclMetadata();

  /// \brief Emit the Clang version as llvm.ident metadata.
  void EmitVersionIdentMetadata();

  /// Emits target specific Metadata for global declarations.
  void EmitTargetMetadata();

  /// Emit the llvm.gcov metadata used to tell LLVM where to emit the .gcno and
  /// .gcda files in a way that persists in .bc files.
  void EmitCoverageFile();

  /// Emits the initializer for a uuidof string.
  llvm::Constant *EmitUuidofInitializer(StringRef uuidstr, QualType IIDType);

  /// Determine if the given decl can be emitted lazily; this is only relevant
  /// for definitions. The given decl must be either a function or var decl.
  bool MayDeferGeneration(const ValueDecl *D);

  /// Check whether we can use a "simpler", more core exceptions personality
  /// function.
  void SimplifyPersonality();
};
}  // end namespace CodeGen
}  // end namespace clang

#endif
