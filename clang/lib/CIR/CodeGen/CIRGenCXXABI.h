//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for C++ code generation. Concrete subclasses
// of this implement code generation for specific C++ ABIs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CIRGENCXXABI_H
#define LLVM_CLANG_LIB_CIR_CIRGENCXXABI_H

#include "CIRGenCall.h"
#include "CIRGenCleanup.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"

#include "clang/AST/Mangle.h"

namespace clang::CIRGen {

/// Implements C++ ABI-specific code generation functions.
class CIRGenCXXABI {
protected:
  CIRGenModule &cgm;
  std::unique_ptr<clang::MangleContext> mangleContext;

  virtual bool requiresArrayCookie(const CXXNewExpr *e);

public:
  // TODO(cir): make this protected when target-specific CIRGenCXXABIs are
  // implemented.
  CIRGenCXXABI(CIRGenModule &cgm)
      : cgm(cgm), mangleContext(cgm.getASTContext().createMangleContext()) {}
  virtual ~CIRGenCXXABI();

  void setCXXABIThisValue(CIRGenFunction &cgf, mlir::Value thisPtr);

  /// Emit the code to initialize hidden members required to handle virtual
  /// inheritance, if needed by the ABI.
  virtual void
  initializeHiddenVirtualInheritanceMembers(CIRGenFunction &cgf,
                                            const CXXRecordDecl *rd) {}

  /// Emit a single constructor/destructor with the gen type from a C++
  /// constructor/destructor Decl.
  virtual void emitCXXStructor(clang::GlobalDecl gd) = 0;

  virtual mlir::Value
  getVirtualBaseClassOffset(mlir::Location loc, CIRGenFunction &cgf,
                            Address thisAddr, const CXXRecordDecl *classDecl,
                            const CXXRecordDecl *baseClassDecl) = 0;

  virtual mlir::Value emitDynamicCast(CIRGenFunction &cgf, mlir::Location loc,
                                      QualType srcRecordTy,
                                      QualType destRecordTy,
                                      cir::PointerType destCIRTy,
                                      bool isRefCast, Address src) = 0;

public:
  /// Similar to AddedStructorArgs, but only notes the number of additional
  /// arguments.
  struct AddedStructorArgCounts {
    unsigned prefix = 0;
    unsigned suffix = 0;
    AddedStructorArgCounts() = default;
    AddedStructorArgCounts(unsigned p, unsigned s) : prefix(p), suffix(s) {}
    static AddedStructorArgCounts withPrefix(unsigned n) { return {n, 0}; }
    static AddedStructorArgCounts withSuffix(unsigned n) { return {0, n}; }
  };

  /// Additional implicit arguments to add to the beginning (Prefix) and end
  /// (Suffix) of a constructor / destructor arg list.
  ///
  /// Note that Prefix should actually be inserted *after* the first existing
  /// arg; `this` arguments always come first.
  struct AddedStructorArgs {
    struct Arg {
      mlir::Value value;
      QualType type;
    };
    llvm::SmallVector<Arg, 1> prefix;
    llvm::SmallVector<Arg, 1> suffix;
    AddedStructorArgs() = default;
    AddedStructorArgs(llvm::SmallVector<Arg, 1> p, llvm::SmallVector<Arg, 1> s)
        : prefix(std::move(p)), suffix(std::move(s)) {}
    static AddedStructorArgs withPrefix(llvm::SmallVector<Arg, 1> args) {
      return {std::move(args), {}};
    }
    static AddedStructorArgs withSuffix(llvm::SmallVector<Arg, 1> args) {
      return {{}, std::move(args)};
    }
  };

  /// Build the signature of the given constructor or destructor vairant by
  /// adding any required parameters. For convenience, ArgTys has been
  /// initialized with the type of 'this'.
  virtual AddedStructorArgCounts
  buildStructorSignature(GlobalDecl gd,
                         llvm::SmallVectorImpl<CanQualType> &argTys) = 0;

  AddedStructorArgCounts
  addImplicitConstructorArgs(CIRGenFunction &cgf, const CXXConstructorDecl *d,
                             CXXCtorType type, bool forVirtualBase,
                             bool delegating, CallArgList &args);

  clang::ImplicitParamDecl *getThisDecl(CIRGenFunction &cgf) {
    return cgf.cxxabiThisDecl;
  }

  virtual AddedStructorArgs
  getImplicitConstructorArgs(CIRGenFunction &cgf, const CXXConstructorDecl *d,
                             CXXCtorType type, bool forVirtualBase,
                             bool delegating) = 0;

  /// Emit the ABI-specific prolog for the function
  virtual void emitInstanceFunctionProlog(SourceLocation loc,
                                          CIRGenFunction &cgf) = 0;

  virtual void emitRethrow(CIRGenFunction &cgf, bool isNoReturn) = 0;
  virtual void emitThrow(CIRGenFunction &cgf, const CXXThrowExpr *e) = 0;

  virtual mlir::Attribute getAddrOfRTTIDescriptor(mlir::Location loc,
                                                  QualType ty) = 0;

  /// Get the type of the implicit "this" parameter used by a method. May return
  /// zero if no specific type is applicable, e.g. if the ABI expects the "this"
  /// parameter to point to some artificial offset in a complete object due to
  /// vbases being reordered.
  virtual const clang::CXXRecordDecl *
  getThisArgumentTypeForMethod(const clang::CXXMethodDecl *md) {
    return md->getParent();
  }

  /// Return whether the given global decl needs a VTT (virtual table table)
  /// parameter.
  virtual bool needsVTTParameter(clang::GlobalDecl gd) { return false; }

  /// Perform ABI-specific "this" argument adjustment required prior to
  /// a call of a virtual function.
  /// The "VirtualCall" argument is true iff the call itself is virtual.
  virtual Address adjustThisArgumentForVirtualFunctionCall(CIRGenFunction &cgf,
                                                           clang::GlobalDecl gd,
                                                           Address thisPtr,
                                                           bool virtualCall) {
    return thisPtr;
  }

  /// Build a parameter variable suitable for 'this'.
  void buildThisParam(CIRGenFunction &cgf, FunctionArgList &params);

  /// Loads the incoming C++ this pointer as it was passed by the caller.
  mlir::Value loadIncomingCXXThis(CIRGenFunction &cgf);

  virtual CatchTypeInfo getCatchAllTypeInfo();

  /// Get the implicit (second) parameter that comes after the "this" pointer,
  /// or nullptr if there is isn't one.
  virtual mlir::Value getCXXDestructorImplicitParam(CIRGenFunction &cgf,
                                                    const CXXDestructorDecl *dd,
                                                    CXXDtorType type,
                                                    bool forVirtualBase,
                                                    bool delegating) = 0;

  /// Emit constructor variants required by this ABI.
  virtual void emitCXXConstructors(const clang::CXXConstructorDecl *d) = 0;

  /// Emit dtor variants required by this ABI.
  virtual void emitCXXDestructors(const clang::CXXDestructorDecl *d) = 0;

  virtual void emitDestructorCall(CIRGenFunction &cgf,
                                  const CXXDestructorDecl *dd, CXXDtorType type,
                                  bool forVirtualBase, bool delegating,
                                  Address thisAddr, QualType thisTy) = 0;

  /// Emit code to force the execution of a destructor during global
  /// teardown.  The default implementation of this uses atexit.
  ///
  /// \param dtor - a function taking a single pointer argument
  /// \param addr - a pointer to pass to the destructor function.
  virtual void registerGlobalDtor(const VarDecl *vd, cir::FuncOp dtor,
                                  mlir::Value addr) = 0;

  /// Checks if ABI requires extra virtual offset for vtable field.
  virtual bool
  isVirtualOffsetNeededForVTableField(CIRGenFunction &cgf,
                                      CIRGenFunction::VPtr vptr) = 0;

  /// Emits the VTable definitions required for the given record type.
  virtual void emitVTableDefinitions(CIRGenVTables &cgvt,
                                     const CXXRecordDecl *rd) = 0;

  using DeleteOrMemberCallExpr =
      llvm::PointerUnion<const CXXDeleteExpr *, const CXXMemberCallExpr *>;

  virtual mlir::Value emitVirtualDestructorCall(CIRGenFunction &cgf,
                                                const CXXDestructorDecl *dtor,
                                                CXXDtorType dtorType,
                                                Address thisAddr,
                                                DeleteOrMemberCallExpr e) = 0;

  /// Emit any tables needed to implement virtual inheritance.  For Itanium,
  /// this emits virtual table tables.
  virtual void emitVirtualInheritanceTables(const CXXRecordDecl *rd) = 0;

  /// Returns true if the given destructor type should be emitted as a linkonce
  /// delegating thunk, regardless of whether the dtor is defined in this TU or
  /// not.
  virtual bool useThunkForDtorVariant(const CXXDestructorDecl *dtor,
                                      CXXDtorType dt) const = 0;

  virtual cir::GlobalLinkageKind
  getCXXDestructorLinkage(GVALinkage linkage, const CXXDestructorDecl *dtor,
                          CXXDtorType dt) const;

  /// Get the address of the vtable for the given record decl which should be
  /// used for the vptr at the given offset in RD.
  virtual cir::GlobalOp getAddrOfVTable(const CXXRecordDecl *rd,
                                        CharUnits vptrOffset) = 0;

  /// Build a virtual function pointer in the ABI-specific way.
  virtual CIRGenCallee getVirtualFunctionPointer(CIRGenFunction &cgf,
                                                 clang::GlobalDecl gd,
                                                 Address thisAddr,
                                                 mlir::Type ty,
                                                 SourceLocation loc) = 0;

  /// Get the address point of the vtable for the given base subobject.
  virtual mlir::Value
  getVTableAddressPoint(BaseSubobject base,
                        const CXXRecordDecl *vtableClass) = 0;

  /// Get the address point of the vtable for the given base subobject while
  /// building a constructor or a destructor.
  virtual mlir::Value getVTableAddressPointInStructor(
      CIRGenFunction &cgf, const CXXRecordDecl *vtableClass, BaseSubobject base,
      const CXXRecordDecl *nearestVBase) = 0;

  /// Insert any ABI-specific implicit parameters into the parameter list for a
  /// function. This generally involves extra data for constructors and
  /// destructors.
  ///
  /// ABIs may also choose to override the return type, which has been
  /// initialized with the type of 'this' if HasThisReturn(CGF.CurGD) is true or
  /// the formal return type of the function otherwise.
  virtual void addImplicitStructorParams(CIRGenFunction &cgf,
                                         clang::QualType &resTy,
                                         FunctionArgList &params) = 0;

  /// Checks if ABI requires to initialize vptrs for given dynamic class.
  virtual bool
  doStructorsInitializeVPtrs(const clang::CXXRecordDecl *vtableClass) = 0;

  /// Returns true if the given constructor or destructor is one of the kinds
  /// that the ABI says returns 'this' (only applies when called non-virtually
  /// for destructors).
  ///
  /// There currently is no way to indicate if a destructor returns 'this' when
  /// called virtually, and CIR generation does not support this case.
  virtual bool hasThisReturn(clang::GlobalDecl gd) const { return false; }

  virtual bool hasMostDerivedReturn(clang::GlobalDecl gd) const {
    return false;
  }

  /// Returns true if the target allows calling a function through a pointer
  /// with a different signature than the actual function (or equivalently,
  /// bitcasting a function or function pointer to a different function type).
  /// In principle in the most general case this could depend on the target, the
  /// calling convention, and the actual types of the arguments and return
  /// value. Here it just means whether the signature mismatch could *ever* be
  /// allowed; in other words, does the target do strict checking of signatures
  /// for all calls.
  virtual bool canCallMismatchedFunctionType() const { return true; }

  /// Gets the mangle context.
  clang::MangleContext &getMangleContext() { return *mangleContext; }

  clang::ImplicitParamDecl *&getStructorImplicitParamDecl(CIRGenFunction &cgf) {
    return cgf.cxxStructorImplicitParamDecl;
  }

  mlir::Value getStructorImplicitParamValue(CIRGenFunction &cgf) {
    return cgf.cxxStructorImplicitParamValue;
  }

  void setStructorImplicitParamValue(CIRGenFunction &cgf, mlir::Value val) {
    cgf.cxxStructorImplicitParamValue = val;
  }

  /**************************** Array cookies ******************************/

  /// Returns the extra size required in order to store the array
  /// cookie for the given new-expression.  May return 0 to indicate that no
  /// array cookie is required.
  ///
  /// Several cases are filtered out before this method is called:
  ///   - non-array allocations never need a cookie
  ///   - calls to \::operator new(size_t, void*) never need a cookie
  ///
  /// \param e - the new-expression being allocated.
  virtual CharUnits getArrayCookieSize(const CXXNewExpr *e);

  /// Initialize the array cookie for the given allocation.
  ///
  /// \param newPtr - a char* which is the presumed-non-null
  ///   return value of the allocation function
  /// \param numElements - the computed number of elements,
  ///   potentially collapsed from the multidimensional array case;
  ///   always a size_t
  /// \param elementType - the base element allocated type,
  ///   i.e. the allocated type after stripping all array types
  virtual Address initializeArrayCookie(CIRGenFunction &cgf, Address newPtr,
                                        mlir::Value numElements,
                                        const CXXNewExpr *e,
                                        QualType elementType) = 0;

protected:
  /// Returns the extra size required in order to store the array
  /// cookie for the given type.  Assumes that an array cookie is
  /// required.
  virtual CharUnits getArrayCookieSizeImpl(QualType elementType) = 0;
};

/// Creates and Itanium-family ABI
CIRGenCXXABI *CreateCIRGenItaniumCXXABI(CIRGenModule &cgm);

} // namespace clang::CIRGen

#endif
