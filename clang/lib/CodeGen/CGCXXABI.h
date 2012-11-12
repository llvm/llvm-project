//===----- CGCXXABI.h - Interface to C++ ABIs -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for C++ code generation. Concrete subclasses
// of this implement code generation for specific C++ ABIs.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CXXABI_H
#define CLANG_CODEGEN_CXXABI_H

#include "clang/Basic/LLVM.h"

#include "CodeGenFunction.h"

namespace llvm {
  class Constant;
  class Type;
  class Value;
}

namespace clang {
  class CastExpr;
  class CXXConstructorDecl;
  class CXXDestructorDecl;
  class CXXMethodDecl;
  class CXXRecordDecl;
  class FieldDecl;
  class MangleContext;

namespace CodeGen {
  class CodeGenFunction;
  class CodeGenModule;

/// \brief Implements C++ ABI-specific code generation functions.
class CGCXXABI {
protected:
  CodeGenModule &CGM;
  OwningPtr<MangleContext> MangleCtx;

  CGCXXABI(CodeGenModule &CGM)
    : CGM(CGM), MangleCtx(CGM.getContext().createMangleContext()) {}

protected:
  ImplicitParamDecl *&getThisDecl(CodeGenFunction &CGF) {
    return CGF.CXXABIThisDecl;
  }
  llvm::Value *&getThisValue(CodeGenFunction &CGF) {
    return CGF.CXXABIThisValue;
  }

  ImplicitParamDecl *&getVTTDecl(CodeGenFunction &CGF) {
    return CGF.CXXVTTDecl;
  }
  llvm::Value *&getVTTValue(CodeGenFunction &CGF) {
    return CGF.CXXVTTValue;
  }

  /// Build a parameter variable suitable for 'this'.
  void BuildThisParam(CodeGenFunction &CGF, FunctionArgList &Params);

  /// Perform prolog initialization of the parameter variable suitable
  /// for 'this' emitted by BuildThisParam.
  void EmitThisParam(CodeGenFunction &CGF);

  ASTContext &getContext() const { return CGM.getContext(); }

  virtual bool requiresArrayCookie(const CXXDeleteExpr *E, QualType eltType);
  virtual bool requiresArrayCookie(const CXXNewExpr *E);

public:

  virtual ~CGCXXABI();

  /// Gets the mangle context.
  MangleContext &getMangleContext() {
    return *MangleCtx;
  }

  /// Find the LLVM type used to represent the given member pointer
  /// type.
  virtual llvm::Type *
  ConvertMemberPointerType(const MemberPointerType *MPT);

  /// Load a member function from an object and a member function
  /// pointer.  Apply the this-adjustment and set 'This' to the
  /// adjusted value.
  virtual llvm::Value *
  EmitLoadOfMemberFunctionPointer(CodeGenFunction &CGF,
                                  llvm::Value *&This,
                                  llvm::Value *MemPtr,
                                  const MemberPointerType *MPT);

  /// Calculate an l-value from an object and a data member pointer.
  virtual llvm::Value *EmitMemberDataPointerAddress(CodeGenFunction &CGF,
                                                    llvm::Value *Base,
                                                    llvm::Value *MemPtr,
                                            const MemberPointerType *MPT);

  /// Perform a derived-to-base, base-to-derived, or bitcast member
  /// pointer conversion.
  virtual llvm::Value *EmitMemberPointerConversion(CodeGenFunction &CGF,
                                                   const CastExpr *E,
                                                   llvm::Value *Src);

  /// Perform a derived-to-base, base-to-derived, or bitcast member
  /// pointer conversion on a constant value.
  virtual llvm::Constant *EmitMemberPointerConversion(const CastExpr *E,
                                                      llvm::Constant *Src);

  /// Return true if the given member pointer can be zero-initialized
  /// (in the C++ sense) with an LLVM zeroinitializer.
  virtual bool isZeroInitializable(const MemberPointerType *MPT);

  /// Create a null member pointer of the given type.
  virtual llvm::Constant *EmitNullMemberPointer(const MemberPointerType *MPT);

  /// Create a member pointer for the given method.
  virtual llvm::Constant *EmitMemberPointer(const CXXMethodDecl *MD);

  /// Create a member pointer for the given field.
  virtual llvm::Constant *EmitMemberDataPointer(const MemberPointerType *MPT,
                                                CharUnits offset);

  /// Create a member pointer for the given member pointer constant.
  virtual llvm::Constant *EmitMemberPointer(const APValue &MP, QualType MPT);

  /// Emit a comparison between two member pointers.  Returns an i1.
  virtual llvm::Value *
  EmitMemberPointerComparison(CodeGenFunction &CGF,
                              llvm::Value *L,
                              llvm::Value *R,
                              const MemberPointerType *MPT,
                              bool Inequality);

  /// Determine if a member pointer is non-null.  Returns an i1.
  virtual llvm::Value *
  EmitMemberPointerIsNotNull(CodeGenFunction &CGF,
                             llvm::Value *MemPtr,
                             const MemberPointerType *MPT);

protected:
  /// A utility method for computing the offset required for the given
  /// base-to-derived or derived-to-base member-pointer conversion.
  /// Does not handle virtual conversions (in case we ever fully
  /// support an ABI that allows this).  Returns null if no adjustment
  /// is required.
  llvm::Constant *getMemberPointerAdjustment(const CastExpr *E);

public:
  /// Adjust the given non-null pointer to an object of polymorphic
  /// type to point to the complete object.
  ///
  /// The IR type of the result should be a pointer but is otherwise
  /// irrelevant.
  virtual llvm::Value *adjustToCompleteObject(CodeGenFunction &CGF,
                                              llvm::Value *ptr,
                                              QualType type) = 0;

  /// Build the signature of the given constructor variant by adding
  /// any required parameters.  For convenience, ResTy has been
  /// initialized to 'void', and ArgTys has been initialized with the
  /// type of 'this' (although this may be changed by the ABI) and
  /// will have the formal parameters added to it afterwards.
  ///
  /// If there are ever any ABIs where the implicit parameters are
  /// intermixed with the formal parameters, we can address those
  /// then.
  virtual void BuildConstructorSignature(const CXXConstructorDecl *Ctor,
                                         CXXCtorType T,
                                         CanQualType &ResTy,
                               SmallVectorImpl<CanQualType> &ArgTys) = 0;

  /// Build the signature of the given destructor variant by adding
  /// any required parameters.  For convenience, ResTy has been
  /// initialized to 'void' and ArgTys has been initialized with the
  /// type of 'this' (although this may be changed by the ABI).
  virtual void BuildDestructorSignature(const CXXDestructorDecl *Dtor,
                                        CXXDtorType T,
                                        CanQualType &ResTy,
                               SmallVectorImpl<CanQualType> &ArgTys) = 0;

  /// Build the ABI-specific portion of the parameter list for a
  /// function.  This generally involves a 'this' parameter and
  /// possibly some extra data for constructors and destructors.
  ///
  /// ABIs may also choose to override the return type, which has been
  /// initialized with the formal return type of the function.
  virtual void BuildInstanceFunctionParams(CodeGenFunction &CGF,
                                           QualType &ResTy,
                                           FunctionArgList &Params) = 0;

  /// Emit the ABI-specific prolog for the function.
  virtual void EmitInstanceFunctionProlog(CodeGenFunction &CGF) = 0;

  virtual void EmitReturnFromThunk(CodeGenFunction &CGF,
                                   RValue RV, QualType ResultType);

  /// Gets the pure virtual member call function.
  virtual StringRef GetPureVirtualCallName() = 0;

  /// Gets the deleted virtual member call name.
  virtual StringRef GetDeletedVirtualCallName() = 0;

  /**************************** Array cookies ******************************/

  /// Returns the extra size required in order to store the array
  /// cookie for the given new-expression.  May return 0 to indicate that no
  /// array cookie is required.
  ///
  /// Several cases are filtered out before this method is called:
  ///   - non-array allocations never need a cookie
  ///   - calls to \::operator new(size_t, void*) never need a cookie
  ///
  /// \param expr - the new-expression being allocated.
  virtual CharUnits GetArrayCookieSize(const CXXNewExpr *expr);

  /// Initialize the array cookie for the given allocation.
  ///
  /// \param NewPtr - a char* which is the presumed-non-null
  ///   return value of the allocation function
  /// \param NumElements - the computed number of elements,
  ///   potentially collapsed from the multidimensional array case;
  ///   always a size_t
  /// \param ElementType - the base element allocated type,
  ///   i.e. the allocated type after stripping all array types
  virtual llvm::Value *InitializeArrayCookie(CodeGenFunction &CGF,
                                             llvm::Value *NewPtr,
                                             llvm::Value *NumElements,
                                             const CXXNewExpr *expr,
                                             QualType ElementType);

  /// Reads the array cookie associated with the given pointer,
  /// if it has one.
  ///
  /// \param Ptr - a pointer to the first element in the array
  /// \param ElementType - the base element type of elements of the array
  /// \param NumElements - an out parameter which will be initialized
  ///   with the number of elements allocated, or zero if there is no
  ///   cookie
  /// \param AllocPtr - an out parameter which will be initialized
  ///   with a char* pointing to the address returned by the allocation
  ///   function
  /// \param CookieSize - an out parameter which will be initialized
  ///   with the size of the cookie, or zero if there is no cookie
  virtual void ReadArrayCookie(CodeGenFunction &CGF, llvm::Value *Ptr,
                               const CXXDeleteExpr *expr,
                               QualType ElementType, llvm::Value *&NumElements,
                               llvm::Value *&AllocPtr, CharUnits &CookieSize);

protected:
  /// Returns the extra size required in order to store the array
  /// cookie for the given type.  Assumes that an array cookie is
  /// required.
  virtual CharUnits getArrayCookieSizeImpl(QualType elementType);

  /// Reads the array cookie for an allocation which is known to have one.
  /// This is called by the standard implementation of ReadArrayCookie.
  ///
  /// \param ptr - a pointer to the allocation made for an array, as a char*
  /// \param cookieSize - the computed cookie size of an array
  ///
  /// Other parameters are as above.
  ///
  /// \return a size_t
  virtual llvm::Value *readArrayCookieImpl(CodeGenFunction &IGF,
                                           llvm::Value *ptr,
                                           CharUnits cookieSize);

public:

  /*************************** Static local guards ****************************/

  /// Emits the guarded initializer and destructor setup for the given
  /// variable, given that it couldn't be emitted as a constant.
  /// If \p PerformInit is false, the initialization has been folded to a
  /// constant and should not be performed.
  ///
  /// The variable may be:
  ///   - a static local variable
  ///   - a static data member of a class template instantiation
  virtual void EmitGuardedInit(CodeGenFunction &CGF, const VarDecl &D,
                               llvm::GlobalVariable *DeclPtr, bool PerformInit);

  /// Emit code to force the execution of a destructor during global
  /// teardown.  The default implementation of this uses atexit.
  ///
  /// \param dtor - a function taking a single pointer argument
  /// \param addr - a pointer to pass to the destructor function.
  virtual void registerGlobalDtor(CodeGenFunction &CGF, llvm::Constant *dtor,
                                  llvm::Constant *addr);

  /***************************** Virtual Tables *******************************/

  /// Generates and emits the virtual tables for a class.
  virtual void EmitVTables(const CXXRecordDecl *Class) = 0;
};

/// Creates an instance of a C++ ABI class.
CGCXXABI *CreateARMCXXABI(CodeGenModule &CGM);
CGCXXABI *CreateItaniumCXXABI(CodeGenModule &CGM);
CGCXXABI *CreateMicrosoftCXXABI(CodeGenModule &CGM);

}
}

#endif
