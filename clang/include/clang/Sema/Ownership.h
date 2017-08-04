//===--- Ownership.h - Parser ownership helpers -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file contains classes for managing ownership of Stmt and Expr nodes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_OWNERSHIP_H
#define LLVM_CLANG_SEMA_OWNERSHIP_H

#include "clang/AST/Expr.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

//===----------------------------------------------------------------------===//
// OpaquePtr
//===----------------------------------------------------------------------===//

namespace clang {
  class CXXCtorInitializer;
  class CXXBaseSpecifier;
  class Decl;
  class Expr;
  class ParsedTemplateArgument;
  class QualType;
  class Stmt;
  class TemplateName;
  class TemplateParameterList;

  /// \brief Wrapper for void* pointer.
  /// \tparam PtrTy Either a pointer type like 'T*' or a type that behaves like
  ///               a pointer.
  ///
  /// This is a very simple POD type that wraps a pointer that the Parser
  /// doesn't know about but that Sema or another client does.  The PtrTy
  /// template argument is used to make sure that "Decl" pointers are not
  /// compatible with "Type" pointers for example.
  template <class PtrTy>
  class OpaquePtr {
    void *Ptr = nullptr;
    explicit OpaquePtr(void *Ptr) : Ptr(Ptr) {}

    typedef llvm::PointerLikeTypeTraits<PtrTy> Traits;

  public:
    OpaquePtr(std::nullptr_t = nullptr) {}

    static OpaquePtr make(PtrTy P) { OpaquePtr OP; OP.set(P); return OP; }

    /// \brief Returns plain pointer to the entity pointed by this wrapper.
    /// \tparam PointeeT Type of pointed entity.
    ///
    /// It is identical to getPtrAs<PointeeT*>.
    template <typename PointeeT> PointeeT* getPtrTo() const {
      return get();
    }

    /// \brief Returns pointer converted to the specified type.
    /// \tparam PtrT Result pointer type.  There must be implicit conversion
    ///              from PtrTy to PtrT.
    ///
    /// In contrast to getPtrTo, this method allows the return type to be
    /// a smart pointer.
    template <typename PtrT> PtrT getPtrAs() const {
      return get();
    }

    PtrTy get() const {
      return Traits::getFromVoidPointer(Ptr);
    }

    void set(PtrTy P) {
      Ptr = Traits::getAsVoidPointer(P);
    }

    explicit operator bool() const { return Ptr != nullptr; }

    void *getAsOpaquePtr() const { return Ptr; }
    static OpaquePtr getFromOpaquePtr(void *P) { return OpaquePtr(P); }
  };

  /// UnionOpaquePtr - A version of OpaquePtr suitable for membership
  /// in a union.
  template <class T> struct UnionOpaquePtr {
    void *Ptr;

    static UnionOpaquePtr make(OpaquePtr<T> P) {
      UnionOpaquePtr OP = { P.getAsOpaquePtr() };
      return OP;
    }

    OpaquePtr<T> get() const { return OpaquePtr<T>::getFromOpaquePtr(Ptr); }
    operator OpaquePtr<T>() const { return get(); }

    UnionOpaquePtr &operator=(OpaquePtr<T> P) {
      Ptr = P.getAsOpaquePtr();
      return *this;
    }
  };
}

namespace llvm {
  template <class T>
  class PointerLikeTypeTraits<clang::OpaquePtr<T> > {
  public:
    static inline void *getAsVoidPointer(clang::OpaquePtr<T> P) {
      // FIXME: Doesn't work? return P.getAs< void >();
      return P.getAsOpaquePtr();
    }
    static inline clang::OpaquePtr<T> getFromVoidPointer(void *P) {
      return clang::OpaquePtr<T>::getFromOpaquePtr(P);
    }
    enum { NumLowBitsAvailable = 0 };
  };

  template <class T>
  struct isPodLike<clang::OpaquePtr<T> > { static const bool value = true; };
}

namespace clang {
  // Basic
  class DiagnosticBuilder;

  // Determines whether the low bit of the result pointer for the
  // given UID is always zero. If so, ActionResult will use that bit
  // for it's "invalid" flag.
  template<class Ptr>
  struct IsResultPtrLowBitFree {
    static const bool value = false;
  };

  /// ActionResult - This structure is used while parsing/acting on
  /// expressions, stmts, etc.  It encapsulates both the object returned by
  /// the action, plus a sense of whether or not it is valid.
  /// When CompressInvalid is true, the "invalid" flag will be
  /// stored in the low bit of the Val pointer.
  template<class PtrTy,
           bool CompressInvalid = IsResultPtrLowBitFree<PtrTy>::value>
  class ActionResult {
    PtrTy Val;
    bool Invalid;

  public:
    ActionResult(bool Invalid = false)
      : Val(PtrTy()), Invalid(Invalid) {}
    ActionResult(PtrTy val) : Val(val), Invalid(false) {}
    ActionResult(const DiagnosticBuilder &) : Val(PtrTy()), Invalid(true) {}

    // These two overloads prevent void* -> bool conversions.
    ActionResult(const void *) = delete;
    ActionResult(volatile void *) = delete;

    bool isInvalid() const { return Invalid; }
    bool isUsable() const { return !Invalid && Val; }
    bool isUnset() const { return !Invalid && !Val; }

    PtrTy get() const { return Val; }
    template <typename T> T *getAs() { return static_cast<T*>(get()); }

    void set(PtrTy V) { Val = V; }

    const ActionResult &operator=(PtrTy RHS) {
      Val = RHS;
      Invalid = false;
      return *this;
    }
  };

  // This ActionResult partial specialization places the "invalid"
  // flag into the low bit of the pointer.
  template<typename PtrTy>
  class ActionResult<PtrTy, true> {
    // A pointer whose low bit is 1 if this result is invalid, 0
    // otherwise.
    uintptr_t PtrWithInvalid;
    typedef llvm::PointerLikeTypeTraits<PtrTy> PtrTraits;
  public:
    ActionResult(bool Invalid = false)
      : PtrWithInvalid(static_cast<uintptr_t>(Invalid)) { }

    ActionResult(PtrTy V) {
      void *VP = PtrTraits::getAsVoidPointer(V);
      PtrWithInvalid = reinterpret_cast<uintptr_t>(VP);
      assert((PtrWithInvalid & 0x01) == 0 && "Badly aligned pointer");
    }
    ActionResult(const DiagnosticBuilder &) : PtrWithInvalid(0x01) { }

    // These two overloads prevent void* -> bool conversions.
    ActionResult(const void *) = delete;
    ActionResult(volatile void *) = delete;

    bool isInvalid() const { return PtrWithInvalid & 0x01; }
    bool isUsable() const { return PtrWithInvalid > 0x01; }
    bool isUnset() const { return PtrWithInvalid == 0; }

    PtrTy get() const {
      void *VP = reinterpret_cast<void *>(PtrWithInvalid & ~0x01);
      return PtrTraits::getFromVoidPointer(VP);
    }
    template <typename T> T *getAs() { return static_cast<T*>(get()); }

    void set(PtrTy V) {
      void *VP = PtrTraits::getAsVoidPointer(V);
      PtrWithInvalid = reinterpret_cast<uintptr_t>(VP);
      assert((PtrWithInvalid & 0x01) == 0 && "Badly aligned pointer");
    }

    const ActionResult &operator=(PtrTy RHS) {
      void *VP = PtrTraits::getAsVoidPointer(RHS);
      PtrWithInvalid = reinterpret_cast<uintptr_t>(VP);
      assert((PtrWithInvalid & 0x01) == 0 && "Badly aligned pointer");
      return *this;
    }

    // For types where we can fit a flag in with the pointer, provide
    // conversions to/from pointer type.
    static ActionResult getFromOpaquePointer(void *P) {
      ActionResult Result;
      Result.PtrWithInvalid = (uintptr_t)P;
      return Result;
    }
    void *getAsOpaquePointer() const { return (void*)PtrWithInvalid; }
  };

  /// An opaque type for threading parsed type information through the
  /// parser.
  typedef OpaquePtr<QualType> ParsedType;
  typedef UnionOpaquePtr<QualType> UnionParsedType;

  // We can re-use the low bit of expression, statement, base, and
  // member-initializer pointers for the "invalid" flag of
  // ActionResult.
  template<> struct IsResultPtrLowBitFree<Expr*> {
    static const bool value = true;
  };
  template<> struct IsResultPtrLowBitFree<Stmt*> {
    static const bool value = true;
  };
  template<> struct IsResultPtrLowBitFree<CXXBaseSpecifier*> {
    static const bool value = true;
  };
  template<> struct IsResultPtrLowBitFree<CXXCtorInitializer*> {
    static const bool value = true;
  };

  typedef ActionResult<Expr*> ExprResult;
  typedef ActionResult<Stmt*> StmtResult;
  typedef ActionResult<ParsedType> TypeResult;
  typedef ActionResult<CXXBaseSpecifier*> BaseResult;
  typedef ActionResult<CXXCtorInitializer*> MemInitResult;

  typedef ActionResult<Decl*> DeclResult;
  typedef OpaquePtr<TemplateName> ParsedTemplateTy;
  typedef UnionOpaquePtr<TemplateName> UnionParsedTemplateTy;

  typedef MutableArrayRef<Expr*> MultiExprArg;
  typedef MutableArrayRef<Stmt*> MultiStmtArg;
  typedef MutableArrayRef<ParsedTemplateArgument> ASTTemplateArgsPtr;
  typedef MutableArrayRef<ParsedType> MultiTypeArg;
  typedef MutableArrayRef<TemplateParameterList*> MultiTemplateParamsArg;

  inline ExprResult ExprError() { return ExprResult(true); }
  inline StmtResult StmtError() { return StmtResult(true); }

  inline ExprResult ExprError(const DiagnosticBuilder&) { return ExprError(); }
  inline StmtResult StmtError(const DiagnosticBuilder&) { return StmtError(); }

  inline ExprResult ExprEmpty() { return ExprResult(false); }
  inline StmtResult StmtEmpty() { return StmtResult(false); }

  inline Expr *AssertSuccess(ExprResult R) {
    assert(!R.isInvalid() && "operation was asserted to never fail!");
    return R.get();
  }

  inline Stmt *AssertSuccess(StmtResult R) {
    assert(!R.isInvalid() && "operation was asserted to never fail!");
    return R.get();
  }
}

#endif
