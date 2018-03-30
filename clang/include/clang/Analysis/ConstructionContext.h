//===- ConstructionContext.h - CFG constructor information ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ConstructionContext class and its sub-classes,
// which represent various different ways of constructing C++ objects
// with the additional information the users may want to know about
// the constructor.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_CONSTRUCTIONCONTEXT_H
#define LLVM_CLANG_ANALYSIS_CONSTRUCTIONCONTEXT_H

#include "clang/Analysis/Support/BumpVector.h"
#include "clang/AST/ExprCXX.h"

namespace clang {

/// Construction context is a linked list of multiple layers. Layers are
/// created gradually while traversing the AST, and layers that represent
/// the outmost AST nodes are built first, while the node that immediately
/// contains the constructor would be built last and capture the previous
/// layers as its parents. Construction context captures the last layer
/// (which has links to the previous layers) and classifies the seemingly
/// arbitrary chain of layers into one of the possible ways of constructing
/// an object in C++ for user-friendly experience.
class ConstructionContextLayer {
public:
  typedef llvm::PointerUnion<Stmt *, CXXCtorInitializer *> TriggerTy;

private:
  /// The construction site - the statement that triggered the construction
  /// for one of its parts. For instance, stack variable declaration statement
  /// triggers construction of itself or its elements if it's an array,
  /// new-expression triggers construction of the newly allocated object(s).
  TriggerTy Trigger;

  /// Sometimes a single trigger is not enough to describe the construction
  /// site. In this case we'd have a chain of "partial" construction context
  /// layers.
  /// Some examples:
  /// - A constructor within in an aggregate initializer list within a variable
  ///   would have a construction context of the initializer list with
  ///   the parent construction context of a variable.
  /// - A constructor for a temporary that needs to be both destroyed
  ///   and materialized into an elidable copy constructor would have a
  ///   construction context of a CXXBindTemporaryExpr with the parent
  ///   construction context of a MaterializeTemproraryExpr.
  /// Not all of these are currently supported.
  const ConstructionContextLayer *Parent = nullptr;

  ConstructionContextLayer(TriggerTy Trigger,
                          const ConstructionContextLayer *Parent)
      : Trigger(Trigger), Parent(Parent) {}

public:
  static const ConstructionContextLayer *
  create(BumpVectorContext &C, TriggerTy Trigger,
         const ConstructionContextLayer *Parent = nullptr);

  const ConstructionContextLayer *getParent() const { return Parent; }
  bool isLast() const { return !Parent; }

  const Stmt *getTriggerStmt() const {
    return Trigger.dyn_cast<Stmt *>();
  }

  const CXXCtorInitializer *getTriggerInit() const {
    return Trigger.dyn_cast<CXXCtorInitializer *>();
  }

  /// Returns true if these layers are equal as individual layers, even if
  /// their parents are different.
  bool isSameLayer(const ConstructionContextLayer *Other) const {
    assert(Other);
    return (Trigger == Other->Trigger);
  }

  /// See if Other is a proper initial segment of this construction context
  /// in terms of the parent chain - i.e. a few first parents coincide and
  /// then the other context terminates but our context goes further - i.e.,
  /// we are providing the same context that the other context provides,
  /// and a bit more above that.
  bool isStrictlyMoreSpecificThan(const ConstructionContextLayer *Other) const;
};


/// ConstructionContext's subclasses describe different ways of constructing
/// an object in C++. The context re-captures the essential parent AST nodes
/// of the CXXConstructExpr it is assigned to and presents these nodes
/// through easy-to-understand accessor methods.
class ConstructionContext {
public:
  enum Kind {
    SimpleVariableKind,
    CXX17ElidedCopyVariableKind,
    VARIABLE_BEGIN = SimpleVariableKind,
    VARIABLE_END = CXX17ElidedCopyVariableKind,
    SimpleConstructorInitializerKind,
    CXX17ElidedCopyConstructorInitializerKind,
    INITIALIZER_BEGIN = SimpleConstructorInitializerKind,
    INITIALIZER_END = CXX17ElidedCopyConstructorInitializerKind,
    NewAllocatedObjectKind,
    TemporaryObjectKind,
    SimpleReturnedValueKind,
    CXX17ElidedCopyReturnedValueKind,
    RETURNED_VALUE_BEGIN = SimpleReturnedValueKind,
    RETURNED_VALUE_END = CXX17ElidedCopyReturnedValueKind
  };

protected:
  Kind K;

  // Do not make public! These need to only be constructed
  // via createFromLayers().
  explicit ConstructionContext(Kind K) : K(K) {}

private:
  // A helper function for constructing an instance into a bump vector context.
  template <typename T, typename... ArgTypes>
  static T *create(BumpVectorContext &C, ArgTypes... Args) {
    auto *CC = C.getAllocator().Allocate<T>();
    return new (CC) T(Args...);
  }

public:
  /// Consume the construction context layer, together with its parent layers,
  /// and wrap it up into a complete construction context. May return null
  /// if layers do not form any supported construction context.
  static const ConstructionContext *
  createFromLayers(BumpVectorContext &C,
                   const ConstructionContextLayer *TopLayer);

  Kind getKind() const { return K; }
};

/// An abstract base class for local variable constructors.
class VariableConstructionContext : public ConstructionContext {
  const DeclStmt *DS;

protected:
  VariableConstructionContext(ConstructionContext::Kind K, const DeclStmt *DS)
      : ConstructionContext(K), DS(DS) {
    assert(classof(this));
    assert(DS);
  }

public:
  const DeclStmt *getDeclStmt() const { return DS; }

  static bool classof(const ConstructionContext *CC) {
    return CC->getKind() >= VARIABLE_BEGIN &&
           CC->getKind() <= VARIABLE_END;
  }
};

/// Represents construction into a simple local variable, eg. T var(123);.
/// If a variable has an initializer, eg. T var = makeT();, then the final
/// elidable copy-constructor from makeT() into var would also be a simple
/// variable constructor handled by this class.
class SimpleVariableConstructionContext : public VariableConstructionContext {
  friend class ConstructionContext; // Allows to create<>() itself.

  explicit SimpleVariableConstructionContext(const DeclStmt *DS)
      : VariableConstructionContext(ConstructionContext::SimpleVariableKind,
                                    DS) {}

public:
  static bool classof(const ConstructionContext *CC) {
    return CC->getKind() == SimpleVariableKind;
  }
};

/// Represents construction into a simple variable with an initializer syntax,
/// with a single constructor, eg. T var = makeT();. Such construction context
/// may only appear in C++17 because previously it was split into a temporary
/// object constructor and an elidable simple variable copy-constructor and
/// we were producing separate construction contexts for these constructors.
/// In C++17 we have a single construction context that combines both.
/// Note that if the object has trivial destructor, then this code is
/// indistinguishable from a simple variable constructor on the AST level;
/// in this case we provide a simple variable construction context.
class CXX17ElidedCopyVariableConstructionContext
    : public VariableConstructionContext {
  const CXXBindTemporaryExpr *BTE;

  friend class ConstructionContext; // Allows to create<>() itself.

  explicit CXX17ElidedCopyVariableConstructionContext(
      const DeclStmt *DS, const CXXBindTemporaryExpr *BTE)
      : VariableConstructionContext(CXX17ElidedCopyVariableKind, DS), BTE(BTE) {
    assert(BTE);
  }

public:
  const CXXBindTemporaryExpr *getCXXBindTemporaryExpr() const { return BTE; }

  static bool classof(const ConstructionContext *CC) {
    return CC->getKind() == CXX17ElidedCopyVariableKind;
  }
};

// An abstract base class for constructor-initializer-based constructors.
class ConstructorInitializerConstructionContext : public ConstructionContext {
  const CXXCtorInitializer *I;

protected:
  explicit ConstructorInitializerConstructionContext(
      ConstructionContext::Kind K, const CXXCtorInitializer *I)
      : ConstructionContext(K), I(I) {
    assert(classof(this));
    assert(I);
  }

public:
  const CXXCtorInitializer *getCXXCtorInitializer() const { return I; }

  static bool classof(const ConstructionContext *CC) {
    return CC->getKind() >= INITIALIZER_BEGIN &&
           CC->getKind() <= INITIALIZER_END;
  }
};

/// Represents construction into a field or a base class within a bigger object
/// via a constructor initializer, eg. T(): field(123) { ... }.
class SimpleConstructorInitializerConstructionContext
    : public ConstructorInitializerConstructionContext {
  friend class ConstructionContext; // Allows to create<>() itself.

  explicit SimpleConstructorInitializerConstructionContext(
      const CXXCtorInitializer *I)
      : ConstructorInitializerConstructionContext(
            ConstructionContext::SimpleConstructorInitializerKind, I) {}

public:
  static bool classof(const ConstructionContext *CC) {
    return CC->getKind() == SimpleConstructorInitializerKind;
  }
};

/// Represents construction into a field or a base class within a bigger object
/// via a constructor initializer, with a single constructor, eg.
/// T(): field(Field(123)) { ... }. Such construction context may only appear
/// in C++17 because previously it was split into a temporary object constructor
/// and an elidable simple constructor-initializer copy-constructor and we were
/// producing separate construction contexts for these constructors. In C++17
/// we have a single construction context that combines both. Note that if the
/// object has trivial destructor, then this code is indistinguishable from
/// a simple constructor-initializer constructor on the AST level; in this case
/// we provide a simple constructor-initializer construction context.
class CXX17ElidedCopyConstructorInitializerConstructionContext
    : public ConstructorInitializerConstructionContext {
  const CXXBindTemporaryExpr *BTE;

  friend class ConstructionContext; // Allows to create<>() itself.

  explicit CXX17ElidedCopyConstructorInitializerConstructionContext(
      const CXXCtorInitializer *I, const CXXBindTemporaryExpr *BTE)
      : ConstructorInitializerConstructionContext(
            CXX17ElidedCopyConstructorInitializerKind, I),
        BTE(BTE) {
    assert(BTE);
  }

public:
  const CXXBindTemporaryExpr *getCXXBindTemporaryExpr() const { return BTE; }

  static bool classof(const ConstructionContext *CC) {
    return CC->getKind() == CXX17ElidedCopyConstructorInitializerKind;
  }
};

/// Represents immediate initialization of memory allocated by operator new,
/// eg. new T(123);.
class NewAllocatedObjectConstructionContext : public ConstructionContext {
  const CXXNewExpr *NE;

  friend class ConstructionContext; // Allows to create<>() itself.

  explicit NewAllocatedObjectConstructionContext(const CXXNewExpr *NE)
      : ConstructionContext(ConstructionContext::NewAllocatedObjectKind),
        NE(NE) {
    assert(NE);
  }

public:
  const CXXNewExpr *getCXXNewExpr() const { return NE; }

  static bool classof(const ConstructionContext *CC) {
    return CC->getKind() == NewAllocatedObjectKind;
  }
};

/// Represents a temporary object, eg. T(123), that does not immediately cross
/// function boundaries "by value"; constructors that construct function
/// value-type arguments or values that are immediately returned from the
/// function that returns a value receive separate construction context kinds.
class TemporaryObjectConstructionContext : public ConstructionContext {
  const CXXBindTemporaryExpr *BTE;
  const MaterializeTemporaryExpr *MTE;

  friend class ConstructionContext; // Allows to create<>() itself.

  explicit TemporaryObjectConstructionContext(
      const CXXBindTemporaryExpr *BTE, const MaterializeTemporaryExpr *MTE)
      : ConstructionContext(ConstructionContext::TemporaryObjectKind),
        BTE(BTE), MTE(MTE) {
    // Both BTE and MTE can be null here, all combinations possible.
    // Even though for now at least one should be non-null, we simply haven't
    // implemented this case yet (this would be a temporary in the middle of
    // nowhere that doesn't have a non-trivial destructor).
  }

public:
  /// CXXBindTemporaryExpr here is non-null as long as the temporary has
  /// a non-trivial destructor.
  const CXXBindTemporaryExpr *getCXXBindTemporaryExpr() const {
    return BTE;
  }

  /// MaterializeTemporaryExpr is non-null as long as the temporary is actually
  /// used after construction, eg. by binding to a reference (lifetime
  /// extension), accessing a field, calling a method, or passing it into
  /// a function (an elidable copy or move constructor would be a common
  /// example) by reference.
  const MaterializeTemporaryExpr *getMaterializedTemporaryExpr() const {
    return MTE;
  }

  static bool classof(const ConstructionContext *CC) {
    return CC->getKind() == TemporaryObjectKind;
  }
};

class ReturnedValueConstructionContext : public ConstructionContext {
  const ReturnStmt *RS;

protected:
  explicit ReturnedValueConstructionContext(ConstructionContext::Kind K,
                                            const ReturnStmt *RS)
      : ConstructionContext(K), RS(RS) {
    assert(classof(this));
    assert(RS);
  }

public:
  const ReturnStmt *getReturnStmt() const { return RS; }

  static bool classof(const ConstructionContext *CC) {
    return CC->getKind() >= RETURNED_VALUE_BEGIN &&
           CC->getKind() <= RETURNED_VALUE_END;
  }
};

/// Represents a temporary object that is being immediately returned from a
/// function by value, eg. return t; or return T(123);. In this case there is
/// always going to be a constructor at the return site. However, the usual
/// temporary-related bureaucracy (CXXBindTemporaryExpr,
/// MaterializeTemporaryExpr) is normally located in the caller function's AST.
class SimpleReturnedValueConstructionContext
    : public ReturnedValueConstructionContext {
  friend class ConstructionContext; // Allows to create<>() itself.

  explicit SimpleReturnedValueConstructionContext(const ReturnStmt *RS)
      : ReturnedValueConstructionContext(
            ConstructionContext::SimpleReturnedValueKind, RS) {}

public:
  static bool classof(const ConstructionContext *CC) {
    return CC->getKind() == SimpleReturnedValueKind;
  }
};

/// Represents a temporary object that is being immediately returned from a
/// function by value, eg. return t; or return T(123); in C++17.
/// In C++17 there is not going to be an elidable copy constructor at the
/// return site.  However, the usual temporary-related bureaucracy (CXXBindTemporaryExpr,
/// MaterializeTemporaryExpr) is normally located in the caller function's AST.
/// Note that if the object has trivial destructor, then this code is
/// indistinguishable from a simple returned value constructor on the AST level;
/// in this case we provide a simple returned value construction context.
class CXX17ElidedCopyReturnedValueConstructionContext
    : public ReturnedValueConstructionContext {
  const CXXBindTemporaryExpr *BTE;

  friend class ConstructionContext; // Allows to create<>() itself.

  explicit CXX17ElidedCopyReturnedValueConstructionContext(
      const ReturnStmt *RS, const CXXBindTemporaryExpr *BTE)
      : ReturnedValueConstructionContext(
            ConstructionContext::CXX17ElidedCopyReturnedValueKind, RS),
        BTE(BTE) {
    assert(BTE);
  }

public:
  const CXXBindTemporaryExpr *getCXXBindTemporaryExpr() const { return BTE; }

  static bool classof(const ConstructionContext *CC) {
    return CC->getKind() == CXX17ElidedCopyReturnedValueKind;
  }
};

} // end namespace clang

#endif // LLVM_CLANG_ANALYSIS_CONSTRUCTIONCONTEXT_H
