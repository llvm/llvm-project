// RUN: %check_clang_tidy -std=c++11-or-later --extra-arg=-Wno-unevaluated-expression %s bugprone-rethrow-caught-exception %t -- -- -fexceptions -fblocks

#include <utility>

struct BaseError {};
struct DerivedError : BaseError {};
struct OtherError {};
using AliasError = BaseError;
typedef BaseError TypedefError;
typedef void (^Thunk)();
typedef const BaseError &BaseRef;
template <class T> using AliasRef = const T &;
// A copy constructor with a defaulted trailing parameter still copies, but
// an explicitly spelled construction is a deliberate new object.
struct DefaultedCopyError {
  int Value;
  explicit DefaultedCopyError(int V) : Value(V) {}
  DefaultedCopyError(const DefaultedCopyError &Other, int = 0)
      : Value(Other.Value + 1) {}
};

void mayThrow();
void consume(int);
int getInt();

void rethrowConstRef() {
  try {
    mayThrow();
  } catch (const BaseError &Err) {
    throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void rethrowNonConstRef() {
  try {
    mayThrow();
  } catch (BaseError &Err) {
    throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void rethrowScalar() {
  try {
    mayThrow();
  } catch (const int &Err) {
    throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'int' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void rethrowParens() {
  try {
    mayThrow();
  } catch (BaseError &Err) {
    throw(Err);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void rethrowMove() {
  try {
    mayThrow();
  } catch (BaseError &Err) {
    throw std::move(Err);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void explicitConstructionIsIgnored() {
  try {
    mayThrow();
  } catch (const BaseError &Err) {
    throw BaseError(Err);
  }
}

// An explicitly spelled construction is a deliberate new object, not a
// rethrow typo. Rewriting it to a bare `throw;` would drop the copy
// constructor side effect below (2 vs 1).
struct CountingError {
  int Value;
  explicit CountingError(int V) : Value(V) {}
  CountingError(const CountingError &Other) : Value(Other.Value + 1) {}
};

void explicitParenConstructionIsIgnored() {
  try {
    mayThrow();
  } catch (const CountingError &Err) {
    throw CountingError(Err);
  }
}

void explicitBracedConstructionIsIgnored() {
  try {
    mayThrow();
  } catch (const CountingError &Err) {
    throw CountingError{Err};
  }
}

void rethrowDefaultedCopy() {
  try {
    mayThrow();
  } catch (const DefaultedCopyError &Err) {
    throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'DefaultedCopyError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void explicitDefaultedParenIsIgnored() {
  try {
    mayThrow();
  } catch (const DefaultedCopyError &Err) {
    throw DefaultedCopyError(Err);
  }
}

void explicitDefaultedBracedIsIgnored() {
  try {
    mayThrow();
  } catch (const DefaultedCopyError &Err) {
    throw DefaultedCopyError{Err};
  }
}

void noexceptOperandIsIgnored() {
  try {
    mayThrow();
  } catch (const BaseError &Err) {
    bool NoThrow = noexcept(throw Err);
    consume(NoThrow ? 1 : 0);
  }
}

void decltypeOperandIsIgnored() {
  try {
    mayThrow();
  } catch (const BaseError &Err) {
    using CommaType = decltype((throw Err, 0));
    consume(sizeof(CommaType));
  }
}

void rethrowAlias() {
  try {
    mayThrow();
  } catch (const AliasError &Err) {
    throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void rethrowTypedef() {
  try {
    mayThrow();
  } catch (TypedefError &Err) {
    throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void rethrowTypedefRef() {
  try {
    mayThrow();
  } catch (BaseRef Err) {
    throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void rethrowAliasTemplateRef() {
  try {
    mayThrow();
  } catch (AliasRef<BaseError> Err) {
    throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void rethrowTwiceInOneHandler(int Selector) {
  try {
    mayThrow();
  } catch (const BaseError &Err) {
    if (Selector == 1)
      throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
    throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void lambdaSiblingDirectWarns() {
  try {
    mayThrow();
  } catch (const BaseError &Err) {
    auto Fn = [&] { throw Err; };
    (void)Fn;
    throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void blockSiblingDirectWarns() {
  try {
    mayThrow();
  } catch (const BaseError &Err) {
    Thunk Run = ^{ throw Err; };
    (void)Run;
    throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void innerOuterSiblingDirectWarns() {
  try {
    mayThrow();
  } catch (const BaseError &Outer) {
    try {
      mayThrow();
    } catch (const OtherError &) {
      throw Outer;
    }
    throw Outer;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void innerOwnAndOuterDirectBothWarn() {
  try {
    mayThrow();
  } catch (const BaseError &Outer) {
    try {
      mayThrow();
    } catch (const OtherError &Inner) {
      throw Inner;
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: throwing a copy of the caught 'OtherError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
      // CHECK-FIXES: throw;
    }
    throw Outer;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void directBeforeLambdaWarns(int Selector) {
  try {
    mayThrow();
  } catch (const BaseError &Err) {
    if (Selector == 1)
      throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
    auto Fn = [&] { throw Err; };
    (void)Fn;
  }
}

void rethrowInBranch() {
  try {
    mayThrow();
  } catch (const BaseError &Err) {
    if (getInt() > 0)
      throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void rethrowAfterNestedTry() {
  try {
    mayThrow();
  } catch (const BaseError &Err) {
    try {
      mayThrow();
    } catch (...) {
    }
    throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void rethrowInnerCatchOwnVar() {
  try {
    mayThrow();
  } catch (const BaseError &Outer) {
    try {
      mayThrow();
    } catch (const OtherError &Inner) {
      throw Inner;
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: throwing a copy of the caught 'OtherError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
      // CHECK-FIXES: throw;
    }
  }
}

void handleOverload(int);
void handleOverload(BaseError &);

void overloadInt(int Value) {
  try {
    mayThrow();
  } catch (int &Err) {
    consume(Err);
    handleOverload(Value);
    throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'int' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void overloadError(BaseError &ErrParam) {
  try {
    mayThrow();
  } catch (BaseError &Err) {
    handleOverload(ErrParam);
    throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

template <typename T> void rethrowTemplate() {
  try {
    mayThrow();
  } catch (const T &Err) {
    throw Err;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'type-parameter-0-0' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: throw;
  }
}

void instantiateTemplate() {
  rethrowTemplate<BaseError>();
  rethrowTemplate<int>();
}

#define RETHROW_CAUGHT(X) throw X

void rethrowMacro() {
  try {
    mayThrow();
  } catch (const BaseError &Err) {
    RETHROW_CAUGHT(Err);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
    // CHECK-FIXES: RETHROW_CAUGHT(Err);
  }
}

void bareThrowIsFine() {
  try {
    mayThrow();
  } catch (const BaseError &) {
    throw;
  }
}

void catchAllIsFine() {
  try {
    mayThrow();
  } catch (...) {
    throw;
  }
}

void catchByValueIsIgnored(int Ignored) {
  try {
    mayThrow();
  } catch (BaseError Err) {
    throw Err;
  } catch (int Count) {
    Count = 67;
    throw Count;
  }
  consume(Ignored);
}

void shadowedVarIsIgnored() {
  try {
    mayThrow();
  } catch (const BaseError &Err) {
    consume(sizeof(Err));
    {
      int Err = 0;
      consume(Err);
      throw Err;
    }
  }
}

void unrelatedThrowIsIgnored() {
  try {
    mayThrow();
  } catch (const BaseError &Err) {
    OtherError Fresh;
    consume(sizeof(Fresh));
    throw Fresh;
  }
}

void convertingConstructionIsIgnored() {
  try {
    mayThrow();
  } catch (const DerivedError &Err) {
    throw BaseError(Err);
  }
}

// Rewriting this to a bare `throw;` would drop the call to mayThrow().
void commaSideEffectIsIgnored() {
  try {
    mayThrow();
  } catch (const BaseError &Err) {
    throw(mayThrow(), Err);
  }
}

void lambdaCaptureIsIgnored() {
  try {
    mayThrow();
  } catch (const BaseError &Err) {
    auto Thunk = [&] { throw Err; };
    (void)Thunk;
  }
}

// A block may run while an unrelated exception is active, so rethrowing the
// caught variable from its body must not become a bare `throw;`.
void blockCaptureIsIgnored() {
  try {
    mayThrow();
  } catch (const BaseError &Err) {
    Thunk Run = ^{ throw Err; };
    (void)Run;
  }
}

void blockScalarCaptureIsIgnored() {
  try {
    mayThrow();
  } catch (int &Err) {
    Thunk Run = ^{ throw Err; };
    (void)Run;
  }
}

void blockOwnCatchWarns() {
  Thunk Run = ^{
    try {
      mayThrow();
    } catch (const BaseError &Err) {
      throw Err;
      // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: throwing a copy of the caught 'BaseError' exception; use a bare 'throw' to rethrow the original exception [bugprone-rethrow-caught-exception]
      // CHECK-FIXES: throw;
    }
  };
  (void)Run;
}

void nestedCallableIsIgnored() {
  try {
    mayThrow();
  } catch (const BaseError &Err) {
    struct Local {
      static void run(const BaseError &Param) { throw Param; }
    };
    Local::run(Err);
  }
}

void innerCatchNamingOuterIsIgnored() {
  try {
    mayThrow();
  } catch (const BaseError &Outer) {
    try {
      mayThrow();
    } catch (const OtherError &) {
      throw Outer;
    }
    consume(sizeof(Outer));
  }
}

void pointerCatchIsIgnored() {
  try {
    mayThrow();
  } catch (const BaseError *Err) {
    throw Err;
  }
}
