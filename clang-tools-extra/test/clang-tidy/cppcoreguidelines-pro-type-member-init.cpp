// RUN: %check_clang_tidy %s cppcoreguidelines-pro-type-member-init %t

struct PositiveFieldBeforeConstructor {
  int F;
  // CHECK-FIXES: int F{};
  PositiveFieldBeforeConstructor() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these built-in/pointer fields: F
  // CHECK-FIXES: PositiveFieldBeforeConstructor() {}
};

struct PositiveFieldAfterConstructor {
  PositiveFieldAfterConstructor() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these built-in/pointer fields: F, G
  // CHECK-FIXES: PositiveFieldAfterConstructor() {}
  int F;
  // CHECK-FIXES: int F{};
  bool G /* with comment */;
  // CHECK-FIXES: bool G{} /* with comment */;
  PositiveFieldBeforeConstructor IgnoredField;
};

struct PositiveSeparateDefinition {
  PositiveSeparateDefinition();
  int F;
  // CHECK-FIXES: int F{};
};

PositiveSeparateDefinition::PositiveSeparateDefinition() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: constructor does not initialize these built-in/pointer fields: F
// CHECK-FIXES: PositiveSeparateDefinition::PositiveSeparateDefinition() {}

struct PositiveMixedFieldOrder {
  PositiveMixedFieldOrder() : J(0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these built-in/pointer fields: I, K
  // CHECK-FIXES: PositiveMixedFieldOrder() : J(0) {}
  int I;
  // CHECK-FIXES: int I{};
  int J;
  int K;
  // CHECK-FIXES: int K{};
};

template <typename T>
struct Template {
  Template() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these built-in/pointer fields: F
  int F;
  // CHECK-FIXES: int F{};
  T T1;
  // CHECK-FIXES: T T1;
};

void instantiate() {
  Template<int> TInt;
}

struct NegativeFieldInitialized {
  int F;

  NegativeFieldInitialized() : F() {}
};

struct NegativeFieldInitializedInDefinition {
  int F;

  NegativeFieldInitializedInDefinition();
};
NegativeFieldInitializedInDefinition::NegativeFieldInitializedInDefinition() : F() {}


struct NegativeInClassInitialized {
  int F = 0;

  NegativeInClassInitialized() {}
};

struct NegativeConstructorDelegated {
  int F;

  NegativeConstructorDelegated(int F) : F(F) {}
  NegativeConstructorDelegated() : NegativeConstructorDelegated(0) {}
};

struct NegativeInitializedInBody {
  NegativeInitializedInBody() { I = 0; }
  int I;
};

#define UNINITIALIZED_FIELD_IN_MACRO_BODY(FIELD) \
  struct UninitializedField##FIELD {		 \
    UninitializedField##FIELD() {}		 \
    int FIELD;					 \
  };						 \
// Ensure FIELD is not initialized since fixes inside of macros are disabled.
// CHECK-FIXES: int FIELD;

UNINITIALIZED_FIELD_IN_MACRO_BODY(F);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: constructor does not initialize these built-in/pointer fields: F
UNINITIALIZED_FIELD_IN_MACRO_BODY(G);
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: constructor does not initialize these built-in/pointer fields: G

#define UNINITIALIZED_FIELD_IN_MACRO_ARGUMENT(ARGUMENT) \
  ARGUMENT						\

UNINITIALIZED_FIELD_IN_MACRO_ARGUMENT(struct UninitializedFieldInMacroArg {
  UninitializedFieldInMacroArg() {}
  int Field;
});
// CHECK-MESSAGES: :[[@LINE-3]]:3: warning: constructor does not initialize these built-in/pointer fields: Field
// Ensure FIELD is not initialized since fixes inside of macros are disabled.
// CHECK-FIXES: int Field;
