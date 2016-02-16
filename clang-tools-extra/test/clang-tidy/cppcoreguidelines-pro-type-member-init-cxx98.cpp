// RUN: %check_clang_tidy %s cppcoreguidelines-pro-type-member-init %t -- -- -std=c++98

struct PositiveFieldBeforeConstructor {
  int F;
  PositiveFieldBeforeConstructor() /* some comment */ {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these built-in/pointer fields: F
  // CHECK-FIXES: PositiveFieldBeforeConstructor() : F() /* some comment */ {}
};

struct PositiveFieldAfterConstructor {
  PositiveFieldAfterConstructor() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these built-in/pointer fields: F, G, H
  // CHECK-FIXES: PositiveFieldAfterConstructor() : F(), G(), H() {}
  int F;
  bool G /* with comment */;
  int *H;
  PositiveFieldBeforeConstructor IgnoredField;
};

struct PositiveSeparateDefinition {
  PositiveSeparateDefinition();
  int F;
};

PositiveSeparateDefinition::PositiveSeparateDefinition() {}
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: constructor does not initialize these built-in/pointer fields: F
// CHECK-FIXES: PositiveSeparateDefinition::PositiveSeparateDefinition() : F() {}

struct PositiveMixedFieldOrder {
  PositiveMixedFieldOrder() : /* some comment */ J(0), L(0), M(0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these built-in/pointer fields: I, K, N
  // CHECK-FIXES: PositiveMixedFieldOrder() : I(), /* some comment */ J(0), K(), L(0), M(0), N() {}
  int I;
  int J;
  int K;
  int L;
  int M;
  int N;
};

struct PositiveAfterBaseInitializer : public PositiveMixedFieldOrder {
  PositiveAfterBaseInitializer() : PositiveMixedFieldOrder() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these built-in/pointer fields: F
  // CHECK-FIXES: PositiveAfterBaseInitializer() : PositiveMixedFieldOrder(), F() {}
  int F;
};

struct NegativeFieldInitialized {
  int F;

  NegativeFieldInitialized() : F() {}
};

struct NegativeFieldInitializedInDefinition {
  int F;

  NegativeFieldInitializedInDefinition();
};

NegativeFieldInitializedInDefinition::NegativeFieldInitializedInDefinition() : F() {}

struct NegativeInitializedInBody {
  NegativeInitializedInBody() { I = 0; }
  int I;
};


