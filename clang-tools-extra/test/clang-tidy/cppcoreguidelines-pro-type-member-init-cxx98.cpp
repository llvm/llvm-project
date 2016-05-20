// RUN: %check_clang_tidy %s cppcoreguidelines-pro-type-member-init %t -- -- -std=c++98 -fno-delayed-template-parsing

struct PositiveFieldBeforeConstructor {
  int F;
  PositiveFieldBeforeConstructor() /* some comment */ {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: F
  // CHECK-FIXES: PositiveFieldBeforeConstructor() : F() /* some comment */ {}
};

struct PositiveFieldAfterConstructor {
  PositiveFieldAfterConstructor() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: F, G, H
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
// CHECK-MESSAGES: :[[@LINE-1]]:1: warning: constructor does not initialize these fields: F
// CHECK-FIXES: PositiveSeparateDefinition::PositiveSeparateDefinition() : F() {}

struct PositiveMixedFieldOrder {
  PositiveMixedFieldOrder() : /* some comment */ J(0), L(0), M(0) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: I, K, N
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
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: F
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

struct NegativeAggregateType {
  int X;
  int Y;
  int Z;
};

struct TrivialType {
  int X;
  int Y;
};

struct PositiveUninitializedBaseOrdering : public NegativeAggregateType,
                                           public TrivialType {
  PositiveUninitializedBaseOrdering() : NegativeAggregateType(), TrivialType(), B() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: A
  // CHECK-FIXES: PositiveUninitializedBaseOrdering() : NegativeAggregateType(), TrivialType(), A(), B() {}

  // This is somewhat pathological with the base class initializer at the end...
  PositiveUninitializedBaseOrdering(int) : B(), TrivialType(), A() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these bases: NegativeAggregateType
  // CHECK-FIXES: PositiveUninitializedBaseOrdering(int) : B(), NegativeAggregateType(), TrivialType(), A() {}

  PositiveUninitializedBaseOrdering(float) : NegativeAggregateType(), A() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these bases: TrivialType
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: constructor does not initialize these fields: B
  // CHECK-FIXES: PositiveUninitializedBaseOrdering(float) : NegativeAggregateType(), TrivialType(), A(), B() {}

  int A, B;
};

template <class T>
class PositiveTemplateBase : T {
public:
  PositiveTemplateBase() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: X
  // CHECK-FIXES: PositiveTemplateBase() : X() {}

  int X;
};

class PositiveIndirectMember {
  struct {
    int *A;
  };

  PositiveIndirectMember() : A() {}
  PositiveIndirectMember(int) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: constructor does not initialize these fields: A
  // CHECK-FIXES: PositiveIndirectMember(int) : A() {}
};
