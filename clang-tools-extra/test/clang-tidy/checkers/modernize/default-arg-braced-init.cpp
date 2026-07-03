// RUN: %check_clang_tidy -std=c++11-or-later %s modernize-default-arg-braced-init %t

struct Box {
  Box() = default;
  Box(int x) {}
  Box(const Box&) {}
};

struct ExplicitBox {
  explicit ExplicitBox() {}
};

struct Base {
  Base() = default;
};

struct Derived : Base {
  Derived() = default;
};

Box makeBox(Box b) { return b; }

// CHECK-MESSAGES: :[[@LINE+2]]:25: warning: use braced initializer list for default argument
// CHECK-FIXES: void BasicSingleParam(Box b = {}) {
void BasicSingleParam(Box b = Box()) {}

// CHECK-MESSAGES: :[[@LINE+3]]:33: warning: use braced initializer list for default argument
// CHECK-MESSAGES: :[[@LINE+2]]:49: warning: use braced initializer list for default argument
// CHECK-FIXES: void MultipleBoxParams(Box b1 = {}, Box b2 = {}) {
void MultipleBoxParams(Box b1 = Box(), Box b2 = Box()) {}

// CHECK-MESSAGES: :[[@LINE+2]]:44: warning: use braced initializer list for default argument
// CHECK-FIXES: void MixedParamTypes(int x = 10, Box b = {}, double d = 3.14) {
void MixedParamTypes(int x = 10, Box b = Box(), double d = 3.14) {}

template<typename T>
// CHECK-MESSAGES: :[[@LINE+2]]:37: warning: use braced initializer list for default argument
// CHECK-FIXES: void TemplateConcreteType(Box b = {}) {
void TemplateConcreteType(Box b = Box()) {}

// CHECK-MESSAGES: :[[@LINE+4]]:38: warning: use braced initializer list for default argument
// CHECK-MESSAGES: :[[@LINE+3]]:61: warning: use braced initializer list for default argument
// CHECK-MESSAGES: :[[@LINE+2]]:79: warning: use braced initializer list for default argument
// CHECK-FIXES: void ConstAndReferenceParams(const Box b1 = {}, const Box& b2 = {}, Box&& b3 = {}) {
void ConstAndReferenceParams(const Box b1 = Box(), const Box& b2 = Box(), Box&& b3 = Box()) {}

struct S {
  // CHECK-MESSAGES: :[[@LINE+2]]:13: warning: use braced initializer list for default argument
  // CHECK-FIXES: S(Box b = {}) {}
  S(Box b = Box()) {}
};

using MyBox = Box;
// CHECK-MESSAGES: :[[@LINE+2]]:29: warning: use braced initializer list for default argument
// CHECK-FIXES: void TypedefAlias(MyBox b = {}) {
void TypedefAlias(MyBox b = Box()) {}

// Explicit constructors should not be transformed
void ExplicitConstructor(ExplicitBox e = ExplicitBox()) {}

// Constructors with arguments should not be transformed
void ConstructorWithArgs(Box b = Box(5)) {}

// Type mismatch should not be transformed
void TypeMismatch(Base b = Derived()) {}

// Already using braced initializer should be left alone
void AlreadyBracedInit(Box b = {}) {}

// Dependent types in templates should not be transformed
template<typename T>
void TemplateDependentType(T t = T()) {}

// Macro expansion should not be transformed
#define DEFAULT_BOX Box()
void MacroExpansion(Box b = DEFAULT_BOX) {}
#undef DEFAULT_BOX

// Nested expressions (ternary) should not be transformed
void TernaryExpression(Box b = true ? Box() : Box()) {}

// Nested function calls should not be transformed
void NestedFunctionCall(Box b = makeBox(Box())) {}
