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



// Helper function for testing nested calls
Box makeBox(Box b) { return b; }

// ========== CASES THAT SHOULD TRANSFORM ==========

// Case 1: Basic transformation
// CHECK-MESSAGES: :[[@LINE+2]]:25: warning: use braced initializer list for default argument
// CHECK-FIXES: void basic_case(Box b = {}) {
void basic_case(Box b = Box()) {
}

// Case 2: Multiple Box parameters
// CHECK-MESSAGES: :[[@LINE+3]]:36: warning: use braced initializer list for default argument
// CHECK-MESSAGES: :[[@LINE+2]]:52: warning: use braced initializer list for default argument
// CHECK-FIXES: void multiple_params_case(Box b1 = {}, Box b2 = {}) {
void multiple_params_case(Box b1 = Box(), Box b2 = Box()) {
}

// Case 3: Mixed parameter types (int, Box, double)
// CHECK-MESSAGES: :[[@LINE+2]]:44: warning: use braced initializer list for default argument
// CHECK-FIXES: void mixed_params_case(int x = 10, Box b = {}, double d = 3.14) {
void mixed_params_case(int x = 10, Box b = Box(), double d = 3.14) {
}



// Case 4: Template with concrete type (Box) - SHOULD transform
// CHECK-MESSAGES: :[[@LINE+3]]:37: warning: use braced initializer list for default argument
// CHECK-FIXES: void template_concrete_case(Box b = {}) {
template<typename T>
void template_concrete_case(Box b = Box()) {
}

// Case 5: const, const&, && parameters (all should transform)
// CHECK-MESSAGES: :[[@LINE+4]]:38: warning: use braced initializer list for default argument
// CHECK-MESSAGES: :[[@LINE+3]]:61: warning: use braced initializer list for default argument
// CHECK-MESSAGES: :[[@LINE+2]]:79: warning: use braced initializer list for default argument
// CHECK-FIXES: void const_ref_params(const Box b1 = {}, const Box& b2 = {}, Box&& b3 = {}) {
void const_ref_params(const Box b1 = Box(), const Box& b2 = Box(), Box&& b3 = Box()) {
}

// Case 6: Constructor default argument (should transform)
struct S {
  // CHECK-MESSAGES: :[[@LINE+2]]:13: warning: use braced initializer list for default argument
  // CHECK-FIXES: S(Box b = {}) {}
  S(Box b = Box()) {}
};

// Case 7: Typedef alias (should transform)
using MyBox = Box;
// CHECK-MESSAGES: :[[@LINE+2]]:29: warning: use braced initializer list for default argument
// CHECK-FIXES: void typedef_case(MyBox b = {}) {
void typedef_case(MyBox b = Box()) {
}

// ========== CASES THAT SHOULD NOT TRANSFORM (SKIP) ==========

// Case 8: Explicit constructor (skip)
void explicit_case(ExplicitBox e = ExplicitBox()) {
}

// Case 9: Constructor with arguments (skip)
void with_args_case(Box b = Box(5)) {
}

// Case 10: Type mismatch (skip)
void type_mismatch_case(Base b = Derived()) {
}

// Case 11: Already braced (skip)
void already_braced_case(Box b = {}) {
}

// Case 12: Template with dependent type T() - SHOULD NOT transform
template<typename T>
void template_dependent_case(T t = T()) {
}

// Case 13: Macro expansion (skip)
#define DEFAULT_BOX Box()
void macro_case(Box b = DEFAULT_BOX) {
}
#undef DEFAULT_BOX

// Case 14: Ternary operator with Box() (skip - nested)
void ternary_case(Box b = true ? Box() : Box()) {
}

// Case 15: Nested function call (skip - nested)
void nested_call_case(Box b = makeBox(Box())) {
}
