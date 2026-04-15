// RUN: %check_clang_tidy -std=c++17-or-later %s llvm-type-switch-case-types %t

namespace llvm {

template <typename T, typename ResultT = int>
class TypeSwitch {
public:
  TypeSwitch(T) {}

  // Single-type Case with explicit template argument.
  template <typename CaseT, typename CallableT>
  TypeSwitch &Case(CallableT &&) { return *this; }

  // Inferred Case: callable's first argument determines the type.
  template <typename CallableT>
  TypeSwitch &Case(CallableT &&) { return *this; }

  // Variadic Case: multiple types with single callable.
  template <typename CaseT, typename CaseT2, typename... CaseTs,
            typename CallableT>
  TypeSwitch &Case(CallableT &&) { return *this; }
};

} // namespace llvm

// Test types for the switch cases.
struct Base {};
struct DerivedA : Base {};
struct DerivedB : Base {};

void test_explicit_type_matches_lambda_param(Base *base) {
  llvm::TypeSwitch<Base *, int>(base)
      .Case<DerivedA>([](DerivedA *a) { return 10; });
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: redundant explicit template argument
  // CHECK-FIXES: .Case([](DerivedA *a) { return 10; });

  llvm::TypeSwitch<Base *>(base)
      .Case<DerivedA>([](DerivedA *) {});
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: redundant explicit template argument
  // CHECK-FIXES: .Case([](DerivedA *) {});
}

void test_value_type_switch() {
  // TypeSwitch on value types (not pointers) -- common in MLIR.
  struct Type {};
  struct Float16Type : Type {};

  llvm::TypeSwitch<Type, int>(Type())
      .Case<Float16Type>([](Float16Type) { return 1; });
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: redundant explicit template argument
  // CHECK-FIXES: .Case([](Float16Type) { return 1; });
}

void test_auto_param_with_explicit_type(Base *base) {
  llvm::TypeSwitch<Base *, int>(base)
      .Case<DerivedA>([](auto a) { return 20; })
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: lambda parameter needlessly uses 'auto', use explicit type instead
  // CHECK-MESSAGES: :[[@LINE-2]]:26: note: replace 'auto' with explicit type
  // CHECK-MESSAGES: :[[@LINE-3]]:13: note: type from template argument can be inferred and removed
      .Case<DerivedB>([](auto *b) { return 80; });
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: lambda parameter needlessly uses 'auto', use explicit type instead
  // CHECK-MESSAGES: :[[@LINE-2]]:26: note: replace 'auto' with explicit type
  // CHECK-MESSAGES: :[[@LINE-3]]:13: note: type from template argument can be inferred and removed
}

void test_already_inferred_case(Base *base) {
  // Already using type-inferred Case - no warning expected.
  llvm::TypeSwitch<Base *, int>(base)
      .Case([](DerivedA *a) { return 1; })
      .Case([](DerivedB *b) { return 2; });
}

void test_const_param(const Base *base) {
  llvm::TypeSwitch<const Base *, int>(base)
      .Case<DerivedA>([](const DerivedA *a) { return 30; });
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: redundant explicit template argument
  // CHECK-FIXES: .Case([](const DerivedA *a) { return 30; });
}

void test_comments_preserved(Base *base) {
  llvm::TypeSwitch<Base *, int>(base)
      .Case<DerivedA>(/*comment*/ [](DerivedA *a) { return 40; });
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: redundant explicit template argument
  // CHECK-FIXES: .Case(/*comment*/ [](DerivedA *a) { return 40; });
}

void test_whitespace(Base *base) {
  llvm::TypeSwitch<Base *, int>(base)
      .Case< DerivedA >([](DerivedA *a) { return 50; });
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: redundant explicit template argument
  // CHECK-FIXES: .Case([](DerivedA *a) { return 50; });
}

template <typename T>
void test_template_keyword(T *base) {
  llvm::TypeSwitch<T *, int>(base)
      .template Case<DerivedA>([](DerivedA *a) { return 90; });
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: redundant explicit template argument
  // CHECK-FIXES: .Case([](DerivedA *a) { return 90; });
}
// Explicit instantiation to trigger the check.
template void test_template_keyword<Base>(Base *);

void test_fully_qualified(Base *base) {
  ::llvm::TypeSwitch<Base *, int>(base)
      .Case<DerivedA>([](DerivedA *a) { return 60; });
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: redundant explicit template argument
  // CHECK-FIXES: .Case([](DerivedA *a) { return 60; });
}

namespace llvm {
void test_inside_llvm_namespace(Base *base) {
  TypeSwitch<Base *, int>(base)
      .Case<DerivedA>([](DerivedA *a) { return 70; });
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: redundant explicit template argument
  // CHECK-FIXES: .Case([](DerivedA *a) { return 70; });

  TypeSwitch<Base *, int>(base)
      .Case<DerivedA>([](auto a) { return 71; });
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: lambda parameter needlessly uses 'auto', use explicit type instead
  // CHECK-MESSAGES: :[[@LINE-2]]:26: note: replace 'auto' with explicit type
  // CHECK-MESSAGES: :[[@LINE-3]]:13: note: type from template argument can be inferred and removed
}
} // namespace llvm

void test_macro_in_type(Base *base) {
#define CASE_TYPE DerivedA
  // Warning + fix-it: angle brackets are real tokens, macro content is deleted.
  llvm::TypeSwitch<Base *, int>(base)
      // CHECK-MESSAGES: :[[@LINE+1]]:8: warning: redundant explicit template argument
      .Case<CASE_TYPE>([](DerivedA *a) { return 1; });
  // CHECK-FIXES: .Case([](DerivedA *a) { return 1; });
#undef CASE_TYPE
}

void test_macro_entire_case(Base *base) {
#define MAKE_CASE(Type) .Case<Type>([](Type *x) { return 1; })
  // Warning emitted but no fix-it when angle brackets are in a macro.
  llvm::TypeSwitch<Base *, int>(base)
      MAKE_CASE(DerivedA);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: redundant explicit template argument
#undef MAKE_CASE
}

//===----------------------------------------------------------------------===//
// Negative test cases - should NOT trigger any warnings.
//===----------------------------------------------------------------------===//

// Non-TypeSwitch Case method -- should not be modified.
struct OtherClass {
  template <typename T, typename F>
  OtherClass &Case(F &&f) { return *this; }
};

void test_negative_non_type_switch() {
  OtherClass()
      .Case<DerivedA>([](DerivedA *a) {});
  // No warning expected -- this is not `llvm::TypeSwitch`.
}

// TypeSwitch in non-llvm namespace.
namespace other {
template <typename T, typename R = int>
struct TypeSwitch {
  TypeSwitch(T) {}
  template <typename CaseT, typename F>
  TypeSwitch &Case(F &&) { return *this; }
};
} // namespace other

void test_negative_non_llvm_namespace(Base *base) {
  // No warning expected -- this is not `llvm::TypeSwitch`.
  other::TypeSwitch<Base *, int>(base)
      .Case<DerivedA>([](DerivedA *a) { return 1; });
}

void test_variadic_case_no_change(Base *base) {
  // Variadic Case with multiple types - do not modify.
  llvm::TypeSwitch<Base *, int>(base)
      .Case<DerivedA, DerivedB>([](auto x) { return 1; });
}
