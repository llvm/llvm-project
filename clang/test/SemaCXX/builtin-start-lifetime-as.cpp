// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

namespace InvalidArgs {
  void test_non_pointers(int x) {
    __builtin_start_lifetime_as(x); // expected-error {{non-pointer argument to '__builtin_start_lifetime_as' is not allowed}}
    __builtin_start_lifetime_as(nullptr); // expected-error {{non-pointer argument to '__builtin_start_lifetime_as' is not allowed}}
  }

  void test_void_and_func(void *p, void (*f)()) {
    __builtin_start_lifetime_as(p); // expected-error {{type 'void' is not an implicit-lifetime type; cannot start lifetime}}
    __builtin_start_lifetime_as(f); // expected-error {{type 'void ()' is not an implicit-lifetime type; cannot start lifetime}}
  }

  struct Incomplete; // expected-note {{forward declaration of 'InvalidArgs::Incomplete'}}
  void test_incomplete(Incomplete *p) {
    __builtin_start_lifetime_as(p); // expected-error {{incomplete type 'Incomplete' where a complete type is required}}
  }

  void test_vla(int n) { // expected-note {{declared here}}
    int vla[n]; // expected-warning {{variable length arrays in C++ are a Clang extension}} \
                // expected-note {{function parameter 'n' with unknown value cannot be used in a constant expression}}
    __builtin_start_lifetime_as(&vla); // expected-error {{variable length arrays are not supported in '__builtin_start_lifetime_as'}}
  }
} // namespace InvalidArgs

namespace ImplicitLifetimeRules {
  // Valid types
  struct Trivial { int x; int y; };
  struct TrivialArray { int arr[5]; };
  union TrivialUnion { int a; float b; };

  struct AggregateNoDtor { int a; ~AggregateNoDtor() = default; };

  // Invalid types
  struct UserDtor { ~UserDtor() {} };
  struct UserCopy { UserCopy(const UserCopy&); };
  struct VirtualBase { virtual void f(); };

  void test_valid_types(void* p) {
    __builtin_start_lifetime_as((int*)p);
    __builtin_start_lifetime_as((Trivial*)p);
    __builtin_start_lifetime_as((TrivialArray*)p);
    __builtin_start_lifetime_as((TrivialUnion*)p);
    __builtin_start_lifetime_as((AggregateNoDtor*)p);

    // Arrays of implicit-lifetime types are implicitly valid
    __builtin_start_lifetime_as((int(*)[5])p);

    // Arrays of non-implicit-lifetime types are also valid under C++23
    __builtin_start_lifetime_as((UserDtor(*)[5])p);
  }

  void test_invalid_types(void *p) {
    __builtin_start_lifetime_as((UserDtor*)p); // expected-error {{type 'UserDtor' is not an implicit-lifetime type; cannot start lifetime}}
    __builtin_start_lifetime_as((UserCopy*)p); // expected-error {{type 'UserCopy' is not an implicit-lifetime type; cannot start lifetime}}
    __builtin_start_lifetime_as((VirtualBase*)p); // expected-error {{type 'VirtualBase' is not an implicit-lifetime type; cannot start lifetime}}
  }
} // namespace ImplicitLifetimeRules

namespace StrictnessFlag {
  struct NonTrivial { ~NonTrivial() {} };

  void test_flag(NonTrivial* p, int runtime_flag) {
    // defaults to strict
    __builtin_start_lifetime_as(p); // expected-error {{type 'NonTrivial' is not an implicit-lifetime type}}

    // explicit strict
    __builtin_start_lifetime_as(p, true); // expected-error {{type 'NonTrivial' is not an implicit-lifetime type}}
    
    // bypasses the implicit-lifetime check
    __builtin_start_lifetime_as(p, false);

    // Flag must be an ICE (Integer Constant Expression)
    __builtin_start_lifetime_as(p, runtime_flag); // expected-error {{expression is not an integer constant expression}}
  }
} // namespace StrictnessFlag

namespace Templates {
  template <typename T>
  T* test_dependent_type(void *p) {
    // Should defer evaluation until instantiation
    return __builtin_start_lifetime_as((T*)p); // expected-error {{type 'Templates::std::string' is not an implicit-lifetime type}}
  }

  namespace std { struct string { ~string() {} }; }

  void instantiate(void *p) {
    test_dependent_type<int>(p);
    test_dependent_type<std::string>(p); // expected-note {{in instantiation of function template specialization 'Templates::test_dependent_type<Templates::std::string>' requested here}}
  }
} // namespace Templates

namespace ConstexprEvaluation {
  struct Trivial { int x; };

  constexpr Trivial* test_constexpr_builtin(Trivial* p) {
    return __builtin_start_lifetime_as(p); // expected-note {{subexpression not valid in a constant expression}}
  }

  constexpr bool test_eval() {
    Trivial t{0};
    Trivial* p = test_constexpr_builtin(&t); // expected-note {{in call to 'test_constexpr_builtin(&t)'}}
    return p != nullptr;
  }

  constexpr bool b = test_eval(); // expected-error {{constexpr variable 'b' must be initialized by a constant expression}} \
                                  // expected-note {{in call to 'test_eval()'}}
} // namespace ConstexprEvaluation
