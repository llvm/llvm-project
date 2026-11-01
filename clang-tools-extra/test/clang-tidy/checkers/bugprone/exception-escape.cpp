// RUN: %check_clang_tidy -std=c++11,c++14 %s bugprone-exception-escape %t -- \
// RUN:     -config="{CheckOptions: { \
// RUN:         bugprone-exception-escape.IgnoredExceptions: 'ignored1,ignored2', \
// RUN:         bugprone-exception-escape.FunctionsThatShouldNotThrow: 'enabled1,enabled2,enabled3' \
// RUN:     }}" \
// RUN:     -- -fexceptions
// FIXME: Fix the checker to work in C++17 or later mode.

struct throwing_destructor {
  ~throwing_destructor() {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: an exception may be thrown in function '~throwing_destructor' which should not throw exceptions
    throw 1;
  }
};
// CHECK-MESSAGES: :[[@LINE-3]]:5: note: frame #0: unhandled exception of type 'int' may be thrown in function '~throwing_destructor' here

struct throwing_move_constructor {
  throwing_move_constructor(throwing_move_constructor&&) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: an exception may be thrown in function 'throwing_move_constructor' which should not throw exceptions
    throw 1;
  }
};
// CHECK-MESSAGES: :[[@LINE-3]]:5: note: frame #0: unhandled exception of type 'int' may be thrown in function 'throwing_move_constructor' here

struct throwing_move_assignment {
  throwing_move_assignment& operator=(throwing_move_assignment&&) {
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: an exception may be thrown in function 'operator=' which should not throw exceptions
    throw 1;
  }
};
// CHECK-MESSAGES: :[[@LINE-3]]:5: note: frame #0: unhandled exception of type 'int' may be thrown in function 'operator=' here

void throwing_noexcept() noexcept {
    // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throwing_noexcept' which should not throw exceptions
  throw 1;
}
// CHECK-MESSAGES: :[[@LINE-2]]:3: note: frame #0: unhandled exception of type 'int' may be thrown in function 'throwing_noexcept' here

void throw_and_catch() noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_and_catch' which should not throw exceptions
  try {
    throw 1;
  } catch(int &) {
  }
}

void throw_and_catch_some(int n) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_and_catch_some' which should not throw exceptions
  try {
    if (n) throw 1;
    throw 1.1;
  } catch(int &) {
  }
}
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: unhandled exception of type 'double' may be thrown in function 'throw_and_catch_some' here

void throw_and_catch_each(int n) noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_and_catch_each' which should not throw exceptions
  try {
    if (n) throw 1;
    throw 1.1;
  } catch(int &) {
  } catch(double &) {
  }
}

void throw_and_catch_all(int n) noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_and_catch_all' which should not throw exceptions
  try {
    if (n) throw 1;
    throw 1.1;
  } catch(...) {
  }
}

void throw_and_rethrow() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_and_rethrow' which should not throw exceptions
  try {
    throw 1;
  } catch(int &) {
    throw;
  }
}
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: frame #0: unhandled exception of type 'int' may be thrown in function 'throw_and_rethrow' here

void throw_catch_throw() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_catch_throw' which should not throw exceptions
  try {
    throw 1;
  } catch(int &) {
    throw 2;
  }
}
// CHECK-MESSAGES: :[[@LINE-3]]:5: note: frame #0: unhandled exception of type 'int' may be thrown in function 'throw_catch_throw' here

void throw_catch_rethrow_the_rest(int n) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_catch_rethrow_the_rest' which should not throw exceptions
  try {
    if (n) throw 1;
    throw 1.1;
  } catch(int &) {
  } catch(...) {
    throw;
  }
}
// CHECK-MESSAGES: :[[@LINE-6]]:5: note: frame #0: unhandled exception of type 'double' may be thrown in function 'throw_catch_rethrow_the_rest' here

void throw_catch_pointer_c() noexcept {
  try {
    int a = 1;
    throw &a;
  } catch(const int *) {}
}

void throw_catch_pointer_v() noexcept {
  try {
    int a = 1;
    throw &a;
  } catch(volatile int *) {}
}

void throw_catch_pointer_cv() noexcept {
  try {
    int a = 1;
    throw &a;
  } catch(const volatile int *) {}
}

void throw_catch_multi_ptr_1() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_catch_multi_ptr_1' which should not throw exceptions
  try {
    char **p = 0;
    throw p;
  } catch (const char **) {
  }
}
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: unhandled exception of type 'char **' may be thrown in function 'throw_catch_multi_ptr_1' here

void throw_catch_multi_ptr_2() noexcept {
  try {
    char **p = 0;
    throw p;
  } catch (const char *const *) {
  }
}

void throw_catch_multi_ptr_3() noexcept {
  try {
    char **p = 0;
    throw p;
  } catch (volatile char *const *) {
  }
}

void throw_catch_multi_ptr_4() noexcept {
  try {
    char **p = 0;
    throw p;
  } catch (volatile const char *const *) {
  }
}

// FIXME: In this case 'a' is convertible to the handler and should be caught
// but in reality it's thrown. Note that clang doesn't report a warning for
// this either.
void throw_catch_multi_ptr_5() noexcept {
  try {
    double *a[2][3];
    throw a;
  } catch (double *(*)[3]) {
  }
}


void throw_c_catch_pointer() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_c_catch_pointer' which should not throw exceptions
  try {
    int a = 1;
    const int *p = &a;
    throw p;
  } catch(int *) {}
}
// CHECK-MESSAGES: :[[@LINE-3]]:5: note: frame #0: unhandled exception of type 'const int *' may be thrown in function 'throw_c_catch_pointer' here

void throw_c_catch_pointer_v() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_c_catch_pointer_v' which should not throw exceptions
  try {
    int a = 1;
    const int *p = &a;
    throw p;
  } catch(volatile int *) {}
}
// CHECK-MESSAGES: :[[@LINE-3]]:5: note: frame #0: unhandled exception of type 'const int *' may be thrown in function 'throw_c_catch_pointer_v' here

void throw_v_catch_pointer() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_v_catch_pointer' which should not throw exceptions
  try {
    int a = 1;
    volatile int *p = &a;
    throw p;
  } catch(int *) {}
}
// CHECK-MESSAGES: :[[@LINE-3]]:5: note: frame #0: unhandled exception of type 'volatile int *' may be thrown in function 'throw_v_catch_pointer' here

void throw_v_catch_pointer_c() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_v_catch_pointer_c' which should not throw exceptions
  try {
    int a = 1;
    volatile int *p = &a;
    throw p;
  } catch(const int *) {}
}
// CHECK-MESSAGES: :[[@LINE-3]]:5: note: frame #0: unhandled exception of type 'volatile int *' may be thrown in function 'throw_v_catch_pointer_c' here

void throw_cv_catch_pointer_c() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_cv_catch_pointer_c' which should not throw exceptions
  try {
    int a = 1;
    const volatile int *p = &a;
    throw p;
  } catch(const int *) {}
}
// CHECK-MESSAGES: :[[@LINE-3]]:5: note: frame #0: unhandled exception of type 'const volatile int *' may be thrown in function 'throw_cv_catch_pointer_c' here

void throw_cv_catch_pointer_v() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_cv_catch_pointer_v' which should not throw exceptions
  try {
    int a = 1;
    const volatile int *p = &a;
    throw p;
  } catch(volatile int *) {}
}
// CHECK-MESSAGES: :[[@LINE-3]]:5: note: frame #0: unhandled exception of type 'const volatile int *' may be thrown in function 'throw_cv_catch_pointer_v' here

class base {};
class derived: public base {};

void throw_derived_catch_base() noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_derived_catch_base' which should not throw exceptions
  try {
    throw derived();
  } catch(base &) {
  }
}

void throw_derived_alias_catch_base() noexcept {
  using alias = derived;

  try {
    throw alias();
  } catch(base &) {
  }
}

void throw_derived_catch_base_alias() noexcept {
  using alias = base;

  try {
    throw derived();
  } catch(alias &) {
  }
}

void throw_derived_catch_base_ptr_c() noexcept {
  try {
    derived d;
    throw &d;
  } catch(const base *) {
  }
}

void throw_derived_catch_base_ptr() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_derived_catch_base_ptr' which should not throw exceptions
  try {
    derived d;
    const derived *p = &d;
    throw p;
  } catch(base *) {
  }
}
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: unhandled exception of type 'const derived *' may be thrown in function 'throw_derived_catch_base_ptr' here

class A {};
class B : A {};

// The following alias hell is deliberately created for testing.
using aliasedA = A;
class C : protected aliasedA {};

typedef aliasedA moreAliasedA;
class D : public moreAliasedA {};

using moreMoreAliasedA = moreAliasedA;
using aliasedD = D;
class E : public moreMoreAliasedA, public aliasedD {};

void throw_derived_catch_base_private() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_derived_catch_base_private' which should not throw exceptions
  try {
    B b;
    throw b;
  } catch(A) {
  }
}
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: unhandled exception of type 'B' may be thrown in function 'throw_derived_catch_base_private' here

void throw_derived_catch_base_private_ptr() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_derived_catch_base_private_ptr' which should not throw exceptions
  try {
    B b;
    throw &b;
  } catch(A *) {
  }
}
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: unhandled exception of type 'B *' may be thrown in function 'throw_derived_catch_base_private_ptr' here

void throw_derived_catch_base_protected() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_derived_catch_base_protected' which should not throw exceptions
  try {
    C c;
    throw c;
  } catch(A) {
  }
}
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: unhandled exception of type 'C' may be thrown in function 'throw_derived_catch_base_protected' here

void throw_derived_catch_base_protected_ptr() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_derived_catch_base_protected_ptr' which should not throw exceptions
  try {
    C c;
    throw &c;
  } catch(A *) {
  }
}
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: unhandled exception of type 'C *' may be thrown in function 'throw_derived_catch_base_protected_ptr' here

void throw_derived_catch_base_ambiguous() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_derived_catch_base_ambiguous' which should not throw exceptions
  try {
    E e;
    throw e;
  } catch(A) {
  }
}
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: unhandled exception of type 'E' may be thrown in function 'throw_derived_catch_base_ambiguous' here

void throw_derived_catch_base_ambiguous_ptr() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_derived_catch_base_ambiguous_ptr' which should not throw exceptions
  try {
    E e;
    throw e;
  } catch(A) {
  }
}
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: unhandled exception of type 'E' may be thrown in function 'throw_derived_catch_base_ambiguous_ptr' here

void throw_alias_catch_original() noexcept {
  using alias = int;

  try {
    alias a = 3;
    throw a;
  } catch (int) {
  }
}

void throw_alias_catch_original_warn() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_alias_catch_original_warn' which should not throw exceptions
  using alias = float;

  try {
    alias a = 3;
    throw a;
  } catch (int) {
  }
}
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: unhandled exception of type 'float' may be thrown in function 'throw_alias_catch_original_warn' here

void throw_original_catch_alias() noexcept {
  using alias = char;

  try {
    char **p = 0;
    throw p;
  } catch (volatile const alias *const *) {
  }
}

void throw_original_catch_alias_warn() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_original_catch_alias_warn' which should not throw exceptions
  using alias = int;

  try {
    char **p = 0;
    throw p;
  } catch (volatile const alias *const *) {
  }
}
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: unhandled exception of type 'char **' may be thrown in function 'throw_original_catch_alias_warn' here

void throw_original_catch_alias_2() noexcept {
  using alias = const char *const;

  try {
    char **p = 0;
    throw p;
  } catch (volatile alias *) {
  }
}

namespace a {
  int foo() { return 0; };

  void throw_regular_catch_regular() noexcept {
    try {
      throw &foo;
    } catch(int (*)()) {
    }
  }
}

namespace b {
  inline int foo() { return 0; };

  void throw_inline_catch_regular() noexcept {
    try {
      throw &foo;
    } catch(int (*)()) {
    }
  }
}

namespace c {
  inline int foo() noexcept { return 0; };

  void throw_noexcept_catch_regular() noexcept {
    try {
      throw &foo;
    } catch(int (*)()) {
    }
  }
}

struct baseMember {
    int *iptr;
    virtual void foo(){};
};

struct derivedMember : baseMember {
    void foo() override {};
};

void throw_basefn_catch_derivedfn() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_basefn_catch_derivedfn' which should not throw exceptions
  try {
    throw &baseMember::foo;
  } catch(void(derivedMember::*)()) {
  }
}
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: unhandled exception of type 'void (baseMember::*)()' may be thrown in function 'throw_basefn_catch_derivedfn' here

void throw_basefn_catch_basefn() noexcept {
  try {
    throw &baseMember::foo;
  } catch(void(baseMember::*)()) {
  }
}

void throw_basem_catch_basem_throw() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_basem_catch_basem_throw' which should not throw exceptions
  try {
    auto ptr = &baseMember::iptr;
    throw &ptr;
  } catch(const int* baseMember::* const *) {
  }
}
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: unhandled exception of type 'int *baseMember::**' may be thrown in function 'throw_basem_catch_basem_throw' here

void throw_basem_catch_basem() noexcept {
  try {
    auto ptr = &baseMember::iptr;
    throw &ptr;
  } catch(const int* const baseMember::* const *) {
  }
}

void throw_basem_catch_derivedm() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_basem_catch_derivedm' which should not throw exceptions
  try {
    auto ptr = &baseMember::iptr;
    throw &ptr;
  } catch(const int* const derivedMember::* const *) {
  }
}
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: unhandled exception of type 'int *baseMember::**' may be thrown in function 'throw_basem_catch_derivedm' here

void throw_derivedm_catch_basem() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_derivedm_catch_basem' which should not throw exceptions
  try {
    int *derivedMember::* ptr = &derivedMember::iptr;
    throw &ptr;
  } catch(const int* const baseMember::* const *) {
  }
}
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: unhandled exception of type 'int *derivedMember::**' may be thrown in function 'throw_derivedm_catch_basem' here

void throw_original_catch_alias_2_warn() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_original_catch_alias_2_warn' which should not throw exceptions
  using alias = const int *const;

  try {
    char **p = 0;
    throw p;
  } catch (volatile alias *) {
  }
}
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: unhandled exception of type 'char **' may be thrown in function 'throw_original_catch_alias_2_warn' here

void try_nested_try(int n) noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'try_nested_try' which should not throw exceptions
  try {
    try {
      if (n) throw 1;
      throw 1.1;
    } catch(int &) {
    }
  } catch(double &) {
  }
}

void bad_try_nested_try(int n) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'bad_try_nested_try' which should not throw exceptions
  try {
    if (n) throw 1;
    try {
      throw 1.1;
    } catch(int &) {
    }
  } catch(double &) {
  }
}
// CHECK-MESSAGES: :[[@LINE-8]]:12: note: frame #0: unhandled exception of type 'int' may be thrown in function 'bad_try_nested_try' here

void try_nested_catch() noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'try_nested_catch' which should not throw exceptions
  try {
    try {
      throw 1;
    } catch(int &) {
      throw 1.1;
    }
  } catch(double &) {
  }
}

void catch_nested_try() noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'catch_nested_try' which should not throw exceptions
  try {
    throw 1;
  } catch(int &) {
    try {
      throw 1;
    } catch(int &) {
    }
  }
}

void bad_catch_nested_try() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'bad_catch_nested_try' which should not throw exceptions
  try {
    throw 1;
  } catch(int &) {
    try {
      throw 1.1;
    } catch(int &) {
    }
  } catch(double &) {
  }
}
// CHECK-MESSAGES: :[[@LINE-6]]:7: note: frame #0: unhandled exception of type 'double' may be thrown in function 'bad_catch_nested_try' here

void implicit_int_thrower() {
  throw 1;
}

void explicit_int_thrower() noexcept(false) {
  throw 1;
}

void indirect_implicit() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'indirect_implicit' which should not throw exceptions
  implicit_int_thrower();
}
// CHECK-MESSAGES: :[[@LINE-11]]:3: note: frame #0: unhandled exception of type 'int' may be thrown in function 'implicit_int_thrower' here
// CHECK-MESSAGES: :[[@LINE-3]]:3: note: frame #1: function 'indirect_implicit' calls function 'implicit_int_thrower' here

void indirect_explicit() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'indirect_explicit' which should not throw exceptions
  explicit_int_thrower();
}
// CHECK-MESSAGES: :[[@LINE-14]]:3: note: frame #0: unhandled exception of type 'int' may be thrown in function 'explicit_int_thrower' here
// CHECK-MESSAGES: :[[@LINE-3]]:3: note: frame #1: function 'indirect_explicit' calls function 'explicit_int_thrower' here

void indirect_catch() noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'indirect_catch' which should not throw exceptions
  try {
    implicit_int_thrower();
  } catch(int&) {
  }
}

template<typename T>
void dependent_throw() noexcept(sizeof(T)<4) {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'dependent_throw' which should not throw exceptions
  if (sizeof(T) > 4)
    throw 1;
}

void swap(int&, int&) {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'swap' which should not throw exceptions
  throw 1;
}
// CHECK-MESSAGES: :[[@LINE-2]]:3: note: frame #0: unhandled exception of type 'int' may be thrown in function 'swap' here

void iter_swap(int&, int&) {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'iter_swap' which should not throw exceptions
  throw 1;
}
// CHECK-MESSAGES: :[[@LINE-2]]:3: note: frame #0: unhandled exception of type 'int' may be thrown in function 'iter_swap' here

void iter_move(int&) {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'iter_move' which should not throw exceptions
  throw 1;
}
// CHECK-MESSAGES: :[[@LINE-2]]:3: note: frame #0: unhandled exception of type 'int' may be thrown in function 'iter_move' here

namespace std {
class bad_alloc {};
}

void alloc() {
  throw std::bad_alloc();
}

void allocator() noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'allocator' which should not throw exceptions
  alloc();
}

void enabled1() {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'enabled1' which should not throw exceptions
  throw 1;
}
// CHECK-MESSAGES: :[[@LINE-2]]:3: note: frame #0: unhandled exception of type 'int' may be thrown in function 'enabled1' here

void enabled2() {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'enabled2' which should not throw exceptions
  enabled1();
}
// CHECK-MESSAGES: :[[@LINE-8]]:3: note: frame #0: unhandled exception of type 'int' may be thrown in function 'enabled1' here
// CHECK-MESSAGES: :[[@LINE-3]]:3: note: frame #1: function 'enabled2' calls function 'enabled1' here

void enabled3() {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'enabled3' which should not throw exceptions
  try {
    enabled1();
  } catch(...) {
  }
}

void deep_level3_thrower() {
  throw 42;
}

void deep_level2_caller() {
  deep_level3_thrower();
}

void deep_level1_caller() {
  deep_level2_caller();
}

void deep_stack_test() noexcept {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'deep_stack_test' which should not throw exceptions
  deep_level1_caller();
}
// CHECK-MESSAGES: :[[@LINE-15]]:3: note: frame #0: unhandled exception of type 'int' may be thrown in function 'deep_level3_thrower' here
// CHECK-MESSAGES: :[[@LINE-12]]:3: note: frame #1: function 'deep_level2_caller' calls function 'deep_level3_thrower' here
// CHECK-MESSAGES: :[[@LINE-9]]:3: note: frame #2: function 'deep_level1_caller' calls function 'deep_level2_caller' here
// CHECK-MESSAGES: :[[@LINE-5]]:3: note: frame #3: function 'deep_stack_test' calls function 'deep_level1_caller' here

// Template function call stack
template<typename T>
void template_thrower(T t) {
  throw T{};
}

template<typename T>
void template_caller(T t) {
  template_thrower<T>(t);
}

void template_stack_test() noexcept {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'template_stack_test' which should not throw exceptions
  template_caller<char>('a');
}
// CHECK-MESSAGES: :[[@LINE-12]]:3: note: frame #0: unhandled exception of type 'char' may be thrown in function 'template_thrower<char>' here
// CHECK-MESSAGES: :[[@LINE-8]]:3: note: frame #1: function 'template_caller<char>' calls function 'template_thrower<char>' here
// CHECK-MESSAGES: :[[@LINE-4]]:3: note: frame #2: function 'template_stack_test' calls function 'template_caller<char>' here

// template function without instantiation is not warned
template<typename T>
void template_function_stack_test(T t) noexcept {
  template_thrower(t);
}

void multiple_exception_types(int choice) noexcept {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'multiple_exception_types' which should not throw exceptions
  if (choice == 1) {
    throw 42;
  } else if (choice == 2) {
    throw 3.14;
  } else if (choice == 3) {
    throw "string";
  }
}
// CHECK-MESSAGES: note: frame #0: unhandled exception of type '{{(int|double|const char \*)}}' may be thrown in function 'multiple_exception_types' here

class ignored1 {};
class ignored2 {};

void this_does_not_count() noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'this_does_not_count' which should not throw exceptions
  throw ignored1();
}

void this_does_not_count_either(int n) noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'this_does_not_count_either' which should not throw exceptions
  try {
    throw 1;
    if (n) throw ignored2();
  } catch(int &) {
  }
}

void this_counts(int n) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'this_counts' which should not throw exceptions
  if (n) throw 1;
  throw ignored1();
}
// CHECK-MESSAGES: :[[@LINE-3]]:10: note: frame #0: unhandled exception of type 'int' may be thrown in function 'this_counts' here

void thrower(int n) {
  throw n;
}

int directly_recursive(int n) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: an exception may be thrown in function 'directly_recursive' which should not throw exceptions
  if (n == 0)
    thrower(n);
  return directly_recursive(n);
}
// CHECK-MESSAGES: :[[@LINE-9]]:3: note: frame #0: unhandled exception of type 'int' may be thrown in function 'thrower' here
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #1: function 'directly_recursive' calls function 'thrower' here

int indirectly_recursive(int n) noexcept;

int recursion_helper(int n) {
  indirectly_recursive(n);
  return 0;
}

int indirectly_recursive(int n) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: an exception may be thrown in function 'indirectly_recursive' which should not throw exceptions
  if (n == 0)
    thrower(n);
  return recursion_helper(n);
}
// CHECK-MESSAGES: :[[@LINE-25]]:3: note: frame #0: unhandled exception of type 'int' may be thrown in function 'thrower' here
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #1: function 'indirectly_recursive' calls function 'thrower' here

struct super_throws {
  super_throws() noexcept(false) { throw 42; }
};

struct sub_throws : super_throws {
  sub_throws() noexcept : super_throws() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: an exception may be thrown in function 'sub_throws' which should not throw exceptions
};
// CHECK-MESSAGES: :[[@LINE-7]]:36: note: frame #0: unhandled exception of type 'int' may be thrown in function 'super_throws' here
// CHECK-MESSAGES: :[[@LINE-4]]:27: note: frame #1: function 'sub_throws' calls function 'super_throws' here

struct init_member_throws {
  super_throws s;

  init_member_throws() noexcept : s() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: an exception may be thrown in function 'init_member_throws' which should not throw exceptions
};
// CHECK-MESSAGES: :[[@LINE-16]]:36: note: frame #0: unhandled exception of type 'int' may be thrown in function 'super_throws' here
// CHECK-MESSAGES: :[[@LINE-4]]:35: note: frame #1: function 'init_member_throws' calls function 'super_throws' here

struct implicit_init_member_throws {
  super_throws s;

  implicit_init_member_throws() noexcept {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: an exception may be thrown in function 'implicit_init_member_throws' which should not throw exceptions
};
// CHECK-MESSAGES: :[[@LINE-25]]:36: note: frame #0: unhandled exception of type 'int' may be thrown in function 'super_throws' here
// CHECK-MESSAGES: :[[@LINE-4]]:3: note: frame #1: function 'implicit_init_member_throws' calls function 'super_throws'

struct init {
  explicit init(int, int) noexcept(false) { throw 42; }
};

struct in_class_init_throws {
  init i{1, 2};

  in_class_init_throws() noexcept {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: an exception may be thrown in function 'in_class_init_throws' which should not throw exceptions
};
// CHECK-MESSAGES: :[[@LINE-9]]:45: note: frame #0: unhandled exception of type 'int' may be thrown in function 'init' here
// CHECK-MESSAGES: :[[@LINE-6]]:9: note: frame #1: function 'in_class_init_throws' calls function 'init' here

int main() {
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: an exception may be thrown in function 'main' which should not throw exceptions
  throw 1;
  return 0;
}
// CHECK-MESSAGES: :[[@LINE-3]]:3: note: frame #0: unhandled exception of type 'int' may be thrown in function 'main' here

// The following function all incorrectly throw exceptions, *but* calling them
// should not yield a warning because they are marked as noexcept.

void test_basic_no_throw() noexcept { throw 42; }
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'test_basic_no_throw' which should not throw exceptions
// CHECK-MESSAGES: :[[@LINE-2]]:39: note: frame #0: unhandled exception of type 'int' may be thrown in function 'test_basic_no_throw' here

void test_basic_throw() noexcept(false) { throw 42; }

void only_calls_non_throwing() noexcept {
  test_basic_no_throw();
}

void calls_non_and_throwing() noexcept {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'calls_non_and_throwing' which should not throw exceptions
  test_basic_no_throw();
  test_basic_throw();
}
// CHECK-MESSAGES: :[[@LINE-11]]:43: note: frame #0: unhandled exception of type 'int' may be thrown in function 'test_basic_throw' here
// CHECK-MESSAGES: :[[@LINE-3]]:3: note: frame #1: function 'calls_non_and_throwing' calls function 'test_basic_throw' here

namespace PR55143 { namespace PR40583 {

struct test_explicit_throw {
    test_explicit_throw() throw(int) { throw 42; }
    test_explicit_throw(const test_explicit_throw&) throw(int) { throw 42; }
    test_explicit_throw(test_explicit_throw&&) throw(int) { throw 42; }
    test_explicit_throw& operator=(const test_explicit_throw&) throw(int) { throw 42; }
    test_explicit_throw& operator=(test_explicit_throw&&) throw(int) { throw 42; }
    ~test_explicit_throw() throw(int) { throw 42; }
};

struct test_implicit_throw {
    test_implicit_throw() { throw 42; }
    test_implicit_throw(const test_implicit_throw&) { throw 42; }
    test_implicit_throw(test_implicit_throw&&) { throw 42; }
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: an exception may be thrown in function 'test_implicit_throw' which should not throw exceptions
    // CHECK-MESSAGES: :[[@LINE-2]]:50: note: frame #0: unhandled exception of type 'int' may be thrown in function 'test_implicit_throw' here
    test_implicit_throw& operator=(const test_implicit_throw&) { throw 42; }
    test_implicit_throw& operator=(test_implicit_throw&&) { throw 42; }
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: an exception may be thrown in function 'operator=' which should not throw exceptions
    // CHECK-MESSAGES: :[[@LINE-2]]:61: note: frame #0: unhandled exception of type 'int' may be thrown in function 'operator='
    ~test_implicit_throw() { throw 42; }
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: an exception may be thrown in function '~test_implicit_throw' which should not throw exceptions
    // CHECK-MESSAGES: :[[@LINE-2]]:30: note: frame #0: unhandled exception of type 'int' may be thrown in function '~test_implicit_throw' here
};

}}

void pointer_exception_can_not_escape_with_const_void_handler() noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'pointer_exception_can_not_escape_with_const_void_handler' which should not throw exceptions
  const int value = 42;
  try {
    throw &value;
  } catch (const void *) {
  }
}

void pointer_exception_can_not_escape_with_void_handler() noexcept {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'pointer_exception_can_not_escape_with_void_handler' which should not throw exceptions
  int value = 42;
  try {
    throw &value;
  } catch (void *) {
  }
}

void throw_in_uninvoked_lambda() noexcept {
  [] { throw 42; };
}

void throw_in_lambda() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_in_lambda' which should not throw exceptions
  [] { throw 42; }();
  // CHECK-MESSAGES: :[[@LINE-1]]:8: note: frame #0: unhandled exception of type 'int' may be thrown in function 'operator()' here
  // CHECK-MESSAGES: :[[@LINE-2]]:19: note: frame #1: function 'throw_in_lambda' calls function 'operator()' here
}

struct copy_constructor_throws {
  copy_constructor_throws(const copy_constructor_throws&) { throw 42; }
};

void throw_in_lambda_default_by_value_capture(const copy_constructor_throws& a) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_in_lambda_default_by_value_capture' which should not throw exceptions
  [=] { a; };
  // CHECK-MESSAGES: :[[@LINE-6]]:61: note: frame #0: unhandled exception of type 'int' may be thrown in function 'copy_constructor_throws' here
  // CHECK-MESSAGES: :[[@LINE-2]]:4: note: frame #1: function 'throw_in_lambda_default_by_value_capture' calls function 'copy_constructor_throws' here
}

void throw_in_lambda_explicit_by_value_capture(const copy_constructor_throws& a) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_in_lambda_explicit_by_value_capture' which should not throw exceptions
  [a] {};
  // CHECK-MESSAGES: :[[@LINE-13]]:61: note: frame #0: unhandled exception of type 'int' may be thrown in function 'copy_constructor_throws' here
  // CHECK-MESSAGES: :[[@LINE-2]]:4: note: frame #1: function 'throw_in_lambda_explicit_by_value_capture' calls function 'copy_constructor_throws' here
}

void no_throw_in_lambda_by_reference_capture(const copy_constructor_throws& a) noexcept {
  [&] { a; };
  [&a] {};
}

void throw_in_lambda_init_capture() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_in_lambda_init_capture' which should not throw exceptions
  [a = [] { throw 42; return 0; }()] {};
  // CHECK-MESSAGES: :[[@LINE-1]]:13: note: frame #0: unhandled exception of type 'int' may be thrown in function 'operator()' here
  // CHECK-MESSAGES: :[[@LINE-2]]:34: note: frame #1: function 'throw_in_lambda_init_capture' calls function 'operator()' here
}

void throw_from_nested_lambda() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_from_nested_lambda' which should not throw exceptions
  [] { [] { throw 42; }(); }();
  // CHECK-MESSAGES: :[[@LINE-1]]:13: note: frame #0: unhandled exception of type 'int' may be thrown in function 'operator()' here
  // CHECK-MESSAGES: :[[@LINE-2]]:24: note: frame #1: function 'operator()' calls function 'operator()' here
  // CHECK-MESSAGES: :[[@LINE-3]]:29: note: frame #2: function 'throw_from_nested_lambda' calls function 'operator()' here
}

const auto throw_in_noexcept_lambda = [] () noexcept { throw 42; };
// CHECK-MESSAGES: :[[@LINE-1]]:39: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
// CHECK-MESSAGES: :[[@LINE-2]]:56: note: frame #0: unhandled exception of type 'int' may be thrown in function 'operator()' here

int thrower() {
  throw 42;
}

const auto indirect_throw_in_noexcept_lambda = [] () noexcept { thrower(); };
// CHECK-MESSAGES: :[[@LINE-1]]:48: warning: an exception may be thrown in function 'operator()' which should not throw exceptions
// CHECK-MESSAGES: :[[@LINE-5]]:3: note: frame #0: unhandled exception of type 'int' may be thrown in function 'thrower' here
// CHECK-MESSAGES: :[[@LINE-3]]:65: note: frame #1: function 'operator()' calls function 'thrower' here

int f(int);
void throw_in_function_arg() noexcept {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_in_function_arg' which should not throw exceptions
  f(false ? 0 : throw 1);
}
// CHECK-MESSAGES: :[[@LINE-2]]:17: note: frame #0: unhandled exception of type 'int' may be thrown in function 'throw_in_function_arg' here

int g(int, int, int);
void throw_in_last_function_arg() noexcept {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_in_last_function_arg' which should not throw exceptions
  g(42, 67, false ? 0 : throw 1);
}
// CHECK-MESSAGES: :[[@LINE-2]]:25: note: frame #0: unhandled exception of type 'int' may be thrown in function 'throw_in_last_function_arg' here

void indirect_throw_in_function_arg() noexcept {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'indirect_throw_in_function_arg' which should not throw exceptions
  f(thrower());
}
// CHECK-MESSAGES: :[[@LINE-26]]:3: note: frame #0: unhandled exception of type 'int' may be thrown in function 'thrower' here
// CHECK-MESSAGES: :[[@LINE-3]]:5: note: frame #1: function 'indirect_throw_in_function_arg' calls function 'thrower' here

void indirect_throw_from_lambda_in_function_arg() noexcept {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'indirect_throw_from_lambda_in_function_arg' which should not throw exceptions
  f([] { throw 1; return 0; }());
}
// CHECK-MESSAGES: :[[@LINE-2]]:10: note: frame #0: unhandled exception of type 'int' may be thrown in function 'operator()' here
// CHECK-MESSAGES: :[[@LINE-3]]:30: note: frame #1: function 'indirect_throw_from_lambda_in_function_arg' calls function 'operator()' here

struct S {
  S(int) noexcept {}
};

void throw_in_constructor_arg() noexcept {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_in_constructor_arg' which should not throw exceptions
  S s(false ? 0 : throw 1);
}
// CHECK-MESSAGES: :[[@LINE-2]]:19: note: frame #0: unhandled exception of type 'int' may be thrown in function 'throw_in_constructor_arg' here

void indirect_throw_in_constructor_arg() noexcept {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'indirect_throw_in_constructor_arg' which should not throw exceptions
  S s = thrower();
}
// CHECK-MESSAGES: :[[@LINE-50]]:3: note: frame #0: unhandled exception of type 'int' may be thrown in function 'thrower' here
// CHECK-MESSAGES: :[[@LINE-3]]:9: note: frame #1: function 'indirect_throw_in_constructor_arg' calls function 'thrower' here

void weird_throw_in_call_subexpression() noexcept {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'weird_throw_in_call_subexpression' which should not throw exceptions
  (false ? []{} : throw 1)();
}
// CHECK-MESSAGES: :[[@LINE-2]]:19: note: frame #0: unhandled exception of type 'int' may be thrown in function 'weird_throw_in_call_subexpression' here
