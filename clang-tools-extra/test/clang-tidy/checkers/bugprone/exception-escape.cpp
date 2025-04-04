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
// CHECK-MESSAGES: :[[@LINE-5]]:3: note: example of unhandled exception throw stack, starting from function '~throwing_destructor'
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: function '~throwing_destructor' throws unhandled exception here

struct throwing_move_constructor {
  throwing_move_constructor(throwing_move_constructor&&) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: an exception may be thrown in function 'throwing_move_constructor' which should not throw exceptions
    throw 1;
  }
};
// CHECK-MESSAGES: :[[@LINE-5]]:3: note: example of unhandled exception throw stack, starting from function 'throwing_move_constructor'
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: function 'throwing_move_constructor' throws unhandled exception here

struct throwing_move_assignment {
  throwing_move_assignment& operator=(throwing_move_assignment&&) {
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: an exception may be thrown in function 'operator=' which should not throw exceptions
    throw 1;
  }
};
// CHECK-MESSAGES: :[[@LINE-5]]:29: note: example of unhandled exception throw stack, starting from function 'operator='
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: function 'operator=' throws unhandled exception here

void throwing_noexcept() noexcept {
    // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throwing_noexcept' which should not throw exceptions
  throw 1;
}
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: example of unhandled exception throw stack, starting from function 'throwing_noexcept'
// CHECK-MESSAGES: :[[@LINE-3]]:3: note: frame #0: function 'throwing_noexcept' throws unhandled exception here

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
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: example of unhandled exception throw stack, starting from function 'throw_and_catch_some'
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: frame #0: function 'throw_and_catch_some' throws unhandled exception here

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
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: example of unhandled exception throw stack, starting from function 'throw_and_rethrow'
// CHECK-MESSAGES: :[[@LINE-6]]:5: note: frame #0: function 'throw_and_rethrow' throws unhandled exception here

void throw_catch_throw() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_catch_throw' which should not throw exceptions
  try {
    throw 1;
  } catch(int &) {
    throw 2;
  }
}
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: example of unhandled exception throw stack, starting from function 'throw_catch_throw'
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: function 'throw_catch_throw' throws unhandled exception here

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
// CHECK-MESSAGES: :[[@LINE-10]]:6: note: example of unhandled exception throw stack, starting from function 'throw_catch_rethrow_the_rest'
// CHECK-MESSAGES: :[[@LINE-7]]:5: note: frame #0: function 'throw_catch_rethrow_the_rest' throws unhandled exception here

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
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: example of unhandled exception throw stack, starting from function 'throw_catch_multi_ptr_1'
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: frame #0: function 'throw_catch_multi_ptr_1' throws unhandled exception here

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
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: example of unhandled exception throw stack, starting from function 'throw_c_catch_pointer'
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: function 'throw_c_catch_pointer' throws unhandled exception here

void throw_c_catch_pointer_v() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_c_catch_pointer_v' which should not throw exceptions
  try {
    int a = 1;
    const int *p = &a;
    throw p;
  } catch(volatile int *) {}
}
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: example of unhandled exception throw stack, starting from function 'throw_c_catch_pointer_v'
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: function 'throw_c_catch_pointer_v' throws unhandled exception here

void throw_v_catch_pointer() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_v_catch_pointer' which should not throw exceptions
  try {
    int a = 1;
    volatile int *p = &a;
    throw p;
  } catch(int *) {}
}
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: example of unhandled exception throw stack, starting from function 'throw_v_catch_pointer'
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: function 'throw_v_catch_pointer' throws unhandled exception here

void throw_v_catch_pointer_c() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_v_catch_pointer_c' which should not throw exceptions
  try {
    int a = 1;
    volatile int *p = &a;
    throw p;
  } catch(const int *) {}
}
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: example of unhandled exception throw stack, starting from function 'throw_v_catch_pointer_c'
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: function 'throw_v_catch_pointer_c' throws unhandled exception here

void throw_cv_catch_pointer_c() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_cv_catch_pointer_c' which should not throw exceptions
  try {
    int a = 1;
    const volatile int *p = &a;
    throw p;
  } catch(const int *) {}
}
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: example of unhandled exception throw stack, starting from function 'throw_cv_catch_pointer_c'
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: function 'throw_cv_catch_pointer_c' throws unhandled exception here

void throw_cv_catch_pointer_v() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_cv_catch_pointer_v' which should not throw exceptions
  try {
    int a = 1;
    const volatile int *p = &a;
    throw p;
  } catch(volatile int *) {}
}
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: example of unhandled exception throw stack, starting from function 'throw_cv_catch_pointer_v'
// CHECK-MESSAGES: :[[@LINE-4]]:5: note: frame #0: function 'throw_cv_catch_pointer_v' throws unhandled exception here

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
// CHECK-MESSAGES: :[[@LINE-9]]:6: note: example of unhandled exception throw stack, starting from function 'throw_derived_catch_base_ptr'
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: frame #0: function 'throw_derived_catch_base_ptr' throws unhandled exception here

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
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: example of unhandled exception throw stack, starting from function 'throw_derived_catch_base_private'
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: frame #0: function 'throw_derived_catch_base_private' throws unhandled exception here

void throw_derived_catch_base_private_ptr() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_derived_catch_base_private_ptr' which should not throw exceptions
  try {
    B b;
    throw &b;
  } catch(A *) {
  }
}
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: example of unhandled exception throw stack, starting from function 'throw_derived_catch_base_private_ptr'
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: frame #0: function 'throw_derived_catch_base_private_ptr' throws unhandled exception here

void throw_derived_catch_base_protected() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_derived_catch_base_protected' which should not throw exceptions
  try {
    C c;
    throw c;
  } catch(A) {
  }
}
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: example of unhandled exception throw stack, starting from function 'throw_derived_catch_base_protected'
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: frame #0: function 'throw_derived_catch_base_protected' throws unhandled exception here

void throw_derived_catch_base_protected_ptr() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_derived_catch_base_protected_ptr' which should not throw exceptions
  try {
    C c;
    throw &c;
  } catch(A *) {
  }
}
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: example of unhandled exception throw stack, starting from function 'throw_derived_catch_base_protected_ptr'
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: frame #0: function 'throw_derived_catch_base_protected_ptr' throws unhandled exception here

void throw_derived_catch_base_ambiguous() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_derived_catch_base_ambiguous' which should not throw exceptions
  try {
    E e;
    throw e;
  } catch(A) {
  }
}
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: example of unhandled exception throw stack, starting from function 'throw_derived_catch_base_ambiguous'
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: frame #0: function 'throw_derived_catch_base_ambiguous' throws unhandled exception here

void throw_derived_catch_base_ambiguous_ptr() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_derived_catch_base_ambiguous_ptr' which should not throw exceptions
  try {
    E e;
    throw e;
  } catch(A) {
  }
}
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: example of unhandled exception throw stack, starting from function 'throw_derived_catch_base_ambiguous_ptr'
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: frame #0: function 'throw_derived_catch_base_ambiguous_ptr' throws unhandled exception here

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
// CHECK-MESSAGES: :[[@LINE-10]]:6: note: example of unhandled exception throw stack, starting from function 'throw_alias_catch_original_warn'
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: frame #0: function 'throw_alias_catch_original_warn' throws unhandled exception here

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
// CHECK-MESSAGES: :[[@LINE-10]]:6: note: example of unhandled exception throw stack, starting from function 'throw_original_catch_alias_warn'
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: frame #0: function 'throw_original_catch_alias_warn' throws unhandled exception here

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
// CHECK-MESSAGES: :[[@LINE-7]]:6: note: example of unhandled exception throw stack, starting from function 'throw_basefn_catch_derivedfn'
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: frame #0: function 'throw_basefn_catch_derivedfn' throws unhandled exception here

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
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: example of unhandled exception throw stack, starting from function 'throw_basem_catch_basem_throw'
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: frame #0: function 'throw_basem_catch_basem_throw' throws unhandled exception here

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
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: example of unhandled exception throw stack, starting from function 'throw_basem_catch_derivedm'
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: frame #0: function 'throw_basem_catch_derivedm' throws unhandled exception here

void throw_derivedm_catch_basem() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_derivedm_catch_basem' which should not throw exceptions
  try {
    int *derivedMember::* ptr = &derivedMember::iptr;
    throw &ptr;
  } catch(const int* const baseMember::* const *) {
  }
}
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: example of unhandled exception throw stack, starting from function 'throw_derivedm_catch_basem'
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: frame #0: function 'throw_derivedm_catch_basem' throws unhandled exception here

void throw_original_catch_alias_2_warn() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throw_original_catch_alias_2_warn' which should not throw exceptions
  using alias = const int *const;

  try {
    char **p = 0;
    throw p;
  } catch (volatile alias *) {
  }
}
// CHECK-MESSAGES: :[[@LINE-10]]:6: note: example of unhandled exception throw stack, starting from function 'throw_original_catch_alias_2_warn'
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: frame #0: function 'throw_original_catch_alias_2_warn' throws unhandled exception here

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
// CHECK-MESSAGES: :[[@LINE-11]]:6: note: example of unhandled exception throw stack, starting from function 'bad_try_nested_try'
// CHECK-MESSAGES: :[[@LINE-9]]:12: note: frame #0: function 'bad_try_nested_try' throws unhandled exception here

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
// CHECK-MESSAGES: :[[@LINE-12]]:6: note: example of unhandled exception throw stack, starting from function 'bad_catch_nested_try'
// CHECK-MESSAGES: :[[@LINE-7]]:7: note: frame #0: function 'bad_catch_nested_try' throws unhandled exception here

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
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: example of unhandled exception throw stack, starting from function 'indirect_implicit'
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: frame #0: function 'indirect_implicit'
// CHECK-MESSAGES: :[[@LINE-13]]:3: note: frame #1: function 'implicit_int_thrower' throws unhandled exception here

void indirect_explicit() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'indirect_explicit' which should not throw exceptions
  explicit_int_thrower();
}
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: example of unhandled exception throw stack, starting from function 'indirect_explicit'
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: frame #0: function 'indirect_explicit'
// CHECK-MESSAGES: :[[@LINE-17]]:3: note: frame #1: function 'explicit_int_thrower' throws unhandled exception here

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
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: example of unhandled exception throw stack, starting from function 'swap'
// CHECK-MESSAGES: :[[@LINE-3]]:3: note: frame #0: function 'swap' throws unhandled exception here

void iter_swap(int&, int&) {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'iter_swap' which should not throw exceptions
  throw 1;
}
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: example of unhandled exception throw stack, starting from function 'iter_swap'
// CHECK-MESSAGES: :[[@LINE-3]]:3: note: frame #0: function 'iter_swap' throws unhandled exception here

void iter_move(int&) {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'iter_move' which should not throw exceptions
  throw 1;
}
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: example of unhandled exception throw stack, starting from function 'iter_move'
// CHECK-MESSAGES: :[[@LINE-3]]:3: note: frame #0: function 'iter_move' throws unhandled exception here

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
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: example of unhandled exception throw stack, starting from function 'enabled1'
// CHECK-MESSAGES: :[[@LINE-3]]:3: note: frame #0: function 'enabled1' throws unhandled exception here

void enabled2() {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'enabled2' which should not throw exceptions
  enabled1();
}
// CHECK-MESSAGES: :[[@LINE-4]]:6: note: example of unhandled exception throw stack, starting from function 'enabled2'
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: frame #0: function 'enabled2'
// CHECK-MESSAGES: :[[@LINE-11]]:3: note: frame #1: function 'enabled1' throws unhandled exception here

void enabled3() {
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'enabled3' which should not throw exceptions
  try {
    enabled1();
  } catch(...) {
  }
}

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
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: example of unhandled exception throw stack, starting from function 'this_counts'
// CHECK-MESSAGES: :[[@LINE-4]]:10: note: frame #0: function 'this_counts' throws unhandled exception here

void thrower(int n) {
  throw n;
}

int directly_recursive(int n) noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: an exception may be thrown in function 'directly_recursive' which should not throw exceptions
  if (n == 0)
    thrower(n);
  return directly_recursive(n);
}
// CHECK-MESSAGES: :[[@LINE-6]]:5: note: example of unhandled exception throw stack, starting from function 'directly_recursive'
// CHECK-MESSAGES: :[[@LINE-7]]:5: note: frame #0: function 'directly_recursive'
// CHECK-MESSAGES: :[[@LINE-11]]:3: note: frame #1: function 'thrower' throws unhandled exception here

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

struct super_throws {
  super_throws() noexcept(false) { throw 42; }
};

struct sub_throws : super_throws {
  sub_throws() noexcept : super_throws() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: an exception may be thrown in function 'sub_throws' which should not throw exceptions
};
// CHECK-MESSAGES: :[[@LINE-3]]:3: note: example of unhandled exception throw stack, starting from function 'sub_throws'
// CHECK-MESSAGES: :[[@LINE-4]]:3: note: frame #0: function 'sub_throws'
// CHECK-MESSAGES: :[[@LINE-9]]:36: note: frame #1: function 'super_throws' throws unhandled exception here

struct init_member_throws {
  super_throws s;

  init_member_throws() noexcept : s() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: an exception may be thrown in function 'init_member_throws' which should not throw exceptions
};
// CHECK-MESSAGES: :[[@LINE-3]]:3: note: example of unhandled exception throw stack, starting from function 'init_member_throws'
// CHECK-MESSAGES: :[[@LINE-4]]:3: note: frame #0: function 'init_member_throws'

struct implicit_init_member_throws {
  super_throws s;

  implicit_init_member_throws() noexcept {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: an exception may be thrown in function 'implicit_init_member_throws' which should not throw exceptions
};
// CHECK-MESSAGES: :[[@LINE-3]]:3: note: example of unhandled exception throw stack, starting from function 'implicit_init_member_throws'
// CHECK-MESSAGES: :[[@LINE-4]]:3: note: frame #0: function 'implicit_init_member_throws'

struct init {
  explicit init(int, int) noexcept(false) { throw 42; }
};

struct in_class_init_throws {
  init i{1, 2};

  in_class_init_throws() noexcept {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: an exception may be thrown in function 'in_class_init_throws' which should not throw exceptions
};
// CHECK-MESSAGES: :[[@LINE-3]]:3: note: example of unhandled exception throw stack, starting from function 'in_class_init_throws'
// CHECK-MESSAGES: :[[@LINE-4]]:3: note: frame #0: function 'in_class_init_throws'
// CHECK-MESSAGES: :[[@LINE-11]]:45: note: frame #1: function 'init' throws unhandled exception here

int main() {
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: an exception may be thrown in function 'main' which should not throw exceptions
  throw 1;
  return 0;
}
// CHECK-MESSAGES: :[[@LINE-5]]:5: note: example of unhandled exception throw stack, starting from function 'main'
// CHECK-MESSAGES: :[[@LINE-4]]:3: note: frame #0: function 'main' throws unhandled exception here

// The following function all incorrectly throw exceptions, *but* calling them
// should not yield a warning because they are marked as noexcept.

void test_basic_no_throw() noexcept { throw 42; }
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'test_basic_no_throw' which should not throw exceptions
// CHECK-MESSAGES: :[[@LINE-2]]:6: note: example of unhandled exception throw stack, starting from function 'test_basic_no_throw'
// CHECK-MESSAGES: :[[@LINE-3]]:39: note: frame #0: function 'test_basic_no_throw' throws unhandled exception here

void test_basic_throw() noexcept(false) { throw 42; }

void only_calls_non_throwing() noexcept {
  test_basic_no_throw();
}

void calls_non_and_throwing() noexcept {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'calls_non_and_throwing' which should not throw exceptions
  test_basic_no_throw();
  test_basic_throw();
}
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: example of unhandled exception throw stack, starting from function 'calls_non_and_throwing'
// CHECK-MESSAGES: :[[@LINE-6]]:6: note: frame #0: function 'calls_non_and_throwing'
// CHECK-MESSAGES: :[[@LINE-13]]:43: note: frame #1: function 'test_basic_throw' throws unhandled exception here

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
    // CHECK-MESSAGES: :[[@LINE-2]]:5: note: example of unhandled exception throw stack, starting from function 'test_implicit_throw'
    // CHECK-MESSAGES: :[[@LINE-3]]:50: note: frame #0: function 'test_implicit_throw'
    test_implicit_throw& operator=(const test_implicit_throw&) { throw 42; }
    test_implicit_throw& operator=(test_implicit_throw&&) { throw 42; }
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: an exception may be thrown in function 'operator=' which should not throw exceptions
    // CHECK-MESSAGES: :[[@LINE-2]]:26: note: example of unhandled exception throw stack, starting from function 'operator='
    // CHECK-MESSAGES: :[[@LINE-3]]:61: note: frame #0: function 'operator='
    ~test_implicit_throw() { throw 42; }
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: an exception may be thrown in function '~test_implicit_throw' which should not throw exceptions
    // CHECK-MESSAGES: :[[@LINE-2]]:5: note: example of unhandled exception throw stack, starting from function '~test_implicit_throw'
    // CHECK-MESSAGES: :[[@LINE-3]]:30: note: frame #0: function '~test_implicit_throw'
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
