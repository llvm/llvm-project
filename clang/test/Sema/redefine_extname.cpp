// RUN: %clang_cc1 -triple=x86_64-unknown-linux -Wpragmas -verify -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -triple=x86_64-unknown-linux -Wpragmas -verify -emit-llvm -o - %s | FileCheck %s --check-prefix=CONTAINS

/// Ensure nothing ever creates any identifiers with check_not_ in them.
// CONTAINS-NOT: check_not_

/// Check that C functions are affected normally.
#pragma redefine_extname check_not_foo_cfunc bar_cfunc
#pragma redefine_extname check_not_foo_cvar bar_cvar
extern "C" int check_not_foo_cfunc() { return 1; }
// CHECK-DAG: @bar_cfunc
extern "C" int check_not_foo_cvar = 1;
// CHECK-DAG: @bar_cvar

/// Check that there is a warning for C++ functions (which are not affected).
#pragma redefine_extname foo_cppfunc check_not_bar_cppfunc
int foo_cppfunc() { return 1; } // expected-warning {{#pragma redefine_extname is applicable to external C declarations only; not applied to function 'foo_cppfunc'}}
// CHECK-DAG: {{@[^ ]*foo_cppfunc}}

/// Check that there is a warning for C++ variables (which are not affected).
#pragma redefine_extname foo_cppvar check_not_bar_cppvar
int foo_cppvar = 1; // expected-warning {{#pragma redefine_extname is applicable to external C declarations only; not applied to variable 'foo_cppvar'}}
// CHECK-DAG: {{@[^ ]*foo_cppvar}}

/// Check that the warning goes away when doing it in a namespace.
/// Such uses are clearly scoped and need no warning (and often can be intentional).
#pragma redefine_extname foo_nsfunc check_not_bar_nsfunc
#pragma redefine_extname foo_nsvar check_not_bar_nsvar
namespace ns {
int foo_nsfunc() { return 1; }
// CHECK-DAG: {{@[^ ]*foo_nsfunc}}
int foo_nsvar = 1;
// CHECK-DAG: {{@[^ ]*foo_nsvar}}
} // namespace ns

/// Check that the warning goes away when doing it in a class.
/// Such uses are clearly scoped and need no warning (and often can be intentional).
#pragma redefine_extname foo_classmethod check_not_bar_classmethod
#pragma redefine_extname foo_staticmethod check_not_bar_staticmethod
#pragma redefine_extname foo_classmember check_not_bar_classmember
#pragma redefine_extname foo_staticmember check_not_bar_staticmember
class C {
public:
  int foo_classmethod();
  // CHECK-DAG: {{@[^ ]*foo_classmethod}}
  static int foo_staticmethod();
  // CHECK-DAG: {{@[^ ]*foo_staticmethod}}
  int foo_classmember = 1;
  // CHECK-DAG: {{%[^ ]*foo_classmember}}
  static int foo_staticmember;
  // CHECK-DAG: {{@[^ ]*foo_staticmember}}
};
int C::foo_classmethod() { return 1; }
int C::foo_staticmethod() { return 1; }
int C::foo_staticmember = 1;

// Force C to be actually instantiated. Emits a reference to foo_classmember.
void instantiate_C(C *p) { p = new C; }

/// Check that the warning remains when doing it in an extern "C++" block.
/// Such blocks do not affect scope.
extern "C++" {
#pragma redefine_extname foo_extcppfunc check_not_bar_extcppfunc
#pragma redefine_extname foo_extcppvar check_not_bar_extcppvar
int foo_extcppfunc() { return 1; } // expected-warning {{#pragma redefine_extname is applicable to external C declarations only; not applied to function 'foo_extcppfunc'}}
// CHECK-DAG: {{@[^ ]*foo_extcppfunc}}
int foo_extcppvar = 1; // expected-warning {{#pragma redefine_extname is applicable to external C declarations only; not applied to variable 'foo_extcppvar'}}
// CHECK-DAG: {{@[^ ]*foo_extcppvar}}
}

/// Check that the warning remains when doing it in C++ class friends.
/// These are actually in global scope.
#pragma redefine_extname foo_friendcppfunc check_not_bar_friendcppfunc
class F {
public:
  friend int foo_friendcppfunc(F f) { return 1; } // expected-warning {{#pragma redefine_extname is applicable to external C declarations only; not applied to function 'foo_friendcppfunc'}}
  // CHECK-DAG: {{@[^ ]*foo_friendcppfunc}}
};
// Force foo_friendcppfunc to be actually instantiated.
int instantiate_friendcppfunc() { return foo_friendcppfunc(F{}); }

/// Check that extern "C" friends can be renamed.
#pragma redefine_extname check_not_foo_friendcfunc bar_friendcfunc
extern "C" {
class CF {
public:
  friend int check_not_foo_friendcfunc(CF cf) { return 1; }
  // CHECK-DAG: @bar_friendcfunc
};
// Force foo_friendcfunc to be actually instantiated.
int instantiate_friendcfunc() { return check_not_foo_friendcfunc(CF{}); }
}

#pragma redefine_extname foo_arg check_not_bar_arg
#pragma redefine_extname foo_local check_not_bar_local
#pragma redefine_extname foo_staticlocal check_not_bar_staticlocal
int func(int foo_arg = 1) {
  // CHECK-DAG: %foo_arg
  int foo_local = 2;
  // CHECK-DAG: %foo_local
  static int foo_staticlocal = 3;
  // CHECK-DAG: {{@[^ ]*foo_staticlocal}}
  /// A hard enough nonsense computation to force foo_local to exist.
  for (int i = 0; i < foo_arg; ++i) {
    foo_local += foo_staticlocal++;
  }
  return foo_arg + foo_local + foo_staticlocal;
}
