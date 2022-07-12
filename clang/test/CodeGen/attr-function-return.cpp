// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -emit-llvm -o - \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CHECK-NOM
// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -emit-llvm -o - \
// RUN:   -mfunction-return=keep | FileCheck %s \
// RUN:   --check-prefixes=CHECK,CHECK-KEEP
// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -emit-llvm -o - \
// RUN:  -mfunction-return=thunk-extern | FileCheck %s \
// RUN:  --check-prefixes=CHECK,CHECK-EXTERN

int foo(void) {
  // CHECK: @"_ZZ3foovENK3$_0clEv"({{.*}}) [[NOATTR:#[0-9]+]]
  return []() {
    return 42;
  }();
}
int bar(void) {
  // CHECK: @"_ZZ3barvENK3$_1clEv"({{.*}}) [[EXTERN:#[0-9]+]]
  return []() __attribute__((function_return("thunk-extern"))) {
    return 42;
  }
  ();
}
int baz(void) {
  // CHECK: @"_ZZ3bazvENK3$_2clEv"({{.*}}) [[KEEP:#[0-9]+]]
  return []() __attribute__((function_return("keep"))) {
    return 42;
  }
  ();
}

class Foo {
public:
  // CHECK: @_ZN3Foo3fooEv({{.*}}) [[EXTERN]]
  __attribute__((function_return("thunk-extern"))) int foo() { return 42; }
};

int quux() {
  Foo my_foo;
  return my_foo.foo();
}

// CHECK: @extern_c() [[EXTERN]]
extern "C" __attribute__((function_return("thunk-extern"))) void extern_c() {}
extern "C" {
// CHECK: @extern_c2() [[EXTERN]]
__attribute__((function_return("thunk-extern"))) void extern_c2() {}
}

// CHECK-NOM-NOT:   [[NOATTR]] = {{.*}}fn_ret_thunk_extern
// CHECK-KEEP-NOT:  [[NOATTR]] = {{.*}}fn_ret_thunk_extern
// CHECK-KEEP-NOT:  [[KEEP]] = {{.*}}fn_ret_thunk_extern
// CHECK-EXTERN:    [[EXTERN]] = {{.*}}fn_ret_thunk_extern
