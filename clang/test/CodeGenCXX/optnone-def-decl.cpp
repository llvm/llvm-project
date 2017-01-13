// RUN: %clang_cc1 %s -triple %itanium_abi_triple -fms-extensions -O2 -disable-llvm-passes -emit-llvm -o - | FileCheck %s

// Test optnone on both function declarations and function definitions.
// Verify also that we don't generate invalid IR functions with
// both alwaysinline and noinline. (optnone implies noinline and wins
// over alwaysinline, in all cases.)

// Test optnone on extern declaration only.
extern int decl_only(int a) __attribute__((optnone));

// This function should be marked 'optnone'.
int decl_only(int a) {
  return a + a + a + a;
}

// CHECK: define {{.*}} @_Z9decl_onlyi({{.*}}) [[OPTNONE:#[0-9]+]]

// Test optnone on definition but not extern declaration.
extern int def_only(int a);

__attribute__((optnone))
int def_only(int a) {
  return a + a + a + a;
}

// Function def_only is a optnone function and therefore it should not be
// inlined inside 'user_of_def_only'.
int user_of_def_only() {
  return def_only(5);
}

// CHECK: define {{.*}} @_Z8def_onlyi({{.*}}) [[OPTNONE]]
// CHECK: define {{.*}} @_Z16user_of_def_onlyv() [[NORMAL:#[0-9]+]]

// Test optnone on both definition and declaration.
extern int def_and_decl(int a) __attribute__((optnone));

__attribute__((optnone))
int def_and_decl(int a) {
  return a + a + a + a;
}

// CHECK: define {{.*}} @_Z12def_and_decli({{.*}}) [[OPTNONE]]

// Check that optnone wins over always_inline.

// Test optnone on definition and always_inline on declaration.
extern int always_inline_function(int a) __attribute__((always_inline));

__attribute__((optnone))
extern int always_inline_function(int a) {
  return a + a + a + a;
}
// CHECK: define {{.*}} @_Z22always_inline_functioni({{.*}}) [[OPTNONE]]

int user_of_always_inline_function() {
  return always_inline_function(4);
}

// CHECK: define {{.*}} @_Z30user_of_always_inline_functionv() [[NORMAL]]

// Test optnone on declaration and always_inline on definition.
extern int optnone_function(int a) __attribute__((optnone));

__attribute__((always_inline))
int optnone_function(int a) {
  return a + a + a + a;
}
// CHECK: define {{.*}} @_Z16optnone_functioni({{.*}}) [[OPTNONE]]

int user_of_optnone_function() {
  return optnone_function(4);
}

// CHECK: define {{.*}} @_Z24user_of_optnone_functionv() [[NORMAL]]

// Test the combination of optnone with forceinline (optnone wins).
extern __forceinline int forceinline_optnone_function(int a, int b);

__attribute__((optnone))
extern int forceinline_optnone_function(int a, int b) {
    return a + b;
}

int user_of_forceinline_optnone_function() {
    return forceinline_optnone_function(4,5);
}

// CHECK: @_Z36user_of_forceinline_optnone_functionv() [[NORMAL]]
// CHECK: @_Z28forceinline_optnone_functionii({{.*}}) [[OPTNONE]]

// CHECK: attributes [[OPTNONE]] = { noinline nounwind optnone {{.*}} }
// CHECK: attributes [[NORMAL]] =
// CHECK-NOT: noinline
// CHECK-NOT: optnone
