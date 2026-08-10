// RUN: %clang_cc1 < %s -triple i386-mingw32 -fms-extensions -emit-llvm -x c++ | FileCheck %s

// optnone wins over inlinehint.
// Test that both func1 and func2 are marked optnone and noinline.

// Definition with both optnone and inlinehint.
__attribute__((optnone))
inline int func1(int a) {
  return a + a + a + a;
}
// CHECK: @_Z5func1i({{.*}}) [[OPTNONE:#[0-9]+]]

// optnone declaration, inlinehint definition.
__attribute__((optnone))
int func2(int a);

inline int func2(int a) {
  return a + a + a + a;
}
// CHECK: @_Z5func2i({{.*}}) [[OPTNONE]]

// Keep alive the definitions of func1 and func2.
int foo() {
  int val = func1(1);
  return val + func2(2);
}

// optnone wins over minsize.
__attribute__((optnone))
int func3(int a);

__attribute__((minsize))
int func3(int a) {
  return a + a + a + a;
}
// Same attribute set as everything else, therefore no 'minsize'.
// CHECK: @_Z5func3i({{.*}}) [[OPTNONE]]


// Verify that noreturn is compatible with optnone.
__attribute__((noreturn))
extern void exit_from_function();

__attribute__((noreturn)) __attribute((optnone))
extern void noreturn_function(int a) { exit_from_function(); }
// CHECK: @_Z17noreturn_functioni({{.*}}) [[NORETURN:#[0-9]+]]


// Verify that __declspec(noinline) is compatible with optnone.
__declspec(noinline) __attribute__((optnone))
void func4() { return; }
// CHECK: @_Z5func4v() [[OPTNONE]]

__declspec(noinline)
extern void func5();

__attribute__((optnone))
void func5() { return; }
// CHECK: @_Z5func5v() [[OPTNONE]]


// Verify also that optnone can be used on dllexport functions.
// Adding attribute optnone on a dllimport function has no effect.

__attribute__((dllimport))
__attribute__((optnone))
int imported_optnone_func(int a);

__attribute__((dllexport))
__attribute__((optnone))
int exported_optnone_func(int a) {
  return imported_optnone_func(a); // use of imported func
}
// CHECK: @_Z21exported_optnone_funci({{.*}}) [[OPTNONE]]
// CHECK: declare dllimport {{.*}} @_Z21imported_optnone_funci({{.*}}) [[DLLIMPORT:#[0-9]+]]


// CHECK: attributes [[OPTNONE]] = { mustprogress noinline {{.*}} optnone
// CHECK: attributes [[NORETURN]] = { mustprogress noinline noreturn {{.*}} optnone

// CHECK: attributes [[DLLIMPORT]] =
// CHECK-NOT: optnone
