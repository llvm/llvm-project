// This test checks if internal linkage symbols get unique names with
// -funique-internal-linkage-names option.
// RUN: %clang_cc1 -triple x86_64 -x c++ -S -emit-llvm -o - < %s | FileCheck %s --check-prefix=PLAIN
// RUN: %clang_cc1 -triple x86_64 -x c++ -O0 -S -emit-llvm -funique-internal-linkage-names -o - < %s | FileCheck %s --check-prefix=UNIQUE
// RUN: %clang_cc1 -triple x86_64 -x c++ -O1 -S -emit-llvm -funique-internal-linkage-names -o - < %s | FileCheck %s --check-prefix=UNIQUEO1
// RUN: %clang_cc1 -triple x86_64 -x c++ -O0 -S -emit-llvm -fexperimental-new-pass-manager -funique-internal-linkage-names -o - < %s | FileCheck %s --check-prefix=UNIQUE
// RUN: %clang_cc1 -triple x86_64 -x c++ -O1 -S -emit-llvm -fexperimental-new-pass-manager -funique-internal-linkage-names -o - < %s | FileCheck %s --check-prefix=UNIQUEO1

static int glob;
static int foo() {
  return 0;
}

int (*bar())() {
  return foo;
}

int getGlob() {
  return glob;
}

// Function local static variable and anonymous namespace namespace variable.
namespace {
int anon_m;
int getM() {
  return anon_m;
}
} // namespace

int retAnonM() {
  static int fGlob;
  return getM() + fGlob;
}

// Multiversioning symbols
__attribute__((target("default"))) static int mver() {
  return 0;
}

__attribute__((target("sse4.2"))) static int mver() {
  return 1;
}

int mver_call() {
  return mver();
}

// PLAIN: @_ZL4glob = internal global
// PLAIN: @_ZZ8retAnonMvE5fGlob = internal global
// PLAIN: @_ZN12_GLOBAL__N_16anon_mE = internal global
// PLAIN: define internal i32 @_ZL3foov()
// PLAIN: define internal i32 @_ZN12_GLOBAL__N_14getMEv
// PLAIN: define weak_odr i32 ()* @_ZL4mverv.resolver()
// PLAIN: define internal i32 @_ZL4mverv()
// PLAIN: define internal i32 @_ZL4mverv.sse4.2()
// UNIQUE: @_ZL4glob.{{[0-9a-f]+}} = internal global
// UNIQUE: @_ZZ8retAnonMvE5fGlob.{{[0-9a-f]+}} = internal global
// UNIQUE: @_ZN12_GLOBAL__N_16anon_mE.{{[0-9a-f]+}} = internal global
// UNIQUE: define internal i32 @_ZL3foov.{{[0-9a-f]+}}()
// UNIQUE: define internal i32 @_ZN12_GLOBAL__N_14getMEv.{{[0-9a-f]+}}
// UNIQUE: define weak_odr i32 ()* @_ZL4mverv.resolver()
// UNIQUE: define internal i32 @_ZL4mverv.{{[0-9a-f]+}}()
// UNIQUE: define internal i32 @_ZL4mverv.sse4.2.{{[0-9a-f]+}}
// UNIQUEO1: define internal i32 @_ZL3foov.{{[0-9a-f]+}}()
// UNIQUEO1: define weak_odr i32 ()* @_ZL4mverv.resolver()
// UNIQUEO1: define internal i32 @_ZL4mverv.{{[0-9a-f]+}}()
// UNIQUEO1: define internal i32 @_ZL4mverv.sse4.2.{{[0-9a-f]+}}
