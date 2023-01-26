// REQUIRES: x86-registered-target

// Test that -fvisibility-from-dllstorageclass maps DLL storage class to visibility
// and that it overrides the effect of visibility options and annotations.

// RUN: %clang_cc1 -triple x86_64-unknown-windows-itanium -fdeclspec \
// RUN:     -fvisibility=hidden \
// RUN:     -fapply-global-visibility-to-externs \
// RUN:     -fvisibility-from-dllstorageclass \
// RUN:     -x c++ %s -S -emit-llvm -o - | \
// RUN:   FileCheck %s --check-prefixes=DEFAULTS

// RUN: %clang_cc1 -triple x86_64-unknown-windows-itanium -fdeclspec \
// RUN:     -fvisibility=hidden \
// RUN:     -fapply-global-visibility-to-externs \
// RUN:     -fvisibility-from-dllstorageclass \
// RUN:     -fvisibility-dllexport=hidden \
// RUN:     -fvisibility-nodllstorageclass=protected \
// RUN:     -fvisibility-externs-dllimport=hidden \
// RUN:     -fvisibility-externs-nodllstorageclass=protected \
// RUN:     -x c++  %s -S -emit-llvm -o - | \
// RUN:   FileCheck %s --check-prefixes=EXPLICIT

// RUN: %clang_cc1 -triple x86_64-unknown-windows-itanium -fdeclspec \
// RUN:     -fvisibility=hidden \
// RUN:     -fapply-global-visibility-to-externs \
// RUN:     -fvisibility-from-dllstorageclass \
// RUN:     -fvisibility-dllexport=default \
// RUN:     -fvisibility-nodllstorageclass=default \
// RUN:     -fvisibility-externs-dllimport=default \
// RUN:     -fvisibility-externs-nodllstorageclass=default \
// RUN:     -x c++  %s -S -emit-llvm -o - | \
// RUN:   FileCheck %s --check-prefixes=ALL_DEFAULT

// Local
static void l() {}
void use_locals(){l();}
// DEFAULTS-DAG: define internal void @_ZL1lv()
// EXPLICIT-DAG: define internal void @_ZL1lv()
// ALL_DEFAULT-DAG: define internal void @_ZL1lv()

// Function
void f() {}
void __declspec(dllexport) exported_f() {}
// DEFAULTS-DAG: define hidden void @_Z1fv()
// DEFAULTS-DAG: define void @_Z10exported_fv()
// EXPLICIT-DAG: define protected void @_Z1fv()
// EXPLICIT-DAG: define hidden void @_Z10exported_fv()
// ALL_DEFAULT-DAG: define void @_Z1fv()
// ALL_DEFAULT-DAG: define void @_Z10exported_fv()

// Variable
int d = 123;
__declspec(dllexport) int exported_d = 123;
// DEFAULTS-DAG: @d = hidden global
// DEFAULTS-DAG: @exported_d = global
// EXPLICIT-DAG: @d = protected global
// EXPLICIT-DAG: @exported_d = hidden global
// ALL_DEFAULT-DAG: @d = global
// ALL_DEFAULT-DAG: @exported_d = global

// Alias
extern "C" void aliased() {}
void a() __attribute__((alias("aliased")));
void __declspec(dllexport) a_exported() __attribute__((alias("aliased")));
// DEFAULTS-DAG: @_Z1av = hidden alias
// DEFAULTS-DAG: @_Z10a_exportedv = alias
// EXPLICIT-DAG: @_Z1av = protected alias
// EXPLICIT-DAG: @_Z10a_exportedv = hidden alias
// ALL_DEFAULT-DAG: @_Z1av = alias
// ALL_DEFAULT-DAG: @_Z10a_exportedv = alias

// Declaration
extern void e();
extern void __declspec(dllimport) imported_e();
// DEFAULTS-DAG: declare hidden void @_Z1ev()
// DEFAULTS-DAG: declare void @_Z10imported_ev()
// EXPLICIT-DAG: declare protected void @_Z1ev()
// EXPLICIT-DAG: declare hidden void @_Z10imported_ev()
// ALL_DEFAULT-DAG: declare void @_Z1ev()
// ALL_DEFAULT-DAG: declare void @_Z10imported_ev()

// Weak Declaration
__attribute__((weak))
extern void w();
__attribute__((weak))
extern void __declspec(dllimport) imported_w();
// DEFAULTS-DAG: declare extern_weak hidden void @_Z1wv()
// DEFAULTS-DAG: declare extern_weak void @_Z10imported_wv()
// EXPLICIT-DAG: declare extern_weak protected void @_Z1wv()
// EXPLICIT-DAG: declare extern_weak hidden void @_Z10imported_wv()
// ALL_DEFAULT-DAG: declare extern_weak void @_Z1wv()
// ALL_DEFAULT-DAG: declare extern_weak void @_Z10imported_wv()

void use_declarations(){e(); imported_e(); w(); imported_w();}

// Show that -fvisibility-from-dllstorageclass overrides the effect of visibility annotations.

struct __attribute__((type_visibility("protected"))) t {
  virtual void foo();
};
void t::foo() {}
// DEFAULTS-DAG: @_ZTV1t = hidden unnamed_addr constant

int v __attribute__ ((__visibility__ ("protected"))) = 123;
// DEFAULTS-DAG: @v = hidden global

#pragma GCC visibility push(protected)
int p = 345;
#pragma GCC visibility pop
// DEFAULTS-DAG: @p = hidden global
