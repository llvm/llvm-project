// REQUIRES: x86-registered-target

// Test that -fvisibility-from-dllstorageclass maps DLL storage class to visibility
// and that it overrides the effect of visibility options and annotations.

// RUN: %clang_cc1 -triple x86_64-unknown-windows-itanium -fdeclspec \
// RUN:     -fvisibility hidden \
// RUN:     -fvisibility-from-dllstorageclass \
// RUN:     -x c++ %s -S -emit-llvm -o - | \
// RUN:   FileCheck %s --check-prefixes=DEFAULT

// RUN: %clang_cc1 -triple x86_64-unknown-windows-itanium -fdeclspec \
// RUN:     -fvisibility hidden \
// RUN:     -fvisibility-from-dllstorageclass \
// RUN:     -fvisibility-dllexport=hidden \
// RUN:     -fvisibility-nodllstorageclass=protected \
// RUN:     -fvisibility-externs-dllimport=hidden \
// RUN:     -fvisibility-externs-nodllstorageclass=protected \
// RUN:     -x c++  %s -S -emit-llvm -o - | \
// RUN:   FileCheck %s --check-prefixes=EXPLICIT

// Local
static void l() {}
void use_locals(){l();}
// DEFAULT-DAG: define internal void @_ZL1lv()
// EXPLICIT-DAG: define internal void @_ZL1lv()

// Function
void f() {}
void __declspec(dllexport) exported_f() {}
// DEFAULT-DAG: define hidden void @_Z1fv()
// DEFAULT-DAG: define dso_local void @_Z10exported_fv()
// EXPLICIT-DAG: define protected void @_Z1fv()
// EXPLICIT-DAG: define hidden void @_Z10exported_fv()

// Variable
int d = 123;
__declspec(dllexport) int exported_d = 123;
// DEFAULT-DAG: @d = hidden global
// DEFAULT-DAG: @exported_d = dso_local global
// EXPLICIT-DAG: @d = protected global
// EXPLICIT-DAG: @exported_d = hidden global

// Alias
extern "C" void aliased() {}
void a() __attribute__((alias("aliased")));
void __declspec(dllexport) a_exported() __attribute__((alias("aliased")));
// DEFAULT-DAG: @_Z1av = hidden alias
// DEFAULT-DAG: @_Z10a_exportedv = dso_local alias
// EXPLICIT-DAG: @_Z1av = protected alias
// EXPLICIT-DAG: @_Z10a_exportedv = hidden alias

// Declaration
extern void e();
extern void __declspec(dllimport) imported_e();
void use_declarations(){e(); imported_e();}
// DEFAULT-DAG: declare hidden void @_Z1ev()
// DEFAULT-DAG: declare void @_Z10imported_ev()
// EXPLICIT-DAG: declare protected void @_Z1ev()
// EXPLICIT-DAG: declare hidden void @_Z10imported_ev()

// Show that -fvisibility-from-dllstorageclass overrides the effect of visibility annotations.

struct __attribute__((type_visibility("protected"))) t {
  virtual void foo();
};
void t::foo() {}
// DEFAULT-DAG: @_ZTV1t = hidden unnamed_addr constant

int v __attribute__ ((__visibility__ ("protected"))) = 123;
// DEFAULT-DAG: @v = hidden global

#pragma GCC visibility push(protected)
int p = 345;
#pragma GCC visibility pop
// DEFAULT-DAG: @p = hidden global
