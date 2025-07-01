// This test is copied from clang/test/CodeGen/dso-local-executable.c with
// unsupported targets, thread_local variables, and function checks removed.

// These are here so we find this test when grepping for missing features.
// cir::MissingFeatures::opGlobalThreadLocal()
// cir::MissingFeatures::opFuncDsoLocal()

/// Static relocation model defaults to -fdirect-access-external-data and sets
/// dso_local on most global objects.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -mrelocation-model static %s -o - | FileCheck --check-prefix=STATIC %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -mrelocation-model static -fdirect-access-external-data %s -o - | FileCheck --check-prefix=STATIC %s
// STATIC:      @baz = dso_local global i32 42
// STATIC-NEXT: @import_var = external dso_local global i32
// STATIC-NEXT: @weak_bar = extern_weak dso_local global i32
// STATIC-NEXT: @bar = external dso_local global i32

/// If -fno-direct-access-external-data is set, drop dso_local from global variable
/// declarations.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -mrelocation-model static -fno-direct-access-external-data  %s -o - | FileCheck --check-prefix=STATIC-INDIRECT %s
// STATIC-INDIRECT:      @baz = dso_local global i32 42
// STATIC-INDIRECT-NEXT: @import_var = external global i32
// STATIC-INDIRECT-NEXT: @weak_bar = extern_weak global i32
// STATIC-INDIRECT-NEXT: @bar = external global i32

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -pic-level 1 -pic-is-pie %s -o - | FileCheck --check-prefix=PIE %s
// PIE:      @baz = dso_local global i32 42
// PIE-NEXT: @import_var = external global i32
// PIE-NEXT: @weak_bar = extern_weak global i32
// PIE-NEXT: @bar = external global i32

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -pic-level 1 -pic-is-pie -fdirect-access-external-data %s -o - | FileCheck --check-prefix=PIE-DIRECT %s
// PIE-DIRECT:      @baz = dso_local global i32 42
// PIE-DIRECT-NEXT: @import_var = external dso_local global i32
// PIE-DIRECT-NEXT: @weak_bar = extern_weak global i32
// PIE-DIRECT-NEXT: @bar = external dso_local global i32

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -mrelocation-model static -fno-plt %s -o - | FileCheck --check-prefix=NOPLT %s
// NOPLT:      @baz = dso_local global i32 42
// NOPLT-NEXT: @import_var = external dso_local global i32
// NOPLT-NEXT: @weak_bar = extern_weak dso_local global i32
// NOPLT-NEXT: @bar = external dso_local global i32

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fno-plt -pic-level 1 -pic-is-pie -fdirect-access-external-data %s -o - | FileCheck --check-prefix=PIE-DIRECT-NOPLT %s
// PIE-DIRECT-NOPLT:      @baz = dso_local global i32 42
// PIE-DIRECT-NOPLT-NEXT: @import_var = external dso_local global i32
// PIE-DIRECT-NOPLT-NEXT: @weak_bar = extern_weak global i32
// PIE-DIRECT-NOPLT-NEXT: @bar = external dso_local global i32

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -pic-level 1 -pic-is-pie -fno-plt %s -o - | FileCheck --check-prefix=PIE-NO-PLT %s
// RUN: %clang_cc1 -triple powerpc64le -fclangir -emit-llvm -mrelocation-model static %s -o - | FileCheck --check-prefix=PIE-NO-PLT %s
// PIE-NO-PLT:      @baz = dso_local global i32 42
// PIE-NO-PLT-NEXT: @import_var = external global i32
// PIE-NO-PLT-NEXT: @weak_bar = extern_weak global i32
// PIE-NO-PLT-NEXT: @bar = external global i32

/// -fdirect-access-external-data is currently ignored for -fPIC.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -pic-level 2 %s -o - | FileCheck --check-prefix=SHARED %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -pic-level 2 -fdirect-access-external-data %s -o - | FileCheck --check-prefix=SHARED %s
// SHARED-DAG: @bar = external global i32
// SHARED-DAG: @weak_bar = extern_weak global i32
// SHARED-DAG: @baz ={{.*}} global i32 42

int baz = 42;
__attribute__((dllimport)) extern int import_var;
__attribute__((weak)) extern int weak_bar;
extern int bar;

int *use_import(void) {
  return &import_var;
}

int *zed(void) {
  if (baz)
    return &weak_bar;
  return &bar;
}
