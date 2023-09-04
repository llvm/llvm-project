// RUN: %clang_cc1 -triple x86_64-pc-win32 -emit-llvm %s -o - | FileCheck --check-prefix=COFF %s
// COFF:      @baz = dso_local global i32 42
// COFF-NEXT: @import_var = external dllimport global i32
// COFF-NEXT: @weak_bar = extern_weak global i32
// COFF-NEXT: @bar = external dso_local global i32
// COFF-NEXT: @local_thread_var = dso_local thread_local global i32 42
// COFF-NEXT: @thread_var = external dso_local thread_local global i32
// COFF-DAG: declare dso_local void @foo()
// COFF-DAG: define dso_local ptr @zed()
// COFF-DAG: declare dllimport void @import_func()

// RUN: %clang_cc1 -triple x86_64-w64-mingw32 -emit-llvm %s -o - | FileCheck --check-prefixes=MINGW,MINGW-NATIVE_TLS,MINGW-AUTO-IMPORT %s
// RUN: %clang_cc1 -triple x86_64-w64-mingw32 -emit-llvm %s -o - -fno-auto-import | FileCheck --check-prefixes=MINGW,MINGW-NATIVE_TLS,MINGW-NO-AUTO-IMPORT %s
// RUN: %clang_cc1 -triple x86_64-w64-mingw32 -emit-llvm %s -o - -femulated-tls | FileCheck --check-prefixes=MINGW,MINGW-EMUTLS,MINGW-AUTO-IMPORT %s
// MINGW:      @baz = dso_local global i32 42
// MINGW-NEXT: @import_var = external dllimport global i32
// MINGW-NEXT: @weak_bar = extern_weak global i32
// MINGW-AUTO-IMPORT-NEXT: @bar = external global i32
// MINGW-NO-AUTO-IMPORT-NEXT: @bar = external dso_local global i32
// MINGW-NEXT: @local_thread_var = dso_local thread_local global i32 42
// MINGW-NATIVE_TLS-NEXT: @thread_var = external dso_local thread_local global i32
// MINGW-EMUTLS-NEXT: @thread_var = external thread_local global i32
// MINGW-DAG: declare dso_local void @foo()
// MINGW-DAG: define dso_local ptr @zed()
// MINGW-DAG: declare dllimport void @import_func()

/// Static relocation model defaults to -fdirect-access-external-data and sets
/// dso_local on most global objects.
// RUN: %clang_cc1 -triple x86_64 -emit-llvm -mrelocation-model static %s -o - | FileCheck --check-prefix=STATIC %s
// RUN: %clang_cc1 -triple x86_64 -emit-llvm -mrelocation-model static -fdirect-access-external-data %s -o - | FileCheck --check-prefix=STATIC %s
// STATIC:      @baz = dso_local global i32 42
// STATIC-NEXT: @import_var = external dso_local global i32
// STATIC-NEXT: @weak_bar = extern_weak dso_local global i32
// STATIC-NEXT: @bar = external dso_local global i32
// STATIC-NEXT: @local_thread_var = dso_local thread_local global i32 42
// STATIC-NEXT: @thread_var = external thread_local global i32
// STATIC-DAG: declare dso_local void @foo()
// STATIC-DAG: define dso_local ptr @zed()
// STATIC-DAG: declare dso_local void @import_func()

// STATIC-NOT: !"direct-access-external-data"

/// If -fno-direct-access-external-data is set, drop dso_local from global variable
/// declarations.
// RUN: %clang_cc1 -triple x86_64 -emit-llvm %s -mrelocation-model static -fno-direct-access-external-data -o - | FileCheck --check-prefix=STATIC-INDIRECT %s
// STATIC-INDIRECT:      @baz = dso_local global i32 42
// STATIC-INDIRECT-NEXT: @import_var = external global i32
// STATIC-INDIRECT-NEXT: @weak_bar = extern_weak global i32
// STATIC-INDIRECT-NEXT: @bar = external global i32
// STATIC-INDIRECT-NEXT: @local_thread_var = dso_local thread_local global i32 42
// STATIC-INDIRECT-NEXT: @thread_var = external thread_local global i32
// STATIC-INDIRECT-DAG:  declare void @import_func()
// STATIC-INDIRECT-DAG:  define dso_local ptr @zed()
// STATIC-INDIRECT-DAG:  declare void @foo()

// STATIC-INDIRECT:      ![[#]] = !{i32 7, !"direct-access-external-data", i32 0}

// RUN: %clang_cc1 -triple x86_64 -emit-llvm -pic-level 1 -pic-is-pie %s -o - | FileCheck --check-prefix=PIE %s
// PIE:      @baz = dso_local global i32 42
// PIE-NEXT: @import_var = external global i32
// PIE-NEXT: @weak_bar = extern_weak global i32
// PIE-NEXT: @bar = external global i32
// PIE-NEXT: @local_thread_var = dso_local thread_local global i32 42
// PIE-NEXT: @thread_var = external thread_local global i32
// PIE-DAG: declare void @foo()
// PIE-DAG: define dso_local ptr @zed()
// PIE-DAG: declare void @import_func()

// PIE-NOT:  !"direct-access-external-data"

// RUN: %clang_cc1 -triple x86_64 -emit-llvm -pic-level 1 -pic-is-pie -fdirect-access-external-data %s -o - | FileCheck --check-prefix=PIE-DIRECT %s
// PIE-DIRECT:      @baz = dso_local global i32 42
// PIE-DIRECT-NEXT: @import_var = external dso_local global i32
// PIE-DIRECT-NEXT: @weak_bar = extern_weak global i32
// PIE-DIRECT-NEXT: @bar = external dso_local global i32
// PIE-DIRECT-NEXT: @local_thread_var = dso_local thread_local global i32 42
// PIE-DIRECT-NEXT: @thread_var = external thread_local global i32
// PIE-DIRECT-DAG: declare void @foo()
// PIE-DIRECT-DAG: define dso_local ptr @zed()
// PIE-DIRECT-DAG: declare void @import_func()

// PIE-DIRECT:      ![[#]] = !{i32 7, !"direct-access-external-data", i32 1}

// RUN: %clang_cc1 -triple x86_64 -emit-llvm -mrelocation-model static -fno-plt %s -o - | FileCheck --check-prefix=NOPLT %s
// NOPLT:      @baz = dso_local global i32 42
// NOPLT-NEXT: @import_var = external dso_local global i32
// NOPLT-NEXT: @weak_bar = extern_weak dso_local global i32
// NOPLT-NEXT: @bar = external dso_local global i32
// NOPLT-NEXT: @local_thread_var = dso_local thread_local global i32 42
// NOPLT-NEXT: @thread_var = external thread_local global i32
// NOPLT-DAG: declare void @foo()
// NOPLT-DAG: define dso_local ptr @zed()
// NOPLT-DAG: declare void @import_func()

// RUN: %clang_cc1 -triple x86_64 -emit-llvm -fno-plt -pic-level 1 -pic-is-pie -fdirect-access-external-data %s -o - | FileCheck --check-prefix=PIE-DIRECT-NOPLT %s
// PIE-DIRECT-NOPLT:      @baz = dso_local global i32 42
// PIE-DIRECT-NOPLT-NEXT: @import_var = external dso_local global i32
// PIE-DIRECT-NOPLT-NEXT: @weak_bar = extern_weak global i32
// PIE-DIRECT-NOPLT-NEXT: @bar = external dso_local global i32
// PIE-DIRECT-NOPLT-NEXT: @local_thread_var = dso_local thread_local global i32 42
// PIE-DIRECT-NOPLT-NEXT: @thread_var = external thread_local global i32
// PIE-DIRECT-NOPLT-DAG: declare void @foo()
// PIE-DIRECT-NOPLT-DAG: define dso_local ptr @zed()
// PIE-DIRECT-NOPLT-DAG: declare void @import_func()

// RUN: %clang_cc1 -triple x86_64 -emit-llvm -pic-level 1 -pic-is-pie -fno-plt %s -o - | FileCheck --check-prefix=PIE-NO-PLT %s
// RUN: %clang_cc1 -triple powerpc64le -emit-llvm -mrelocation-model static %s -o - | FileCheck --check-prefix=PIE-NO-PLT %s
// PIE-NO-PLT:      @baz = dso_local global i32 42
// PIE-NO-PLT-NEXT: @import_var = external global i32
// PIE-NO-PLT-NEXT: @weak_bar = extern_weak global i32
// PIE-NO-PLT-NEXT: @bar = external global i32
// PIE-NO-PLT-NEXT: @local_thread_var = dso_local thread_local global i32 42
// PIE-NO-PLT-NEXT: @thread_var = external thread_local global i32
// PIE-NO-PLT-DAG:  declare void @import_func()
// PIE-NO-PLT-DAG:  define dso_local ptr @zed()
// PIE-NO-PLT-DAG:  declare void @foo()

/// -fdirect-access-external-data is currently ignored for -fPIC.
// RUN: %clang_cc1 -triple x86_64 -emit-llvm -pic-level 2 %s -o - | FileCheck --check-prefix=SHARED %s
// RUN: %clang_cc1 -triple x86_64 -emit-llvm -pic-level 2 -fdirect-access-external-data %s -o - | FileCheck --check-prefix=SHARED %s
// SHARED-DAG: @bar = external global i32
// SHARED-DAG: @weak_bar = extern_weak global i32
// SHARED-DAG: declare void @foo()
// SHARED-DAG: @baz ={{.*}} global i32 42
// SHARED-DAG: define{{.*}} ptr @zed()
// SHARED-DAG: @thread_var = external thread_local global i32
// SHARED-DAG: @local_thread_var ={{.*}} thread_local global i32 42

int baz = 42;
__attribute__((dllimport)) extern int import_var;
__attribute__((weak)) extern int weak_bar;
extern int bar;
__attribute__((dllimport)) void import_func(void);

int *use_import(void) {
  import_func();
  return &import_var;
}

void foo(void);

int *zed(void) {
  foo();
  return baz ? &weak_bar : &bar;
}

__thread int local_thread_var = 42;
extern __thread int thread_var;
int *get_thread_var(int a) {
  return a ? &thread_var : &local_thread_var;
}
