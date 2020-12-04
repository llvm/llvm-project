// RUN: %clang_cc1 -triple x86_64-pc-win32 -emit-llvm %s -o - | FileCheck --check-prefix=COFF %s
// COFF:      @baz = dso_local global i32 42
// COFF-NEXT: @import_var = external dllimport global i32
// COFF-NEXT: @weak_bar = extern_weak global i32
// COFF-NEXT: @bar = external dso_local global i32
// COFF-NEXT: @local_thread_var = dso_local thread_local global i32 42
// COFF-NEXT: @thread_var = external dso_local thread_local global i32
// COFF-DAG: declare dso_local void @foo()
// COFF-DAG: define dso_local i32* @zed()
// COFF-DAG: declare dllimport void @import_func()

// RUN: %clang_cc1 -triple x86_64-w64-mingw32 -emit-llvm %s -o - | FileCheck --check-prefix=MINGW %s
// MINGW:      @baz = dso_local global i32 42
// MINGW-NEXT: @import_var = external dllimport global i32
// MINGW-NEXT: @weak_bar = extern_weak global i32
// MINGW-NEXT: @bar = external global i32
// MINGW-NEXT: @local_thread_var = dso_local thread_local global i32 42
// MINGW-NEXT: @thread_var = external dso_local thread_local global i32
// MINGW-DAG: declare dso_local void @foo()
// MINGW-DAG: define dso_local i32* @zed()
// MINGW-DAG: declare dllimport void @import_func()

// RUN: %clang_cc1 -triple x86_64 -emit-llvm -mrelocation-model static %s -o - | FileCheck --check-prefix=STATIC %s
// STATIC:      @baz = dso_local global i32 42
// STATIC-NEXT: @import_var = external dso_local global i32
// STATIC-NEXT: @weak_bar = extern_weak dso_local global i32
// STATIC-NEXT: @bar = external dso_local global i32
// STATIC-NEXT: @local_thread_var = dso_local thread_local global i32 42
// STATIC-NEXT: @thread_var = external thread_local global i32
// STATIC-DAG: declare dso_local void @foo()
// STATIC-DAG: define dso_local i32* @zed()
// STATIC-DAG: declare dso_local void @import_func()

// RUN: %clang_cc1 -triple x86_64 -emit-llvm -pic-level 1 -pic-is-pie %s -o - | FileCheck --check-prefix=PIE %s
// PIE:      @baz = dso_local global i32 42
// PIE-NEXT: @import_var = external global i32
// PIE-NEXT: @weak_bar = extern_weak global i32
// PIE-NEXT: @bar = external global i32
// PIE-NEXT: @local_thread_var = dso_local thread_local global i32 42
// PIE-NEXT: @thread_var = external thread_local global i32
// PIE-DAG: declare void @foo()
// PIE-DAG: define dso_local i32* @zed()
// PIE-DAG: declare void @import_func()

// RUN: %clang_cc1 -triple x86_64 -emit-llvm -pic-level 1 -pic-is-pie -mpie-copy-relocations %s -o - | FileCheck --check-prefix=PIE-DIRECT %s
// PIE-DIRECT:      @baz = dso_local global i32 42
// PIE-DIRECT-NEXT: @import_var = external dso_local global i32
// PIE-DIRECT-NEXT: @weak_bar = extern_weak global i32
// PIE-DIRECT-NEXT: @bar = external dso_local global i32
// PIE-DIRECT-NEXT: @local_thread_var = dso_local thread_local global i32 42
// PIE-DIRECT-NEXT: @thread_var = external thread_local global i32
// PIE-DIRECT-DAG: declare void @foo()
// PIE-DIRECT-DAG: define dso_local i32* @zed()
// PIE-DIRECT-DAG: declare void @import_func()

// RUN: %clang_cc1 -triple x86_64 -emit-llvm -mrelocation-model static -fno-plt %s -o - | FileCheck --check-prefix=NOPLT %s
// NOPLT:      @baz = dso_local global i32 42
// NOPLT-NEXT: @import_var = external dso_local global i32
// NOPLT-NEXT: @weak_bar = extern_weak dso_local global i32
// NOPLT-NEXT: @bar = external dso_local global i32
// NOPLT-NEXT: @local_thread_var = dso_local thread_local global i32 42
// NOPLT-NEXT: @thread_var = external thread_local global i32
// NOPLT-DAG: declare void @foo()
// NOPLT-DAG: define dso_local i32* @zed()
// NOPLT-DAG: declare void @import_func()

// RUN: %clang_cc1 -triple x86_64 -emit-llvm -fno-plt -pic-level 1 -pic-is-pie -mpie-copy-relocations %s -o - | FileCheck --check-prefix=PIE-DIRECT-NOPLT %s
// PIE-DIRECT-NOPLT:      @baz = dso_local global i32 42
// PIE-DIRECT-NOPLT-NEXT: @import_var = external dso_local global i32
// PIE-DIRECT-NOPLT-NEXT: @weak_bar = extern_weak global i32
// PIE-DIRECT-NOPLT-NEXT: @bar = external dso_local global i32
// PIE-DIRECT-NOPLT-NEXT: @local_thread_var = dso_local thread_local global i32 42
// PIE-DIRECT-NOPLT-NEXT: @thread_var = external thread_local global i32
// PIE-DIRECT-NOPLT-DAG: declare void @foo()
// PIE-DIRECT-NOPLT-DAG: define dso_local i32* @zed()
// PIE-DIRECT-NOPLT-DAG: declare void @import_func()

// RUN: %clang_cc1 -triple x86_64 -emit-llvm -pic-is-pie -fno-plt %s -o - | FileCheck --check-prefix=PIE-NO-PLT %s
// RUN: %clang_cc1 -triple powerpc64le -emit-llvm -mrelocation-model static %s -o - | FileCheck --check-prefix=PIE-NO-PLT %s
// PIE-NO-PLT:      @baz = dso_local global i32 42
// PIE-NO-PLT-NEXT: @import_var = external global i32
// PIE-NO-PLT-NEXT: @weak_bar = extern_weak global i32
// PIE-NO-PLT-NEXT: @bar = external global i32
// PIE-NO-PLT-NEXT: @local_thread_var = dso_local thread_local global i32 42
// PIE-NO-PLT-NEXT: @thread_var = external thread_local global i32
// PIE-NO-PLT-DAG:  declare void @import_func()
// PIE-NO-PLT-DAG:  define dso_local i32* @zed()
// PIE-NO-PLT-DAG:  declare void @foo()

// RUN: %clang_cc1 -triple x86_64 -emit-llvm -pic-level 2 %s -o - | FileCheck --check-prefix=SHARED %s
// SHARED-DAG: @bar = external global i32
// SHARED-DAG: @weak_bar = extern_weak global i32
// SHARED-DAG: declare void @foo()
// SHARED-DAG: @baz = global i32 42
// SHARED-DAG: define i32* @zed()
// SHARED-DAG: @thread_var = external thread_local global i32
// SHARED-DAG: @local_thread_var = thread_local global i32 42

int baz = 42;
__attribute__((dllimport)) extern int import_var;
__attribute__((weak)) extern int weak_bar;
extern int bar;
__attribute__((dllimport)) void import_func(void);

int *use_import() {
  import_func();
  return &import_var;
}

void foo(void);

int *zed() {
  foo();
  return baz ? &weak_bar : &bar;
}

__thread int local_thread_var = 42;
extern __thread int thread_var;
int *get_thread_var(int a) {
  return a ? &thread_var : &local_thread_var;
}
