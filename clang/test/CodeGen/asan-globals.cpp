// RUN: echo "int extra_global;" > %t.extra-source.cpp
// RUN: echo "global:*ignorelisted_global*" > %t.ignorelist
// RUN: %clang_cc1 -include %t.extra-source.cpp -fsanitize=address -fsanitize-ignorelist=%t.ignorelist -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,GLOBS,ASAN
// RUN: %clang_cc1 -include %t.extra-source.cpp -fsanitize=kernel-address -fsanitize-ignorelist=%t.ignorelist -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,GLOBS,KASAN
// The ignorelist file uses regexps, so Windows path backslashes.
// RUN: echo "src:%s" | sed -e 's/\\/\\\\/g' > %t.ignorelist-src
// RUN: %clang_cc1 -include %t.extra-source.cpp -fsanitize=address -fsanitize-ignorelist=%t.ignorelist-src -emit-llvm -o - %s | FileCheck %s --check-prefix=IGNORELIST-SRC
// RUN: %clang_cc1 -include %t.extra-source.cpp -fsanitize=kernel-address -fsanitize-ignorelist=%t.ignorelist-src -emit-llvm -o - %s | FileCheck %s --check-prefix=IGNORELIST-SRC

int global;
int dyn_init_global = global;
int __attribute__((no_sanitize("address"))) attributed_global;
int __attribute__((disable_sanitizer_instrumentation)) disable_instrumentation_global;
int ignorelisted_global;
extern int __attribute__((no_sanitize("address"))) external_global;

int __attribute__((section("__DATA, __common"))) sectioned_global; // KASAN - ignore globals in a section
extern "C" {
int __special_global; // KASAN - ignore globals with __-prefix
}

void func() {
  static int static_var = 0;
  const char *literal = "Hello, world!";
  external_global = 1;
}

// GLOBS:     @{{.*}}extra_global{{.*}} ={{.*}} global
// GLOBS-NOT: no_sanitize_address
// GLOBS:     @{{.*}}global{{.*}} ={{.*}} global
// GLOBS-NOT: no_sanitize_address
// GLOBS:     @{{.*}}dyn_init_global{{.*}} ={{.*}} global {{.*}}, sanitize_address_dyninit
// GLOBS-NOT: no_sanitize_address

// GLOBS:     @{{.*}}attributed_global{{.*}} ={{.*}} global {{.*}} no_sanitize_address
// GLOBS:     @{{.*}}disable_instrumentation_global{{.*}} ={{.*}} global {{.*}} no_sanitize_address
// GLOBS:     @{{.*}}ignorelisted_global{{.*}} ={{.*}} global {{.*}} no_sanitize_address

// ASAN:     @{{.*}}sectioned_global{{.*}} ={{.*}} global { i32, [28 x i8] }{{.*}}, align 32
// ASAN-NOT: no_sanitize_address
// ASAN:     @{{.*}}__special_global{{.*}} ={{.*}} global { i32, [28 x i8] }{{.*}}, align 32
// ASAN-NOT: no_sanitize_address

/// Note: No attribute is added by the IR pass, but the type didn't change, so
/// that checks our assertions that the globals didn't get instrumented.
// KASAN:    @{{.*}}sectioned_global{{.*}} ={{.*}} global i32 {{.*}}
// KASAN:    @{{.*}}__special_global{{.*}} ={{.*}} global i32 {{.*}}

// GLOBS:     @{{[^ ]*}}static_var{{[^ ]*}} ={{.*}} global {{.*}}
// GLOBS-NOT: no_sanitize_address
// GLOBS:     @{{.*}} = {{.*}}c"Hello, world!\00"
// GLOBS-NOT: no_sanitize_address

// GLOBS: @{{.*}}external_global{{.*}} ={{.*}} no_sanitize_address

/// Without -fasynchronous-unwind-tables, ctor and dtor get the uwtable attribute.
// CHECK-LABEL: define internal void @asan.module_ctor() #[[#ATTR:]] {
// ASAN-NEXT: call void @__asan_init
// ASAN-NEXT: call void @__asan_version_mismatch_check
// KASAN-NOT: call void @__asan_init
// KASAN-NOT: call void @__asan_version_mismatch_check
// ASAN-NEXT: call void @__asan_register_globals({{.*}}, i{{32|64}} 7)
// KASAN-NEXT: call void @__asan_register_globals({{.*}}, i{{32|64}} 5)
// CHECK-NEXT: ret void

// CHECK:      define internal void @asan.module_dtor() #[[#ATTR]] {
// CHECK-NEXT: call void @__asan_unregister_globals
// CHECK-NEXT: ret void

// CHECK: attributes #[[#ATTR]] = { nounwind

/// If -fasynchronous-unwind-tables, set the module flag "uwtable". ctor/dtor
/// will thus get the uwtable attribute.
// RUN: %clang_cc1 -emit-llvm -fsanitize=address -funwind-tables=2 -o - %s | FileCheck %s --check-prefixes=UWTABLE
// UWTABLE: define internal void @asan.module_dtor() #[[#ATTR:]] {
// UWTABLE: attributes #[[#ATTR]] = { nounwind uwtable
// UWTABLE: ![[#]] = !{i32 7, !"uwtable", i32 2}

// IGNORELIST-SRC:     @{{.*}}extra_global{{.*}} ={{.*}} global
// IGNORELIST-SRC-NOT: no_sanitize_address
// IGNORELIST-SRC:     @{{.*}}global{{.*}} ={{.*}} global {{.*}} no_sanitize_address
// IGNORELIST-SRC:     @{{.*}}dyn_init_global{{.*}} ={{.*}} global {{.*}} no_sanitize_address
// IGNORELIST-SRC:     @{{.*}}attributed_global{{.*}} ={{.*}} global {{.*}} no_sanitize_address
// IGNORELIST-SRC:     @{{.*}}disable_instrumentation_global{{.*}} ={{.*}} global {{.*}} no_sanitize_address
// IGNORELIST-SRC:     @{{.*}}ignorelisted_global{{.*}} ={{.*}} global {{.*}} no_sanitize_address
// IGNORELIST-SRC:     @{{.*}}sectioned_global{{.*}} ={{.*}} global {{.*}} no_sanitize_address
// IGNORELIST-SRC:     @{{.*}}__special_global{{.*}} ={{.*}} global {{.*}} no_sanitize_address
// IGNORELIST-SRC:     @{{.*}}static_var{{.*}} ={{.*}} global {{.*}} no_sanitize_address
// IGNORELIST-SRC:     @{{.*}} ={{.*}} c"Hello, world!\00"{{.*}} no_sanitize_address
// IGNORELIST-SRC:     @{{.*}}external_global{{.*}} ={{.*}} no_sanitize_address
