// RUN: echo "int extra_global;" > %t.extra-source.cpp
// RUN: echo "global:*ignorelisted_global*" > %t.ignorelist
// RUN: %clang_cc1 -include %t.extra-source.cpp -fsanitize=address -fsanitize-ignorelist=%t.ignorelist -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,ASAN
// RUN: %clang_cc1 -include %t.extra-source.cpp -fsanitize=kernel-address -fsanitize-ignorelist=%t.ignorelist -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,KASAN
// The ignorelist file uses regexps, so Windows path backslashes.
// RUN: echo "src:%s" | sed -e 's/\\/\\\\/g' > %t.ignorelist-src
// RUN: %clang_cc1 -include %t.extra-source.cpp -fsanitize=address -fsanitize-ignorelist=%t.ignorelist-src -emit-llvm -o - %s | FileCheck %s --check-prefix=IGNORELIST-SRC
// RUN: %clang_cc1 -include %t.extra-source.cpp -fsanitize=kernel-address -fsanitize-ignorelist=%t.ignorelist-src -emit-llvm -o - %s | FileCheck %s --check-prefix=IGNORELIST-SRC

int global;
int dyn_init_global = global;
int __attribute__((no_sanitize("address"))) attributed_global;
int ignorelisted_global;

int __attribute__((section("__DATA, __common"))) sectioned_global; // KASAN - ignore globals in a section
extern "C" {
int __special_global; // KASAN - ignore globals with __-prefix
}

void func() {
  static int static_var = 0;
  const char *literal = "Hello, world!";
}

// ASAN: sectioned_global{{.*}} global { i32, [28 x i8] }{{.*}}, align 32
// KASAN: sectioned_global{{.*}} global i32
// ASAN: @__special_global{{.*}} global { i32, [28 x i8] }{{.*}}, align 32
// KASAN: @__special_global{{.*}} global i32

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

// CHECK: attributes #[[#ATTR]] = { nounwind }

/// If -fasynchronous-unwind-tables, set the module flag "uwtable". ctor/dtor
/// will thus get the uwtable attribute.
// RUN: %clang_cc1 -emit-llvm -fsanitize=address -funwind-tables=2 -o - %s | FileCheck %s --check-prefixes=UWTABLE
// UWTABLE: define internal void @asan.module_dtor() #[[#ATTR:]] {
// UWTABLE: attributes #[[#ATTR]] = { nounwind uwtable }
// UWTABLE: ![[#]] = !{i32 7, !"uwtable", i32 2}

// CHECK: !llvm.asan.globals = !{![[EXTRA_GLOBAL:[0-9]+]], ![[GLOBAL:[0-9]+]], ![[DYN_INIT_GLOBAL:[0-9]+]], ![[ATTR_GLOBAL:[0-9]+]], ![[IGNORELISTED_GLOBAL:[0-9]+]], ![[SECTIONED_GLOBAL:[0-9]+]], ![[SPECIAL_GLOBAL:[0-9]+]], ![[STATIC_VAR:[0-9]+]], ![[LITERAL:[0-9]+]]}
// CHECK: ![[EXTRA_GLOBAL]] = !{{{.*}} ![[EXTRA_GLOBAL_LOC:[0-9]+]], !"extra_global", i1 false, i1 false}
// CHECK: ![[EXTRA_GLOBAL_LOC]] = !{!"{{.*}}extra-source.cpp", i32 1, i32 5}
// CHECK: ![[GLOBAL]] = !{{{.*}} ![[GLOBAL_LOC:[0-9]+]], !"global", i1 false, i1 false}
// CHECK: ![[GLOBAL_LOC]] = !{!"{{.*}}asan-globals.cpp", i32 10, i32 5}
// CHECK: ![[DYN_INIT_GLOBAL]] = !{{{.*}} ![[DYN_INIT_LOC:[0-9]+]], !"dyn_init_global", i1 true, i1 false}
// CHECK: ![[DYN_INIT_LOC]] = !{!"{{.*}}asan-globals.cpp", i32 11, i32 5}
// CHECK: ![[ATTR_GLOBAL]] = !{{{.*}}, null, null, i1 false, i1 true}
// CHECK: ![[IGNORELISTED_GLOBAL]] = !{{{.*}}, null, null, i1 false, i1 true}
// CHECK: ![[SECTIONED_GLOBAL]] = !{{{.*}} ![[SECTIONED_GLOBAL_LOC:[0-9]+]], !"sectioned_global", i1 false, i1 false}
// CHECK: ![[SECTIONED_GLOBAL_LOC]] = !{!"{{.*}}asan-globals.cpp", i32 15, i32 50}
// CHECK: ![[SPECIAL_GLOBAL]] = !{{{.*}} ![[SPECIAL_GLOBAL_LOC:[0-9]+]], !"__special_global", i1 false, i1 false}
// CHECK: ![[SPECIAL_GLOBAL_LOC]] = !{!"{{.*}}asan-globals.cpp", i32 17, i32 5}
// CHECK: ![[STATIC_VAR]] = !{{{.*}} ![[STATIC_LOC:[0-9]+]], !"static_var", i1 false, i1 false}
// CHECK: ![[STATIC_LOC]] = !{!"{{.*}}asan-globals.cpp", i32 21, i32 14}
// CHECK: ![[LITERAL]] = !{{{.*}} ![[LITERAL_LOC:[0-9]+]], !"<string literal>", i1 false, i1 false}
// CHECK: ![[LITERAL_LOC]] = !{!"{{.*}}asan-globals.cpp", i32 22, i32 25}

// IGNORELIST-SRC: !llvm.asan.globals = !{![[EXTRA_GLOBAL:[0-9]+]], ![[GLOBAL:[0-9]+]], ![[DYN_INIT_GLOBAL:[0-9]+]], ![[ATTR_GLOBAL:[0-9]+]], ![[IGNORELISTED_GLOBAL:[0-9]+]], ![[SECTIONED_GLOBAL:[0-9]+]], ![[SPECIAL_GLOBAL:[0-9]+]], ![[STATIC_VAR:[0-9]+]], ![[LITERAL:[0-9]+]]}
// IGNORELIST-SRC: ![[EXTRA_GLOBAL]] = !{{{.*}} ![[EXTRA_GLOBAL_LOC:[0-9]+]], !"extra_global", i1 false, i1 false}
// IGNORELIST-SRC: ![[EXTRA_GLOBAL_LOC]] = !{!"{{.*}}extra-source.cpp", i32 1, i32 5}
// IGNORELIST-SRC: ![[GLOBAL]] = !{{{.*}} null, null, i1 false, i1 true}
// IGNORELIST-SRC: ![[DYN_INIT_GLOBAL]] = !{{{.*}} null, null, i1 true, i1 true}
// IGNORELIST-SRC: ![[ATTR_GLOBAL]] = !{{{.*}}, null, null, i1 false, i1 true}
// IGNORELIST-SRC: ![[IGNORELISTED_GLOBAL]] = !{{{.*}}, null, null, i1 false, i1 true}
// IGNORELIST-SRC: ![[SECTIONED_GLOBAL]] = !{{{.*}} null, null, i1 false, i1 true}
// IGNORELIST-SRC: ![[SPECIAL_GLOBAL]] = !{{{.*}} null, null, i1 false, i1 true}
// IGNORELIST-SRC: ![[STATIC_VAR]] = !{{{.*}} null, null, i1 false, i1 true}
// IGNORELIST-SRC: ![[LITERAL]] = !{{{.*}} null, null, i1 false, i1 true}
