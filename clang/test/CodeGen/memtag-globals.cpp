// RUN: %clang_cc1 -include %S/Inputs/sanitizer-extra-source.cpp \
// RUN:   -fsanitize-ignorelist=%S/Inputs/sanitizer-ignorelist-global.txt \
// RUN:   -fsanitize=memtag-globals -emit-llvm -o - %s | FileCheck %s

// RUN: %clang_cc1 -include %S/Inputs/sanitizer-extra-source.cpp \
// RUN:   -fsanitize-ignorelist=%S/Inputs/sanitizer-ignorelist-src.txt \
// RUN:   -fsanitize=memtag-globals -emit-llvm -o - %s | \
// RUN:   FileCheck %s --check-prefix=IGNORELIST

int global;
int __attribute__((no_sanitize("memtag"))) attributed_global;
int __attribute__((disable_sanitizer_instrumentation)) disable_instrumentation_global;
int ignorelisted_global;

void func() {
  static int static_var = 0;
  const char *literal = "Hello, world!";
}

// CHECK: @{{.*}}extra_global{{.*}} =
// CHECK-NOT: no_sanitize_memtag
// CHECK: @{{.*}}global{{.*}} =
// CHECK-NOT: no_sanitize_memtag
// CHECK: @{{.*}}attributed_global{{.*}} ={{.*}} global {{.*}}, no_sanitize_memtag
// CHECK: @{{.*}}disable_instrumentation_global{{.*}} ={{.*}} global {{.*}}, no_sanitize_memtag
// CHECK: @{{.*}}ignorelisted_global{{.*}} ={{.*}} global {{.*}}, no_sanitize_memtag
// CHECK: @{{.*}}static_var{{.*}} =
// CHECK-NOT: no_sanitize_memtag
// CHECK: @{{.*}} = {{.*}} c"Hello, world!\00"
// CHECK-NOT: no_sanitize_memtag

// CHECK: !llvm.asan.globals = !{![[EXTRA_GLOBAL:[0-9]+]], ![[GLOBAL:[0-9]+]], ![[ATTR_GLOBAL:[0-9]+]], ![[DISABLE_INSTR_GLOBAL:[0-9]+]], ![[IGNORELISTED_GLOBAL:[0-9]+]], ![[STATIC_VAR:[0-9]+]], ![[LITERAL:[0-9]+]]}
// CHECK: ![[EXTRA_GLOBAL]] = !{{{.*}} ![[EXTRA_GLOBAL_LOC:[0-9]+]], !"extra_global", i1 false, i1 false}
// CHECK: ![[EXTRA_GLOBAL_LOC]] = !{!"{{.*}}extra-source.cpp", i32 1, i32 5}
// CHECK: ![[GLOBAL]] = !{{{.*}} ![[GLOBAL_LOC:[0-9]+]], !"global", i1 false, i1 false}
// CHECK: ![[GLOBAL_LOC]] = !{!"{{.*}}memtag-globals.cpp", i32 10, i32 5}
// CHECK: ![[ATTR_GLOBAL]] = !{{{.*attributed_global.*}}, null, null, i1 false, i1 true}
// CHECK: ![[DISABLE_INSTR_GLOBAL]] = !{{{.*disable_instrumentation_global.*}}, null, null, i1 false, i1 true}
// CHECK: ![[IGNORELISTED_GLOBAL]] = !{{{.*ignorelisted_global.*}}, null, null, i1 false, i1 true}
// CHECK: ![[STATIC_VAR]] = !{{{.*}} ![[STATIC_LOC:[0-9]+]], !"static_var", i1 false, i1 false}
// CHECK: ![[STATIC_LOC]] = !{!"{{.*}}memtag-globals.cpp", i32 16, i32 14}
// CHECK: ![[LITERAL]] = !{{{.*}} ![[LITERAL_LOC:[0-9]+]], !"<string literal>", i1 false, i1 false}
// CHECK: ![[LITERAL_LOC]] = !{!"{{.*}}memtag-globals.cpp", i32 17, i32 25}

// IGNORELIST: @{{.*}}extra_global{{.*}} ={{.*}} global
// IGNORELIST-NOT: no_sanitize_memtag
// IGNORELIST: @{{.*}}global{{.*}} ={{.*}} global {{.*}}, no_sanitize_memtag
// IGNORELIST: @{{.*}}attributed_global{{.*}} ={{.*}} global {{.*}}, no_sanitize_memtag
// IGNORELIST: @{{.*}}disable_instrumentation_global{{.*}} ={{.*}} global {{.*}}, no_sanitize_memtag
// IGNORELIST: @{{.*}}ignorelisted_globa{{.*}} ={{.*}} global {{.*}}, no_sanitize_memtag
// IGNORELIST: @{{.*}}static_var{{.*}} ={{.*}} global {{.*}}, no_sanitize_memtag
// IGNORELIST: @{{.*}} = {{.*}} c"Hello, world!\00"{{.*}}, no_sanitize_memtag

// IGNORELIST: !llvm.asan.globals = !{![[EXTRA_GLOBAL:[0-9]+]], ![[GLOBAL:[0-9]+]], ![[ATTR_GLOBAL:[0-9]+]], ![[DISABLE_INSTR_GLOBAL:[0-9]+]], ![[IGNORELISTED_GLOBAL:[0-9]+]], ![[STATIC_VAR:[0-9]+]], ![[LITERAL:[0-9]+]]}
// IGNORELIST: ![[EXTRA_GLOBAL]] = !{{{.*}} ![[EXTRA_GLOBAL_LOC:[0-9]+]], !"extra_global", i1 false, i1 false}
// IGNORELIST: ![[EXTRA_GLOBAL_LOC]] = !{!"{{.*}}extra-source.cpp", i32 1, i32 5}
// IGNORELIST: ![[GLOBAL]] = !{{{.*}} null, null, i1 false, i1 true}
// IGNORELIST: ![[ATTR_GLOBAL]] = !{{{.*attributed_global.*}}, null, null, i1 false, i1 true}
// IGNORELIST: ![[DISABLE_INSTR_GLOBAL]] = !{{{.*disable_instrumentation_global.*}}, null, null, i1 false, i1 true}
// IGNORELIST: ![[IGNORELISTED_GLOBAL]] = !{{{.*ignorelisted_global.*}}, null, null, i1 false, i1 true}
// IGNORELIST: ![[STATIC_VAR]] = !{{{.*}} null, null, i1 false, i1 true}
// IGNORELIST: ![[LITERAL]] = !{{{.*}} null, null, i1 false, i1 true}
