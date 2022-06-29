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

// IGNORELIST: @{{.*}}extra_global{{.*}} ={{.*}} global
// IGNORELIST-NOT: no_sanitize_memtag
// IGNORELIST: @{{.*}}global{{.*}} ={{.*}} global {{.*}}, no_sanitize_memtag
// IGNORELIST: @{{.*}}attributed_global{{.*}} ={{.*}} global {{.*}}, no_sanitize_memtag
// IGNORELIST: @{{.*}}disable_instrumentation_global{{.*}} ={{.*}} global {{.*}}, no_sanitize_memtag
// IGNORELIST: @{{.*}}ignorelisted_globa{{.*}} ={{.*}} global {{.*}}, no_sanitize_memtag
// IGNORELIST: @{{.*}}static_var{{.*}} ={{.*}} global {{.*}}, no_sanitize_memtag
// IGNORELIST: @{{.*}} = {{.*}} c"Hello, world!\00"{{.*}}, no_sanitize_memtag
