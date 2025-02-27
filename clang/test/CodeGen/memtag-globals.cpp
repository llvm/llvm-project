// RUN: %clang_cc1 -triple aarch64-linux-android34 \
// RUN:   -include %S/Inputs/sanitizer-extra-source.cpp \
// RUN:   -fsanitize-ignorelist=%S/Inputs/sanitizer-ignorelist-global.txt \
// RUN:   -fsanitize=memtag-globals -emit-llvm -o - %s | FileCheck %s

// RUN: %clang_cc1 -triple aarch64-linux-android34 \
// RUN:    -include %S/Inputs/sanitizer-extra-source.cpp \
// RUN:   -fsanitize-ignorelist=%S/Inputs/sanitizer-ignorelist-src.txt \
// RUN:   -fsanitize=memtag-globals -emit-llvm -o - %s | \
// RUN:   FileCheck %s --check-prefix=IGNORELIST

int global;
int __attribute__((__section__("my_section"))) section_global;
int __attribute__((no_sanitize("memtag"))) attributed_global;
int __attribute__((disable_sanitizer_instrumentation)) disable_instrumentation_global;
int ignorelisted_global;
extern int external_global;

void func() {
  static int static_var = 0;
  const char *literal = "Hello, world!";
  external_global = 1;
}

// CHECK: @{{.*}}extra_global{{.*}} ={{.*}} sanitize_memtag
// CHECK: @{{.*}}global{{.*}} ={{.*}} sanitize_memtag

// CHECK:     @{{.*}}section_global{{.*}} =
// CHECK-NOT: sanitize_memtag
// CHECK:     @{{.*}}attributed_global{{.*}} =
// CHECK-NOT: sanitize_memtag
// CHECK:     @{{.*}}disable_instrumentation_global{{.*}} =
// CHECK-NOT: sanitize_memtag
// CHECK:     @{{.*}}ignorelisted_global{{.*}} =
// CHECK-NOT: sanitize_memtag

// CHECK: @{{.*}}static_var{{.*}} ={{.*}} sanitize_memtag
// CHECK: @{{.*}} = {{.*}} c"Hello, world!\00"{{.*}}
// CHECK: @{{.*}}external_global{{.*}} ={{.*}} sanitize_memtag

// IGNORELIST: @{{.*}}extra_global{{.*}} ={{.*}} sanitize_memtag

// IGNORELIST:     @{{.*}}global{{.*}} =
// IGNORELIST-NOT: sanitize_memtag
// IGNORELIST:     @{{.*}}attributed_global{{.*}} =
// IGNORELIST-NOT: sanitize_memtag
// IGNORELIST:     @{{.*}}disable_instrumentation_global{{.*}} =
// IGNORELIST-NOT: sanitize_memtag
// IGNORELIST:     @{{.*}}ignorelisted_globa{{.*}} =
// IGNORELIST-NOT: sanitize_memtag
// IGNORELIST:     @{{.*}}static_var{{.*}} =
// IGNORELIST-NOT: sanitize_memtag
// IGNORELIST:     @{{.*}} = {{.*}} c"Hello, world!\00"{{.*}}
// IGNORELIST-NOT: sanitize_memtag
// IGNORELIST:     @{{.*}}external_global{{.*}} =
// IGNORELIST-NOT: sanitize_memtag
