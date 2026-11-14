// RUN: %clang_cc1 -triple aarch64-linux-android34 -target-feature +mte \
// RUN:   -include %S/Inputs/sanitizer-extra-source.cpp \
// RUN:   -fsanitize-ignorelist=%S/Inputs/sanitizer-ignorelist-global.txt \
// RUN:   -fsanitize=memtag-globals -emit-llvm -o - %s | FileCheck %s

// RUN: %clang_cc1 -triple aarch64-linux-android34 -target-feature +mte \
// RUN:    -include %S/Inputs/sanitizer-extra-source.cpp \
// RUN:   -fsanitize-ignorelist=%S/Inputs/sanitizer-ignorelist-src.txt \
// RUN:   -fsanitize=memtag-globals -emit-llvm -o - %s | \
// RUN:   FileCheck %s --check-prefix=IGNORELIST

// RUN: %clang_cc1 -triple aarch64-linux-android34 -target-feature +mte \
// RUN:   -include %S/Inputs/sanitizer-extra-source.cpp \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=ATTR

// RUN: %clang_cc1 -triple aarch64-linux-android34 \
// RUN:   -include %S/Inputs/sanitizer-extra-source.cpp \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=NOMTE

int global;
int __attribute__((memtag)) explicitly_tagged_global;
int __attribute__((__section__("my_section"))) section_global;
int __attribute__((no_sanitize("memtag"))) attributed_global;
int __attribute__((disable_sanitizer_instrumentation)) disable_instrumentation_global;
int ignorelisted_global;
extern int external_global;
extern __attribute__((memtag)) int explicitly_tagged_external_global;

void func() {
  static int static_var = 0;
  __attribute__((memtag)) static int explicitly_tagged_static_var = 0;
  const char *literal = "Hello, world!";
  external_global = 1;
  explicitly_tagged_external_global = 1;
}

// CHECK: @{{.*}}extra_global{{.*}} ={{.*}} sanitize_memtag
// CHECK: @{{.*}}global{{.*}} ={{.*}} sanitize_memtag
// CHECK: @{{.*}}explicitly_tagged_global{{.*}} ={{.*}} sanitize_memtag

// This DOES NOT mean we are instrumenting the section global,
// but we are ignoring it in AsmPrinter rather than in clang.
// CHECK:     @{{.*}}section_global{{.*}} ={{.*}} sanitize_memtag
// CHECK:     @{{.*}}attributed_global{{.*}} =
// CHECK-NOT: sanitize_memtag
// CHECK:     @{{.*}}disable_instrumentation_global{{.*}} =
// CHECK-NOT: sanitize_memtag
// CHECK:     @{{.*}}ignorelisted_global{{.*}} =
// CHECK-NOT: sanitize_memtag

// CHECK: @{{.*}}static_var{{.*}} ={{.*}} sanitize_memtag
// CHECK: @{{.*}}explicitly_tagged_static_var{{.*}} ={{.*}} sanitize_memtag
// CHECK: @{{.*}} = {{.*}} c"Hello, world!\00"{{.*}} sanitize_memtag
// CHECK: @{{.*}}external_global{{.*}} ={{.*}} sanitize_memtag
// CHECK: @{{.*}}explicitly_tagged_external_global{{.*}} ={{.*}} sanitize_memtag

// IGNORELIST: @{{.*}}extra_global{{.*}} ={{.*}} sanitize_memtag

// IGNORELIST:     @{{.*}}global{{.*}} =
// IGNORELIST-NOT: sanitize_memtag
// IGNORELIST:     @{{.*}}explicitly_tagged_global{{.*}} =
// IGNORELIST-NOT: sanitize_memtag
// IGNORELIST:     @{{.*}}attributed_global{{.*}} =
// IGNORELIST-NOT: sanitize_memtag
// IGNORELIST:     @{{.*}}disable_instrumentation_global{{.*}} =
// IGNORELIST-NOT: sanitize_memtag
// IGNORELIST:     @{{.*}}ignorelisted_globa{{.*}} =
// IGNORELIST-NOT: sanitize_memtag
// IGNORELIST:     @{{.*}}static_var{{.*}} =
// IGNORELIST-NOT: sanitize_memtag
// IGNORELIST:     @{{.*}}explicitly_tagged_static_var{{.*}} =
// IGNORELIST-NOT: sanitize_memtag
// IGNORELIST:     @{{.*}} = {{.*}} c"Hello, world!\00"{{.*}}
// IGNORELIST-NOT: sanitize_memtag
// IGNORELIST:     @{{.*}}external_global{{.*}} =
// IGNORELIST-NOT: sanitize_memtag
// IGNORELIST:     @{{.*}}explicitly_tagged_external_global{{.*}} =
// IGNORELIST-NOT: sanitize_memtag

// ATTR:     @{{.*}}extra_global{{.*}} =
// ATTR-NOT: sanitize_memtag
// ATTR:     @{{.*}}global{{.*}} =
// ATTR-NOT: sanitize_memtag
// ATTR:     @{{.*}}explicitly_tagged_global{{.*}} ={{.*}} sanitize_memtag
// ATTR:     @{{.*}}section_global{{.*}} =
// ATTR-NOT: sanitize_memtag
// ATTR:     @{{.*}}attributed_global{{.*}} =
// ATTR-NOT: sanitize_memtag
// ATTR:     @{{.*}}disable_instrumentation_global{{.*}} =
// ATTR-NOT: sanitize_memtag
// ATTR:     @{{.*}}ignorelisted_global{{.*}} =
// ATTR-NOT: sanitize_memtag
// ATTR:     @{{.*}}static_var{{.*}} =
// ATTR-NOT: sanitize_memtag
// ATTR:     @{{.*}}explicitly_tagged_static_var{{.*}} ={{.*}} sanitize_memtag
// ATTR:     @{{.*}} = {{.*}} c"Hello, world!\00"{{.*}}
// ATTR-NOT: sanitize_memtag
// ATTR:     @{{.*}}external_global{{.*}} =
// ATTR-NOT: sanitize_memtag
// ATTR:     @{{.*}}explicitly_tagged_external_global{{.*}} ={{.*}} sanitize_memtag

// NOMTE-NOT: sanitize_memtag
