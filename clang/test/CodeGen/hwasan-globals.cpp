// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -include %S/Inputs/sanitizer-extra-source.cpp \
// RUN:   -fsanitize-ignorelist=%S/Inputs/sanitizer-ignorelist-global.txt \
// RUN:   -fsanitize=hwaddress -emit-llvm -triple aarch64-linux-android31 -o -\
// RUN:    %s | FileCheck %s

// RUN: %clang_cc1 -include %S/Inputs/sanitizer-extra-source.cpp \
// RUN:   -fsanitize-ignorelist=%S/Inputs/sanitizer-ignorelist-src.txt \
// RUN:   -fsanitize=hwaddress -emit-llvm -triple aarch64-linux-android31 -o -\
// RUN:   %s | FileCheck %s --check-prefix=IGNORELIST

int global;
int __attribute__((no_sanitize("hwaddress"))) attributed_global;
int __attribute__((disable_sanitizer_instrumentation)) disable_instrumentation_global;
int ignorelisted_global;

void func() {
  static int static_var = 0;
  const char *literal = "Hello, world!";
}

// CHECK: @{{.*}}attributed_global{{.*}} ={{.*}} global {{.*}}, no_sanitize_hwaddress
// CHECK: @{{.*}}disable_instrumentation_global{{.*}} ={{.*}} global {{.*}}, no_sanitize_hwaddress
// CHECK: @{{.*}}ignorelisted_global{{.*}} ={{.*}} global {{.*}}, no_sanitize_hwaddress
// CHECK: @{{.*}}extra_global{{.*}}.hwasan{{.*}} =
// CHECK: @{{.*}}global{{.*}}.hwasan{{.*}} =
// CHECK: @{{.*}}static_var{{.*}}.hwasan{{.*}} =
// CHECK: @{{.*}}.hwasan{{.*}} = {{.*}} c"Hello, world!\00"

// IGNORELIST: @{{.*}}global{{.*}} ={{.*}} global {{.*}}, no_sanitize_hwaddress
// IGNORELIST: @{{.*}}attributed_global{{.*}} ={{.*}} global {{.*}}, no_sanitize_hwaddress
// IGNORELIST: @{{.*}}disable_instrumentation_global{{.*}} ={{.*}} global {{.*}}, no_sanitize_hwaddress
// IGNORELIST: @{{.*}}ignorelisted_globa{{.*}} ={{.*}} global {{.*}}, no_sanitize_hwaddress
// IGNORELIST: @{{.*}}static_var{{.*}} ={{.*}} global {{.*}}, no_sanitize_hwaddress
// IGNORELIST: @{{.*}} = {{.*}} c"Hello, world!\00"{{.*}}, no_sanitize_hwaddress
// IGNORELIST: @extra_global.hwasan =
