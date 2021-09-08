// RUN: %clang -target i386-apple-darwin9 %s -### --emit-static-lib 2>&1 | FileCheck %s
// CHECK: "{{.*}}libtool" "-static" "-D" "-no_warning_for_no_symbols" "-o" "a.out" "{{.*o}}"

// RUN: %clang -target i386-apple-darwin9 %s -### --emit-static-lib -o libfoo.a 2>&1 | FileCheck %s --check-prefix=OUTPUT
// OUTPUT: "{{.*}}libtool" "-static" "-D" "-no_warning_for_no_symbols" "-o" "libfoo.a" "{{.*o}}"
