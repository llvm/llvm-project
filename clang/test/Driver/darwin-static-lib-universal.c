// REQUIRES: system-darwin && target={{.*}}-{{darwin|macos}}{{.*}}

// RUN: touch %t1.o %t2.o

// Multiple -arch produces one libtool job per arch plus a lipo. -Xarch_ is only
// forwarded to the matching architecture libtool. --static-lib-target-arch-only
// should pick up the effective triple for each -arch.
// RUN: %clang -### --emit-static-lib %t1.o %t2.o \
// RUN:     -arch x86_64 -arch arm64 --static-lib-target-arch-only \
// RUN:     -Xarch_arm64 --no-static-lib-deterministic \
// RUN:     -Xarch_x86_64 --static-lib-warn-no-symbols -o libfoo.a 2>&1 \
// RUN:   | FileCheck %s
// CHECK: "{{.*}}libtool" "-static" "-arch_only" "x86_64" "-D" "-o" "{{.*}}x86_64.out" "{{.*}}1.o" "{{.*}}2.o"
// CHECK: "{{.*}}libtool" "-static" "-arch_only" "arm64" "-no_warning_for_no_symbols" "-o" "{{.*}}arm64.out" "{{.*}}1.o" "{{.*}}2.o"
// CHECK: "{{.*}}lipo" "-create" "-output" "libfoo.a"
