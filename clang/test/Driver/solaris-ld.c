// Test ld invocation on Solaris targets.

// Check sparc-sun-solaris2.1
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=sparc-sun-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/sparc-sun-solaris2.11 \
// RUN:   | FileCheck %s
// CHECK: "-cc1" "-triple" "sparc-sun-solaris2.11"
// CHECK: ld{{.*}}"
// CHECK: "--dynamic-linker" "{{.*}}/usr/lib/ld.so.1"
// CHECK: "{{.*}}/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2{{/|\\\\}}crt1.o"
// CHECK: "{{.*}}/usr/lib/crti.o"
// CHECK: "{{.*}}/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2{{/|\\\\}}crtbegin.o"
// CHECK: "{{.*}}/usr/gcc/4.8/lib/gcc/sparc-sun-solaris2.11/4.8.2{{/|\\\\}}crtend.o"
// CHECK: "{{.*}}/usr/lib/crtn.o"
// CHECK "-lc"
// CHECK "-lgcc_s"
// CHECK "-lgcc"
// CHECK "-lm"

// Check the right -l flags are present with -shared
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o -shared 2>&1 \
// RUN:     --target=sparc-sun-solaris2.11 \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/sparc-sun-solaris2.11 \
// RUN:   | FileCheck --check-prefix=CHECK-SHARED %s

// CHECK-SHARED: ld{{.*}}"
// CHECK-SHARED "-lc"
// CHECK-SHARED "-lgcc_s"
// CHECK-SHARED-NOT "-lgcc"
// CHECK-SHARED-NOT: "-lm"
