// Check handling -mhard-float / -msoft-float options
// when build for SystemZ platforms.

// Default
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target s390x-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-DEF %s
// CHECK-DEF-NOT: "-msoft-float"
// CHECK-DEF-NOT: "-mfloat-abi" "soft"

// -mhard-float
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target s390x-linux-gnu -mhard-float \
// RUN:   | FileCheck --check-prefix=CHECK-HARD %s
// CHECK-HARD-NOT: "-msoft-float"
// CHECK-HARD-NOT: "-mfloat-abi" "soft"

// -msoft-float
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target s390x-linux-gnu -msoft-float \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT %s
// CHECK-SOFT: "-msoft-float" "-mfloat-abi" "soft"

// -mfloat-abi=soft
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target s390x-linux-gnu -mfloat-abi=soft \
// RUN:   | FileCheck --check-prefix=CHECK-FLOATABISOFT %s
// CHECK-FLOATABISOFT: error: unsupported option '-mfloat-abi=soft'

// -mfloat-abi=hard
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target s390x-linux-gnu -mfloat-abi=hard \
// RUN:   | FileCheck --check-prefix=CHECK-FLOATABIHARD %s
// CHECK-FLOATABIHARD: error: unsupported option '-mfloat-abi=hard'

// check invalid -mfloat-abi
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target s390x-linux-gnu -mfloat-abi=x \
// RUN:   | FileCheck --check-prefix=CHECK-ERRMSG %s
// CHECK-ERRMSG: error: unsupported option '-mfloat-abi=x'

int foo(void) {
  return 0;
}

