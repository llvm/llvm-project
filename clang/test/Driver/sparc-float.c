// Check handling -mhard-float / -msoft-float options
// when build for SPARC platforms.

// Default sparc
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target sparc-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-DEF %s
// CHECK-DEF-NOT: "-target-feature" "+soft-float"
// CHECK-DEF-NOT: "-msoft-float"

// -mhard-float
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target sparc-linux-gnu -mhard-float \
// RUN:   | FileCheck --check-prefix=CHECK-HARD %s
// CHECK-HARD-NOT: "-msoft-float"

// -msoft-float
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target sparc-linux-gnu -msoft-float \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT %s
// CHECK-SOFT: "-target-feature" "+soft-float"

// -mfloat-abi=soft
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target sparc-linux-gnu -mfloat-abi=soft \
// RUN:   | FileCheck --check-prefix=CHECK-FLOATABISOFT %s
// CHECK-FLOATABISOFT: "-target-feature" "+soft-float"

// -mfloat-abi=hard
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target sparc-linux-gnu -mfloat-abi=hard \
// RUN:   | FileCheck --check-prefix=CHECK-FLOATABIHARD %s
// CHECK-FLOATABIHARD-NOT: "-target-feature" "+soft-float"

// check invalid -mfloat-abi
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target sparc-linux-gnu -mfloat-abi=x \
// RUN:   | FileCheck --check-prefix=CHECK-ERRMSG %s
// CHECK-ERRMSG: error: invalid float ABI '-mfloat-abi=x'

// Default sparc64
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target sparc64-linux-gnu \
// RUN:   | FileCheck --check-prefix=CHECK-DEF-SPARC64 %s
// CHECK-DEF-SPARC64-NOT: "-target-feature" "+soft-float"
// CHECK-DEF-SPARC64-NOT: "-msoft-float"

// -mhard-float
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target sparc64-linux-gnu -mhard-float \
// RUN:   | FileCheck --check-prefix=CHECK-HARD-SPARC64 %s
// CHECK-HARD-SPARC64-NOT: "-msoft-float"

// -msoft-float
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target sparc64-linux-gnu -msoft-float \
// RUN:   | FileCheck --check-prefix=CHECK-SOFT-SPARC64 %s
// CHECK-SOFT-SPARC64: "-target-feature" "+soft-float"

// -mfloat-abi=soft
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target sparc64-linux-gnu -mfloat-abi=soft \
// RUN:   | FileCheck --check-prefix=CHECK-FLOATABISOFT64 %s
// CHECK-FLOATABISOFT64: "-target-feature" "+soft-float"

// -mfloat-abi=hard
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target sparc64-linux-gnu -mfloat-abi=hard \
// RUN:   | FileCheck --check-prefix=CHECK-FLOATABIHARD64 %s
// CHECK-FLOATABIHARD64-NOT: "-target-feature" "+soft-float"

// check invalid -mfloat-abi
// RUN: %clang -c %s -### -o %t.o 2>&1 \
// RUN:     -target sparc64-linux-gnu -mfloat-abi=x \
// RUN:   | FileCheck --check-prefix=CHECK-ERRMSG64 %s
// CHECK-ERRMSG64: error: invalid float ABI '-mfloat-abi=x'
