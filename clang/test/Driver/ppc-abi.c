// Check passing PowerPC ABI options to the backend.

// RUN: %clang -target powerpc64-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ELFv1 %s
// RUN: %clang -target powerpc64-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   -mabi=elfv1 | FileCheck -check-prefix=CHECK-ELFv1 %s
// RUN: %clang -target powerpc64-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   -mabi=elfv2 | FileCheck -check-prefix=CHECK-ELFv2-BE %s

// RUN: %clang -target powerpc64le-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ELFv2 %s
// RUN: %clang -target powerpc64le-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   -mabi=elfv1 | FileCheck -check-prefix=CHECK-ELFv1-LE %s
// RUN: %clang -target powerpc64le-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   -mabi=elfv2 | FileCheck -check-prefix=CHECK-ELFv2 %s
// RUN: %clang -target powerpc64le-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   -mabi=altivec | FileCheck -check-prefix=CHECK-ELFv2 %s

// RUN: %clang -target powerpc64-unknown-freebsd11 %s -### 2>&1 | FileCheck --check-prefix=CHECK-ELFv1 %s
// RUN: %clang -target powerpc64-unknown-freebsd12 %s -### 2>&1 | FileCheck --check-prefix=CHECK-ELFv1 %s
// RUN: %clang -target powerpc64-unknown-freebsd13 %s -### 2>&1 | FileCheck --check-prefix=CHECK-ELFv2-BE %s
// RUN: %clang -target powerpc64-unknown-freebsd14 %s -### 2>&1 | FileCheck --check-prefix=CHECK-ELFv2-BE %s
// RUN: %clang -target powerpc64le-unknown-freebsd13 %s -### 2>&1 | FileCheck --check-prefix=CHECK-ELFv2 %s
// RUN: %clang -target powerpc64-unknown-openbsd %s -### 2>&1 | FileCheck --check-prefix=CHECK-ELFv2-BE-PIE %s
// RUN: %clang -target powerpc64-linux-musl %s -### 2>&1 | FileCheck --check-prefix=CHECK-ELFv2-BE-PIE %s

// CHECK-ELFv1: "-target-abi" "elfv1"
// CHECK-ELFv1-LE: "-target-abi" "elfv1"
// CHECK-ELFv2: "-target-abi" "elfv2"
// CHECK-ELFv2-BE: "-target-abi" "elfv2"
// CHECK-ELFv2-BE-PIE: "-mrelocation-model" "pic" "-pic-level" "2" "-pic-is-pie"
// CHECK-ELFv2-BE-PIE: "-target-abi" "elfv2"

// RUN: %clang -fPIC -target powerpc64-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ELFv1-PIC %s
// RUN: %clang -fPIC -target powerpc64-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   -mabi=elfv1 | FileCheck -check-prefix=CHECK-ELFv1-PIC %s
// RUN: %clang -fPIC -target powerpc64-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   -mabi=elfv2 | FileCheck -check-prefix=CHECK-ELFv2-PIC %s

// RUN: %clang -fPIC -target powerpc64le-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ELFv2-PIC %s
// RUN: %clang -fPIC -target powerpc64le-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   -mabi=elfv1 | FileCheck -check-prefix=CHECK-ELFv1-PIC %s
// RUN: %clang -fPIC -target powerpc64le-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   -mabi=elfv2 | FileCheck -check-prefix=CHECK-ELFv2-PIC %s
// RUN: %clang -fPIC -target powerpc64le-unknown-linux-gnu %s -### -o %t.o 2>&1 \
// RUN:   -mabi=altivec | FileCheck -check-prefix=CHECK-ELFv2-PIC %s

// CHECK-ELFv1-PIC: "-mrelocation-model" "pic" "-pic-level" "2"
// CHECK-ELFv1-PIC: "-target-abi" "elfv1"
// CHECK-ELFv2-PIC: "-mrelocation-model" "pic" "-pic-level" "2"
// CHECK-ELFv2-PIC: "-target-abi" "elfv2"

// Check -mabi=ieeelongdouble is passed through but it does not change -target-abi.
// RUN: %clang -target powerpc64le-linux-gnu %s -mabi=ieeelongdouble -mabi=elfv1 -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ELFv1-IEEE %s
// RUN: %clang -target powerpc64le-linux-gnu %s -mabi=elfv1 -mabi=ieeelongdouble -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ELFv1-IEEE %s
// RUN: %clang -target powerpc64le-linux-gnu %s -mabi=elfv2 -mabi=elfv1 -mabi=ibmlongdouble -mabi=ieeelongdouble -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ELFv1-IEEE %s

// CHECK-ELFv1-IEEE: "-mabi=ieeelongdouble"
// CHECK-ELFv1-IEEE: "-target-abi" "elfv1"

// Check -mabi=ibmlongdouble is the default.
// RUN: %clang -target powerpc64le-linux-gnu %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ELFv2-IBM128 %s
// RUN: %clang -target powerpc64le-linux-gnu %s -mabi=ibmlongdouble -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-ELFv2-IBM128 %s

// CHECK-ELFv2-IBM128-NOT: "-mabi=ieeelongdouble"
// CHECK-ELFv2-IBM128: "-target-abi" "elfv2"
