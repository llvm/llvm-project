// RUN: %clang -O3 --target=aarch64 %s -S -o- \
// RUN:   | FileCheck --check-prefix=CHECK-NO --check-prefix=CHECK %s
// RUN: %clang -O3 --target=aarch64 -mfix-cortex-a53-835769 %s -S -o- 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-YES --check-prefix=CHECK %s
// RUN: %clang -O3 --target=aarch64 -mno-fix-cortex-a53-835769 %s -S -o- 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO --check-prefix=CHECK %s

// RUN: %clang -O3 --target=aarch64-linux-androideabi %s -S -o- \
// RUN:   | FileCheck --check-prefix=CHECK-YES --check-prefix=CHECK %s
// RUN: %clang -O3 -target aarch64-linux-ohos %s -S -o- \
// RUN:   | FileCheck --check-prefix=CHECK-YES --check-prefix=CHECK %s
// RUN: %clang -O3 --target=aarch64-linux-androideabi -mfix-cortex-a53-835769 %s -S -o- \
// RUN:   | FileCheck --check-prefix=CHECK-YES --check-prefix=CHECK %s
// RUN: %clang -O3 --target=aarch64-linux-androideabi -mno-fix-cortex-a53-835769 %s -S -o- \
// RUN:   | FileCheck --check-prefix=CHECK-NO --check-prefix=CHECK %s

// REQUIRES: aarch64-registered-target

typedef long int64_t;

int64_t f_load_madd_64(int64_t a, int64_t b, int64_t *c) {
    int64_t result = a+b*(*c);
    return result;
}

// CHECK: ldr
// CHECK-YES-NEXT: nop
// CHECK-NO-NOT: nop
// CHECK-NEXT: madd
