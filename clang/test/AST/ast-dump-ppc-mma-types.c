// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -target-cpu future \
// RUN:   -ast-dump -ast-dump-filter __vector %s | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -target-cpu future \
// RUN:   -target-feature -mma -ast-dump %s | FileCheck %s \
// RUN:   --check-prefix=CHECK-NO-MMA
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -target-cpu pwr9 \
// RUN:   -ast-dump %s | FileCheck %s --check-prefix=CHECK-PWR9
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ast-dump %s | FileCheck %s \
// RUN:   --check-prefix=CHECK-X86_64
// RUN: %clang_cc1 -triple arm-unknown-unknown -ast-dump %s | FileCheck %s \
// RUN:   --check-prefix=CHECK-ARM
// RUN: %clang_cc1 -triple riscv64-unknown-unknown -ast-dump %s | FileCheck %s \
// RUN:   --check-prefix=CHECK-RISCV64

// This test case checks that the PowerPC __vector_pair and __vector_quad types
// are correctly defined. These types should only be defined on PowerPC targets
// supporting the mma feature. We also added checks on a couple of other targets
// to ensure the types are target-dependent.

// CHECK: TypedefDecl {{.*}} implicit __vector_quad '__vector_quad'
// CHECK-NEXT: -BuiltinType {{.*}} '__vector_quad'
// CHECK: TypedefDecl {{.*}} implicit __vector_pair '__vector_pair'
// CHECK-NEXT: -BuiltinType {{.*}} '__vector_pair'

// CHECK-NO-MMA-NOT: __vector_quad
// CHECK-NO-MMA-NOT: __vector_pair

// CHECK-PWR9-NOT: __vector_quad
// CHECK-PWR9-NOT: __vector_pair

// CHECK-X86_64-NOT: __vector_quad
// CHECK-X86_64-NOT: __vector_pair

// CHECK-ARM-NOT: __vector_quad
// CHECK-ARM-NOT: __vector_pair

// CHECK-RISCV64-NOT: __vector_quad
// CHECK-RISCV64-NOT: __vector_pair
