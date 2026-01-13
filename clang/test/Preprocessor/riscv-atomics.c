// RUN: %clang --target=riscv32-unknown-linux-gnu -march=rv32ia -x c -E -dM %s \
// RUN: -o - | FileCheck %s
// RUN: %clang --target=riscv32-unknown-linux-gnu -march=rv32i_zalrsc -x c -E \
// RUN: -dM %s -o - | FileCheck %s
// RUN: %clang --target=riscv64-unknown-linux-gnu -march=rv64ia -x c -E -dM %s \
// RUN: -o - | FileCheck %s --check-prefixes=CHECK,CHECK-RV64
// RUN: %clang --target=riscv64-unknown-linux-gnu -march=rv64i_zalrsc -x c -E \
// RUN: -dM %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-RV64

// CHECK: #define __GCC_ATOMIC_BOOL_LOCK_FREE 2
// CHECK: #define __GCC_ATOMIC_CHAR16_T_LOCK_FREE 2
// CHECK: #define __GCC_ATOMIC_CHAR32_T_LOCK_FREE 2
// CHECK: #define __GCC_ATOMIC_CHAR_LOCK_FREE 2
// CHECK: #define __GCC_ATOMIC_INT_LOCK_FREE 2
// CHECK-RV64: #define __GCC_ATOMIC_LLONG_LOCK_FREE 2
// CHECK: #define __GCC_ATOMIC_LONG_LOCK_FREE 2
// CHECK: #define __GCC_ATOMIC_POINTER_LOCK_FREE 2
// CHECK: #define __GCC_ATOMIC_SHORT_LOCK_FREE 2
// CHECK: #define __GCC_ATOMIC_TEST_AND_SET_TRUEVAL 1
// CHECK: #define __GCC_ATOMIC_WCHAR_T_LOCK_FREE 2
// CHECK: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_1 1
// CHECK: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_2 1
// CHECK: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_4 1
// CHECK-RV64: #define __GCC_HAVE_SYNC_COMPARE_AND_SWAP_8 1
