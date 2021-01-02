// RUN: %clang_cc1 -triple sparc-unknown-unknown -emit-llvm -o - %s | FileCheck %s

// Ensure that we pass proper alignment to llvm in the call
// instruction. The proper alignment for the type is sometimes known
// only by clang, and is not manifest in the LLVM-type. So, it must be
// explicitly passed through. (Besides the case of the user specifying
// alignment, as here, this situation also occurrs for non-POD C++
// structs with tail-padding: clang emits these as packed llvm-structs
// for ABI reasons.)

struct s1 {
  int x;
} __attribute__((aligned(8)));

struct s1 x1;


// Ensure the align 8 is passed through:
// CHECK-LABEL: define{{.*}} void @f1()
// CHECK: call void @f1_helper(%struct.s1* byval(%struct.s1) align 8 @x1)
// Also ensure the declaration of f1_helper includes it
// CHECK: declare void @f1_helper(%struct.s1* byval(%struct.s1) align 8)

void f1_helper(struct s1);
void f1() {
  f1_helper(x1);
}
