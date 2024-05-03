// Test lowering FIR to LLVM IR for recursive types
  
// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s
// RUN: fir-opt --fir-to-llvm-ir="target=aarch64-unknown-linux-gnu" %s | FileCheck %s
// RUN: fir-opt --fir-to-llvm-ir="target=i386-unknown-linux-gnu" %s | FileCheck %s
// RUN: fir-opt --fir-to-llvm-ir="target=powerpc64le-unknown-linux-gn" %s | FileCheck %s

!t1 = !fir.type<t1 {a1:!fir.ptr<!fir.type<t1>>}>
!t2 = !fir.type<t2 {b1:f32,b2:!fir.ptr<!fir.type<t2>>,b3:i32,b4:!fir.ptr<!fir.type<t2>>}>
!t3 = !fir.type<t3 {c1:!fir.ptr<!fir.type<t4>>}>
!t4 = !fir.type<t4 {d1:!fir.ptr<!fir.type<t3>>}>

// CHECK-LABEL: llvm.func @recursiveTypes
// CHECK-SAME: %{{.*}}: !llvm.struct<"t1", (ptr)>
// CHECK-SAME: %{{.*}}: !llvm.struct<"t2", (f32, ptr, i32, ptr)>
// CHECK-SAME: %{{.*}}: !llvm.struct<"t3", (ptr)>, %{{.*}}: !llvm.struct<"t4", (ptr)>)
func.func @recursiveTypes(%a : !t1, %b : !t2, %c : !t3, %d : !t4) {
  return
}
