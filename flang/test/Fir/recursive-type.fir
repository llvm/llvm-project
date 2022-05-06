// Test lowering FIR to LLVM IR for recursive types
  
// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s
// RUN: fir-opt --fir-to-llvm-ir="target=aarch64-unknown-linux-gnu" %s | FileCheck %s
// RUN: fir-opt --fir-to-llvm-ir="target=i386-unknown-linux-gnu" %s | FileCheck %s
// RUN: fir-opt --fir-to-llvm-ir="target=powerpc64le-unknown-linux-gn" %s | FileCheck %s

!t1 = type !fir.type<t1 {a1:!fir.ptr<!fir.type<t1>>}>
!t2 = type !fir.type<t2 {b1:f32,b2:!fir.ptr<!fir.type<t2>>,b3:i32,b4:!fir.ptr<!fir.type<t2>>}>
!t3 = type !fir.type<t3 {c1:!fir.ptr<!fir.type<t4>>}>
!t4 = type !fir.type<t4 {d1:!fir.ptr<!fir.type<t3>>}>

// CHECK-LABEL: llvm.func @recursiveTypes
// CHECK-SAME: %{{.*}}: !llvm.struct<"[[T1:.*]]", (ptr<struct<"[[T1]]">>)>
// CHECK-SAME: %{{.*}}: !llvm.struct<"[[T2:.*]]", (f32, ptr<struct<"[[T2]]">>, i32, ptr<struct<"[[T2]]">>)>
// CHECK-SAME: %{{.*}}: !llvm.struct<"[[T3:.*]]", (ptr<struct<"[[T4:.*]]", (ptr<struct<"[[T3]]">>)>>)>, %{{.*}}: !llvm.struct<"[[T4]]", (ptr<struct<"[[T3]]", (ptr<struct<"[[T4]]">>)>>)>)
func.func @recursiveTypes(%a : !t1, %b : !t2, %c : !t3, %d : !t4) {
  return
}
