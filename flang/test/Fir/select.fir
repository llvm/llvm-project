// Test lowering FIR to LLVM IR of fir.select{|_rank|_case}

// RUN: tco %s | FileCheck %s
// RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: @f
func.func @f(%a : i32) -> i32 {
   %1 = arith.constant 1 : i32
   %2 = arith.constant 42 : i32
// CHECK: switch i64 %{{.*}}, label %{{.*}} [
// CHECK:   i64 1, label %{{.*}}
// CHECK: ]
   fir.select %a : i32 [1, ^bb2(%1:i32), unit, ^bb3(%2:i32)]
^bb2(%3 : i32) :
   return %3 : i32
^bb3(%4 : i32) :
   %5 = arith.addi %4, %4 : i32
   // CHECK: ret i32
   return %5 : i32 
}

// CHECK-LABEL: @g
func.func @g(%a : i32) -> i32 {
   %1 = arith.constant 1 : i32
   %2 = arith.constant 42 : i32

// CHECK: switch i64 %{{.*}}, label %{{.*}} [
// CHECK:  i64 1, label %{{.*}}
// CHECK:  i64 -1, label %{{.*}}
// CHECK: ]
   fir.select_rank %a : i32 [1, ^bb2(%1:i32), -1, ^bb4, unit, ^bb3(%2:i32)]
^bb2(%3 : i32) :
   return %3 : i32
^bb3(%4 : i32) :
   %5 = arith.addi %4, %4 : i32
   return %5 : i32 
^bb4:
   // CHECK: ret i32
   return %a : i32
}

// CHECK-LABEL: @h
func.func @h(%a : i32) -> i32 {
   %1 = arith.constant 1 : i32
   %2 = arith.constant 42 : i32
   %b1 = arith.constant 4 : i32
   %b2 = arith.constant 14 : i32
   %b3 = arith.constant 82 : i32
   %b4 = arith.constant 96 : i32
   // CHECK-DAG: icmp eq i32 %{{.*}}, 1
   // CHECK-DAG: icmp sle i32 4, %{{.*}}
   // CHECK-DAG: icmp sle i32 %{{.*}}, 14
   // CHECK-DAG: icmp sle i32 82, %{{.*}}
   // CHECK-DAG: icmp sle i32 %{{.*}}, 96
   fir.select_case %a : i32 [#fir.point, %1, ^bb2(%1:i32), #fir.lower, %b1, ^bb4, #fir.upper, %b2, ^bb6, #fir.interval, %b3, %b4, ^bb5, unit, ^bb3(%2:i32)]
^bb2(%3 : i32) :
   return %3 : i32
^bb3(%4 : i32) :
   %5 = arith.addi %4, %4 : i32
   cf.br ^bb2(%5 : i32) 
^bb4:
   return %a : i32
^bb5:
   return %1 : i32
^bb6:
   %x = arith.addi %b4, %b3 : i32
   // CHECK: ret i32
   return %x : i32
}
