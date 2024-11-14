// RUN: fir-opt --fir-to-llvm-ir %s | FileCheck %s

#access_group = #llvm.access_group<id = distinct[0]<>>
// CHECK: #[[ACCESS:.*]] = #llvm.access_group<id = distinct[0]<>>
#loop_vectorize = #llvm.loop_vectorize<disable = false>
// CHECK: #[[VECTORIZE:.*]] = #llvm.loop_vectorize<disable = false>
#loop_annotation = #llvm.loop_annotation<vectorize = #loop_vectorize, parallelAccesses = #access_group>
// CHECK: #[[ANNOTATION:.*]] = #llvm.loop_annotation<vectorize = #[[VECTORIZE]], parallelAccesses = #[[ACCESS]]>

func.func @_QPvector_always() -> i32 {
  %c1 = arith.constant 1 : index
  %c10_i32 = arith.constant 10 : i32
  %c1_i32 = arith.constant 1 : i32
  %c10 = arith.constant 10 : index
  %0 = arith.subi %c10, %c1 : index
  %1 = arith.addi %0, %c1 : index
  %2 = arith.divsi %1, %c1 : index
  cf.br ^bb1(%c1, %c1_i32, %2 : index, i32, index)
^bb1(%3: index, %4: i32, %5: index):  // 2 preds: ^bb0, ^bb2
  %c0 = arith.constant 0 : index
  %6 = arith.cmpi sgt, %5, %c0 : index
  cf.cond_br %6, ^bb2, ^bb3 {loop_annotation = #loop_annotation}
// CHECK:   llvm.cond_br %{{.*}}, ^{{.*}}, ^{{.*}} {loop_annotation = #[[ANNOTATION]]}
^bb2:  // pred: ^bb1
  %7 = arith.addi %3, %c1 : index
  %c1_0 = arith.constant 1 : index
  %8 = arith.subi %5, %c1_0 : index
  cf.br ^bb1(%7, %c1_i32, %8 : index, i32, index)
^bb3:  // pred: ^bb1
  return %4 : i32
}

