// RUN: fir-opt --cfg-conversion %s | FileCheck %s

#access_group = #llvm.access_group<id = distinct[0]<>>
// CHECK: #[[ACCESS:.*]] = #llvm.access_group<id = distinct[0]<>>
#loop_vectorize = #llvm.loop_vectorize<disable = false>
// CHECK: #[[VECTORIZE:.*]] = #llvm.loop_vectorize<disable = false>
#loop_annotation = #llvm.loop_annotation<vectorize = #loop_vectorize, parallelAccesses = #access_group>
// CHECK: #[[ANNOTATION:.*]] = #llvm.loop_annotation<vectorize = #[[VECTORIZE]], parallelAccesses = #[[ACCESS]]>

// CHECK-LABEL: @_QPvector_always
func.func @_QPvector_always() -> i32 {
    %c1 = arith.constant 1 : index
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10 = arith.constant 10 : index
// CHECK:   cf.cond_br %{{.*}}, ^{{.*}}, ^{{.*}} {loop_annotation = #[[ANNOTATION]]}
    %8:2 = fir.do_loop %arg0 = %c1 to %c10 step %c1 iter_args(%arg1 = %c1_i32) -> (index, i32) attributes {loopAnnotation = #loop_annotation} {
      fir.result %c1, %c1_i32 : index, i32
    }
    return %8#1 : i32
  }
