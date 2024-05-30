// RUN: %flang_fc1 -emit-llvm -o - %s | FileCheck %s

#access_group = #llvm.access_group<id = distinct[0]<>>
#loop_vectorize = #llvm.loop_vectorize<disable = false>
#loop_annotation = #llvm.loop_annotation<vectorize = #loop_vectorize, parallelAccesses = #access_group>

// CHECK-LABEL: @vector_always_
// CHECK:   br i1 {{.*}}, label {{.*}}, label {{.*}}, !llvm.loop ![[ANNOTATION:.*]]
func.func @_QPvector_always() {
    %c1 = arith.constant 1 : index
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10 = arith.constant 10 : index
    %0 = fir.alloca !fir.array<10xi32> {bindc_name = "a", uniq_name = "_QFvector_alwaysEa"}
    %1 = fir.shape %c10 : (index) -> !fir.shape<1>
    %2 = fir.declare %0(%1) {uniq_name = "_QFvector_alwaysEa"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
    %3 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFvector_alwaysEi"}
    %4 = fir.declare %3 {uniq_name = "_QFvector_alwaysEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %5 = fir.convert %c1_i32 : (i32) -> index
    %6 = fir.convert %c10_i32 : (i32) -> index
    %7 = fir.convert %5 : (index) -> i32
    %8:2 = fir.do_loop %arg0 = %5 to %6 step %c1 iter_args(%arg1 = %7) -> (index, i32) attributes {loop_annotation = #loop_annotation} {
      fir.store %arg1 to %4 : !fir.ref<i32>
      %9 = fir.load %4 : !fir.ref<i32>
      %10 = fir.load %4 : !fir.ref<i32>
      %11 = fir.convert %10 : (i32) -> i64
      %12 = fir.array_coor %2(%1) %11 : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
      fir.store %9 to %12 : !fir.ref<i32>
      %13 = arith.addi %arg0, %c1 : index
      %14 = fir.convert %c1 : (index) -> i32
      %15 = fir.load %4 : !fir.ref<i32>
      %16 = arith.addi %15, %14 : i32
      fir.result %13, %16 : index, i32
    }
    fir.store %8#1 to %4 : !fir.ref<i32>
    return
  }

// CHECK: ![[ANNOTATION]] = distinct !{![[ANNOTATION]], ![[VECTORIZE:.*]], ![[PAR_ACCESS:.*]]}
// CHECK: ![[VECTORIZE]] = !{!"llvm.loop.vectorize.enable", i1 true}
// CHECK: ![[PAR_ACCESS]] = !{!"llvm.loop.parallel_accesses", ![[DISTINCT:.*]]}
// CHECK: ![[DISTINCT]] = distinct !{}
