// RUN: fir-opt --simplify-intrinsics %s | FileCheck %s --check-prefixes=CHECK,PLAIN
// RUN: fir-opt --simplify-intrinsics="compensated-sum=false" %s | FileCheck %s --check-prefixes=CHECK,PLAIN
// RUN: fir-opt --simplify-intrinsics="compensated-sum=true" %s | FileCheck %s --check-prefixes=CHECK,COMPENSATED

// When compensated-sum is true, disable floating-point SUM simplification
// while still attempting to simplify integer SUM.
func.func @compensated_sum_mixed(%array: !fir.ref<!fir.array<10x10xf32>>,
                                 %integers: !fir.ref<!fir.array<10xi32>>) -> (f32, i32) {
  %c10 = arith.constant 10 : index
  %c0 = arith.constant 0 : index
  %dim = fir.convert %c0 : (index) -> i32
  %line = arith.constant 1 : i32
  %file = fir.zero_bits !fir.ref<i8>
  %absent = fir.absent !fir.box<i1>
  %mask = fir.convert %absent : (!fir.box<i1>) -> !fir.box<none>

  %shape2d = fir.shape %c10, %c10 : (index, index) -> !fir.shape<2>
  %box2d = fir.embox %array(%shape2d) : (!fir.ref<!fir.array<10x10xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<10x10xf32>>
  %array_box = fir.convert %box2d : (!fir.box<!fir.array<10x10xf32>>) -> !fir.box<none>
  %sum_real = fir.call @_FortranASumReal4(%array_box, %file, %line, %dim, %mask) : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> f32

  %shape1d = fir.shape %c10 : (index) -> !fir.shape<1>
  %ibox = fir.embox %integers(%shape1d) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
  %integer_box = fir.convert %ibox : (!fir.box<!fir.array<10xi32>>) -> !fir.box<none>
  %sum_integer = fir.call @_FortranASumInteger4(%integer_box, %file, %line, %dim, %mask) : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32

  return %sum_real, %sum_integer : f32, i32
}

// CHECK-LABEL: func.func @compensated_sum_mixed(
// PLAIN:       %[[SUM_REAL:.*]] = fir.call @_FortranASumReal4x2_simplified(%{{.*}}) : (!fir.box<none>) -> f32
// PLAIN-NOT: fir.call @_FortranASumReal{{[0-9]+}}(
// COMPENSATED: %[[SUM_REAL:.*]] = fir.call @_FortranASumReal4(%{{.*}}) : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> f32
// COMPENSATED-NOT: _FortranASumReal{{.*}}_simplified
// CHECK:       %[[SUM_INTEGER:.*]] = fir.call @_FortranASumInteger4x1_simplified(%{{.*}}) : (!fir.box<none>) -> i32
// CHECK:       return %[[SUM_REAL]], %[[SUM_INTEGER]] : f32, i32

func.func private @_FortranASumReal4(!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> f32 attributes {fir.runtime}
func.func private @_FortranASumInteger4(!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32 attributes {fir.runtime}
