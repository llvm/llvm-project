// RUN: fir-opt --lower-workdistribute %s | FileCheck %s

// CHECK-LABEL:   func.func @test_nested_derived_type_map_operand_and_block_addition(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>) {
// CHECK:           %[[VAL_0:.*]] = fir.declare %[[ARG0]] {uniq_name = "_QFmaptype_derived_nested_explicit_multiple_membersEsa"} : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>) -> !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>
// CHECK:           %[[VAL_1:.*]] = fir.coordinate_of %[[VAL_0]], n : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>) -> !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>>
// CHECK:           %[[VAL_2:.*]] = fir.coordinate_of %[[VAL_1]], i : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>>) -> !fir.ref<i32>
// CHECK:           %[[VAL_3:.*]] = omp.map.info var_ptr(%[[VAL_2]] : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "sa%[[VAL_4:.*]]%[[VAL_5:.*]]"}
// CHECK:           %[[VAL_6:.*]] = fir.coordinate_of %[[VAL_0]], n : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>) -> !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>>
// CHECK:           %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_6]], r : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>>) -> !fir.ref<f32>
// CHECK:           %[[VAL_8:.*]] = omp.map.info var_ptr(%[[VAL_7]] : !fir.ref<f32>, f32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<f32> {name = "sa%[[VAL_4]]%[[VAL_9:.*]]"}
// CHECK:           %[[VAL_10:.*]] = omp.map.info var_ptr(%[[VAL_0]] : !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>, !fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>) map_clauses(tofrom) capture(ByRef) members(%[[VAL_3]], %[[VAL_8]] : [1, 0], [1, 1] : !fir.ref<i32>, !fir.ref<f32>) -> !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>> {name = "sa", partial_map = true}
// CHECK:           omp.target map_entries(%[[VAL_10]] -> %[[VAL_11:.*]] : !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>) {
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @test_nested_derived_type_map_operand_and_block_addition(%arg0: !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>) {        
  %0 = fir.declare %arg0 {uniq_name = "_QFmaptype_derived_nested_explicit_multiple_membersEsa"} : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>) -> !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>
  %2 = fir.coordinate_of %0, n : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>) -> !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>>
  %4 = fir.coordinate_of %2, i : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>>) -> !fir.ref<i32>
  %5 = omp.map.info var_ptr(%4 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "sa%n%i"}
  %7 = fir.coordinate_of %0, n : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>) -> !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>>
  %9 = fir.coordinate_of %7, r : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>>) -> !fir.ref<f32>
  %10 = omp.map.info var_ptr(%9 : !fir.ref<f32>, f32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<f32> {name = "sa%n%r"}
  %11 = omp.map.info var_ptr(%0 : !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>, !fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>) map_clauses(tofrom) capture(ByRef) members(%5, %10 : [1,0], [1,1] : !fir.ref<i32>, !fir.ref<f32>) -> !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>> {name = "sa", partial_map = true}
  omp.target map_entries(%11 -> %arg1 : !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>) {
    omp.terminator
  }
  return
}
