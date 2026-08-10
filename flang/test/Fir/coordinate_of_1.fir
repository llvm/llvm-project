// RUN: fir-opt --split-input-file --verify-diagnostics %s

func.func @_QPcoordinate_box_derived_with_fir_len(%arg0: !fir.box<!fir.type<derived_2{len1:i32}>>) -> i32 {
  %lp = arith.constant 22 : i32
  %e = fir.len_param_index len1, !fir.type<derived_1{len1:i32}>(%lp : i32)
  // expected-error@+1 {{'fir.coordinate_of' op len_param_index type not compatible with reference type}}
  %q = fir.coordinate_of %arg0, %e : (!fir.box<!fir.type<derived_2{len1:i32}>>, !fir.len) -> !fir.ref<i32>
  %val = fir.load %q : !fir.ref<i32>
  return %val : i32
}

// -----

func.func @_QPcoordinate_box_derived_with_fir_len2(%arg0: !fir.box<!fir.array<?x!fir.type<derived_2{len1:i32}>>>) -> i32 {
  %lp = arith.constant 22 : i32
  %e = fir.len_param_index len1, !fir.type<derived_1{len1:i32}>(%lp : i32)
  // expected-error@+1 {{'fir.coordinate_of' op too many operands for len_param_index case}}
  %q = fir.coordinate_of %arg0, %lp, %e : (!fir.box<!fir.array<?x!fir.type<derived_2{len1:i32}>>>, i32, !fir.len) -> !fir.ref<i32>
  %val = fir.load %q : !fir.ref<i32>
  return %val : i32
}

// -----

func.func @_QPcoordinate_box_derived_with_fir_len3(%arg0: !fir.box<!fir.type<derived_2{len1:i32}>>) -> i32 {
  %lp = arith.constant 22 : i32
  %e = fir.len_param_index len1, !fir.type<derived_1{len1:i32}>(%lp : i32)
  // expected-error@+1 {{'fir.coordinate_of' op len_param_index must be last argument}}
  %q = fir.coordinate_of %arg0, %e, %e : (!fir.box<!fir.type<derived_2{len1:i32}>>, !fir.len, !fir.len) -> !fir.ref<i32>
  %val = fir.load %q : !fir.ref<i32>
  return %val : i32
}

// -----

func.func @_QPtest_coordinate_of(%arr : !fir.ref<!fir.array<2 x !fir.char<10, 2>>>, %arg1: index) {
  %1 = arith.constant 10 : i32
  // expected-error@+1 {{'fir.coordinate_of' op character type mismatch}}
  %2 = fir.coordinate_of %arr, %arg1, %1 : (!fir.ref<!fir.array<2 x !fir.char<10, 2>>>, index, i32) -> !fir.ref<!fir.char<1,10>>
  return
}

// -----

func.func @_QPtest_coordinate_of(%arr : !fir.ref<!fir.array<2 x i32>>, %arg1: index) {
  // expected-error@+1 {{'fir.coordinate_of' op invalid parameters (too many)}}
  %2 = fir.coordinate_of %arr, %arg1, %arg1 : (!fir.ref<!fir.array<2 x i32>>, index, index) -> !fir.ref<i32>
  return
}
