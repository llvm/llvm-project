// FIR ops diagnotic tests

// RUN: fir-opt -split-input-file -verify-diagnostics %s

// expected-error@+1{{custom op 'fir.string_lit' must have character type}}
%0 = fir.string_lit "Hello, World!"(13) : !fir.int<32>

// -----

// expected-error@+1{{custom op 'fir.string_lit' found an invalid constant}}
%0 = fir.string_lit 20(13) : !fir.int<32>

// -----

// expected-error@+1{{'fir.string_lit' op values in list must be integers}}
%2 = fir.string_lit [158, 2.0](2) : !fir.char<2>

// -----

func @bad_rebox_1(%arg0: !fir.ref<!fir.array<?x?xf32>>) {
  %c10 = constant 10 : index
  %0 = fir.shape %c10 : (index) -> !fir.shape<1>
  // expected-error@+1{{op operand #0 must be The type of a Fortran descriptor, but got '!fir.ref<!fir.array<?x?xf32>>'}}
  %1 = fir.rebox %arg0(%0) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  return
}

// -----

func @bad_rebox_2(%arg0: !fir.box<!fir.array<?x?xf32>>) {
  %c10 = constant 10 : index
  %0 = fir.shape %c10 : (index) -> !fir.shape<1>
  // expected-error@+1{{op result #0 must be The type of a Fortran descriptor, but got '!fir.ref<!fir.array<?xf32>>'}}
  %1 = fir.rebox %arg0(%0) : (!fir.box<!fir.array<?x?xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf32>>
  return
}

// -----

func @bad_rebox_3(%arg0: !fir.box<!fir.array<*:f32>>) {
  %c10 = constant 10 : index
  %0 = fir.shape %c10 : (index) -> !fir.shape<1>
  // expected-error@+1{{op box operand must not have unknown rank or type}}
  %1 = fir.rebox %arg0(%0) : (!fir.box<!fir.array<*:f32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  return
}

// -----

func @bad_rebox_4(%arg0: !fir.box<!fir.array<?xf32>>) {
  // expected-error@+1{{op result type must not have unknown rank or type}}
  %0 = fir.rebox %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<*:f32>>
  return
}

// -----

func @bad_rebox_5(%arg0: !fir.box<!fir.array<?x?xf32>>) {
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  %0 = fir.slice %c1, %c10, %c1 : (index, index, index) -> !fir.slice<1>
  // expected-error@+1{{op slice operand rank must match box operand rank}}
  %1 = fir.rebox %arg0 [%0] : (!fir.box<!fir.array<?x?xf32>>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  return
}

// -----

func @bad_rebox_6(%arg0: !fir.box<!fir.array<?xf32>>) {
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  %0 = fir.slice %c1, %c10, %c1 : (index, index, index) -> !fir.slice<1>
  %1 = fir.shift %c1, %c1 : (index, index) -> !fir.shift<2>
  // expected-error@+1{{shape operand and input box ranks must match when there is a slice}}
  %2 = fir.rebox %arg0(%1) [%0] : (!fir.box<!fir.array<?xf32>>, !fir.shift<2>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  return
}

// -----

func @bad_rebox_7(%arg0: !fir.box<!fir.array<?xf32>>) {
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  %0 = fir.slice %c1, %c10, %c1 : (index, index, index) -> !fir.slice<1>
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  // expected-error@+1{{shape operand must absent or be a fir.shift when there is a slice}}
  %2 = fir.rebox %arg0(%1) [%0] : (!fir.box<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  return
}

// -----

func @bad_rebox_8(%arg0: !fir.box<!fir.array<?x?xf32>>) {
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  %undef = fir.undefined index
  %0 = fir.slice %c1, %undef, %undef, %c1, %c10, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  // expected-error@+1{{result type rank and rank after applying slice operand must match}}
  %1 = fir.rebox %arg0 [%0] : (!fir.box<!fir.array<?x?xf32>>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xf32>>
  return
}

// -----

func @bad_rebox_9(%arg0: !fir.box<!fir.array<?xf32>>) {
  %c10 = constant 10 : index
  %0 = fir.shift %c10, %c10 : (index, index) -> !fir.shift<2>
  // expected-error@+1{{shape operand and input box ranks must match when the shape is a fir.shift}}
  %1 = fir.rebox %arg0(%0) : (!fir.box<!fir.array<?xf32>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?xf32>>
  return
}

// -----

func @bad_rebox_10(%arg0: !fir.box<!fir.array<?xf32>>) {
  %c10 = constant 10 : index
  %0 = fir.shape %c10, %c10 : (index, index) -> !fir.shape<2>
  // expected-error@+1{{result type and shape operand ranks must match}}
  %1 = fir.rebox %arg0(%0) : (!fir.box<!fir.array<?xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<?xf32>>
  return
}

// -----

func @bad_rebox_11(%arg0: !fir.box<!fir.array<?x?xf32>>) {
  %c42 = constant 42 : index
  %0 = fir.shape %c42 : (index) -> !fir.shape<1>
  // expected-error@+1{{op input and output element types must match for intrinsic types}}
  %1 = fir.rebox %arg0(%0) : (!fir.box<!fir.array<?x?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
  return
}

// -----

func @array_access(%arr : !fir.ref<!fir.array<?x?xf32>>) {
  %c1 = constant 1 : index
  %c100 = constant 100 : index
  %c50 = constant 50 : index
  %shape = fir.shape %c100, %c50 : (index, index) -> !fir.shape<2>
  // expected-error@+1 {{'fir.array_coor' op operand #0 must be any reference or box, but got 'index'}}
  %p = fir.array_coor %c100(%shape) %c1, %c1 : (index, !fir.shape<2>, index, index) -> !fir.ref<f32>
  return
}

// -----

func @array_access(%arr : !fir.ref<f32>) {
  %c1 = constant 1 : index
  %c100 = constant 100 : index
  %c50 = constant 50 : index
  %shape = fir.shape %c100, %c50 : (index, index) -> !fir.shape<2>
  // expected-error@+1 {{'fir.array_coor' op must be a reference to an array}}
  %p = fir.array_coor %arr(%shape) %c1, %c1 : (!fir.ref<f32>, !fir.shape<2>, index, index) -> !fir.ref<f32>
  return
}

// -----

func @array_access(%arr : !fir.ref<!fir.array<?x?xf32>>) {
  %c1 = constant 1 : index
  %c100 = constant 100 : index
  %c50 = constant 50 : index
  %shape = fir.shape %c100, %c50 : (index, index) -> !fir.shape<2>
  %c47 = constant 47 : index
  %c78 = constant 78 : index
  %c3 = constant 3 : index
  %slice = fir.slice %c47, %c78, %c3 : (index,index,index) -> !fir.slice<1>
  // expected-error@+1 {{'fir.array_coor' op rank of dimension in slice mismatched}}
  %p = fir.array_coor %arr(%shape)[%slice] %c1, %c1 : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, !fir.slice<1>, index, index) -> !fir.ref<f32>
  return
}

// -----

func @array_access(%arr : !fir.ref<!fir.array<?x?xf32>>) {
  %c1 = constant 1 : index
  %c100 = constant 100 : index
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  // expected-error@+1 {{'fir.array_coor' op rank of dimension mismatched}}
  %p = fir.array_coor %arr(%shape) %c1, %c1 : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<1>, index, index) -> !fir.ref<f32>
  return
}

// -----

func @array_access(%arr : !fir.ref<!fir.array<?x?xf32>>) {
  %c1 = constant 1 : index
  %c100 = constant 100 : index
  %shift = fir.shift %c1 : (index) -> !fir.shift<1>
  // expected-error@+1 {{'fir.array_coor' op shift can only be provided with fir.box memref}}
  %p = fir.array_coor %arr(%shift) %c1, %c1 : (!fir.ref<!fir.array<?x?xf32>>, !fir.shift<1>, index, index) -> !fir.ref<f32>
  return
}

// -----

func @array_access(%arr : !fir.ref<!fir.array<?x?xf32>>) {
  %c1 = constant 1 : index
  %c100 = constant 100 : index
  %c50 = constant 50 : index
  %shape = fir.shape %c100, %c50 : (index, index) -> !fir.shape<2>
  // expected-error@+1 {{'fir.array_coor' op number of indices do not match dim rank}}
  %p = fir.array_coor %arr(%shape) %c1 : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, index) -> !fir.ref<f32>
  return
}

// -----

func @test_misc_ops(%arr1 : !fir.ref<!fir.array<?x?xf32>>, %m : index, %n : index, %o : index, %p : index) {
  %c2 = constant 2 : index
  %s = fir.shape_shift %m, %n, %o, %p : (index, index, index, index) -> !fir.shapeshift<2>
  // expected-error@+1 {{'fir.array_load' op operand #0 must be any reference or box, but got 'index'}}
  %av1 = fir.array_load %c2(%s) : (index, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  return
}

// -----

func @test_misc_ops(%arr1 : !fir.ref<f32>, %m : index, %n : index, %o : index, %p : index) {
  %s = fir.shape_shift %m, %n, %o, %p : (index, index, index, index) -> !fir.shapeshift<2>
  // expected-error@+1 {{'fir.array_load' op must be a reference to an array}}
  %av1 = fir.array_load %arr1(%s) : (!fir.ref<f32>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  return
}

// -----

func @test_misc_ops(%arr1 : !fir.ref<!fir.array<?x?xf32>>, %m : index, %n : index, %o : index, %p : index) {
  %s = fir.shape_shift %m, %n: (index, index) -> !fir.shapeshift<1>
  // expected-error@+1 {{'fir.array_load' op rank of dimension mismatched}}
  %av1 = fir.array_load %arr1(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<1>) -> !fir.array<?x?xf32>
  return
}

// -----

func @test_misc_ops(%arr1 : !fir.ref<!fir.array<?x?xf32>>, %m : index, %n : index, %o : index, %p : index) {
  %c2 = constant 2 : index
  %shift = fir.shift %c2 : (index) -> !fir.shift<1>
  // expected-error@+1 {{'fir.array_load' op shift can only be provided with fir.box memref}}
  %av1 = fir.array_load %arr1(%shift) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shift<1>) -> !fir.array<?x?xf32>
  return
}

// -----

func @test_misc_ops(%arr1 : !fir.ref<!fir.array<?x?xf32>>, %m : index, %n : index, %o : index, %p : index) {
  %c47 = constant 47 : index
  %c78 = constant 78 : index
  %c3 = constant 3 : index
  %slice = fir.slice %c47, %c78, %c3 : (index,index,index) -> !fir.slice<1>
  %s = fir.shape_shift %m, %n, %o, %p: (index, index, index, index) -> !fir.shapeshift<2>
  // expected-error@+1 {{'fir.array_load' op rank of dimension in slice mismatched}}
  %av1 = fir.array_load %arr1(%s)[%slice] : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>, !fir.slice<1>) -> !fir.array<?x?xf32>
  return
}

// -----

func @test_coordinate_of(%arr : !fir.ref<!fir.array<?x?xf32>>) {
  %1 = constant 10 : i32
  // expected-error@+1 {{'fir.coordinate_of' op cannot find coordinate with unknown extents}}
  %2 = fir.coordinate_of %arr, %1 : (!fir.ref<!fir.array<?x?xf32>>, i32) -> !fir.ref<f32>
  return
}

// -----

func @test_coordinate_of(%arr : !fir.ref<!fir.array<*:f32>>) {
  %1 = constant 10 : i32
  // expected-error@+1 {{'fir.coordinate_of' op cannot find coordinate in unknown shape}}
  %2 = fir.coordinate_of %arr, %1 : (!fir.ref<!fir.array<*:f32>>, i32) -> !fir.ref<f32>
  return
}

// -----

func @test_coordinate_of(%arr : !fir.ref<!fir.char<10>>) {
  %1 = constant 10 : i32
  // expected-error@+1 {{'fir.coordinate_of' op cannot apply coordinate_of to this type}}
  %2 = fir.coordinate_of %arr, %1 : (!fir.ref<!fir.char<10>>, i32) -> !fir.ref<f32>
  return
}

// -----

%0 = constant 22 : i32
// expected-error@+1 {{'fir.embox' op operand #0 must be any reference, but got 'i32'}}
%1 = fir.embox %0 : (i32) -> !fir.box<i32>

// -----

func @fun(%0 : !fir.ref<i32>) {
  %c_100 = constant 100 : index
  %1 = fir.shape %c_100 : (index) -> !fir.shape<1>
  // expected-error@+1 {{'fir.embox' op shape must not be provided for a scalar}}
  %2 = fir.embox %0(%1) : (!fir.ref<i32>, !fir.shape<1>) -> !fir.box<i32>
}

// -----

func @fun(%0 : !fir.ref<i32>) {
  %c_100 = constant 100 : index
  %1 = fir.slice %c_100, %c_100, %c_100 : (index, index, index) -> !fir.slice<1>
  // expected-error@+1 {{'fir.embox' op operand #1 must be any legal shape type, but got '!fir.slice<1>'}}
  %2 = fir.embox %0(%1) : (!fir.ref<i32>, !fir.slice<1>) -> !fir.box<i32>
}

// -----

func @fun(%0 : !fir.ref<i32>) {
  %c_100 = constant 100 : index
  %1 = fir.shape %c_100 : (index) -> !fir.shape<1>
  // expected-error@+1 {{'fir.embox' op operand #1 must be FIR slice, but got '!fir.shape<1>'}}
  %2 = fir.embox %0[%1] : (!fir.ref<i32>, !fir.shape<1>) -> !fir.box<i32>
}

// -----

func @fun(%0 : !fir.ref<i32>) {
  %c_100 = constant 100 : index
  %1 = fir.slice %c_100, %c_100, %c_100 : (index, index, index) -> !fir.slice<1>
  // expected-error@+1 {{'fir.embox' op slice must not be provided for a scalar}}
  %2 = fir.embox %0[%1] : (!fir.ref<i32>, !fir.slice<1>) -> !fir.box<i32>
}

// -----

%lo = constant 1 : index
%c1 = constant 1 : index
%up = constant 10 : index
%okIn = constant 1 : i1
%shIn = constant 1 : i16
// expected-error@+1 {{'fir.iterate_while' op expected body first argument to be an index argument for the induction variable}}
%v:3 = fir.iterate_while (%i = %lo to %up step %c1) and (%ok = %okIn) iter_args(%sh = %shIn) -> (i16, i1, i16) {
  %shNew = fir.call @bar(%sh) : (i16) -> i16
  %okNew = fir.call @foo(%sh) : (i16) -> i1
  fir.result %shNew, %okNew, %shNew : i16, i1, i16
}

// -----

%lo = constant 1 : index
%c1 = constant 1 : index
%up = constant 10 : index
%okIn = constant 1 : i1
%shIn = constant 1 : i16
// expected-error@+1 {{'fir.iterate_while' op expected body second argument to be an index argument for the induction variable}}
%v:3 = fir.iterate_while (%i = %lo to %up step %c1) and (%ok = %okIn) iter_args(%sh = %shIn) -> (index, f32, i16) {
  %shNew = fir.call @bar(%sh) : (i16) -> i16
  %dummy = fir.call @foo(%sh) : (i16) -> f32
  fir.result %i, %dummy, %shNew : index, f32, i16
}

// -----

%c1 = constant 1 : index
%c10 = constant 10 : index
// expected-error@+1 {{'fir.do_loop' op unordered loop has no final value}}
fir.do_loop %i = %c1 to %c10 step %c1 unordered -> index {
}

// -----

%c1 = constant 1 : index
%c10 = constant 10 : index
fir.do_loop %i = %c1 to %c10 step %c1 -> index {
  %f1 = constant 1.0 : f32
  // expected-error@+1 {{'fir.result' op types mismatch between result op and its parent}}
  fir.result %f1 : f32
}

// -----

%c1 = constant 1 : index
%c10 = constant 10 : index
// expected-error@+1 {{'fir.result' op parent of result must have same arity}}
fir.do_loop %i = %c1 to %c10 step %c1 -> index {
}
