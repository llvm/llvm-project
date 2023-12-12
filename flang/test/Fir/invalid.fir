// FIR ops diagnotic tests

// RUN: fir-opt -split-input-file -verify-diagnostics %s

// expected-error@+1{{custom op 'fir.string_lit' must have character type}}
%0 = fir.string_lit "Hello, World!"(13) : !fir.int<32>

// -----

// expected-error@+1{{custom op 'fir.string_lit' found an invalid constant}}
%0 = fir.string_lit 20(13) : !fir.int<32>

// -----

// expected-error@+1{{'fir.string_lit' op values in initializer must be integers}}
%2 = fir.string_lit [158, 2.0](2) : !fir.char<2>

// -----

func.func @bad_rebox_1(%arg0: !fir.ref<!fir.array<?x?xf32>>) {
  %c10 = arith.constant 10 : index
  %0 = fir.shape %c10 : (index) -> !fir.shape<1>
  // expected-error@+1{{op operand #0 must be box or class, but got '!fir.ref<!fir.array<?x?xf32>>'}}
  %1 = fir.rebox %arg0(%0) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  return
}

// -----

func.func @bad_rebox_2(%arg0: !fir.box<!fir.array<?x?xf32>>) {
  %c10 = arith.constant 10 : index
  %0 = fir.shape %c10 : (index) -> !fir.shape<1>
  // expected-error@+1{{op result #0 must be box or class, but got '!fir.ref<!fir.array<?xf32>>'}}
  %1 = fir.rebox %arg0(%0) : (!fir.box<!fir.array<?x?xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xf32>>
  return
}

// -----

func.func @bad_rebox_3(%arg0: !fir.box<!fir.array<*:f32>>) {
  %c10 = arith.constant 10 : index
  %0 = fir.shape %c10 : (index) -> !fir.shape<1>
  // expected-error@+1{{op box operand must not have unknown rank or type}}
  %1 = fir.rebox %arg0(%0) : (!fir.box<!fir.array<*:f32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  return
}

// -----

func.func @bad_rebox_4(%arg0: !fir.box<!fir.array<?xf32>>) {
  // expected-error@+1{{op result type must not have unknown rank or type}}
  %0 = fir.rebox %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<*:f32>>
  return
}

// -----

func.func @bad_rebox_5(%arg0: !fir.box<!fir.array<?x?xf32>>) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = fir.slice %c1, %c10, %c1 : (index, index, index) -> !fir.slice<1>
  // expected-error@+1{{op slice operand rank must match box operand rank}}
  %1 = fir.rebox %arg0 [%0] : (!fir.box<!fir.array<?x?xf32>>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  return
}

// -----

func.func @bad_rebox_6(%arg0: !fir.box<!fir.array<?xf32>>) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = fir.slice %c1, %c10, %c1 : (index, index, index) -> !fir.slice<1>
  %1 = fir.shift %c1, %c1 : (index, index) -> !fir.shift<2>
  // expected-error@+1{{shape operand and input box ranks must match when there is a slice}}
  %2 = fir.rebox %arg0(%1) [%0] : (!fir.box<!fir.array<?xf32>>, !fir.shift<2>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  return
}

// -----

func.func @bad_rebox_7(%arg0: !fir.box<!fir.array<?xf32>>) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = fir.slice %c1, %c10, %c1 : (index, index, index) -> !fir.slice<1>
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  // expected-error@+1{{shape operand must absent or be a fir.shift when there is a slice}}
  %2 = fir.rebox %arg0(%1) [%0] : (!fir.box<!fir.array<?xf32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  return
}

// -----

func.func @bad_rebox_8(%arg0: !fir.box<!fir.array<?x?xf32>>) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %undef = fir.undefined index
  %0 = fir.slice %c1, %undef, %undef, %c1, %c10, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  // expected-error@+1{{result type rank and rank after applying slice operand must match}}
  %1 = fir.rebox %arg0 [%0] : (!fir.box<!fir.array<?x?xf32>>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xf32>>
  return
}

// -----

func.func @bad_rebox_9(%arg0: !fir.box<!fir.array<?xf32>>) {
  %c10 = arith.constant 10 : index
  %0 = fir.shift %c10, %c10 : (index, index) -> !fir.shift<2>
  // expected-error@+1{{shape operand and input box ranks must match when the shape is a fir.shift}}
  %1 = fir.rebox %arg0(%0) : (!fir.box<!fir.array<?xf32>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?xf32>>
  return
}

// -----

func.func @bad_rebox_10(%arg0: !fir.box<!fir.array<?xf32>>) {
  %c10 = arith.constant 10 : index
  %0 = fir.shape %c10, %c10 : (index, index) -> !fir.shape<2>
  // expected-error@+1{{result type and shape operand ranks must match}}
  %1 = fir.rebox %arg0(%0) : (!fir.box<!fir.array<?xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<?xf32>>
  return
}

// -----

func.func @bad_rebox_11(%arg0: !fir.box<!fir.array<?x?xf32>>) {
  %c42 = arith.constant 42 : index
  %0 = fir.shape %c42 : (index) -> !fir.shape<1>
  // expected-error@+1{{op input and output element types must match for intrinsic types}}
  %1 = fir.rebox %arg0(%0) : (!fir.box<!fir.array<?x?xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xf64>>
  return
}

// -----

func.func @test_rebox_char(%arg0: !fir.box<!fir.array<?x!fir.char<1,20>>>) {
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10, %c10 : (index, index) -> !fir.shape<2>
  // expected-error@+1{{op input and output element types must match for intrinsic types}}
  %2 = fir.rebox %arg0(%1) : (!fir.box<!fir.array<?x!fir.char<1,20>>>, !fir.shape<2>) -> !fir.box<!fir.array<10x10x!fir.char<1,10>>>
  return
}

// -----

func.func @array_access(%arr : !fir.ref<!fir.array<?x?xf32>>) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c50 = arith.constant 50 : index
  %shape = fir.shape %c100, %c50 : (index, index) -> !fir.shape<2>
  // expected-error@+1 {{'fir.array_coor' op operand #0 must be any reference or box, but got 'index'}}
  %p = fir.array_coor %c100(%shape) %c1, %c1 : (index, !fir.shape<2>, index, index) -> !fir.ref<f32>
  return
}

// -----

func.func @array_access(%arr : !fir.ref<f32>) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c50 = arith.constant 50 : index
  %shape = fir.shape %c100, %c50 : (index, index) -> !fir.shape<2>
  // expected-error@+1 {{'fir.array_coor' op must be a reference to an array}}
  %p = fir.array_coor %arr(%shape) %c1, %c1 : (!fir.ref<f32>, !fir.shape<2>, index, index) -> !fir.ref<f32>
  return
}

// -----

func.func @array_access(%arr : !fir.ref<!fir.array<?x?xf32>>) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c50 = arith.constant 50 : index
  %shape = fir.shape %c100, %c50 : (index, index) -> !fir.shape<2>
  %c47 = arith.constant 47 : index
  %c78 = arith.constant 78 : index
  %c3 = arith.constant 3 : index
  %slice = fir.slice %c47, %c78, %c3 : (index,index,index) -> !fir.slice<1>
  // expected-error@+1 {{'fir.array_coor' op rank of dimension in slice mismatched}}
  %p = fir.array_coor %arr(%shape)[%slice] %c1, %c1 : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, !fir.slice<1>, index, index) -> !fir.ref<f32>
  return
}

// -----

func.func @array_access(%arr : !fir.ref<!fir.array<?x?xf32>>) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  // expected-error@+1 {{'fir.array_coor' op rank of dimension mismatched}}
  %p = fir.array_coor %arr(%shape) %c1, %c1 : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<1>, index, index) -> !fir.ref<f32>
  return
}

// -----

func.func @array_access(%arr : !fir.ref<!fir.array<?x?xf32>>) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %shift = fir.shift %c1 : (index) -> !fir.shift<1>
  // expected-error@+1 {{'fir.array_coor' op shift can only be provided with fir.box memref}}
  %p = fir.array_coor %arr(%shift) %c1, %c1 : (!fir.ref<!fir.array<?x?xf32>>, !fir.shift<1>, index, index) -> !fir.ref<f32>
  return
}

// -----

func.func @array_access(%arr : !fir.ref<!fir.array<?x?xf32>>) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c50 = arith.constant 50 : index
  %shape = fir.shape %c100, %c50 : (index, index) -> !fir.shape<2>
  // expected-error@+1 {{'fir.array_coor' op number of indices do not match dim rank}}
  %p = fir.array_coor %arr(%shape) %c1 : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, index) -> !fir.ref<f32>
  return
}

// -----

func.func @test_misc_ops(%arr1 : !fir.ref<!fir.array<?x?xf32>>, %m : index, %n : index, %o : index, %p : index) {
  %c2 = arith.constant 2 : index
  %s = fir.shape_shift %m, %n, %o, %p : (index, index, index, index) -> !fir.shapeshift<2>
  // expected-error@+1 {{'fir.array_load' op operand #0 must be any reference or box, but got 'index'}}
  %av1 = fir.array_load %c2(%s) : (index, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  return
}

// -----

func.func @test_misc_ops(%arr1 : !fir.ref<f32>, %m : index, %n : index, %o : index, %p : index) {
  %s = fir.shape_shift %m, %n, %o, %p : (index, index, index, index) -> !fir.shapeshift<2>
  // expected-error@+1 {{'fir.array_load' op must be a reference to an array}}
  %av1 = fir.array_load %arr1(%s) : (!fir.ref<f32>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  return
}

// -----

func.func @test_misc_ops(%arr1 : !fir.ref<!fir.array<?x?xf32>>, %m : index, %n : index, %o : index, %p : index) {
  %s = fir.shape_shift %m, %n: (index, index) -> !fir.shapeshift<1>
  // expected-error@+1 {{'fir.array_load' op rank of dimension mismatched}}
  %av1 = fir.array_load %arr1(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<1>) -> !fir.array<?x?xf32>
  return
}

// -----

func.func @test_misc_ops(%arr1 : !fir.ref<!fir.array<?x?xf32>>, %m : index, %n : index, %o : index, %p : index) {
  %c2 = arith.constant 2 : index
  %shift = fir.shift %c2 : (index) -> !fir.shift<1>
  // expected-error@+1 {{'fir.array_load' op shift can only be provided with fir.box memref}}
  %av1 = fir.array_load %arr1(%shift) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shift<1>) -> !fir.array<?x?xf32>
  return
}

// -----

func.func @test_misc_ops(%arr1 : !fir.ref<!fir.array<?x?xf32>>, %m : index, %n : index, %o : index, %p : index) {
  %c47 = arith.constant 47 : index
  %c78 = arith.constant 78 : index
  %c3 = arith.constant 3 : index
  %slice = fir.slice %c47, %c78, %c3 : (index,index,index) -> !fir.slice<1>
  %s = fir.shape_shift %m, %n, %o, %p: (index, index, index, index) -> !fir.shapeshift<2>
  // expected-error@+1 {{'fir.array_load' op rank of dimension in slice mismatched}}
  %av1 = fir.array_load %arr1(%s)[%slice] : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>, !fir.slice<1>) -> !fir.array<?x?xf32>
  return
}

// -----

func.func @test_coordinate_of(%arr : !fir.ref<!fir.array<?x?xf32>>) {
  %1 = arith.constant 10 : i32
  // expected-error@+1 {{'fir.coordinate_of' op cannot find coordinate with unknown extents}}
  %2 = fir.coordinate_of %arr, %1 : (!fir.ref<!fir.array<?x?xf32>>, i32) -> !fir.ref<f32>
  return
}

// -----

func.func @test_coordinate_of(%arr : !fir.ref<!fir.array<*:f32>>) {
  %1 = arith.constant 10 : i32
  // expected-error@+1 {{'fir.coordinate_of' op cannot find coordinate in unknown shape}}
  %2 = fir.coordinate_of %arr, %1 : (!fir.ref<!fir.array<*:f32>>, i32) -> !fir.ref<f32>
  return
}

// -----

func.func @test_coordinate_of(%arr : !fir.ref<!fir.char<10>>) {
  %1 = arith.constant 10 : i32
  // expected-error@+1 {{'fir.coordinate_of' op cannot apply to this element type}}
  %2 = fir.coordinate_of %arr, %1 : (!fir.ref<!fir.char<10>>, i32) -> !fir.ref<!fir.char<10>>
  return
}

// -----

func.func @test_coordinate_of(%arr : !fir.ref<!fir.char<10, 2>>) {
  %1 = arith.constant 10 : i32
  // expected-error@+1 {{'fir.coordinate_of' op cannot apply to character singleton}}
  %2 = fir.coordinate_of %arr, %1, %1 : (!fir.ref<!fir.char<10, 2>>, i32, i32) -> !fir.ref<!fir.char<10>>
  return
}

// -----

%0 = arith.constant 22 : i32
// expected-error@+1 {{'fir.embox' op operand #0 must be any reference, but got 'i32'}}
%1 = fir.embox %0 : (i32) -> !fir.box<i32>

// -----

func.func @fun(%0 : !fir.ref<i32>) {
  %c_100 = arith.constant 100 : index
  %1 = fir.shape %c_100 : (index) -> !fir.shape<1>
  // expected-error@+1 {{'fir.embox' op shape must not be provided for a scalar}}
  %2 = fir.embox %0(%1) : (!fir.ref<i32>, !fir.shape<1>) -> !fir.box<i32>
}

// -----

func.func @fun(%0 : !fir.ref<i32>) {
  %c_100 = arith.constant 100 : index
  %1 = fir.slice %c_100, %c_100, %c_100 : (index, index, index) -> !fir.slice<1>
  // expected-error@+1 {{'fir.embox' op operand #1 must be any legal shape type, but got '!fir.slice<1>'}}
  %2 = fir.embox %0(%1) : (!fir.ref<i32>, !fir.slice<1>) -> !fir.box<i32>
}

// -----

func.func @fun(%0 : !fir.ref<i32>) {
  %c_100 = arith.constant 100 : index
  %1 = fir.shape %c_100 : (index) -> !fir.shape<1>
  // expected-error@+1 {{'fir.embox' op operand #1 must be FIR slice, but got '!fir.shape<1>'}}
  %2 = fir.embox %0[%1] : (!fir.ref<i32>, !fir.shape<1>) -> !fir.box<i32>
}

// -----

func.func @fun(%0 : !fir.ref<i32>) {
  %c_100 = arith.constant 100 : index
  %1 = fir.slice %c_100, %c_100, %c_100 : (index, index, index) -> !fir.slice<1>
  // expected-error@+1 {{'fir.embox' op slice must not be provided for a scalar}}
  %2 = fir.embox %0[%1] : (!fir.ref<i32>, !fir.slice<1>) -> !fir.box<i32>
}

// -----

func.func @embox_tdesc(%arg0: !fir.class<!fir.array<10x!fir.type<derived{a:i32,b:i32}>>>) {
  %0 = fir.alloca i32
  %c1_i32 = arith.constant 1 : i32
  %1 = fir.convert %c1_i32 : (i32) -> index
  %c10_i32 = arith.constant 10 : i32
  %2 = fir.convert %c10_i32 : (i32) -> index
  %c1 = arith.constant 1 : index
  %3 = fir.convert %1 : (index) -> i32
  %4:2 = fir.do_loop %arg2 = %1 to %2 step %c1 iter_args(%arg3 = %3) -> (index, i32) {
    fir.store %arg3 to %0 : !fir.ref<i32>
    %9 = fir.load %0 : !fir.ref<i32>
    %10 = fir.convert %9 : (i32) -> i64
    %c1_i64 = arith.constant 1 : i64
    %11 = arith.subi %10, %c1_i64 : i64
    %12 = fir.coordinate_of %arg0, %11 : (!fir.class<!fir.array<10x!fir.type<derived{a:i32,b:i32}>>>, i64) -> !fir.ref<!fir.type<derived{a:i32,b:i32}>>
    // expected-error@+1 {{'fir.embox' op source_box must be used with fir.class result type}}
    %13 = fir.embox %12 source_box %arg0 : (!fir.ref<!fir.type<derived{a:i32,b:i32}>>, !fir.class<!fir.array<10x!fir.type<derived{a:i32,b:i32}>>>) -> !fir.box<!fir.type<derived{a:i32,b:i32}>>
    %14 = arith.addi %arg2, %c1 : index
    %15 = fir.convert %c1 : (index) -> i32
    %16 = fir.load %0 : !fir.ref<i32>
    %17 = arith.addi %16, %15 : i32
    fir.result %14, %17 : index, i32
  }
  fir.store %4#1 to %0 : !fir.ref<i32>
  return
}

// -----

%lo = arith.constant 1 : index
%c1 = arith.constant 1 : index
%up = arith.constant 10 : index
%okIn = arith.constant 1 : i1
%shIn = arith.constant 1 : i16
// expected-error@+1 {{'fir.iterate_while' op expected body first argument to be an index argument for the induction variable}}
%v:3 = fir.iterate_while (%i = %lo to %up step %c1) and (%ok = %okIn) iter_args(%sh = %shIn) -> (i16, i1, i16) {
  %shNew = fir.call @bar(%sh) : (i16) -> i16
  %okNew = fir.call @foo(%sh) : (i16) -> i1
  fir.result %shNew, %okNew, %shNew : i16, i1, i16
}

// -----

%lo = arith.constant 1 : index
%c1 = arith.constant 1 : index
%up = arith.constant 10 : index
%okIn = arith.constant 1 : i1
%shIn = arith.constant 1 : i16
// expected-error@+1 {{'fir.iterate_while' op expected body second argument to be an index argument for the induction variable}}
%v:3 = fir.iterate_while (%i = %lo to %up step %c1) and (%ok = %okIn) iter_args(%sh = %shIn) -> (index, f32, i16) {
  %shNew = fir.call @bar(%sh) : (i16) -> i16
  %dummy = fir.call @foo(%sh) : (i16) -> f32
  fir.result %i, %dummy, %shNew : index, f32, i16
}

// -----

%c1 = arith.constant 1 : index
%c10 = arith.constant 10 : index
// expected-error@+1 {{'fir.do_loop' op unordered loop has no final value}}
fir.do_loop %i = %c1 to %c10 step %c1 unordered -> index {
}

// -----

%c1 = arith.constant 1 : index
%c10 = arith.constant 10 : index
fir.do_loop %i = %c1 to %c10 step %c1 -> index {
  %f1 = arith.constant 1.0 : f32
  // expected-error@+1 {{'fir.result' op types mismatch between result op and its parent}}
  fir.result %f1 : f32
}

// -----

%c1 = arith.constant 1 : index
%c10 = arith.constant 10 : index
// expected-error@+1 {{'fir.result' op parent of result must have same arity}}
fir.do_loop %i = %c1 to %c10 step %c1 -> index {
}

// -----

func.func @ugly_char_convert() {
  %1 = fir.undefined i32
  %2 = fir.undefined !fir.ref<!fir.char<1>>
  %3 = fir.undefined !fir.ref<!fir.array<?x!fir.char<1>>>
  // expected-error@+1 {{'fir.char_convert' op buffers must have different KIND values}}
  fir.char_convert %2 for %1 to %3 : !fir.ref<!fir.char<1>>, i32, !fir.ref<!fir.array<?x!fir.char<1>>>
  return
}

// -----

func.func @ugly_char_convert() {
  %1 = fir.undefined i32
  %2 = fir.undefined !fir.ref<!fir.char<1>>
  %3 = fir.undefined !fir.ref<!fir.array<?xf32>>
  // expected-error@+1 {{'fir.char_convert' op not a reference to a character}}
  fir.char_convert %2 for %1 to %3 : !fir.ref<!fir.char<1>>, i32, !fir.ref<!fir.array<?xf32>>
  return
}

// -----

func.func @ugly_char_convert() {
  %1 = fir.undefined i32
  %2 = fir.undefined !fir.ref<!fir.char<1>>
  %3 = fir.undefined !fir.ref<!fir.array<?x!fir.char<2,?>>>
  // expected-error@+1 {{'fir.char_convert' op operand #0 must be any reference, but got 'i32'}}
  fir.char_convert %1 for %1 to %3 : i32, i32, !fir.ref<!fir.array<?x!fir.char<2,?>>>
  return
}

// -----

func.func @ugly_char_convert() {
  %1 = fir.undefined i32
  %2 = fir.undefined !fir.ref<!fir.char<1>>
  %3 = fir.undefined !fir.ref<!fir.array<?x!fir.char<2,?>>>
  // expected-error@+1 {{'fir.char_convert' op operand #1 must be any integer, but got '!fir.ref<!fir.char<1>>'}}
  fir.char_convert %2 for %2 to %3 : !fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>, !fir.ref<!fir.array<?x!fir.char<2,?>>>
  return
}

// -----

fir.global internal @_QEmultiarray : !fir.array<32x32xi32> {
  %c0_i32 = arith.constant 1 : i32
  %0 = fir.undefined !fir.array<32x32xi32>
  // expected-error@+1 {{'fir.insert_on_range' op has uneven number of values in ranges}}
  %2 = "fir.insert_on_range"(%0, %c0_i32) { coor = dense<[0, 31, 0]> : tensor<3xindex> } : (!fir.array<32x32xi32>, i32) -> !fir.array<32x32xi32>
  fir.has_value %2 : !fir.array<32x32xi32>
}

// -----

fir.global internal @_QEmultiarray : !fir.array<32x32xi32> {
  %c0_i32 = arith.constant 1 : i32
  %0 = fir.undefined !fir.array<32x32xi32>
  // expected-error@+1 {{'fir.insert_on_range' op has uneven number of values in ranges}}
  %2 = "fir.insert_on_range"(%0, %c0_i32) { coor = dense<[0]> : tensor<1xindex> }  : (!fir.array<32x32xi32>, i32) -> !fir.array<32x32xi32>
  fir.has_value %2 : !fir.array<32x32xi32>
}

// -----

fir.global internal @_QEmultiarray : !fir.array<32x32xi32> {
  %c0_i32 = arith.constant 1 : i32
  %0 = fir.undefined !fir.array<32x32xi32>
  // expected-error@+1 {{'fir.insert_on_range' op negative range bound}}
  %2 = fir.insert_on_range %0, %c0_i32 from (-1) to (0) : (!fir.array<32x32xi32>, i32) -> !fir.array<32x32xi32>
  fir.has_value %2 : !fir.array<32x32xi32>
}

// -----

fir.global internal @_QEmultiarray : !fir.array<32x32xi32> {
  %c0_i32 = arith.constant 1 : i32
  %0 = fir.undefined !fir.array<32x32xi32>
  // expected-error@+1 {{'fir.insert_on_range' op empty range}}
  %2 = fir.insert_on_range %0, %c0_i32 from (10) to (9) : (!fir.array<32x32xi32>, i32) -> !fir.array<32x32xi32>
  fir.has_value %2 : !fir.array<32x32xi32>
}

// -----

fir.global internal @_QEmultiarray : !fir.array<?xi32> {
  %c0_i32 = arith.constant 1 : i32
  %0 = fir.undefined !fir.array<?xi32>
  // expected-error@+1 {{'fir.insert_on_range' op must have constant shape and size}}
  %2 = fir.insert_on_range %0, %c0_i32 from (0) to (10) : (!fir.array<?xi32>, i32) -> !fir.array<?xi32>
  fir.has_value %2 : !fir.array<?xi32>
}

// -----

fir.global internal @_QEmultiarray : !fir.array<*:i32> {
  %c0_i32 = arith.constant 1 : i32
  %0 = fir.undefined !fir.array<*:i32>
  // expected-error@+1 {{'fir.insert_on_range' op must have constant shape and size}}
  %2 = fir.insert_on_range %0, %c0_i32 from (0) to (10) : (!fir.array<*:i32>, i32) -> !fir.array<*:i32>
  fir.has_value %2 : !fir.array<*:i32>
}

// -----

func.func @bad_save_result(%buffer : !fir.ref<!fir.array<?xf64>>, %n :index) {
  %res = fir.call @array_func() : () -> !fir.array<?xf32>
  %shape = fir.shape %n : (index) -> !fir.shape<1>
  // expected-error@+1 {{'fir.save_result' op value type must match memory reference type}}
  fir.save_result %res to %buffer(%shape) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf64>>, !fir.shape<1>
  return
}

// -----

func.func @bad_save_result(%buffer : !fir.ref<!fir.box<!fir.array<*:f32>>>) {
  %res = fir.call @array_func() : () -> !fir.box<!fir.array<*:f32>>
  // expected-error@+1 {{'fir.save_result' op cannot save !fir.box of unknown rank or type}}
  fir.save_result %res to %buffer : !fir.box<!fir.array<*:f32>>, !fir.ref<!fir.box<!fir.array<*:f32>>>
  return
}

// -----

func.func @bad_save_result(%buffer : !fir.ref<f64>) {
  %res = fir.call @array_func() : () -> f64
  // expected-error@+1 {{'fir.save_result' op operand #0 must be fir.box, fir.array or fir.type, but got 'f64'}}
  fir.save_result %res to %buffer : f64, !fir.ref<f64>
  return
}

// -----

func.func @bad_save_result(%buffer : !fir.ref<!fir.box<!fir.array<?xf32>>>, %n : index) {
  %res = fir.call @array_func() : () -> !fir.box<!fir.array<?xf32>>
  %shape = fir.shape %n : (index) -> !fir.shape<1>
  // expected-error@+1 {{'fir.save_result' op must not have shape or length operands if the value is a fir.box}}
  fir.save_result %res to %buffer(%shape) : !fir.box<!fir.array<?xf32>>, !fir.ref<!fir.box<!fir.array<?xf32>>>, !fir.shape<1>
  return
}

// -----

func.func @bad_save_result(%buffer : !fir.ref<!fir.array<?xf32>>, %n :index) {
  %res = fir.call @array_func() : () -> !fir.array<?xf32>
  %shape = fir.shape %n, %n : (index, index) -> !fir.shape<2>
  // expected-error@+1 {{'fir.save_result' op shape operand must be provided and have the value rank when the value is a fir.array}}
  fir.save_result %res to %buffer(%shape) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<2>
  return
}

// -----

func.func @bad_save_result(%buffer : !fir.ref<!fir.type<t{x:f32}>>, %n :index) {
  %res = fir.call @array_func() : () -> !fir.type<t{x:f32}>
  %shape = fir.shape %n : (index) -> !fir.shape<1>
  // expected-error@+1 {{'fir.save_result' op shape operand should only be provided if the value is a fir.array}}
  fir.save_result %res to %buffer(%shape) : !fir.type<t{x:f32}>, !fir.ref<!fir.type<t{x:f32}>>, !fir.shape<1>
  return
}

// -----

func.func @bad_save_result(%buffer : !fir.ref<!fir.type<t{x:f32}>>, %n :index) {
  %res = fir.call @array_func() : () -> !fir.type<t{x:f32}>
  // expected-error@+1 {{'fir.save_result' op length parameters number must match with the value type length parameters}}
  fir.save_result %res to %buffer typeparams %n : !fir.type<t{x:f32}>, !fir.ref<!fir.type<t{x:f32}>>, index
  return
}

// -----

func.func @bad_save_result(%buffer : !fir.ref<!fir.array<?xf32>>, %n :index) {
  %res = fir.call @array_func() : () -> !fir.array<?xf32>
  %shape = fir.shape %n : (index) -> !fir.shape<1>
  // expected-error@+1 {{'fir.save_result' op length parameters must not be provided for this value type}}
  fir.save_result %res to %buffer(%shape) typeparams %n : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index
  return
}

// -----

func.func @test_misc_ops(%arr1 : !fir.ref<!fir.array<?x?xf32>>, %m : index, %n : index, %o : index, %p : index) {
  %s = fir.shape_shift %m, %n, %o, %p : (index, index, index, index) -> !fir.shapeshift<2>
  %av1 = fir.array_load %arr1(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  // expected-error@+1 {{'fir.array_fetch' op number of indices != dimension of array}}
  %f = fir.array_fetch %av1, %m : (!fir.array<?x?xf32>, index) -> f32
  return
}

// -----

func.func @test_misc_ops(%arr1 : !fir.ref<!fir.array<?x?xf32>>, %m : index, %n : index, %o : index, %p : index) {
  %s = fir.shape_shift %m, %n, %o, %p : (index, index, index, index) -> !fir.shapeshift<2>
  %av1 = fir.array_load %arr1(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  // expected-error@+1 {{'fir.array_fetch' op return type does not match array}}
  %f = fir.array_fetch %av1, %m, %n : (!fir.array<?x?xf32>, index, index) -> i32
  return
}

// -----

func.func @test_misc_ops(%arr1 : !fir.ref<!fir.array<?x?xf32>>, %m : index, %n : index, %o : index, %p : index) {
  %s = fir.shape_shift %m, %n, %o, %p : (index, index, index, index) -> !fir.shapeshift<2>
  %av1 = fir.array_load %arr1(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  %f = fir.array_fetch %av1, %m, %n : (!fir.array<?x?xf32>, index, index) -> f32
  // expected-error@+1 {{'fir.array_update' op number of indices != dimension of array}}
  %av2 = fir.array_update %av1, %f, %m : (!fir.array<?x?xf32>, f32, index) -> !fir.array<?x?xf32>
  return
}

// -----

func.func @test_misc_ops(%arr1 : !fir.ref<!fir.array<?x?xf32>>, %m : index, %n : index, %o : index, %p : index) {
  %s = fir.shape_shift %m, %n, %o, %p : (index, index, index, index) -> !fir.shapeshift<2>
  %av1 = fir.array_load %arr1(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  %c0 = arith.constant 0 : i32
  // expected-error@+1 {{'fir.array_update' op merged value does not have element type}}
  %av2 = fir.array_update %av1, %c0, %m, %n : (!fir.array<?x?xf32>, i32, index, index) -> !fir.array<?x?xf32>
  return
}

// -----

func.func @test_misc_ops(%arr1 : !fir.ref<!fir.array<?x?xf32>>, %m : index, %n : index, %o : index, %p : index, %f: !fir.ref<i32>) {
  %s = fir.shape_shift %m, %n, %o, %p : (index, index, index, index) -> !fir.shapeshift<2>
  %av1 = fir.array_load %arr1(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  // expected-error@+1 {{'fir.array_update' op does not support reference type for merge}}
  %av2 = fir.array_update %av1, %f, %m, %n : (!fir.array<?x?xf32>, !fir.ref<i32>, index, index) -> !fir.array<?x?xf32>
  return
}

// -----

func.func @test_misc_ops(%arr1 : !fir.ref<!fir.array<?x?xf32>>, %m : index, %n : index, %o : index, %p : index) {
  %s = fir.shape_shift %m, %n, %o, %p : (index, index, index, index) -> !fir.shapeshift<2>
  %av1 = fir.array_load %arr1(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  %f = fir.array_fetch %av1, %m, %n : (!fir.array<?x?xf32>, index, index) -> f32
  %av2 = fir.array_update %av1, %f, %m, %n : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
  // expected-error@+1 {{'fir.array_merge_store' op operand #0 must be result of a fir.array_load op}}
  fir.array_merge_store %av2, %av2 to %arr1 : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.ref<!fir.array<?x?xf32>>
  return
}

// -----

func.func @bad_array_modify(%arr1 : !fir.ref<!fir.array<?x?xf32>>, %m : index, %n : index, %o : index, %p : index, %f : f32) {
  %i10 = arith.constant 10 : index
  %j20 = arith.constant 20 : index
  %s = fir.shape_shift %m, %n, %o, %p : (index, index, index, index) -> !fir.shapeshift<2>
  %av1 = fir.array_load %arr1(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  // expected-error@+1 {{'fir.array_modify' op number of indices must match array dimension}}
  %addr, %av2 = fir.array_modify %av1, %i10 : (!fir.array<?x?xf32>, index) -> (!fir.ref<f32>, !fir.array<?x?xf32>)
  fir.store %f to %addr : !fir.ref<f32>
  fir.array_merge_store %av1, %av2 to %arr1 : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.ref<!fir.array<?x?xf32>>
  return
}

// -----

func.func @slice_must_be_integral() {
  %0 = arith.constant 42 : i32
  %1 = fir.field_index field, !fir.type<t(param:i32){field:i32}> (%0 : i32)
  // expected-error@+1 {{'fir.slice' op operand #0 must be variadic of any integer, but got '!fir.field'}}
  %2 = fir.slice %1, %1, %1 : (!fir.field, !fir.field, !fir.field) -> !fir.slice<1>
  return
}

// -----

func.func @array_coor_no_slice_substr(%a : !fir.ref<!fir.array<?x?xf32>>) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %slice = fir.slice %c1, %c10, %c1 substr %c1, %c10 : (index, index, index, index, index) -> !fir.slice<1>
  // expected-error@+1 {{'fir.array_coor' op array_coor cannot take a slice with substring}}
  %p = fir.array_coor %a[%slice] %c1, %c1 : (!fir.ref<!fir.array<?x?xf32>>, !fir.slice<1>, index, index) -> !fir.ref<f32>
  return
}

// -----

func.func @array_coor_no_slice_substr(%a : !fir.ref<!fir.array<?x?xf32>>) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %slice = fir.slice %c1, %c10, %c1 substr %c1, %c10 : (index, index, index, index, index) -> !fir.slice<1>
  // expected-error@+1 {{'fir.array_load' op array_load cannot take a slice with substring}}
  %v = fir.array_load %a[%slice] : (!fir.ref<!fir.array<?x?xf32>>, !fir.slice<1>) -> !fir.array<?x?xf32>
  return
}

// -----

func.func @array_merge_store_no_slice_substr(%arr1 : !fir.ref<!fir.array<?x?xf32>>, %m : index, %n : index, %o : index, %p : index, %f : f32) {
  %i10 = arith.constant 10 : index
  %j20 = arith.constant 20 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %s = fir.shape_shift %m, %n, %o, %p : (index, index, index, index) -> !fir.shapeshift<2>
  %slice = fir.slice %c1, %c10, %c1 substr %c1, %c10 : (index, index, index, index, index) -> !fir.slice<1>
  %av1 = fir.array_load %arr1(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  %addr, %av2 = fir.array_modify %av1, %i10, %i10 : (!fir.array<?x?xf32>, index, index) -> (!fir.ref<f32>, !fir.array<?x?xf32>)
  fir.store %f to %addr : !fir.ref<f32>
  // expected-error@+1 {{'fir.array_merge_store' op array_merge_store cannot take a slice with substring}}
  fir.array_merge_store %av1, %av2 to %arr1[%slice] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.ref<!fir.array<?x?xf32>>, !fir.slice<1>
  return
}

// -----

func.func @array_access(%a : !fir.ref<!fir.array<?x?xf32>>) {
  %c1 = arith.constant 1 : index
  %n = arith.constant 0 : index
  %m = arith.constant 50 : index
  %s = fir.shape %n, %m : (index, index) -> !fir.shape<2>
  %v = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
  // expected-error@+1 {{'fir.array_access' op number of indices != dimension of array}}
  %p = fir.array_access %v, %c1 : (!fir.array<?x?xf32>, index) -> !fir.ref<f32>
  return
}

// -----

func.func @array_access(%a : !fir.ref<!fir.array<?x?xf32>>) {
  %c1 = arith.constant 1 : index
  %n = arith.constant 0 : index
  %m = arith.constant 50 : index
  %s = fir.shape %n, %m : (index, index) -> !fir.shape<2>
  %v = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
  // expected-error@+1 {{'fir.array_access' op return type does not match array}}
  %p = fir.array_access %v, %c1, %c1 : (!fir.array<?x?xf32>, index, index) -> !fir.ref<f64>
  return
}

// -----

func.func @foo(%arg0: !fir.ref<!fir.array<30x!fir.type<t{c:!fir.array<20xi32>}>>>) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c9 = arith.constant 9 : index
  %c19 = arith.constant 19 : index
  %c30 = arith.constant 30 : index
  %0 = fir.shape %c30 : (index) -> !fir.shape<1>
  %1 = fir.array_load %arg0(%0) : (!fir.ref<!fir.array<30x!fir.type<t{c:!fir.array<20xi32>}>>>, !fir.shape<1>) -> !fir.array<30x!fir.type<t{c:!fir.array<20xi32>}>>
  %2 = fir.do_loop %arg1 = %c1 to %c9 step %c1 unordered iter_args(%arg2 = %1) -> (!fir.array<30x!fir.type<t{c:!fir.array<20xi32>}>>) {
    %3 = fir.field_index c, !fir.type<t{c:!fir.array<20xi32>}>
    %4 = fir.do_loop %arg3 = %c0 to %c19 step %c1 unordered iter_args(%arg4 = %arg2) -> (!fir.array<30x!fir.type<t{c:!fir.array<20xi32>}>>) {
      // expected-error@+1 {{'fir.array_access' op return type and/or indices do not type check}}
      %5 = fir.array_access %1, %arg1, %3, %arg3 : (!fir.array<30x!fir.type<t{c:!fir.array<20xi32>}>>, index, !fir.field, index) -> !fir.ref<f32>
      %6 = fir.call @ifoo(%5) : (!fir.ref<f32>) -> i32
      %7 = fir.array_update %arg4, %6, %arg1, %3, %arg3 : (!fir.array<30x!fir.type<t{c:!fir.array<20xi32>}>>, i32, index, !fir.field, index) -> !fir.array<30x!fir.type<t{c:!fir.array<20xi32>}>>
      fir.result %7 : !fir.array<30x!fir.type<t{c:!fir.array<20xi32>}>>
    }
    fir.result %4 : !fir.array<30x!fir.type<t{c:!fir.array<20xi32>}>>
  }
  fir.array_merge_store %1, %2 to %arg0 : !fir.array<30x!fir.type<t{c:!fir.array<20xi32>}>>, !fir.array<30x!fir.type<t{c:!fir.array<20xi32>}>>, !fir.ref<!fir.array<30x!fir.type<t{c:!fir.array<20xi32>}>>>
  return
}
func.func private @ifoo(!fir.ref<f32>) -> i32

// -----

func.func private @dispatch(%arg0: !fir.class<!fir.type<derived{a:i32,b:i32}>>) -> () {
  // expected-error@+1 {{'fir.dispatch' op pass_arg_pos must be smaller than the number of operands}}
  fir.dispatch "proc1"(%arg0 : !fir.class<!fir.type<derived{a:i32,b:i32}>>) (%arg0 : !fir.class<!fir.type<derived{a:i32,b:i32}>>) {pass_arg_pos = 1 : i32}
  return
}

// -----

func.func private @dispatch(%arg0: !fir.class<!fir.type<derived{a:i32,b:i32}>>, %arg1: i32) -> () {
  // expected-error@+1 {{'fir.dispatch' op pass_arg_pos must be a polymorphic operand}}
  fir.dispatch "proc1"(%arg0 : !fir.class<!fir.type<derived{a:i32,b:i32}>>) (%arg0, %arg1 : !fir.class<!fir.type<derived{a:i32,b:i32}>>, i32) {pass_arg_pos = 1 : i32}
  return
}

// -----
func.func @test_fortran_var_attrs() {
  // expected-error@+1 {{Unknown fortran variable attribute: volatypo}}
  %0 = fir.alloca f32 {fortran_attrs = #fir.var_attrs<volatypo>}
}

// -----
func.func @bad_numeric_declare(%arg0: !fir.ref<f32>) {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'fir.declare' op requires attribute 'uniq_name'}}
  %0 = fir.declare %arg0 typeparams %c1 {uniq_typo = "x"} : (!fir.ref<f32>, index) -> !fir.ref<f32>
  return
}

// -----
func.func @bad_numeric_declare(%arg0: !fir.ref<f32>) {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'fir.declare' op of numeric, logical, or assumed type entity must not have length parameters}}
  %0 = fir.declare %arg0 typeparams %c1 {uniq_name = "x"} : (!fir.ref<f32>, index) -> !fir.ref<f32>
  return
}

// -----
func.func @bad_char_declare(%arg0: !fir.boxchar<1> ) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  // expected-error@+1 {{'fir.declare' op must be provided exactly one type parameter when its base is a character that is not a box}}
  %1 = fir.declare %0#0 {uniq_name = "c"} : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
  return
}

// -----
func.func @bad_char_declare(%arg0: !fir.boxchar<1> ) {
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  // expected-error@+1 {{'fir.declare' op of character entity must have at most one length parameter}}
  %1 = fir.declare %0#0 typeparams %0#1, %0#1 {uniq_name = "c"} : (!fir.ref<!fir.char<1,?>>, index, index) -> !fir.ref<!fir.char<1,?>>
  return
}

// -----
func.func @bad_derived_declare(%arg0: !fir.ref<!fir.type<t{field:i32}>>) {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'fir.declare' op has too many length parameters}}
  %0 = fir.declare %arg0 typeparams %c1 {uniq_name = "x"} : (!fir.ref<!fir.type<t{field:i32}>>, index) -> !fir.ref<!fir.type<t{field:i32}>>
  return
}

// -----
func.func @bad_pdt_declare(%arg0: !fir.ref<!fir.type<pdt(param:i32){field:i32}>>) {
  // expected-error@+1 {{'fir.declare' op must be provided all the derived type length parameters when the base is not a box}}
  %0 = fir.declare %arg0 {uniq_name = "x"} : (!fir.ref<!fir.type<pdt(param:i32){field:i32}>>) -> !fir.ref<!fir.type<pdt(param:i32){field:i32}>>
  return
}

// -----
func.func @bad_pdt_declare_2(%arg0: !fir.ref<!fir.type<pdt(param:i32){field:i32}>>) {
  %c1 = arith.constant 1 : index
  // expected-error@+1 {{'fir.declare' op has too many length parameters}}
  %0 = fir.declare %arg0 typeparams %c1, %c1 {uniq_name = "x"} : (!fir.ref<!fir.type<pdt(param:i32){field:i32}>>, index, index) -> !fir.ref<!fir.type<pdt(param:i32){field:i32}>>
  return
}


// -----
func.func @bad_array_declare(%arg0: !fir.ref<!fir.array<?x?xf32>>) {
  // expected-error@+1 {{'fir.declare' op of array entity with a raw address base must have a shape operand that is a shape or shapeshift}}
  %0 = fir.declare %arg0 {uniq_name = "x"} : (!fir.ref<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
  return
}

// -----
func.func @bad_array_declare_2(%arg0: !fir.ref<!fir.array<?x?xf32>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %shift = fir.shift %c1, %c2 : (index, index) -> !fir.shift<2>
  // expected-error@+1 {{'fir.declare' op of array entity with a raw address base must have a shape operand that is a shape or shapeshift}}
  %0 = fir.declare %arg0(%shift) {uniq_name = "x"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shift<2>) -> !fir.ref<!fir.array<?x?xf32>>
  return
}

// -----
func.func @bad_array_declare_3(%arg0: !fir.ref<!fir.array<?x?xf32>>) {
  %c1 = arith.constant 1 : index
  %shape = fir.shape %c1 : (index) -> !fir.shape<1>
  // expected-error@+1 {{'fir.declare' op has conflicting shape and base operand ranks}}
  %0 = fir.declare %arg0(%shape) {uniq_name = "x"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?x?xf32>>
  return
}

// -----
func.func @bad_array_declare_4(%arg0: !fir.ref<!fir.array<?x?xf32>>) {
  %c1 = arith.constant 1 : index
  %shape = fir.shape_shift %c1, %c1 : (index, index) -> !fir.shapeshift<1>
  // expected-error@+1 {{'fir.declare' op has conflicting shape and base operand ranks}}
  %0 = fir.declare %arg0(%shape) {uniq_name = "x"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<1>) -> !fir.ref<!fir.array<?x?xf32>>
  return
}

// -----
func.func @bad_array_declare_box(%arg0: !fir.box<!fir.array<?x?xf32>>) {
  %c1 = arith.constant 1 : index
  %shape = fir.shift %c1 : (index) -> !fir.shift<1>
  // expected-error@+1 {{'fir.declare' op has conflicting shape and base operand ranks}}
  %0 = fir.declare %arg0(%shape) {uniq_name = "x"} : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<1>) -> !fir.box<!fir.array<?x?xf32>>
  return
}

// -----
func.func @bad_array_declare_char_boxaddr(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %shape = fir.shift %c1, %c2 : (index, index) -> !fir.shift<2>
  // expected-error@+1 {{'fir.declare' op for box address must not have a shape operand}}
  %0 = fir.declare %arg0(%shape) {uniq_name = "x"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>, !fir.shift<2>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?x!fir.char<1,?>>>>>
  return
}

// -----
func.func @bad_array_declare_unlimited_polymorphic_boxaddr(%arg0: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x?xnone>>>>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %shape = fir.shift %c1, %c2 : (index, index) -> !fir.shift<2>
  // expected-error@+1 {{'fir.declare' op for box address must not have a shape operand}}
  %0 = fir.declare %arg0(%shape) {uniq_name = "x"} : (!fir.ref<!fir.class<!fir.ptr<!fir.array<?x?xnone>>>>, !fir.shift<2>) -> !fir.ref<!fir.class<!fir.ptr<!fir.array<?x?xnone>>>>
  return
}

// -----

func.func @invalid_selector(%arg : !fir.box<!fir.ref<i32>>) -> i32 {
  %0 = arith.constant 1 : i32
  %2 = arith.constant 3 : i32
  // expected-error@+1{{'fir.select_type' op selector must be polymorphic}}
  fir.select_type %arg : !fir.box<!fir.ref<i32>> [
    #fir.type_is<!fir.int<4>>,^bb1(%0:i32),
    #fir.type_is<!fir.int<8>>,^bb2(%2:i32),
    unit,^bb5 ]
^bb1(%a : i32) :
  return %a : i32
^bb2(%b : i32) :
  return %b : i32
^bb5 :
  %zero = arith.constant 0 : i32
  return %zero : i32
}

// -----

func.func @logical_to_fp(%arg0: !fir.logical<4>) -> f32 {
  // expected-error@+1{{'fir.convert' op invalid type conversion}}
  %0 = fir.convert %arg0 : (!fir.logical<4>) -> f32
  return %0 : f32
}

// -----

func.func @fp_to_logical(%arg0: f32) -> !fir.logical<4> {
  // expected-error@+1{{'fir.convert' op invalid type conversion}}
  %0 = fir.convert %arg0 : (f32) -> !fir.logical<4>
  return %0 : !fir.logical<4>
}

// -----

func.func @bad_box_offset(%not_a_box : !fir.ref<i32>) {
  // expected-error@+1{{'fir.box_offset' op box_ref operand must have !fir.ref<!fir.box<T>> type}}
  %addr1 = fir.box_offset %not_a_box base_addr : (!fir.ref<i32>) -> !fir.llvm_ptr<!fir.ref<i32>>
  return
}

// -----

func.func @bad_box_offset(%no_addendum : !fir.ref<!fir.box<i32>>) {
  // expected-error@+1{{'fir.box_offset' op can only address derived_type field of derived type or unlimited polymorphic fir.box}}
  %addr1 = fir.box_offset %no_addendum derived_type : (!fir.ref<!fir.box<i32>>) -> !fir.llvm_ptr<!fir.tdesc<!fir.type<none>>>
  return
}
