// RUN: fir-opt --mif-convert %s | FileCheck %s

func.func @_QPtest_coarray_put_scalar() {
  %0 = fir.alloca f32
  %1 = fir.alloca f32
  %2 = fir.dummy_scope : !fir.dscope
  %3 = fir.address_of(@_QFtest_coarray_put_scalarEa) : !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>
  %4:2 = hlfir.declare %3 {uniq_name = "_QFtest_coarray_put_scalarEa"} : (!fir.ref<!fir.box<!fir.heap<f32>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<f32>, corank:1>>, !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>)
  %cst = arith.constant 2.000000e+00 : f32
  fir.store %cst to %1 : !fir.ref<f32>
  mif.put_coarray from %1 to %4#0 : (!fir.ref<!fir.box<!fir.heap<f32>, corank:1>>, !fir.ref<f32>) -> ()
  %5 = fir.load %4#0 : !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>
  %6 = fir.box_addr %5 : (!fir.box<!fir.heap<f32>, corank:1>) -> !fir.heap<f32>
  %7 = hlfir.designate %6   : (!fir.heap<f32>) -> !fir.ref<f32>
  %cst_0 = arith.constant 3.000000e+00 : f32
  fir.store %cst_0 to %0 : !fir.ref<f32>
  %c2_i64 = arith.constant 2 : i64
  mif.put_coarray from %0 to %7[%c2_i64] : (!fir.ref<f32>, i64, !fir.ref<f32>) -> ()
  return
}

func.func @_QPtest_coarray_put_array() {
  %0 = fir.alloca i32
  %1 = fir.dummy_scope : !fir.dscope
  %2 = fir.address_of(@_QFtest_coarray_put_arrayEa) : !fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>
  %3:2 = hlfir.declare %2 {uniq_name = "_QFtest_coarray_put_arrayEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>, !fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>)
  %4 = fir.alloca i32 {bindc_name = "me", uniq_name = "_QFtest_coarray_put_arrayEme"}
  %5:2 = hlfir.declare %4 {uniq_name = "_QFtest_coarray_put_arrayEme"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6 = mif.this_image : () -> i32
  hlfir.assign %6 to %5#0 : i32, !fir.ref<i32>
  %7 = fir.load %5#0 : !fir.ref<i32>
  %c1_i32 = arith.constant 1 : i32
  %8 = arith.cmpi eq, %7, %c1_i32 : i32
  fir.if %8 {
    %9 = fir.address_of(@_QQro.3x4xi4.0) : !fir.ref<!fir.array<3x4xi32>>
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %10 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
    %11:2 = hlfir.declare %9(%10) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.3x4xi4.0"} : (!fir.ref<!fir.array<3x4xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x4xi32>>, !fir.ref<!fir.array<3x4xi32>>)
    mif.put_coarray from %11#0 to %3#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>, !fir.ref<!fir.array<3x4xi32>>) -> ()
  } else {
    %9 = fir.load %3#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %10 = hlfir.designate %9 (%c2, %c3)  : (!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>, index, index) -> !fir.ref<i32>
    %c2_i32 = arith.constant 2 : i32
    fir.store %c2_i32 to %0 : !fir.ref<i32>
    %c2_i64 = arith.constant 2 : i64
    mif.put_coarray from %0 to %10[%c2_i64] : (!fir.ref<i32>, i64, !fir.ref<i32>) -> ()
  }
  return
}

// CHECK-LABEL: func.func @_QPtest_coarray_put_scalar
// CHECK:    %0 = fir.alloca !fir.box<f32>
// CHECK:    %1 = fir.alloca !fir.array<1xi64>
// CHECK:    %2 = fir.alloca i32
// CHECK:    %3 = fir.alloca i64
// CHECK:    %4 = fir.alloca i64
// CHECK:    %5 = fir.alloca !fir.box<f32>
// CHECK:    %6 = fir.alloca i32
// CHECK:    %7 = fir.alloca none
// CHECK:    %8 = fir.alloca i32
// CHECK:    %9 = fir.alloca i64
// CHECK:    %10 = fir.alloca i64
// CHECK:    %11 = fir.alloca f32
// CHECK:    %12 = fir.alloca f32
// CHECK:    %13 = fir.dummy_scope : !fir.dscope
// CHECK:    %14 = fir.address_of(@_QFtest_coarray_put_scalarEa) : !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>
// CHECK:    %15:2 = hlfir.declare %14 {uniq_name = "_QFtest_coarray_put_scalarEa"} : (!fir.ref<!fir.box<!fir.heap<f32>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<f32>, corank:1>>, !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>)
// CHECK:    %cst = arith.constant 2.000000e+00 : f32
// CHECK:    fir.store %cst to %12 : !fir.ref<f32>
// CHECK:    %16 = fir.load %15#0 : !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>
// CHECK:    %17 = fir.box_elesize %16 : (!fir.box<!fir.heap<f32>, corank:1>) -> i64
// CHECK:    fir.store %17 to %10 : !fir.ref<i64>
// CHECK:    %18 = fir.address_of(@_QFtest_coarray_put_scalarEa_coarray_handle) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>
// CHECK:    %19 = fir.absent !fir.ref<i32>
// CHECK:    %20 = fir.absent !fir.box<!fir.char<1,?>>
// CHECK:    %c0_i64 = arith.constant 0 : i64
// CHECK:    fir.store %c0_i64 to %9 : !fir.ref<i64>
// CHECK:    %c-2_i32 = arith.constant -2 : i32
// CHECK:    fir.store %c-2_i32 to %8 : !fir.ref<i32>
// CHECK:    fir.call @_QMprifPprif_get_team(%8, %7) : (!fir.ref<i32>, !fir.ref<none>) -> ()
// CHECK:    fir.call @_QMprifPprif_this_image_no_coarray(%7, %6) : (!fir.ref<none>, !fir.ref<i32>) -> ()
// CHECK:    %21 = fir.embox %12 : (!fir.ref<f32>) -> !fir.box<f32>
// CHECK:    fir.store %21 to %5 : !fir.ref<!fir.box<f32>>
// CHECK:    %22 = fir.convert %18 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>) -> !fir.ref<none>
// CHECK:    %23 = fir.convert %5 : (!fir.ref<!fir.box<f32>>) -> !fir.ptr<none>
// CHECK:    fir.call @_QMprifPprif_put(%6, %22, %9, %23, %10, %19, %20, %20) : (!fir.ref<i32>, !fir.ref<none>, !fir.ref<i64>, !fir.ptr<none>, !fir.ref<i64>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK:    %24 = fir.load %15#0 : !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>
// CHECK:    %25 = fir.box_addr %24 : (!fir.box<!fir.heap<f32>, corank:1>) -> !fir.heap<f32>
// CHECK:    %26 = hlfir.designate %25   : (!fir.heap<f32>) -> !fir.ref<f32>
// CHECK:    %cst_0 = arith.constant 3.000000e+00 : f32
// CHECK:    fir.store %cst_0 to %11 : !fir.ref<f32>
// CHECK:    %c2_i64 = arith.constant 2 : i64
// CHECK:    %c4_i64 = arith.constant 4 : i64
// CHECK:    fir.store %c4_i64 to %4 : !fir.ref<i64>
// CHECK:    %27 = fir.address_of(@_QFtest_coarray_put_scalarEa_coarray_handle) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>
// CHECK:    %28 = fir.absent !fir.ref<i32>
// CHECK:    %29 = fir.absent !fir.box<!fir.char<1,?>>
// CHECK:    %c0_i64_1 = arith.constant 0 : i64
// CHECK:    fir.store %c0_i64_1 to %3 : !fir.ref<i64>
// CHECK:    %c0 = arith.constant 0 : index
// CHECK:    %30 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
// CHECK:    fir.store %c2_i64 to %30 : !fir.ref<i64>
// CHECK:    %31 = fir.embox %1 : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
// CHECK:    %32 = fir.absent !fir.ref<i32>
// CHECK:    %33 = fir.convert %27 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>) -> !fir.ref<none>
// CHECK:    %34 = fir.convert %31 : (!fir.box<!fir.array<1xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK:    fir.call @_QMprifPprif_initial_team_index(%33, %34, %2, %32) : (!fir.ref<none>, !fir.box<!fir.array<?xi64>>, !fir.ref<i32>, !fir.ref<i32>) -> ()
// CHECK:    %35 = fir.embox %11 : (!fir.ref<f32>) -> !fir.box<f32>
// CHECK:    fir.store %35 to %0 : !fir.ref<!fir.box<f32>>
// CHECK:    %36 = fir.convert %27 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>) -> !fir.ref<none>
// CHECK:    %37 = fir.convert %0 : (!fir.ref<!fir.box<f32>>) -> !fir.ptr<none>
// CHECK:    fir.call @_QMprifPprif_put(%2, %36, %3, %37, %4, %28, %29, %29) : (!fir.ref<i32>, !fir.ref<none>, !fir.ref<i64>, !fir.ptr<none>, !fir.ref<i64>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()


// CHECK-LABEL:  func.func @_QPtest_coarray_put_array() {
// CHECK:    %0 = fir.alloca !fir.box<i32>
// CEHCK:    %1 = fir.alloca !fir.array<1xi64>
// CEHCK:    %2 = fir.alloca i32
// CEHCK:    %3 = fir.alloca i64
// CEHCK:    %4 = fir.alloca i64
// CEHCK:    %5 = fir.alloca !fir.box<!fir.array<3x4xi32>>
// CEHCK:    %6 = fir.alloca i32
// CEHCK:    %7 = fir.alloca none
// CEHCK:    %8 = fir.alloca i32
// CEHCK:    %9 = fir.alloca i64
// CEHCK:    %10 = fir.alloca i64
// CEHCK:    %11 = fir.alloca i32
// CEHCK:    %12 = fir.alloca i32
// CEHCK:    %13 = fir.dummy_scope : !fir.dscope
// CEHCK:    %14 = fir.address_of(@_QFtest_coarray_put_arrayEa) : !fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>
// CEHCK:    %15:2 = hlfir.declare %14 {uniq_name = "_QFtest_coarray_put_arrayEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>, !fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>)
// CEHCK:    %16 = fir.alloca i32 {bindc_name = "me", uniq_name = "_QFtest_coarray_put_arrayEme"}
// CEHCK:    %17:2 = hlfir.declare %16 {uniq_name = "_QFtest_coarray_put_arrayEme"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CEHCK:    %18 = fir.absent !fir.ref<none>
// CEHCK:    fir.call @_QMprifPprif_this_image_no_coarray(%18, %11) : (!fir.ref<none>, !fir.ref<i32>) -> ()
// CEHCK:    %19 = fir.load %11 : !fir.ref<i32>
// CEHCK:    hlfir.assign %19 to %17#0 : i32, !fir.ref<i32>
// CEHCK:    %20 = fir.load %17#0 : !fir.ref<i32>
// CEHCK:    %c1_i32 = arith.constant 1 : i32
// CEHCK:    %21 = arith.cmpi eq, %20, %c1_i32 : i32
// CEHCK:    fir.if %21 {
// CEHCK:      %22 = fir.address_of(@_QQro.3x4xi4.0) : !fir.ref<!fir.array<3x4xi32>>
// CEHCK:      %c3 = arith.constant 3 : index
// CEHCK:      %c4 = arith.constant 4 : index
// CEHCK:      %23 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
// CEHCK:      %24:2 = hlfir.declare %22(%23) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.3x4xi4.0"} : (!fir.ref<!fir.array<3x4xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x4xi32>>, !fir.ref<!fir.array<3x4xi32>>)
// CEHCK:      %25 = fir.load %15#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>
// CEHCK:      %26 = fir.box_elesize %25 : (!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>) -> i64
// CEHCK:      %c12_i64 = arith.constant 12 : i64
// CEHCK:      %27 = arith.muli %26, %c12_i64 : i64
// CEHCK:      fir.store %27 to %10 : !fir.ref<i64>
// CEHCK:      %28 = fir.address_of(@_QFtest_coarray_put_arrayEa_coarray_handle) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>
// CEHCK:      %29 = fir.absent !fir.ref<i32>
// CEHCK:      %30 = fir.absent !fir.box<!fir.char<1,?>>
// CEHCK:      %c0_i64 = arith.constant 0 : i64
// CEHCK:      fir.store %c0_i64 to %9 : !fir.ref<i64>
// CEHCK:      %c-2_i32 = arith.constant -2 : i32
// CEHCK:      fir.store %c-2_i32 to %8 : !fir.ref<i32>
// CEHCK:      fir.call @_QMprifPprif_get_team(%8, %7) : (!fir.ref<i32>, !fir.ref<none>) -> ()
// CEHCK:      fir.call @_QMprifPprif_this_image_no_coarray(%7, %6) : (!fir.ref<none>, !fir.ref<i32>) -> ()
// CEHCK:      %31 = fir.embox %24#0(%23) : (!fir.ref<!fir.array<3x4xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<3x4xi32>>
// CEHCK:      fir.store %31 to %5 : !fir.ref<!fir.box<!fir.array<3x4xi32>>>
// CEHCK:      %32 = fir.convert %28 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>) -> !fir.ref<none>
// CEHCK:      %33 = fir.convert %5 : (!fir.ref<!fir.box<!fir.array<3x4xi32>>>) -> !fir.ptr<none>
// CEHCK:      fir.call @_QMprifPprif_put(%6, %32, %9, %33, %10, %29, %30, %30) : (!fir.ref<i32>, !fir.ref<none>, !fir.ref<i64>, !fir.ptr<none>, !fir.ref<i64>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CEHCK:    } else {
// CEHCK:      %22 = fir.load %15#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>
// CEHCK:      %c2 = arith.constant 2 : index
// CEHCK:      %c3 = arith.constant 3 : index
// CEHCK:      %23 = hlfir.designate %22 (%c2, %c3)  : (!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>, index, index) -> !fir.ref<i32>
// CEHCK:      %c2_i32 = arith.constant 2 : i32
// CEHCK:      fir.store %c2_i32 to %12 : !fir.ref<i32>
// CEHCK:      %c2_i64 = arith.constant 2 : i64
// CEHCK:      %c4_i64 = arith.constant 4 : i64
// CEHCK:      fir.store %c4_i64 to %4 : !fir.ref<i64>
// CEHCK:      %24 = fir.address_of(@_QFtest_coarray_put_arrayEa_coarray_handle) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>
// CEHCK:      %25 = fir.absent !fir.ref<i32>
// CEHCK:      %26 = fir.absent !fir.box<!fir.char<1,?>>
// CEHCK:      %c0_i64 = arith.constant 0 : i64
// CEHCK:      fir.store %c0_i64 to %3 : !fir.ref<i64>
// CEHCK:      %c0 = arith.constant 0 : index
// CEHCK:      %27 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
// CEHCK:      fir.store %c2_i64 to %27 : !fir.ref<i64>
// CEHCK:      %28 = fir.embox %1 : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
// CEHCK:      %29 = fir.absent !fir.ref<i32>
// CEHCK:      %30 = fir.convert %24 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>) -> !fir.ref<none>
// CEHCK:      %31 = fir.convert %28 : (!fir.box<!fir.array<1xi64>>) -> !fir.box<!fir.array<?xi64>>
// CEHCK:      fir.call @_QMprifPprif_initial_team_index(%30, %31, %2, %29) : (!fir.ref<none>, !fir.box<!fir.array<?xi64>>, !fir.ref<i32>, !fir.ref<i32>) -> ()
// CEHCK:      %32 = fir.embox %12 : (!fir.ref<i32>) -> !fir.box<i32>
// CEHCK:      fir.store %32 to %0 : !fir.ref<!fir.box<i32>>
// CEHCK:      %33 = fir.convert %24 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>) -> !fir.ref<none>
// CEHCK:      %34 = fir.convert %0 : (!fir.ref<!fir.box<i32>>) -> !fir.ptr<none>
// CEHCK:      fir.call @_QMprifPprif_put(%2, %33, %3, %34, %4, %25, %26, %26) : (!fir.ref<i32>, !fir.ref<none>, !fir.ref<i64>, !fir.ptr<none>, !fir.ref<i64>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CEHCK:    }

