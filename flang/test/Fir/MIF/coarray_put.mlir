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
// CHECK:   %0 = fir.alloca !fir.box<f32>
// CHECK:   %1 = fir.alloca !fir.array<1xi64>
// CHECK:   %2 = fir.alloca i32
// CHECK:   %3 = fir.alloca i64
// CHECK:   %4 = fir.alloca i64
// CHECK:   %5 = fir.alloca !fir.box<f32>
// CHECK:   %6 = fir.alloca i32
// CHECK:   %7 = fir.alloca i32
// CHECK:   %8 = fir.alloca i64
// CHECK:   %9 = fir.alloca i64
// CHECK:   %10 = fir.alloca f32
// CHECK:   %11 = fir.alloca f32
// CHECK:   %12 = fir.dummy_scope : !fir.dscope
// CHECK:   %13 = fir.address_of(@_QFtest_coarray_put_scalarEa) : !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>
// CHECK:   %14:2 = hlfir.declare %13 {uniq_name = "_QFtest_coarray_put_scalarEa"} : (!fir.ref<!fir.box<!fir.heap<f32>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<f32>, corank:1>>, !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>)
// CHECK:   %cst = arith.constant 2.000000e+00 : f32
// CHECK:   fir.store %cst to %11 : !fir.ref<f32>
// CHECK:   %15 = fir.load %14#0 : !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>
// CHECK:   %16 = fir.box_elesize %15 : (!fir.box<!fir.heap<f32>, corank:1>) -> i64
// CHECK:   fir.store %16 to %9 : !fir.ref<i64>
// CHECK:   %17 = fir.address_of(@_QFtest_coarray_put_scalarEa_coarray_handle) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>
// CHECK:   %18 = fir.absent !fir.ref<i32>
// CHECK:   %19 = fir.absent !fir.box<!fir.char<1,?>>
// CHECK:   %c0_i64 = arith.constant 0 : i64
// CHECK:   fir.store %c0_i64 to %8 : !fir.ref<i64>
// CHECK:   %c-2_i32 = arith.constant -2 : i32
// CHECK:   %20 = mif.get_team level %c-2_i32 : (i32) -> !fir.ref<none>
// CHECK:   %21 = mif.this_image team %20 : (!fir.ref<none>) -> i32
// CHECK:   fir.store %21 to %6 : !fir.ref<i32>
// CHECK:   %22 = fir.embox %11 : (!fir.ref<f32>) -> !fir.box<f32>
// CHECK:   fir.store %22 to %5 : !fir.ref<!fir.box<f32>>
// CHECK:   %23 = fir.convert %17 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>) -> !fir.ref<none>
// CHECK:   %24 = fir.convert %5 : (!fir.ref<!fir.box<f32>>) -> !fir.ptr<none>
// CHECK:   fir.call @_QMprifPprif_put(%6, %23, %8, %24, %9, %18, %19, %19) : (!fir.ref<i32>, !fir.ref<none>, !fir.ref<i64>, !fir.ptr<none>, !fir.ref<i64>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK:   %25 = fir.load %14#0 : !fir.ref<!fir.box<!fir.heap<f32>, corank:1>>
// CHECK:   %26 = fir.box_addr %25 : (!fir.box<!fir.heap<f32>, corank:1>) -> !fir.heap<f32>
// CHECK:   %27 = hlfir.designate %26   : (!fir.heap<f32>) -> !fir.ref<f32>
// CHECK:   %cst_0 = arith.constant 3.000000e+00 : f32
// CHECK:   fir.store %cst_0 to %10 : !fir.ref<f32>
// CHECK:   %c2_i64 = arith.constant 2 : i64
// CHECK:   %c4_i64 = arith.constant 4 : i64
// CHECK:   fir.store %c4_i64 to %4 : !fir.ref<i64>
// CHECK:   %28 = fir.address_of(@_QFtest_coarray_put_scalarEa_coarray_handle) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>
// CHECK:   %29 = fir.absent !fir.ref<i32>
// CHECK:   %30 = fir.absent !fir.box<!fir.char<1,?>>
// CHECK:   %c0_i64_1 = arith.constant 0 : i64
// CHECK:   fir.store %c0_i64_1 to %3 : !fir.ref<i64>
// CHECK:   %c0 = arith.constant 0 : index
// CHECK:   %31 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
// CHECK:   fir.store %c2_i64 to %31 : !fir.ref<i64>
// CHECK:   %32 = fir.embox %1 : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
// CHECK:   %33 = fir.absent !fir.ref<i32>
// CHECK:   %34 = fir.convert %28 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>) -> !fir.ref<none>
// CHECK:   %35 = fir.convert %32 : (!fir.box<!fir.array<1xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK:   fir.call @_QMprifPprif_initial_team_index(%34, %35, %2, %33) : (!fir.ref<none>, !fir.box<!fir.array<?xi64>>, !fir.ref<i32>, !fir.ref<i32>) -> ()
// CHECK:   %36 = fir.embox %10 : (!fir.ref<f32>) -> !fir.box<f32>
// CHECK:   fir.store %36 to %0 : !fir.ref<!fir.box<f32>>
// CHECK:   %37 = fir.convert %28 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>) -> !fir.ref<none>
// CHECK:   %38 = fir.convert %0 : (!fir.ref<!fir.box<f32>>) -> !fir.ptr<none>
// CHECK:   fir.call @_QMprifPprif_put(%2, %37, %3, %38, %4, %29, %30, %30) : (!fir.ref<i32>, !fir.ref<none>, !fir.ref<i64>, !fir.ptr<none>, !fir.ref<i64>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()


// CHECK-LABEL:  func.func @_QPtest_coarray_put_array() {
// CHECK:  %0 = fir.alloca !fir.box<i32>
// CHECK:  %1 = fir.alloca !fir.array<1xi64>
// CHECK:  %2 = fir.alloca i32
// CHECK:  %3 = fir.alloca i64
// CHECK:  %4 = fir.alloca i64
// CHECK:  %5 = fir.alloca !fir.box<!fir.array<3x4xi32>>
// CHECK:  %6 = fir.alloca i32
// CHECK:  %7 = fir.alloca i32
// CHECK:  %8 = fir.alloca i64
// CHECK:  %9 = fir.alloca i64
// CHECK:  %10 = fir.alloca i32
// CHECK:  %11 = fir.alloca i32
// CHECK:  %12 = fir.dummy_scope : !fir.dscope
// CHECK:  %13 = fir.address_of(@_QFtest_coarray_put_arrayEa) : !fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>
// CHECK:  %14:2 = hlfir.declare %13 {uniq_name = "_QFtest_coarray_put_arrayEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>, !fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>)
// CHECK:  %15 = fir.alloca i32 {bindc_name = "me", uniq_name = "_QFtest_coarray_put_arrayEme"}
// CHECK:  %16:2 = hlfir.declare %15 {uniq_name = "_QFtest_coarray_put_arrayEme"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:  %17 = fir.absent !fir.ref<none>
// CHECK:  fir.call @_QMprifPprif_this_image_no_coarray(%17, %10) : (!fir.ref<none>, !fir.ref<i32>) -> ()
// CHECK:  %18 = fir.load %10 : !fir.ref<i32>
// CHECK:  hlfir.assign %18 to %16#0 : i32, !fir.ref<i32>
// CHECK:  %19 = fir.load %16#0 : !fir.ref<i32>
// CHECK:  %c1_i32 = arith.constant 1 : i32
// CHECK:  %20 = arith.cmpi eq, %19, %c1_i32 : i32
// CHECK:  fir.if %20 {
// CHECK:    %21 = fir.address_of(@_QQro.3x4xi4.0) : !fir.ref<!fir.array<3x4xi32>>
// CHECK:    %c3 = arith.constant 3 : index
// CHECK:    %c4 = arith.constant 4 : index
// CHECK:    %22 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
// CHECK:    %23:2 = hlfir.declare %21(%22) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.3x4xi4.0"} : (!fir.ref<!fir.array<3x4xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<3x4xi32>>, !fir.ref<!fir.array<3x4xi32>>)
// CHECK:    %24 = fir.load %14#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>
// CHECK:    %25 = fir.box_elesize %24 : (!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>) -> i64
// CHECK:    %c12_i64 = arith.constant 12 : i64
// CHECK:    %26 = arith.muli %25, %c12_i64 : i64
// CHECK:    fir.store %26 to %9 : !fir.ref<i64>
// CHECK:    %27 = fir.address_of(@_QFtest_coarray_put_arrayEa_coarray_handle) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>
// CHECK:    %28 = fir.absent !fir.ref<i32>
// CHECK:    %29 = fir.absent !fir.box<!fir.char<1,?>>
// CHECK:    %c0_i64 = arith.constant 0 : i64
// CHECK:    fir.store %c0_i64 to %8 : !fir.ref<i64>
// CHECK:    %c-2_i32 = arith.constant -2 : i32
// CHECK:    %30 = mif.get_team level %c-2_i32 : (i32) -> !fir.ref<none>
// CHECK:    %31 = mif.this_image team %30 : (!fir.ref<none>) -> i32
// CHECK:    fir.store %31 to %6 : !fir.ref<i32>
// CHECK:    %32 = fir.embox %23#0(%22) : (!fir.ref<!fir.array<3x4xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<3x4xi32>>
// CHECK:    fir.store %32 to %5 : !fir.ref<!fir.box<!fir.array<3x4xi32>>>
// CHECK:    %33 = fir.convert %27 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>) -> !fir.ref<none>
// CHECK:    %34 = fir.convert %5 : (!fir.ref<!fir.box<!fir.array<3x4xi32>>>) -> !fir.ptr<none>
// CHECK:    fir.call @_QMprifPprif_put(%6, %33, %8, %34, %9, %28, %29, %29) : (!fir.ref<i32>, !fir.ref<none>, !fir.ref<i64>, !fir.ptr<none>, !fir.ref<i64>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK:  } else {
// CHECK:    %21 = fir.load %14#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>>
// CHECK:    %c2 = arith.constant 2 : index
// CHECK:    %c3 = arith.constant 3 : index
// CHECK:    %22 = hlfir.designate %21 (%c2, %c3)  : (!fir.box<!fir.heap<!fir.array<3x4xi32>>, corank:1>, index, index) -> !fir.ref<i32>
// CHECK:    %c2_i32 = arith.constant 2 : i32
// CHECK:    fir.store %c2_i32 to %11 : !fir.ref<i32>
// CHECK:    %c2_i64 = arith.constant 2 : i64
// CHECK:    %c4_i64 = arith.constant 4 : i64
// CHECK:    fir.store %c4_i64 to %4 : !fir.ref<i64>
// CHECK:    %23 = fir.address_of(@_QFtest_coarray_put_arrayEa_coarray_handle) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>
// CHECK:    %24 = fir.absent !fir.ref<i32>
// CHECK:    %25 = fir.absent !fir.box<!fir.char<1,?>>
// CHECK:    %c0_i64 = arith.constant 0 : i64
// CHECK:    fir.store %c0_i64 to %3 : !fir.ref<i64>
// CHECK:    %c0 = arith.constant 0 : index
// CHECK:    %26 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
// CHECK:    fir.store %c2_i64 to %26 : !fir.ref<i64>
// CHECK:    %27 = fir.embox %1 : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
// CHECK:    %28 = fir.absent !fir.ref<i32>
// CHECK:    %29 = fir.convert %23 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>) -> !fir.ref<none>
// CHECK:    %30 = fir.convert %27 : (!fir.box<!fir.array<1xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK:    fir.call @_QMprifPprif_initial_team_index(%29, %30, %2, %28) : (!fir.ref<none>, !fir.box<!fir.array<?xi64>>, !fir.ref<i32>, !fir.ref<i32>) -> ()
// CHECK:    %31 = fir.embox %11 : (!fir.ref<i32>) -> !fir.box<i32>
// CHECK:    fir.store %31 to %0 : !fir.ref<!fir.box<i32>>
// CHECK:    %32 = fir.convert %23 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{info:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__c_ptr_c_address:i64}>}>>) -> !fir.ref<none>
// CHECK:    %33 = fir.convert %0 : (!fir.ref<!fir.box<i32>>) -> !fir.ptr<none>
// CHECK:    fir.call @_QMprifPprif_put(%2, %32, %3, %33, %4, %24, %25, %25) : (!fir.ref<i32>, !fir.ref<none>, !fir.ref<i64>, !fir.ptr<none>, !fir.ref<i64>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK:  }
