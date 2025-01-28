// RUN: fir-opt --simplify-intrinsics %s | FileCheck %s

func.func @_QPsum_in_device(%arg0: !fir.ref<!fir.array<?xi32>> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "a"}, %arg1: i32 {fir.bindc_name = "n"}) attributes {cuf.proc_attr = #cuf.cuda_proc<global>} {
  %c5_i32 = arith.constant 5 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c-1 = arith.constant -1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c-1 : (index) -> !fir.shape<1>
  %2 = fir.declare %arg0(%1) dummy_scope %0 {data_attr = #cuf.cuda<device>, uniq_name = "_QFsum_in_deviceEa"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<?xi32>>
  %3 = fir.embox %2(%1) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
  %4 = fir.alloca i32
  fir.store %arg1 to %4 : !fir.ref<i32>
  %5 = fir.declare %4 dummy_scope %0 {fortran_attrs = #fir.var_attrs<value>, uniq_name = "_QFsum_in_deviceEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %12 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsum_in_deviceEi"}
  %13 = fir.declare %12 {uniq_name = "_QFsum_in_deviceEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %14 = fir.address_of(@_QM__fortran_builtinsE__builtin_threadidx) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %18 = fir.load %5 : !fir.ref<i32>
  %19 = fir.convert %18 : (i32) -> index
  %20 = arith.cmpi sgt, %19, %c0 : index
  %21 = arith.select %20, %19, %c0 : index
  %22 = fir.alloca !fir.array<?xi32>, %21 {bindc_name = "auto", uniq_name = "_QFsum_in_deviceEauto"}
  %23 = fir.shape %21 : (index) -> !fir.shape<1>
  %24 = fir.declare %22(%23) {uniq_name = "_QFsum_in_deviceEauto"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.ref<!fir.array<?xi32>>
  %25 = fir.embox %24(%23) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
  %26 = fir.undefined index
  %27 = fir.slice %c1, %19, %c1 : (index, index, index) -> !fir.slice<1>
  %28 = fir.embox %24(%23) [%27] : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xi32>>
  %29 = fir.absent !fir.box<i1>
  %30 = fir.address_of(@_QQclX91d13f6e74caa2f03965d7a7c6a8fdd5) : !fir.ref<!fir.char<1,50>>
  %31 = fir.convert %28 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
  %32 = fir.convert %30 : (!fir.ref<!fir.char<1,50>>) -> !fir.ref<i8>
  %33 = fir.convert %c0 : (index) -> i32
  %34 = fir.convert %29 : (!fir.box<i1>) -> !fir.box<none>
  %35 = fir.call @_FortranASumInteger4(%31, %32, %c5_i32, %33, %34) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
  %36 = fir.load %13 : !fir.ref<i32>
  %37 = fir.convert %36 : (i32) -> i64
  %38 = fir.array_coor %2(%1) %37 : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
  fir.store %35 to %38 : !fir.ref<i32>
  return
}

// Check that intrinsic simplification is disabled in CUDA Fortran context. The simplified intrinsic is
// created in the module op but the device func will be migrated into a gpu module op resulting in a
// missing symbol error. 
// The simplified intrinsic could also be migrated to the gpu module but the choice has not be made
// at this point.
// CHECK-LABEL: func.func @_QPsum_in_device
// CHECK-NOT: fir.call @_FortranASumInteger4x1_contract_simplified
