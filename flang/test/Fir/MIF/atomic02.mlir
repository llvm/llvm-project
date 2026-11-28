// RUN: fir-opt --mif-convert %s | FileCheck %s

func.func @_QPtest_atomic_fetch() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QFtest_atomic_fetchEatom) : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %2:2 = hlfir.declare %1 uniq_name("_QFtest_atomic_fetchEatom") : (!fir.ref<!fir.box<!fir.heap<i64>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<i64>, corank:1>>, !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>)
  %3 = fir.address_of(@_QM__fortran_builtinsEC__builtin_atomic_int_kind) : !fir.ref<i32>
  %4:2 = hlfir.declare %3 uniq_name("_QM__fortran_builtinsEC__builtin_atomic_int_kind") fortran_attrs<parameter> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5 = fir.alloca i64 <{bindc_name = "old_val", uniq_name = "_QFtest_atomic_fetchEold_val"}>
  %6:2 = hlfir.declare %5 uniq_name("_QFtest_atomic_fetchEold_val") : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %c10_i64 = arith.constant 10 : i64
  %7 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %8 = fir.box_addr %7 : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
  hlfir.assign %c10_i64 to %8 : i64, !fir.heap<i64>
  %c5_i32 = arith.constant 5 : i32
  %9 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %10 = fir.absent !fir.ref<i32>
  mif.atomic_add atom %9 value %c5_i32 old %6#0 stat %10 : (!fir.box<!fir.heap<i64>, corank:1>, i32, !fir.ref<i64>, !fir.ref<i32>) -> ()
  %c12_i32 = arith.constant 12 : i32
  %11 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %12 = fir.absent !fir.ref<i32>
  mif.atomic_and atom %11 value %c12_i32 old %6#0 stat %12 : (!fir.box<!fir.heap<i64>, corank:1>, i32, !fir.ref<i64>, !fir.ref<i32>) -> ()
  %c3_i32 = arith.constant 3 : i32
  %13 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %14 = fir.absent !fir.ref<i32>
  mif.atomic_or atom %13 value %c3_i32 old %6#0 stat %14 : (!fir.box<!fir.heap<i64>, corank:1>, i32, !fir.ref<i64>, !fir.ref<i32>) -> ()
  %c5_i32_0 = arith.constant 5 : i32
  %15 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %16 = fir.absent !fir.ref<i32>
  mif.atomic_xor atom %15 value %c5_i32_0 old %6#0 stat %16 : (!fir.box<!fir.heap<i64>, corank:1>, i32, !fir.ref<i64>, !fir.ref<i32>) -> ()
  return
}

// CHECK-LABEL: @_QPtest_atomic_fetch()
// CHECK: @_QMprifPprif_atomic_fetch_add({{.*}})
// CHECK: @_QMprifPprif_atomic_fetch_and({{.*}})
// CHECK: @_QMprifPprif_atomic_fetch_or({{.*}})
// CHECK: @_QMprifPprif_atomic_fetch_xor({{.*}})

func.func @_QPtest_atomic_fetch_with_coindex() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QFtest_atomic_fetch_with_coindexEatom) : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %2:2 = hlfir.declare %1 uniq_name("_QFtest_atomic_fetch_with_coindexEatom") : (!fir.ref<!fir.box<!fir.heap<i64>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<i64>, corank:1>>, !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>)
  %3 = fir.address_of(@_QM__fortran_builtinsEC__builtin_atomic_int_kind) : !fir.ref<i32>
  %4:2 = hlfir.declare %3 uniq_name("_QM__fortran_builtinsEC__builtin_atomic_int_kind") fortran_attrs<parameter> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5 = fir.address_of(@_QFtest_atomic_fetch_with_coindexEi) : !fir.ref<i32>
  %6:2 = hlfir.declare %5 uniq_name("_QFtest_atomic_fetch_with_coindexEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7 = fir.alloca i64 <{bindc_name = "old_val", uniq_name = "_QFtest_atomic_fetch_with_coindexEold_val"}>
  %8:2 = hlfir.declare %7 uniq_name("_QFtest_atomic_fetch_with_coindexEold_val") : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %c10_i64 = arith.constant 10 : i64
  %9 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %10 = fir.box_addr %9 : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
  hlfir.assign %c10_i64 to %10 : i64, !fir.heap<i64>
  %11 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %12 = fir.box_addr %11 : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
  %c1_i64 = arith.constant 1 : i64
  %13 = hlfir.designate %12 [%c1_i64]  : (!fir.heap<i64>, i64) -> !fir.ref<i64>
  %c5_i32 = arith.constant 5 : i32
  %14 = fir.embox %13 : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
  %15 = fir.absent !fir.ref<i32>
  mif.atomic_add atom %14[%c1_i64] value %c5_i32 old %8#0 stat %15 : (!fir.box<i64, corank:1>, i64, i32, !fir.ref<i64>, !fir.ref<i32>) -> ()
  %16 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %17 = fir.box_addr %16 : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
  %18 = fir.load %6#0 : !fir.ref<i32>
  %19 = fir.convert %18 : (i32) -> i64
  %20 = hlfir.designate %17 [%19]  : (!fir.heap<i64>, i64) -> !fir.ref<i64>
  %c12_i32 = arith.constant 12 : i32
  %21 = fir.embox %20 : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
  %22 = fir.absent !fir.ref<i32>
  mif.atomic_and atom %21[%19] value %c12_i32 old %8#0 stat %22 : (!fir.box<i64, corank:1>, i64, i32, !fir.ref<i64>, !fir.ref<i32>) -> ()
  %23 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %24 = fir.box_addr %23 : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
  %c1_i64_0 = arith.constant 1 : i64
  %25 = hlfir.designate %24 [%c1_i64_0]  : (!fir.heap<i64>, i64) -> !fir.ref<i64>
  %c3_i32 = arith.constant 3 : i32
  %26 = fir.embox %25 : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
  %27 = fir.absent !fir.ref<i32>
  mif.atomic_or atom %26[%c1_i64_0] value %c3_i32 old %8#0 stat %27 : (!fir.box<i64, corank:1>, i64, i32, !fir.ref<i64>, !fir.ref<i32>) -> ()
  %28 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %29 = fir.box_addr %28 : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
  %30 = fir.load %6#0 : !fir.ref<i32>
  %31 = fir.convert %30 : (i32) -> i64
  %32 = hlfir.designate %29 [%31]  : (!fir.heap<i64>, i64) -> !fir.ref<i64>
  %c5_i32_1 = arith.constant 5 : i32
  %33 = fir.embox %32 : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
  %34 = fir.absent !fir.ref<i32>
  mif.atomic_xor atom %33[%31] value %c5_i32_1 old %8#0 stat %34 : (!fir.box<i64, corank:1>, i64, i32, !fir.ref<i64>, !fir.ref<i32>) -> ()
  return
}

// CHECK-LABEL: @_QPtest_atomic_fetch_with_coindex()
// CHECK: @_QMprifPprif_atomic_fetch_add({{.*}})
// CHECK: @_QMprifPprif_atomic_fetch_and({{.*}})
// CHECK: @_QMprifPprif_atomic_fetch_or({{.*}})
// CHECK: @_QMprifPprif_atomic_fetch_xor({{.*}})
