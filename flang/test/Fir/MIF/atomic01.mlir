// RUN: fir-opt --mif-convert %s | FileCheck %s

func.func @_QPtest_atomic() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QFtest_atomicEatom) : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %2:2 = hlfir.declare %1 uniq_name("_QFtest_atomicEatom") : (!fir.ref<!fir.box<!fir.heap<i64>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<i64>, corank:1>>, !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>)
  %3 = fir.address_of(@_QM__fortran_builtinsEC__builtin_atomic_int_kind) : !fir.ref<i32>
  %4:2 = hlfir.declare %3 uniq_name("_QM__fortran_builtinsEC__builtin_atomic_int_kind") fortran_attrs<parameter> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %c10_i64 = arith.constant 10 : i64
  %5 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %6 = fir.box_addr %5 : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
  hlfir.assign %c10_i64 to %6 : i64, !fir.heap<i64>
  %c5_i32 = arith.constant 5 : i32
  %7 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %8 = fir.absent !fir.ref<i32>
  mif.atomic_add atom %7 value %c5_i32 stat %8 : (!fir.box<!fir.heap<i64>, corank:1>, i32, !fir.ref<i32>) -> ()
  %c12_i32 = arith.constant 12 : i32
  %9 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %10 = fir.absent !fir.ref<i32>
  mif.atomic_and atom %9 value %c12_i32 stat %10 : (!fir.box<!fir.heap<i64>, corank:1>, i32, !fir.ref<i32>) -> ()
  %c3_i32 = arith.constant 3 : i32
  %11 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %12 = fir.absent !fir.ref<i32>
  mif.atomic_or atom %11 value %c3_i32 stat %12 : (!fir.box<!fir.heap<i64>, corank:1>, i32, !fir.ref<i32>) -> ()
  %c5_i32_0 = arith.constant 5 : i32
  %13 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %14 = fir.absent !fir.ref<i32>
  mif.atomic_xor atom %13 value %c5_i32_0 stat %14 : (!fir.box<!fir.heap<i64>, corank:1>, i32, !fir.ref<i32>) -> ()
  return
}

// CHECK-LABEL: @_QPtest_atomic()
// CHECK: @_QMprifPprif_atomic_add({{.*}})
// CHECK: @_QMprifPprif_atomic_and({{.*}})
// CHECK: @_QMprifPprif_atomic_or({{.*}})
// CHECK: @_QMprifPprif_atomic_xor({{.*}})

func.func @_QPtest_atomic_with_coindex() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QFtest_atomic_with_coindexEatom) : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %2:2 = hlfir.declare %1 uniq_name("_QFtest_atomic_with_coindexEatom") : (!fir.ref<!fir.box<!fir.heap<i64>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<i64>, corank:1>>, !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>)
  %3 = fir.address_of(@_QM__fortran_builtinsEC__builtin_atomic_int_kind) : !fir.ref<i32>
  %4:2 = hlfir.declare %3 uniq_name("_QM__fortran_builtinsEC__builtin_atomic_int_kind") fortran_attrs<parameter> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5 = fir.address_of(@_QFtest_atomic_with_coindexEi) : !fir.ref<i32>
  %6:2 = hlfir.declare %5 uniq_name("_QFtest_atomic_with_coindexEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %c10_i64 = arith.constant 10 : i64
  %7 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %8 = fir.box_addr %7 : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
  hlfir.assign %c10_i64 to %8 : i64, !fir.heap<i64>
  %9 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %10 = fir.box_addr %9 : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
  %c1_i64 = arith.constant 1 : i64
  %11 = hlfir.designate %10 [%c1_i64]  : (!fir.heap<i64>, i64) -> !fir.ref<i64>
  %c5_i32 = arith.constant 5 : i32
  %12 = fir.embox %11 : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
  %13 = fir.absent !fir.ref<i32>
  mif.atomic_add atom %12[%c1_i64] value %c5_i32 stat %13 : (!fir.box<i64, corank:1>, i64, i32, !fir.ref<i32>) -> ()
  %14 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %15 = fir.box_addr %14 : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
  %16 = fir.load %6#0 : !fir.ref<i32>
  %17 = fir.convert %16 : (i32) -> i64
  %18 = hlfir.designate %15 [%17]  : (!fir.heap<i64>, i64) -> !fir.ref<i64>
  %c12_i32 = arith.constant 12 : i32
  %19 = fir.embox %18 : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
  %20 = fir.absent !fir.ref<i32>
  mif.atomic_and atom %19[%17] value %c12_i32 stat %20 : (!fir.box<i64, corank:1>, i64, i32, !fir.ref<i32>) -> ()
  %21 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %22 = fir.box_addr %21 : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
  %c1_i64_0 = arith.constant 1 : i64
  %23 = hlfir.designate %22 [%c1_i64_0]  : (!fir.heap<i64>, i64) -> !fir.ref<i64>
  %c3_i32 = arith.constant 3 : i32
  %24 = fir.embox %23 : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
  %25 = fir.absent !fir.ref<i32>
  mif.atomic_or atom %24[%c1_i64_0] value %c3_i32 stat %25 : (!fir.box<i64, corank:1>, i64, i32, !fir.ref<i32>) -> ()
  %26 = fir.load %2#0 : !fir.ref<!fir.box<!fir.heap<i64>, corank:1>>
  %27 = fir.box_addr %26 : (!fir.box<!fir.heap<i64>, corank:1>) -> !fir.heap<i64>
  %28 = fir.load %6#0 : !fir.ref<i32>
  %29 = fir.convert %28 : (i32) -> i64
  %30 = hlfir.designate %27 [%29]  : (!fir.heap<i64>, i64) -> !fir.ref<i64>
  %c5_i32_1 = arith.constant 5 : i32
  %31 = fir.embox %30 : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
  %32 = fir.absent !fir.ref<i32>
  mif.atomic_xor atom %31[%29] value %c5_i32_1 stat %32 : (!fir.box<i64, corank:1>, i64, i32, !fir.ref<i32>) -> ()
  return
}

// CHECK-LABEL: @_QPtest_atomic_with_coindex()
// CHECK: @_QMprifPprif_atomic_add({{.*}})
// CHECK: @_QMprifPprif_atomic_and({{.*}})
// CHECK: @_QMprifPprif_atomic_or({{.*}})
// CHECK: @_QMprifPprif_atomic_xor({{.*}})
