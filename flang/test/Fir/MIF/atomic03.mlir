// RUN: fir-opt --mif-convert %s | FileCheck %s

func.func @_QQmain() attributes {fir.bindc_name = "ATOMIC_ADD_ARRAY1D"} {
  %0 = fir.alloca !fir.array<0xi64>
  %1 = fir.alloca !fir.array<1xi64>
  %2 = fir.dummy_scope : !fir.dscope
  %3 = fir.address_of(@_QFEatom) : !fir.ref<!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>>
  %c1_i64 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %4 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
  fir.store %c1_i64 to %4 : !fir.ref<i64>
  %5 = fir.embox %1 : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
  %6 = fir.embox %0 : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
  mif.alloc_coarray %3 lcobounds %5 ucobounds %6 uniq_name("_QFEatom") : (!fir.ref<!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>) -> ()
  %7:2 = hlfir.declare %3 uniq_name("_QFEatom") : (!fir.ref<!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>>, !fir.ref<!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>>)
  %8 = fir.address_of(@_QM__fortran_builtinsEC__builtin_atomic_int_kind) : !fir.ref<i32>
  %9:2 = hlfir.declare %8 uniq_name("_QM__fortran_builtinsEC__builtin_atomic_int_kind") fortran_attrs<parameter> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %10 = fir.alloca i32 <{bindc_name = "i", uniq_name = "_QFEi"}>
  %11:2 = hlfir.declare %10 uniq_name("_QFEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %12 = fir.address_of(@_QFECm) : !fir.ref<i32>
  %13:2 = hlfir.declare %12 uniq_name("_QFECm") fortran_attrs<parameter> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %14 = fir.alloca i32 <{bindc_name = "me", uniq_name = "_QFEme"}>
  %15:2 = hlfir.declare %14 uniq_name("_QFEme") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %16 = fir.alloca i32 <{bindc_name = "n", uniq_name = "_QFEn"}>
  %17:2 = hlfir.declare %16 uniq_name("_QFEn") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %18 = fir.alloca i64 <{bindc_name = "val", uniq_name = "_QFEval"}>
  %19:2 = hlfir.declare %18 uniq_name("_QFEval") : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %20 = mif.this_image : () -> i32
  hlfir.assign %20 to %15#0 : i32, !fir.ref<i32>
  %21 = mif.num_images : () -> i32
  hlfir.assign %21 to %17#0 : i32, !fir.ref<i32>
  %c0_i64 = arith.constant 0 : i64
  %22 = fir.load %7#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>>
  hlfir.assign %c0_i64 to %22 : i64, !fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>
  %c1_i32 = arith.constant 1 : i32
  %c5_i32 = arith.constant 5 : i32
  %c1_i32_0 = arith.constant 1 : i32
  fir.do_loop %arg0 = %c1_i32 to %c5_i32 step %c1_i32_0 : i32 {
    fir.store %arg0 to %11#0 : !fir.ref<i32>
    %34 = fir.load %7#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>>
    %c1_i64_2 = arith.constant 1 : i64
    %35 = fir.load %11#0 : !fir.ref<i32>
    %36 = fir.convert %35 : (i32) -> i64
    %37 = hlfir.designate %34 (%36)[%c1_i64_2]  : (!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>, i64, i64) -> !fir.ref<i64>
    %38 = fir.load %15#0 : !fir.ref<i32>
    %39 = fir.load %11#0 : !fir.ref<i32>
    %40 = arith.muli %38, %39 : i32
    %41 = fir.convert %40 : (i32) -> i64
    %42 = fir.embox %37 : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
    %43 = fir.absent !fir.ref<i32>
    mif.atomic_add atom %42[%c1_i64_2] value %41 stat %43 : (!fir.box<i64, corank:1>, i64, i64, !fir.ref<i32>) -> ()
    %44 = fir.load %7#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>>
    %45 = fir.load %11#0 : !fir.ref<i32>
    %46 = fir.convert %45 : (i32) -> i64
    %47 = hlfir.designate %44 (%46)  : (!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>, i64) -> !fir.ref<i64>
    %48 = fir.load %15#0 : !fir.ref<i32>
    %49 = fir.load %11#0 : !fir.ref<i32>
    %50 = arith.muli %48, %49 : i32
    %51 = fir.convert %50 : (i32) -> i64
    %52 = fir.embox %47 : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
    %53 = fir.absent !fir.ref<i32>
    mif.atomic_and atom %52 value %51 stat %53 : (!fir.box<i64, corank:1>, i64, !fir.ref<i32>) -> ()
    %54 = fir.load %7#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>>
    %55 = fir.load %11#0 : !fir.ref<i32>
    %56 = fir.convert %55 : (i32) -> i64
    %57 = hlfir.designate %54 (%56)  : (!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>, i64) -> !fir.ref<i64>
    %58 = fir.load %15#0 : !fir.ref<i32>
    %59 = fir.load %11#0 : !fir.ref<i32>
    %60 = arith.muli %58, %59 : i32
    %61 = fir.convert %60 : (i32) -> i64
    %62 = fir.embox %57 : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
    %63 = fir.absent !fir.ref<i32>
    mif.atomic_or atom %62 value %61 stat %63 : (!fir.box<i64, corank:1>, i64, !fir.ref<i32>) -> ()
    %64 = fir.load %7#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>>
    %c1_i64_3 = arith.constant 1 : i64
    %65 = fir.load %11#0 : !fir.ref<i32>
    %66 = fir.convert %65 : (i32) -> i64
    %67 = hlfir.designate %64 (%66)[%c1_i64_3]  : (!fir.box<!fir.heap<!fir.array<5xi64>>, corank:1>, i64, i64) -> !fir.ref<i64>
    %68 = fir.load %15#0 : !fir.ref<i32>
    %69 = fir.load %11#0 : !fir.ref<i32>
    %70 = arith.muli %68, %69 : i32
    %71 = fir.convert %70 : (i32) -> i64
    %72 = fir.embox %67 : (!fir.ref<i64>) -> !fir.box<i64, corank:1>
    %73 = fir.absent !fir.ref<i32>
    mif.atomic_xor atom %72[%c1_i64_3] value %71 stat %73 : (!fir.box<i64, corank:1>, i64, i64, !fir.ref<i32>) -> ()
  }
  %23 = fir.convert %c1_i32 : (i32) -> index
  %24 = fir.convert %c5_i32 : (i32) -> index
  %25 = fir.convert %c1_i32_0 : (i32) -> index
  %c0_1 = arith.constant 0 : index
  %26 = arith.subi %24, %23 : index
  %27 = arith.addi %26, %25 : index
  %28 = arith.divsi %27, %25 : index
  %29 = arith.cmpi slt, %28, %c0_1 : index
  %30 = arith.select %29, %c0_1, %28 : index
  %31 = arith.muli %30, %25 : index
  %32 = arith.addi %23, %31 : index
  %33 = fir.convert %32 : (index) -> i32
  fir.store %33 to %11#0 : !fir.ref<i32>
  return
}
  
// CHECK: @_QMprifPprif_atomic_add({{.*}})
// CHECK: @_QMprifPprif_atomic_and({{.*}})
// CHECK: @_QMprifPprif_atomic_or({{.*}})
// CHECK: @_QMprifPprif_atomic_xor({{.*}})
