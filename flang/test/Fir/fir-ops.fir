// Test the FIR operations

// RUN: fir-opt %s | fir-opt | FileCheck %s

// CHECK-LABEL: func private @it1() -> !fir.int<4>
// CHECK: func private @box1() -> !fir.boxchar<2>
// CHECK: func private @box2() -> !fir.boxproc<(i32, i32) -> i64>
// CHECK: func private @box3() -> !fir.box<!fir.type<derived3{f:f32}>>
func.func private @it1() -> !fir.int<4>
func.func private @box1() -> !fir.boxchar<2>
func.func private @box2() -> !fir.boxproc<(i32, i32) -> i64>
func.func private @box3() -> !fir.box<!fir.type<derived3{f:f32}>>

// Fortran SUBROUTINE and FUNCTION
// CHECK-LABEL: func private @print_index3(index, index, index)
// CHECK: func private @user_i64(i64)
// CHECK: func private @user_tdesc(!fir.tdesc<!fir.type<x>>)
func.func private @print_index3(index, index, index)
func.func private @user_i64(i64)
func.func private @user_tdesc(!fir.tdesc<!fir.type<x>>)

// expect the void return to be omitted
// CHECK-LABEL: func private @store_tuple(tuple<!fir.type<qq1{f1:i32}>>)
// CHECK: func private @get_method_box() -> !fir.box<!fir.type<derived3{f:f32}>>
// CHECK: func private @method_impl(!fir.box<!fir.type<derived3{f:f32}>>)
func.func private @store_tuple(tuple<!fir.type<qq1{f1:i32}>>) -> ()
func.func private @get_method_box() -> !fir.box<!fir.type<derived3{f:f32}>>
func.func private @method_impl(!fir.box<!fir.type<derived3{f:f32}>>)

// CHECK-LABEL: func private @nop()
// CHECK-LABEL: func private @get_func() -> (() -> ())
func.func private @nop()
func.func private @get_func() -> (() -> ())

// CHECK-LABEL:       func @instructions() {
func.func @instructions() {
// CHECK: [[VAL_0:%.*]] = fir.alloca !fir.array<10xi32>
// CHECK: [[VAL_1:%.*]] = fir.load [[VAL_0]] : !fir.ref<!fir.array<10xi32>>
// CHECK: [[VAL_2:%.*]] = fir.alloca i32
// CHECK: [[VAL_3:%.*]] = arith.constant 22 : i32
  %0 = fir.alloca !fir.array<10xi32>
  %1 = fir.load %0 : !fir.ref<!fir.array<10xi32>>
  %2 = fir.alloca i32
  %3 = arith.constant 22 : i32

// CHECK: fir.store [[VAL_3]] to [[VAL_2]] : !fir.ref<i32>
// CHECK: [[VAL_4:%.*]] = fir.undefined i32
// CHECK: [[VAL_5:%.*]] = fir.allocmem !fir.array<100xf32>
// CHECK: [[VAL_6:%.*]] = fir.embox [[VAL_5]] : (!fir.heap<!fir.array<100xf32>>) -> !fir.box<!fir.array<100xf32>>
  fir.store %3 to %2 : !fir.ref<i32>
  %4 = fir.undefined i32
  %5 = fir.allocmem !fir.array<100xf32>
  %6 = fir.embox %5 : (!fir.heap<!fir.array<100xf32>>) -> !fir.box<!fir.array<100xf32>>

// CHECK: [[VAL_7:%.*]] = fir.box_addr [[VAL_6]] : (!fir.box<!fir.array<100xf32>>) -> !fir.ref<!fir.array<100xf32>>
  %7 = fir.box_addr %6 : (!fir.box<!fir.array<100xf32>>) -> !fir.ref<!fir.array<100xf32>>
// CHECK: %[[WAL_2:.*]] = fir.undefined !fir.boxproc<() -> ()>
  %ba1 = fir.undefined !fir.boxproc<() -> ()>
// CHECK: %{{.*}} = fir.box_addr %[[WAL_2]] : (!fir.boxproc<() -> ()>) -> (() -> ())
  %ba2 = fir.box_addr %ba1 : (!fir.boxproc<() -> ()>) -> (() -> ())
  %ba3 = fir.undefined !fir.boxchar<1>
// CHECK: %{{.*}} = fir.box_addr %{{.*}} : (!fir.boxchar<1>) -> !fir.ref<!fir.char<1>>
  %ba4 = fir.box_addr %ba3 : (!fir.boxchar<1>) -> !fir.ref<!fir.char<1>>
  %c0 = arith.constant 0 : index
  %d1:3 = fir.box_dims %6, %c0 : (!fir.box<!fir.array<100xf32>>, index) -> (index, index, index)
// CHECK: [[VAL_8:%.*]] = arith.constant 0 : index
// CHECK: [[VAL_9:%.*]]:3 = fir.box_dims [[VAL_6]], [[VAL_8]] : (!fir.box<!fir.array<100xf32>>, index) -> (index, index, index)
// CHECK: fir.call @print_index3([[VAL_9]]#0, [[VAL_9]]#1, [[VAL_9]]#2) : (index, index, index) -> ()
// CHECK: [[VAL_10:%.*]] = fir.call @it1() : () -> !fir.int<4>
  fir.call @print_index3(%d1#0, %d1#1, %d1#2) : (index, index, index) -> ()
  %8 = fir.call @it1() : () -> !fir.int<4>

// CHECK: [[VAL_11:%.*]] = fir.box_elesize [[VAL_6]] : (!fir.box<!fir.array<100xf32>>) -> i64
// CHECK: [[VAL_12:%.*]] = fir.box_isalloc [[VAL_6]] : (!fir.box<!fir.array<100xf32>>) -> i1
// CHECK: [[VAL_13:%.*]] = fir.box_isarray [[VAL_6]] : (!fir.box<!fir.array<100xf32>>) -> i1
// CHECK: [[VAL_14:%.*]] = fir.box_isptr [[VAL_6]] : (!fir.box<!fir.array<100xf32>>) -> i1
// CHECK: [[VAL_15:%.*]] = fir.box_rank [[VAL_6]] : (!fir.box<!fir.array<100xf32>>) -> i64
  %9 = fir.box_elesize %6 : (!fir.box<!fir.array<100xf32>>) -> i64
  %10 = fir.box_isalloc %6 : (!fir.box<!fir.array<100xf32>>) -> i1
  %11 = fir.box_isarray %6 : (!fir.box<!fir.array<100xf32>>) -> i1
  %12 = fir.box_isptr %6 : (!fir.box<!fir.array<100xf32>>) -> i1
  %13 = fir.box_rank %6 : (!fir.box<!fir.array<100xf32>>) -> i64

// CHECK: [[VAL_16:%.*]] = fir.box_tdesc [[VAL_6]] : (!fir.box<!fir.array<100xf32>>) -> !fir.tdesc<!fir.array<100xf32>>
// CHECK: [[VAL_17:%.*]] = fir.call @box1() : () -> !fir.boxchar<2>
// CHECK: [[VAL_18:%.*]] = fir.boxchar_len [[VAL_17]] : (!fir.boxchar<2>) -> i32
// CHECK: [[VAL_19:%.*]] = fir.call @box2() : () -> !fir.boxproc<(i32, i32) -> i64>
// CHECK: [[VAL_20:%.*]] = fir.boxproc_host [[VAL_19]] : (!fir.boxproc<(i32, i32) -> i64>) -> !fir.ref<i32>
  %14 = fir.box_tdesc %6 : (!fir.box<!fir.array<100xf32>>) -> !fir.tdesc<!fir.array<100xf32>>
  %15 = fir.call @box1() : () -> !fir.boxchar<2>
  %16 = fir.boxchar_len %15 : (!fir.boxchar<2>) -> i32
  %17 = fir.call @box2() : () -> !fir.boxproc<(i32, i32) -> i64>
  %18 = fir.boxproc_host %17 : (!fir.boxproc<(i32, i32) -> i64>) -> !fir.ref<i32>

// CHECK: [[VAL_21:%.*]] = arith.constant 10 : i32
// CHECK: [[VAL_22:%.*]] = fir.coordinate_of [[VAL_5]], [[VAL_21]] : (!fir.heap<!fir.array<100xf32>>, i32) -> !fir.ref<f32>
// CHECK: [[VAL_23:%.*]] = fir.field_index f, !fir.type<derived{f:f32}>
// CHECK: [[VAL_24:%.*]] = fir.undefined !fir.type<derived{f:f32}>
// CHECK: [[VAL_25:%.*]] = fir.extract_value [[VAL_24]], ["f", !fir.type<derived{f:f32}>] : (!fir.type<derived{f:f32}>) -> f32
  %19 = arith.constant 10 : i32
  %20 = fir.coordinate_of %5, %19 : (!fir.heap<!fir.array<100xf32>>, i32) -> !fir.ref<f32>
  %21 = fir.field_index f, !fir.type<derived{f:f32}>
  %22 = fir.undefined !fir.type<derived{f:f32}>
  %23 = fir.extract_value %22, ["f", !fir.type<derived{f:f32}>] : (!fir.type<derived{f:f32}>) -> f32

// CHECK: [[VAL_26:%.*]] = arith.constant 1 : i32
// CHECK: [[VAL_27:%.*]] = fir.shape [[VAL_21]] : (i32) -> !fir.shape<1>
// CHECK: [[VAL_28:%.*]] = arith.constant 1.0
// CHECK: [[VAL_29:%.*]] = fir.insert_value [[VAL_24]], [[VAL_28]], ["f", !fir.type<derived{f:f32}>] : (!fir.type<derived{f:f32}>, f32) -> !fir.type<derived{f:f32}>
// CHECK: [[VAL_30:%.*]] = fir.len_param_index f, !fir.type<derived3{f:f32}>
  %c1 = arith.constant 1 : i32
  %24 = fir.shape %19 : (i32) -> !fir.shape<1>
  %cf1 = arith.constant 1.0 : f32
  %25 = fir.insert_value %22, %cf1, ["f", !fir.type<derived{f:f32}>] : (!fir.type<derived{f:f32}>, f32) -> !fir.type<derived{f:f32}>
  %26 = fir.len_param_index f, !fir.type<derived3{f:f32}>

// CHECK: [[VAL_31:%.*]] = fir.call @box3() : () -> !fir.class<!fir.type<derived3{f:f32}>>
// CHECK: [[VAL_32:%.*]] = fir.dispatch "method"([[VAL_31]] : !fir.class<!fir.type<derived3{f:f32}>>) -> i32
// CHECK: [[VAL_33:%.*]] = fir.convert [[VAL_32]] : (i32) -> i64
// CHECK: [[VAL_34:%.*]] = fir.gentypedesc !fir.type<x>
// CHECK: fir.call @user_tdesc([[VAL_34]]) : (!fir.tdesc<!fir.type<x>>) -> ()
// CHECK: [[VAL_35:%.*]] = fir.no_reassoc [[VAL_33]] : i64
  %27 = fir.call @box3() : () -> !fir.class<!fir.type<derived3{f:f32}>>
  %28 = fir.dispatch "method"(%27 : !fir.class<!fir.type<derived3{f:f32}>>) -> i32
  %29 = fir.convert %28 : (i32) -> i64
  %30 = fir.gentypedesc !fir.type<x>
  fir.call @user_tdesc(%30) : (!fir.tdesc<!fir.type<x>>) -> ()
  %31 = fir.no_reassoc %29 : i64

// CHECK: fir.call @user_i64([[VAL_35]]) : (i64) -> ()
// CHECK: fir.freemem [[VAL_5]] : !fir.heap<!fir.array<100xf32>>
// CHECK: [[VAL_36:%.*]] = fir.call @get_func() : () -> (() -> ())
// CHECK: fir.call [[VAL_36]]() : () -> ()
// CHECK: [[VAL_37:%.*]] = fir.address_of(@it1) : !fir.ref<() -> !fir.int<4>>
// CHECK: return
// CHECK: }
  fir.call @user_i64(%31) : (i64) -> ()
  fir.freemem %5 : !fir.heap<!fir.array<100xf32>>
  %32 = fir.call @get_func() : () -> (() -> ())
  fir.call %32() : () -> ()
  %33 = fir.address_of (@it1) : !fir.ref<() -> !fir.int<4>>
  return
}

// CHECK-LABEL: func @boxing_match() {
func.func @boxing_match() {
// CHECK: [[VAL_38:%.*]] = fir.alloca i32
// CHECK: [[VAL_39:%.*]] = fir.alloca !fir.type<qq2{f1:i32,f2:f64}>
// CHECK: [[VAL_40:%.*]] = fir.alloca !fir.char<1>
// CHECK: [[VAL_41:%.*]] = fir.alloca tuple<i32, f64>
// CHECK: [[VAL_42:%.*]] = fir.embox [[VAL_38]] : (!fir.ref<i32>) -> !fir.box<i32>
// CHECK: [[VAL_44:%.*]] = arith.constant 8 : i32
// CHECK: [[VAL_45:%.*]] = fir.undefined !fir.char<1>
// CHECK: [[VAL_46:%.*]] = fir.emboxchar [[VAL_40]], [[VAL_44]] : (!fir.ref<!fir.char<1>>, i32) -> !fir.boxchar<1>
// CHECK: [[VAL_47:%.*]]:2 = fir.unboxchar [[VAL_46]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1>>, i32)
// CHECK: [[VAL_48:%.*]] = fir.undefined !fir.type<qq2{f1:i32,f2:f64}>
// CHECK: [[VAL_49:%.*]] = arith.constant 0 : i32
// CHECK: [[VAL_50:%.*]] = arith.constant 12 : i32
// CHECK: [[VAL_51:%.*]] = fir.insert_value [[VAL_48]], [[VAL_50]], [0 : i32] : (!fir.type<qq2{f1:i32,f2:f64}>, i32) -> !fir.type<qq2{f1:i32,f2:f64}>
// CHECK: [[VAL_52:%.*]] = arith.constant 1 : i32
// CHECK: [[VAL_53:%.*]] = arith.constant 4.213000e+01 : f64
// CHECK: [[VAL_54:%.*]] = fir.insert_value [[VAL_48]], [[VAL_53]], [1 : i32] : (!fir.type<qq2{f1:i32,f2:f64}>, f64) -> !fir.type<qq2{f1:i32,f2:f64}>
// CHECK: fir.store [[VAL_54]] to [[VAL_39]] : !fir.ref<!fir.type<qq2{f1:i32,f2:f64}>>
// CHECK: %[[WAL_1:.*]] = fir.address_of(@method_impl)
// CHECK: [[VAL_55:%.*]] = fir.emboxproc %[[WAL_1]], [[VAL_41]] : ((!fir.box<!fir.type<derived3{f:f32}>>) -> (), !fir.ref<tuple<i32, f64>>) -> !fir.boxproc<(!fir.box<!fir.type<derived3{f:f32}>>) -> ()>
// CHECK: [[VAL_56:%.*]], [[VAL_57:%.*]] = fir.unboxproc [[VAL_55]] : (!fir.boxproc<(!fir.box<!fir.type<derived3{f:f32}>>) -> ()>) -> ((!fir.box<!fir.type<derived3{f:f32}>>) -> (), !fir.ref<tuple<!fir.type<qq2{f1:i32,f2:f64}>>>)
// CHECK: [[VAL_58:%.*]] = fir.call @box2() : () -> !fir.boxproc<(i32, i32) -> i64>
// CHECK: [[VAL_59:%.*]], [[VAL_60:%.*]] = fir.unboxproc [[VAL_58]] : (!fir.boxproc<(i32, i32) -> i64>) -> ((i32, i32) -> i64, !fir.ref<tuple<!fir.type<qq1{f1:i32}>>>)
// CHECK: [[VAL_61:%.*]] = fir.load [[VAL_60]] : !fir.ref<tuple<!fir.type<qq1{f1:i32}>>>
// CHECK: fir.call @store_tuple([[VAL_61]]) : (tuple<!fir.type<qq1{f1:i32}>>) -> ()
// CHECK: return
// CHECK: }
  %0 = fir.alloca i32
  %d6 = fir.alloca !fir.type<qq2{f1:i32,f2:f64}>
  %d3 = fir.alloca !fir.char<1>
  %e6 = fir.alloca tuple<i32,f64>
  %1 = fir.embox %0 : (!fir.ref<i32>) -> !fir.box<i32>
  %c8 = arith.constant 8 : i32
  %3 = fir.undefined !fir.char<1>
  %4 = fir.emboxchar %d3, %c8 : (!fir.ref<!fir.char<1>>, i32) -> !fir.boxchar<1>
  %5:2 = fir.unboxchar %4 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1>>, i32)
  %6 = fir.undefined !fir.type<qq2{f1:i32,f2:f64}>
  %z = arith.constant 0 : i32
  %c12 = arith.constant 12 : i32
  %a2 = fir.insert_value %6, %c12, [0 : i32] : (!fir.type<qq2{f1:i32,f2:f64}>, i32) -> !fir.type<qq2{f1:i32,f2:f64}>
  %z1 = arith.constant 1 : i32
  %c42 = arith.constant 42.13 : f64
  %a3 = fir.insert_value %6, %c42, [1 : i32] : (!fir.type<qq2{f1:i32,f2:f64}>, f64) -> !fir.type<qq2{f1:i32,f2:f64}>
  fir.store %a3 to %d6 : !fir.ref<!fir.type<qq2{f1:i32,f2:f64}>>
  %mi = fir.address_of(@method_impl) : (!fir.box<!fir.type<derived3{f:f32}>>) -> ()
  %7 = fir.emboxproc %mi, %e6 : ((!fir.box<!fir.type<derived3{f:f32}>>) -> (), !fir.ref<tuple<i32,f64>>) -> !fir.boxproc<(!fir.box<!fir.type<derived3{f:f32}>>) -> ()>
  %8:2 = fir.unboxproc %7 : (!fir.boxproc<(!fir.box<!fir.type<derived3{f:f32}>>) -> ()>) -> ((!fir.box<!fir.type<derived3{f:f32}>>) -> (), !fir.ref<tuple<!fir.type<qq2{f1:i32,f2:f64}>>>)
  %9 = fir.call @box2() : () -> !fir.boxproc<(i32, i32) -> i64>
  %10:2 = fir.unboxproc %9 : (!fir.boxproc<(i32, i32) -> i64>) -> ((i32, i32) -> i64, !fir.ref<tuple<!fir.type<qq1{f1:i32}>>>)
  %11 = fir.load %10#1 : !fir.ref<tuple<!fir.type<qq1{f1:i32}>>>
  fir.call @store_tuple(%11) : (tuple<!fir.type<qq1{f1:i32}>>) -> ()
  return
}

// CHECK-LABEL: func @loop() {
func.func @loop() {
// CHECK: [[VAL_62:%.*]] = arith.constant 1 : index
// CHECK: [[VAL_63:%.*]] = arith.constant 10 : index
// CHECK: [[VAL_64:%.*]] = arith.constant true
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %ct = arith.constant true

// CHECK: fir.do_loop [[VAL_65:%.*]] = [[VAL_62]] to [[VAL_63]] step [[VAL_62]] {
// CHECK: fir.if [[VAL_64]] {
// CHECK: fir.call @nop() : () -> ()
// CHECK: } else {
// CHECK: fir.call @nop() : () -> ()
// CHECK: }
// CHECK: }
// CHECK: fir.unreachable
// CHECK: }
  fir.do_loop %i = %c1 to %c10 step %c1 {
    fir.if %ct {
      fir.call @nop() : () -> ()
    } else {
      fir.call @nop() : () -> ()
    }
  }
  fir.unreachable
}

// CHECK: func @bar_select([[VAL_66:%.*]]: i32, [[VAL_67:%.*]]: i32) -> i32 {
func.func @bar_select(%arg : i32, %arg2 : i32) -> i32 {
// CHECK: [[VAL_68:%.*]] = arith.constant 1 : i32
// CHECK: [[VAL_69:%.*]] = arith.constant 2 : i32
// CHECK: [[VAL_70:%.*]] = arith.constant 3 : i32
// CHECK: [[VAL_71:%.*]] = arith.constant 4 : i32
  %0 = arith.constant 1 : i32
  %1 = arith.constant 2 : i32
  %2 = arith.constant 3 : i32
  %3 = arith.constant 4 : i32

// CHECK: fir.select [[VAL_66]] : i32 [1, ^bb1([[VAL_68]] : i32), 2, ^bb2([[VAL_70]], [[VAL_66]], [[VAL_67]] : i32, i32, i32), -3, ^bb3([[VAL_67]], [[VAL_70]] : i32, i32), 4, ^bb4([[VAL_69]] : i32), unit, ^bb5]
// CHECK: ^bb1([[VAL_72:%.*]]: i32):
// CHECK: return [[VAL_72]] : i32
// CHECK: ^bb2([[VAL_73:%.*]]: i32, [[VAL_74:%.*]]: i32, [[VAL_75:%.*]]: i32):
// CHECK: [[VAL_76:%.*]] = arith.addi [[VAL_73]], [[VAL_74]] : i32
// CHECK: [[VAL_77:%.*]] = arith.addi [[VAL_76]], [[VAL_75]] : i32
// CHECK: return [[VAL_77]] : i32
// CHECK: ^bb3([[VAL_78:%.*]]: i32, [[VAL_79:%.*]]: i32):
// CHECK: [[VAL_80:%.*]] = arith.addi [[VAL_78]], [[VAL_79]] : i32
// CHECK: return [[VAL_80]] : i32
// CHECK: ^bb4([[VAL_81:%.*]]: i32):
// CHECK: return [[VAL_81]] : i32
// CHECK: ^bb5:
// CHECK: [[VAL_82:%.*]] = arith.constant 0 : i32
// CHECK: return [[VAL_82]] : i32
// CHECK: }
  fir.select %arg:i32 [ 1,^bb1(%0:i32), 2,^bb2(%2,%arg,%arg2:i32,i32,i32), -3,^bb3(%arg2,%2:i32,i32), 4,^bb4(%1:i32), unit,^bb5 ]
^bb1(%a : i32) :
  return %a : i32
^bb2(%b : i32, %b2 : i32, %b3:i32) :
  %4 = arith.addi %b, %b2 : i32
  %5 = arith.addi %4, %b3 : i32
  return %5 : i32
^bb3(%c:i32, %c2:i32) :
  %6 = arith.addi %c, %c2 : i32
  return %6 : i32
^bb4(%d : i32) :
  return %d : i32
^bb5 :
  %zero = arith.constant 0 : i32
  return %zero : i32
}

// CHECK-LABEL: func @bar_select_rank(
// CHECK-SAME: [[VAL_83:%.*]]: i32, [[VAL_84:%.*]]: i32) -> i32 {
func.func @bar_select_rank(%arg : i32, %arg2 : i32) -> i32 {
// CHECK: [[VAL_85:%.*]] = arith.constant 1 : i32
// CHECK: [[VAL_86:%.*]] = arith.constant 2 : i32
// CHECK: [[VAL_87:%.*]] = arith.constant 3 : i32
// CHECK: [[VAL_88:%.*]] = arith.constant 4 : i32
  %0 = arith.constant 1 : i32
  %1 = arith.constant 2 : i32
  %2 = arith.constant 3 : i32
  %3 = arith.constant 4 : i32

// CHECK: fir.select_rank [[VAL_83]] : i32 [1, ^bb1([[VAL_85]] : i32), 2, ^bb2([[VAL_87]], [[VAL_83]], [[VAL_84]] : i32, i32, i32), 3, ^bb3([[VAL_84]], [[VAL_87]] : i32, i32), -1, ^bb4([[VAL_86]] : i32), unit, ^bb5]
// CHECK: ^bb1([[VAL_89:%.*]]: i32):
// CHECK: return [[VAL_89]] : i32
// CHECK: ^bb2([[VAL_90:%.*]]: i32, [[VAL_91:%.*]]: i32, [[VAL_92:%.*]]: i32):
// CHECK: [[VAL_93:%.*]] = arith.addi [[VAL_90]], [[VAL_91]] : i32
// CHECK: [[VAL_94:%.*]] = arith.addi [[VAL_93]], [[VAL_92]] : i32
// CHECK: return [[VAL_94]] : i32
  fir.select_rank %arg:i32 [ 1,^bb1(%0:i32), 2,^bb2(%2,%arg,%arg2:i32,i32,i32), 3,^bb3(%arg2,%2:i32,i32), -1,^bb4(%1:i32), unit,^bb5 ]
^bb1(%a : i32) :
  return %a : i32
^bb2(%b : i32, %b2 : i32, %b3:i32) :
  %4 = arith.addi %b, %b2 : i32
  %5 = arith.addi %4, %b3 : i32
  return %5 : i32

// CHECK: ^bb3([[VAL_95:%.*]]: i32, [[VAL_96:%.*]]: i32):
// CHECK: [[VAL_97:%.*]] = arith.addi [[VAL_95]], [[VAL_96]] : i32
// CHECK: return [[VAL_97]] : i32
// CHECK: ^bb4([[VAL_98:%.*]]: i32):
// CHECK: return [[VAL_98]] : i32
^bb3(%c:i32, %c2:i32) :
  %6 = arith.addi %c, %c2 : i32
  return %6 : i32
^bb4(%d : i32) :
  return %d : i32

// CHECK: ^bb5:
// CHECK: [[VAL_99:%.*]] = arith.constant 0 : i32
// CHECK: [[VAL_100:%.*]] = fir.call @get_method_box() : () -> !fir.class<!fir.type<derived3{f:f32}>>
// CHECK: fir.dispatch "method"([[VAL_100]] : !fir.class<!fir.type<derived3{f:f32}>>)
^bb5 :
  %zero = arith.constant 0 : i32
  %7 = fir.call @get_method_box() : () -> !fir.class<!fir.type<derived3{f:f32}>>
  fir.dispatch "method"(%7 : !fir.class<!fir.type<derived3{f:f32}>>)

// CHECK: return [[VAL_99]] : i32
// CHECK: }
  return %zero : i32
}

// CHECK-LABEL: func @bar_select_type(
// CHECK-SAME: [[VAL_101:%.*]]: !fir.box<!fir.type<name(param1:i32){fld:!fir.char<1>}>>) -> i32 {
func.func @bar_select_type(%arg : !fir.box<!fir.type<name(param1:i32){fld:!fir.char<1>}>>) -> i32 {

// CHECK: [[VAL_102:%.*]] = arith.constant 1 : i32
// CHECK: [[VAL_103:%.*]] = arith.constant 2 : i32
// CHECK: [[VAL_104:%.*]] = arith.constant 3 : i32
// CHECK: [[VAL_105:%.*]] = arith.constant 4 : i32
  %0 = arith.constant 1 : i32
  %1 = arith.constant 2 : i32
  %2 = arith.constant 3 : i32
  %3 = arith.constant 4 : i32

// CHECK: fir.select_type [[VAL_101]] : !fir.box<!fir.type<name(param1:i32){fld:!fir.char<1>}>> [#fir.instance<!fir.int<4>>, ^bb1([[VAL_102]] : i32), #fir.instance<!fir.int<8>>, ^bb2([[VAL_104]] : i32), #fir.subsumed<!fir.int<2>>, ^bb3([[VAL_104]] : i32), #fir.instance<!fir.int<1>>, ^bb4([[VAL_103]] : i32), unit, ^bb5]
  fir.select_type %arg : !fir.box<!fir.type<name(param1:i32){fld:!fir.char<1>}>> [ #fir.instance<!fir.int<4>>,^bb1(%0:i32), #fir.instance<!fir.int<8>>,^bb2(%2:i32), #fir.subsumed<!fir.int<2>>,^bb3(%2:i32), #fir.instance<!fir.int<1>>,^bb4(%1:i32), unit,^bb5 ]

// CHECK: ^bb1([[VAL_106:%.*]]: i32):
// CHECK: return [[VAL_106]] : i32
// CHECK: ^bb2([[VAL_107:%.*]]: i32):
// CHECK: return [[VAL_107]] : i32
// CHECK: ^bb3([[VAL_108:%.*]]: i32):
// CHECK: return [[VAL_108]] : i32
// CHECK: ^bb4([[VAL_109:%.*]]: i32):
// CHECK: return [[VAL_109]] : i32
^bb1(%a : i32) :
  return %a : i32
^bb2(%b : i32) :
  return %b : i32
^bb3(%c : i32) :
  return %c : i32
^bb4(%d : i32) :
  return %d : i32

// CHECK: ^bb5:
// CHECK: [[VAL_110:%.*]] = arith.constant 0 : i32
// CHECK: return [[VAL_110]] : i32
// CHECK: }
^bb5 :
  %zero = arith.constant 0 : i32
  return %zero : i32
}

// CHECK-LABEL: func @bar_select_case(
// CHECK-SAME: [[VAL_111:%.*]]: i32, [[VAL_112:%.*]]: i32) -> i32 {
// CHECK: [[VAL_113:%.*]] = arith.constant 1 : i32
// CHECK: [[VAL_114:%.*]] = arith.constant 2 : i32
// CHECK: [[VAL_115:%.*]] = arith.constant 3 : i32
// CHECK: [[VAL_116:%.*]] = arith.constant 4 : i32
func.func @bar_select_case(%arg : i32, %arg2 : i32) -> i32 {
  %0 = arith.constant 1 : i32
  %1 = arith.constant 2 : i32
  %2 = arith.constant 3 : i32
  %3 = arith.constant 4 : i32

// CHECK: fir.select_case [[VAL_111]] : i32 [#fir.point, [[VAL_113]], ^bb1([[VAL_113]] : i32), #fir.lower, [[VAL_114]], ^bb2([[VAL_115]], [[VAL_111]], [[VAL_112]], [[VAL_114]] : i32, i32, i32, i32), #fir.interval, [[VAL_115]], [[VAL_116]], ^bb3([[VAL_115]], [[VAL_112]] : i32, i32), #fir.upper, [[VAL_111]], ^bb4([[VAL_114]] : i32), unit, ^bb5]
  fir.select_case %arg : i32 [#fir.point, %0, ^bb1(%0:i32), #fir.lower, %1, ^bb2(%2,%arg,%arg2,%1:i32,i32,i32,i32), #fir.interval, %2, %3, ^bb3(%2,%arg2:i32,i32), #fir.upper, %arg, ^bb4(%1:i32), unit, ^bb5]

// CHECK: ^bb1([[VAL_117:%.*]]: i32):
// CHECK: return [[VAL_117]] : i32
// CHECK: ^bb2([[VAL_118:%.*]]: i32, [[VAL_119:%.*]]: i32, [[VAL_120:%.*]]: i32, [[VAL_121:%.*]]: i32):
// CHECK: [[VAL_122:%.*]] = arith.addi [[VAL_118]], [[VAL_119]] : i32
// CHECK: [[VAL_123:%.*]] = arith.muli [[VAL_122]], [[VAL_120]] : i32
// CHECK: [[VAL_124:%.*]] = arith.addi [[VAL_123]], [[VAL_121]] : i32
// CHECK: return [[VAL_124]] : i32
// CHECK: ^bb3([[VAL_125:%.*]]: i32, [[VAL_126:%.*]]: i32):
// CHECK: [[VAL_127:%.*]] = arith.addi [[VAL_125]], [[VAL_126]] : i32
// CHECK: return [[VAL_127]] : i32
// CHECK: ^bb4([[VAL_128:%.*]]: i32):
// CHECK: return [[VAL_128]] : i32
^bb1(%a : i32) :
  return %a : i32
^bb2(%b : i32, %b2:i32, %b3:i32, %b4:i32) :
  %4 = arith.addi %b, %b2 : i32
  %5 = arith.muli %4, %b3 : i32
  %6 = arith.addi %5, %b4 : i32
  return %6 : i32
^bb3(%c : i32, %c2 : i32) :
  %7 = arith.addi %c, %c2 : i32
  return %7 : i32
^bb4(%d : i32) :
  return %d : i32

// CHECK: ^bb5:
// CHECK: [[VAL_129:%.*]] = arith.constant 0 : i32
// CHECK: return [[VAL_129]] : i32
// CHECK: }
^bb5 :
  %zero = arith.constant 0 : i32
  return %zero : i32
}

// CHECK-LABEL: fir.global @global_var : i32 {
// CHECK: [[VAL_130:%.*]] = arith.constant 1 : i32
// CHECK: fir.has_value [[VAL_130]] : i32
// CHECK: }
fir.global @global_var : i32 {
  %0 = arith.constant 1 : i32
  fir.has_value %0 : i32
}

// CHECK-LABEL: fir.global @global_constant constant : i32 {
// CHECK: [[VAL_131:%.*]] = arith.constant 934 : i32
// CHECK: fir.has_value [[VAL_131]] : i32
// CHECK: }
fir.global @global_constant constant : i32 {
  %0 = arith.constant 934 : i32
  fir.has_value %0 : i32
}

// CHECK-LABEL: fir.global @global_derived : !fir.type<minez(f:i32)> {
// CHECK: fir.global_len "f", 1 : i32
// CHECK: [[VAL_132:%.*]] = fir.undefined !fir.type<minez(f:i32)>
// CHECK: fir.has_value [[VAL_132]] : !fir.type<minez(f:i32)>
// CHECK: }
fir.global @global_derived : !fir.type<minez(f:i32)> {
  fir.global_len f, 1 : i32
  %0 = fir.undefined !fir.type<minez>
  fir.has_value %0 : !fir.type<minez>
}

// CHECK-LABEL: fir.global linkonce_odr @global_linkonce_odr : i32 {
// CHECK: [[VAL_130:%.*]] = arith.constant 1 : i32
// CHECK: fir.has_value [[VAL_130]] : i32
// CHECK: }
fir.global linkonce_odr @global_linkonce_odr : i32 {
  %0 = arith.constant 1 : i32
  fir.has_value %0 : i32
}

// CHECK-LABEL: fir.dispatch_table @dispatch_tbl {
// CHECK: fir.dt_entry "method", @method_impl
// CHECK: }
fir.dispatch_table @dispatch_tbl {
  fir.dt_entry "method", @method_impl
}

// CHECK-LABEL: func @compare_complex(
// CHECK-SAME: [[VAL_151:%.*]]: !fir.complex<16>, [[VAL_152:%.*]]: !fir.complex<16>) {
func.func @compare_complex(%a : !fir.complex<16>, %b : !fir.complex<16>) {

// CHECK: [[VAL_153:%.*]] = fir.cmpc "false", [[VAL_151]], [[VAL_152]] : !fir.complex<16>
// CHECK: [[VAL_154:%.*]] = fir.cmpc "oeq", [[VAL_151]], [[VAL_152]] : !fir.complex<16>
// CHECK: [[VAL_155:%.*]] = fir.cmpc "ogt", [[VAL_151]], [[VAL_152]] : !fir.complex<16>
// CHECK: [[VAL_156:%.*]] = fir.cmpc "oge", [[VAL_151]], [[VAL_152]] : !fir.complex<16>
  %d0 = fir.cmpc "false", %a, %b : !fir.complex<16>
  %d1 = fir.cmpc "oeq", %a, %b : !fir.complex<16>
  %d2 = fir.cmpc "ogt", %a, %b : !fir.complex<16>
  %d3 = fir.cmpc "oge", %a, %b : !fir.complex<16>

// CHECK: [[VAL_157:%.*]] = fir.cmpc "olt", [[VAL_151]], [[VAL_152]] : !fir.complex<16>
// CHECK: [[VAL_158:%.*]] = fir.cmpc "ole", [[VAL_151]], [[VAL_152]] : !fir.complex<16>
// CHECK: [[VAL_159:%.*]] = fir.cmpc "one", [[VAL_151]], [[VAL_152]] : !fir.complex<16>
// CHECK: [[VAL_160:%.*]] = fir.cmpc "ord", [[VAL_151]], [[VAL_152]] : !fir.complex<16>
  %a0 = fir.cmpc "olt", %a, %b : !fir.complex<16>
  %a1 = fir.cmpc "ole", %a, %b : !fir.complex<16>
  %a2 = fir.cmpc "one", %a, %b : !fir.complex<16>
  %a3 = fir.cmpc "ord", %a, %b : !fir.complex<16>

// CHECK: [[VAL_161:%.*]] = fir.cmpc "ueq", [[VAL_151]], [[VAL_152]] : !fir.complex<16>
// CHECK: [[VAL_162:%.*]] = fir.cmpc "ugt", [[VAL_151]], [[VAL_152]] : !fir.complex<16>
// CHECK: [[VAL_163:%.*]] = fir.cmpc "uge", [[VAL_151]], [[VAL_152]] : !fir.complex<16>
// CHECK: [[VAL_164:%.*]] = fir.cmpc "ult", [[VAL_151]], [[VAL_152]] : !fir.complex<16>
  %b0 = fir.cmpc "ueq", %a, %b : !fir.complex<16>
  %b1 = fir.cmpc "ugt", %a, %b : !fir.complex<16>
  %b2 = fir.cmpc "uge", %a, %b : !fir.complex<16>
  %b3 = fir.cmpc "ult", %a, %b : !fir.complex<16>

// CHECK: [[VAL_165:%.*]] = fir.cmpc "ule", [[VAL_151]], [[VAL_152]] : !fir.complex<16>
// CHECK: [[VAL_166:%.*]] = fir.cmpc "une", [[VAL_151]], [[VAL_152]] : !fir.complex<16>
// CHECK: [[VAL_167:%.*]] = fir.cmpc "uno", [[VAL_151]], [[VAL_152]] : !fir.complex<16>
// CHECK: [[VAL_168:%.*]] = fir.cmpc "true", [[VAL_151]], [[VAL_152]] : !fir.complex<16>
  %c0 = fir.cmpc "ule", %a, %b : !fir.complex<16>
  %c1 = fir.cmpc "une", %a, %b : !fir.complex<16>
  %c2 = fir.cmpc "uno", %a, %b : !fir.complex<16>
  %c3 = fir.cmpc "true", %a, %b : !fir.complex<16>
// CHECK: return
// CHECK: }
  return
}

// CHECK-LABEL: func @arith_real(
// CHECK-SAME: [[VAL_169:%.*]]: f128, [[VAL_170:%.*]]: f128) -> f128 {
func.func @arith_real(%a : f128, %b : f128) -> f128 {

// CHECK: [[VAL_171:%.*]] = arith.constant 1.0
// CHECK: [[VAL_172:%.*]] = fir.convert [[VAL_171]] : (f32) -> f128
// CHECK: [[VAL_173:%.*]] = arith.negf [[VAL_169]] : f128
// CHECK: [[VAL_174:%.*]] = arith.addf [[VAL_172]], [[VAL_173]] : f128
// CHECK: [[VAL_175:%.*]] = arith.subf [[VAL_174]], [[VAL_170]] : f128
// CHECK: [[VAL_176:%.*]] = arith.mulf [[VAL_173]], [[VAL_175]] : f128
// CHECK: [[VAL_177:%.*]] = arith.divf [[VAL_176]], [[VAL_169]] : f128
  %c1 = arith.constant 1.0 : f32
  %0 = fir.convert %c1 : (f32) -> f128
  %1 = arith.negf %a : f128
  %2 = arith.addf %0, %1 : f128
  %3 = arith.subf %2, %b : f128
  %4 = arith.mulf %1, %3 : f128
  %5 = arith.divf %4, %a : f128
// CHECK: return [[VAL_177]] : f128
// CHECK: }
  return %5 : f128
}

// CHECK-LABEL: func @arith_complex(
// CHECK-SAME: [[VAL_179:%.*]]: !fir.complex<16>, [[VAL_180:%.*]]: !fir.complex<16>) -> !fir.complex<16> {
func.func @arith_complex(%a : !fir.complex<16>, %b : !fir.complex<16>) -> !fir.complex<16> {
// CHECK: [[VAL_181:%.*]] = fir.negc [[VAL_179]] : !fir.complex<16>
// CHECK: [[VAL_182:%.*]] = fir.addc [[VAL_180]], [[VAL_181]] : !fir.complex<16>
// CHECK: [[VAL_183:%.*]] = fir.subc [[VAL_182]], [[VAL_180]] : !fir.complex<16>
// CHECK: [[VAL_184:%.*]] = fir.mulc [[VAL_181]], [[VAL_183]] : !fir.complex<16>
// CHECK: [[VAL_185:%.*]] = fir.divc [[VAL_184]], [[VAL_179]] : !fir.complex<16>
  %1 = fir.negc %a : !fir.complex<16>
  %2 = fir.addc %b, %1 : !fir.complex<16>
  %3 = fir.subc %2, %b : !fir.complex<16>
  %4 = fir.mulc %1, %3 : !fir.complex<16>
  %5 = fir.divc %4, %a : !fir.complex<16>
// CHECK: return [[VAL_185]] : !fir.complex<16>
// CHECK: }
  return %5 : !fir.complex<16>
}

// CHECK-LABEL: func @character_literal() -> !fir.char<1,13> {
func.func @character_literal() -> !fir.char<1,13> {
// CHECK: [[VAL_186:%.*]] = fir.string_lit "Hello, World!"(13) : !fir.char<1,13>
  %0 = fir.string_lit "Hello, World!"(13) : !fir.char<1,13>
// CHECK: return [[VAL_186]] : !fir.char<1,13>
  return %0 : !fir.char<1,13>
// CHECK: }
}

// CHECK-LABEL: func private @earlyexit2(i32) -> i1
func.func private @earlyexit2(%a : i32) -> i1

// CHECK-LABEL: func @early_exit(
// CHECK-SAME: [[VAL_187:%.*]]: i1, [[VAL_188:%.*]]: i32) -> i1 {
func.func @early_exit(%ok : i1, %k : i32) -> i1 {
// CHECK: [[VAL_189:%.*]] = arith.constant 1 : index
// CHECK: [[VAL_190:%.*]] = arith.constant 100 : index
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index

// CHECK: %[[VAL_191:.*]]:2 = fir.iterate_while ([[VAL_192:%.*]] = [[VAL_189]] to [[VAL_190]] step [[VAL_189]]) and ([[VAL_193:%.*]] = [[VAL_187]]) iter_args([[VAL_194:%.*]] = [[VAL_188]]) -> (i32) {
// CHECK: [[VAL_195:%.*]] = func.call @earlyexit2([[VAL_194]]) : (i32) -> i1
// CHECK: fir.result [[VAL_195]], [[VAL_194]] : i1, i32
// CHECK: }
  %newOk:2 = fir.iterate_while (%i = %c1 to %c100 step %c1) and (%ok_ = %ok) iter_args(%v = %k) -> (i32) {
    %stop = func.call @earlyexit2(%v) : (i32) -> i1
    fir.result %stop, %v : i1, i32
  }
// CHECK: return %[[VAL_191]]#0 : i1
// CHECK: }
  return %newOk#0 : i1
}

// CHECK-LABEL: @array_access
func.func @array_access(%arr : !fir.ref<!fir.array<?x?xf32>>) {
  // CHECK-DAG: %[[c1:.*]] = arith.constant 100
  // CHECK-DAG: %[[c2:.*]] = arith.constant 50
  %c100 = arith.constant 100 : index
  %c50 = arith.constant 50 : index
  // CHECK: %[[sh:.*]] = fir.shape %[[c1]], %[[c2]] : {{.*}} -> !fir.shape<2>
  %shape = fir.shape %c100, %c50 : (index, index) -> !fir.shape<2>
  %c47 = arith.constant 47 : index
  %c78 = arith.constant 78 : index
  %c3 = arith.constant 3 : index
  %c18 = arith.constant 18 : index
  %c36 = arith.constant 36 : index
  %c4 = arith.constant 4 : index
  // CHECK: %[[sl:.*]] = fir.slice {{.*}} -> !fir.slice<2>
  %slice = fir.slice %c47, %c78, %c3, %c18, %c36, %c4 : (index,index,index,index,index,index) -> !fir.slice<2>
  %c0 = arith.constant 0 : index
  %c99 = arith.constant 99 : index
  %c1 = arith.constant 1 : index
  fir.do_loop %i = %c0 to %c99 step %c1 {
    %c49 = arith.constant 49 : index
    fir.do_loop %j = %c0 to %c49 step %c1 {
      // CHECK: fir.array_coor %{{.*}}(%[[sh]]) [%[[sl]]] %{{.*}}, %{{.*}} :
      %p = fir.array_coor %arr(%shape)[%slice] %i, %j : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
      %x = arith.constant 42.0 : f32
      fir.store %x to %p : !fir.ref<f32>
    }
  }
  return
}

// CHECK-LABEL: @test_is_present
func.func @test_is_present(%arg0: !fir.box<!fir.array<?xf32>>) -> i1 {
  // CHECK: fir.is_present %{{.*}} : (!fir.box<!fir.array<?xf32>>) -> i1
  %0 = fir.is_present %arg0 : (!fir.box<!fir.array<?xf32>>) -> i1
  return %0 : i1
}
// CHECK-LABEL: @test_absent
func.func @test_absent() -> i1 {
  // CHECK: fir.absent !fir.box<!fir.array<?xf32>>
  %0 = fir.absent !fir.box<!fir.array<?xf32>>
  %1 = fir.call @_QPfoo(%0) : (!fir.box<!fir.array<?xf32>>) -> i1
  return %1 : i1
}

// CHECK-LABEL: @test_misc_ops(
// CHECK-SAME: [[ARR1:%.*]]: !fir.ref<!fir.array<?x?xf32>>, [[INDXM:%.*]]: index, [[INDXN:%.*]]: index, [[INDXO:%.*]]: index, [[INDXP:%.*]]: index)
func.func @test_misc_ops(%arr1 : !fir.ref<!fir.array<?x?xf32>>, %m : index, %n : index, %o : index, %p : index) {
  // CHECK: [[I10:%.*]] = arith.constant 10 : index
  // CHECK: [[J20:%.*]] = arith.constant 20 : index
  // CHECK: [[C2:%.*]] = arith.constant 2 : index
  // CHECK: [[C9:%.*]] = arith.constant 9 : index
  // CHECK: [[C1_I32:%.*]] = arith.constant 9 : i32
  %i10 = arith.constant 10 : index
  %j20 = arith.constant 20 : index
  %c2 = arith.constant 2 : index
  %c9 = arith.constant 9 : index
  %c1_i32 = arith.constant 9 : i32

  // CHECK: [[ARR2:%.*]] = fir.zero_bits !fir.array<10xi32>
  // CHECK: [[ARR3:%.*]] = fir.insert_on_range [[ARR2]], [[C1_I32]] from (2) to (9) : (!fir.array<10xi32>, i32) -> !fir.array<10xi32>
  // CHECK: fir.call @noret1([[ARR3]]) : (!fir.array<10xi32>) -> ()
  %arr2 = fir.zero_bits !fir.array<10xi32>
  %arr3 = fir.insert_on_range %arr2, %c1_i32 from (2) to (9) : (!fir.array<10xi32>, i32) -> !fir.array<10xi32>
  fir.call @noret1(%arr3) : (!fir.array<10xi32>) -> ()

  // CHECK: [[SHAPE:%.*]] = fir.shape_shift [[INDXM:%.*]], [[INDXN:%.*]], [[INDXO:%.*]], [[INDXP:%.*]] : (index, index, index, index) -> !fir.shapeshift<2>
  // CHECK: [[AV1:%.*]] = fir.array_load [[ARR1]]([[SHAPE]]) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  // CHECK: [[FVAL:%.*]] = fir.array_fetch [[AV1]], [[I10]], [[J20]] : (!fir.array<?x?xf32>, index, index) -> f32
  // CHECK: [[AV2:%.*]] = fir.array_update [[AV1]], [[FVAL]], [[I10]], [[J20]] : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
  // CHECK: fir.array_merge_store [[AV1]], [[AV2]] to [[ARR1]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.ref<!fir.array<?x?xf32>>
  %s = fir.shape_shift %m, %n, %o, %p : (index, index, index, index) -> !fir.shapeshift<2>
  %av1 = fir.array_load %arr1(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  %f = fir.array_fetch %av1, %i10, %j20 : (!fir.array<?x?xf32>, index, index) -> f32
  %av2 = fir.array_update %av1, %f, %i10, %j20 : (!fir.array<?x?xf32>, f32, index, index) -> !fir.array<?x?xf32>
  fir.array_merge_store %av1, %av2 to %arr1 : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.ref<!fir.array<?x?xf32>>

  // CHECK: [[AV3:%.*]] = fir.array_load [[ARR1]]([[SHAPE]]) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  // CHECK: [[FVAL2:%.*]] = fir.array_fetch [[AV3]], [[I10]], [[J20]] : (!fir.array<?x?xf32>, index, index) -> f32
  // CHECK: [[AV4:%.*]]:2 = fir.array_modify [[AV3]], [[I10]], [[J20]] : (!fir.array<?x?xf32>, index, index) -> (!fir.ref<f32>, !fir.array<?x?xf32>)
  // CHECK: fir.store [[FVAL2]] to [[AV4]]#0 : !fir.ref<f32>
  // CHECK: fir.array_merge_store [[AV3]], [[AV4]]#1 to [[ARR1]] : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.ref<!fir.array<?x?xf32>>
  %av3 = fir.array_load %arr1(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.array<?x?xf32>
  %f2 = fir.array_fetch %av3, %i10, %j20 : (!fir.array<?x?xf32>, index, index) -> f32
  %addr, %av4 = fir.array_modify %av3, %i10, %j20 : (!fir.array<?x?xf32>, index, index) -> (!fir.ref<f32>, !fir.array<?x?xf32>)
  fir.store %f2 to %addr : !fir.ref<f32>
  fir.array_merge_store %av3, %av4 to %arr1 : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.ref<!fir.array<?x?xf32>>
  return
}

// CHECK-LABEL: @test_const_complex
func.func @test_const_complex() {
 // CHECK-DAG: {{%.*}} = fir.constc(#fir.real<2, i x3000>, #fir.real<2, i x4C40>) : !fir.complex<2>
 // CHECK-DAG: {{%.*}} = fir.constc(#fir.real<3, i x3E80>, #fir.real<3, i x4202>) : !fir.complex<3>
 // CHECK-DAG: {{%.*}} = fir.constc(#fir.real<4, i x3E800000>, #fir.real<4, i x42028000>) : !fir.complex<4>
 // CHECK-DAG: {{%.*}} = fir.constc(#fir.real<8, i x3FD0000000000000>, #fir.real<8, i x4040500000000000>) : !fir.complex<8>
 // CHECK-DAG: {{%.*}} = fir.constc(#fir.real<10, i x3FFD8000000000000000>, #fir.real<10, i x40048280000000000000>) : !fir.complex<10>
 // CHECK-DAG: {{%.*}} = fir.constc(#fir.real<16, i x3FFD0000000000000000000000000000>, #fir.real<16, i x40040500000000000000000000000000>) : !fir.complex<16>
  %c2 = fir.constc (#fir.real<2, 0.125>, #fir.real<2, 17.0>) : !fir.complex<2>
  %c3 = fir.constc (#fir.real<3, 0.25>, #fir.real<3, 32.625>) : !fir.complex<3>
  %c4 = fir.constc (#fir.real<4, 0.25>, #fir.real<4, 32.625>) : !fir.complex<4>
  %c8 = fir.constc (#fir.real<8, 0.25>, #fir.real<8, 32.625>) : !fir.complex<8>
  %c10 = fir.constc (#fir.real<10, 0.25>, #fir.real<10, 32.625>) : !fir.complex<10>
  %c16 = fir.constc (#fir.real<16, 0.25>, #fir.real<16, 32.625>) : !fir.complex<16>
  return
}

// CHECK-LABEL: @insert_on_range_multi_dim
// CHECK-SAME: %[[ARR:.*]]: !fir.array<10x20xi32>, %[[CST:.*]]: i32
func.func @insert_on_range_multi_dim(%arr : !fir.array<10x20xi32>, %cst : i32) {
  // CHECK: fir.insert_on_range %[[ARR]], %[[CST]] from (2, 3) to (5, 6) : (!fir.array<10x20xi32>, i32) -> !fir.array<10x20xi32>
  %arr3 = fir.insert_on_range %arr, %cst from (2, 3) to (5, 6) : (!fir.array<10x20xi32>, i32) -> !fir.array<10x20xi32>
  return
}

// CHECK-LABEL: @test_shift
func.func @test_shift(%arg0: !fir.box<!fir.array<?xf32>>) -> !fir.ref<f32> {
  %c4 = arith.constant 4 : index
  %c100 = arith.constant 100 : index
  // CHECK: fir.shift %{{.*}} : (index) -> !fir.shift<1>
  %0 = fir.shift %c4 : (index) -> !fir.shift<1>
  %1 = fir.array_coor %arg0(%0) %c100 : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>, index) -> !fir.ref<f32>
  return %1 : !fir.ref<f32>
} 

func.func private @bar_rebox_test(!fir.box<!fir.array<?x?xf32>>)
// CHECK-LABEL: @test_rebox(
func.func @test_rebox(%arg0: !fir.box<!fir.array<?xf32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c10 = arith.constant 10 : index
  %c33 = arith.constant 33 : index
  %0 = fir.slice %c10, %c33, %c2 : (index, index, index) -> !fir.slice<1>
  %1 = fir.shift %c0 : (index) -> !fir.shift<1>
  // CHECK: fir.rebox %{{.*}}(%{{.*}}) [%{{.*}}] : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  %2 = fir.rebox %arg0(%1) [%0] : (!fir.box<!fir.array<?xf32>>, !fir.shift<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xf32>>
  %3 = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  // CHECK: fir.rebox %{{.*}}(%{{.*}}) : (!fir.box<!fir.array<?xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
  %4 = fir.rebox %2(%3) : (!fir.box<!fir.array<?xf32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
  fir.call @bar_rebox_test(%4) : (!fir.box<!fir.array<?x?xf32>>) -> ()
  return
}

func.func private @bar_rebox_test_char(!fir.box<!fir.array<?x!fir.char<1,?>>>)
// CHECK-LABEL: @test_rebox_char(
func.func @test_rebox_char(%arg0: !fir.box<!fir.array<?x!fir.char<1,20>>>) {
  %c7_i64 = arith.constant 7 : i64
  %c1_i64 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.array<?x!fir.char<1,20>>>, index) -> (index, index, index)
  %1 = fir.slice %c1, %0#1, %c1_i64 substr %c1_i64, %c7_i64 : (index, index, i64, i64, i64) -> !fir.slice<1>
  // CHECK: fir.rebox %{{.*}} [%{{.*}}] : (!fir.box<!fir.array<?x!fir.char<1,20>>>, !fir.slice<1>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
  %2 = fir.rebox %arg0 [%1] : (!fir.box<!fir.array<?x!fir.char<1,20>>>, !fir.slice<1>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
  fir.call @bar_rebox_test_char(%2) : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> ()
  return
}


func.func private @array_func() -> !fir.array<?x!fir.char<1,?>>
// CHECK-LABEL: @test_save_result(
func.func @test_save_result(%buffer: !fir.ref<!fir.array<?x!fir.char<1,?>>>) {
  %c100 = arith.constant 100 : index
  %c50 = arith.constant 50 : index
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res = fir.call @array_func() : () -> !fir.array<?x!fir.char<1,?>>
  // CHECK: fir.save_result %{{.*}} to %{{.*}}(%{{.*}}) typeparams %{{.*}} : !fir.array<?x!fir.char<1,?>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index
  fir.save_result %res to %buffer(%shape) typeparams %c50 : !fir.array<?x!fir.char<1,?>>, !fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index
  return
}

func.func @char_convert() {
  %1 = fir.undefined i32
  %2 = fir.undefined !fir.ref<!fir.char<1>>
  %3 = fir.undefined !fir.ref<!fir.array<?x!fir.char<2,?>>>
  // CHECK: fir.char_convert %{{.*}} for %{{.*}} to %{{.*}} : !fir.ref<!fir.char<1>>, i32, !fir.ref<!fir.array<?x!fir.char<2,?>>>
  fir.char_convert %2 for %1 to %3 : !fir.ref<!fir.char<1>>, i32, !fir.ref<!fir.array<?x!fir.char<2,?>>>
  return
}

func.func @slice_substr() {
  %lb = arith.constant 0 : index
  %ub = arith.constant 42 : index
  %c1 = arith.constant 1 : index
  %offset = arith.constant 10 : index
  %0 = fir.slice %lb, %ub, %c1 substr %offset, %c1 : (index, index, index, index, index) -> !fir.slice<1>
  // CHECK: fir.slice %{{.*}}, %{{.*}}, %{{.*}} substr %{{.*}}, %{{.*}} : (index, index, index, index, index) -> !fir.slice<1>
  return
}

// Test load, store, coordinate_of with llvmptr type
// CHECK-LABEL: llvm_ptr_load_store_coordinate
// CHECK-SAME: (%[[ARG0:.*]]: !fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>>>, %[[ARG1:.*]]: !fir.ref<!fir.box<!fir.ptr<f32>>>)
func.func @llvm_ptr_load_store_coordinate(%arg0: !fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>>>, %arg1: !fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>> {
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-NEXT: %[[LLVMPTR:.*]] = fir.coordinate_of %[[ARG0]], %[[C0]] : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<f32>>>>
  %0 = fir.coordinate_of %arg0, %c0_i32 : (!fir.ref<tuple<!fir.ref<!fir.box<!fir.ptr<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>>>, i32) -> !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<f32>>>>
  // CHECK-NEXT: fir.store %[[ARG1]] to %[[LLVMPTR]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<f32>>>>
  fir.store %arg1 to %0 : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<f32>>>>
  // CHECK-NEXT: fir.load %[[LLVMPTR]] : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<f32>>>>
  %1 = fir.load %0 : !fir.llvm_ptr<!fir.ref<!fir.box<!fir.ptr<f32>>>>
  return %1 : !fir.ref<!fir.box<!fir.ptr<f32>>>
}

func.func @array_access_ops(%a : !fir.ref<!fir.array<?x?xf32>>) {
  %c1 = arith.constant 1 : index
  %n = arith.constant 0 : index
  %m = arith.constant 50 : index
  %s = fir.shape %n, %m : (index, index) -> !fir.shape<2>
  %v = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
  %p = fir.array_access %v, %c1, %c1 : (!fir.array<?x?xf32>, index, index) -> !fir.ref<f32>
  // CHECK: %{{.*}} = fir.array_access %{{.*}}, %{{.*}}, %{{.*}} : (!fir.array<?x?xf32>, index, index) -> !fir.ref<f32>
  return
}

func.func @array_amend_ops(%a : !fir.ref<!fir.array<?x?xf32>>) {
  %c1 = arith.constant 1 : index
  %n = arith.constant 0 : index
  %m = arith.constant 50 : index
  %s = fir.shape %n, %m : (index, index) -> !fir.shape<2>
  %v = fir.array_load %a(%s) : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
  %p = fir.array_access %v, %c1, %c1 : (!fir.array<?x?xf32>, index, index) -> !fir.ref<f32>
  %res = fir.array_amend %v, %p : (!fir.array<?x?xf32>, !fir.ref<f32>) -> !fir.array<?x?xf32>
  // CHECK: %{{.*}} = fir.array_amend %{{.*}}, %{{.*}} : (!fir.array<?x?xf32>, !fir.ref<f32>) -> !fir.array<?x?xf32>
  return
}

func.func private @dispatch(%arg0: !fir.class<!fir.type<dispatch_derived1{a:i32,b:i32}>>, %arg1: i32) -> () {
  // CHECK-LABEL: func.func private @dispatch(
  // CHECK-SAME: %[[CLASS:.*]]: !fir.class<!fir.type<dispatch_derived1{a:i32,b:i32}>>, %[[INTARG:.*]]: i32)
  fir.dispatch "proc1"(%arg0 : !fir.class<!fir.type<dispatch_derived1{a:i32,b:i32}>>) (%arg0 : !fir.class<!fir.type<dispatch_derived1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}
  // CHECK: fir.dispatch "proc1"(%[[CLASS]] : !fir.class<!fir.type<dispatch_derived1{a:i32,b:i32}>>) (%[[CLASS]] : !fir.class<!fir.type<dispatch_derived1{a:i32,b:i32}>>) {pass_arg_pos = 0 : i32}

  fir.dispatch "proc2"(%arg0 : !fir.class<!fir.type<dispatch_derived1{a:i32,b:i32}>>)
  // CHECK: fir.dispatch "proc2"(%[[CLASS]] : !fir.class<!fir.type<dispatch_derived1{a:i32,b:i32}>>)

  fir.dispatch "proc3"(%arg0 : !fir.class<!fir.type<dispatch_derived1{a:i32,b:i32}>>) (%arg1, %arg0 : i32, !fir.class<!fir.type<dispatch_derived1{a:i32,b:i32}>>) {pass_arg_pos = 1 : i32}
  // CHECK: fir.dispatch "proc3"(%[[CLASS]] : !fir.class<!fir.type<dispatch_derived1{a:i32,b:i32}>>) (%[[INTARG]], %[[CLASS]] : i32, !fir.class<!fir.type<dispatch_derived1{a:i32,b:i32}>>) {pass_arg_pos = 1 : i32}
  return
}


// CHECK-LABEL: func.func @embox_tdesc
func.func @embox_tdesc(%arg0: !fir.class<!fir.array<10x!fir.type<derived_poly{a:i32,b:i32}>>>) {
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
    %12 = fir.coordinate_of %arg0, %11 : (!fir.class<!fir.array<10x!fir.type<derived_poly{a:i32,b:i32}>>>, i64) -> !fir.ref<!fir.type<derived_poly{a:i32,b:i32}>>
    %tdesc = fir.box_tdesc %arg0 : (!fir.class<!fir.array<10x!fir.type<derived_poly{a:i32,b:i32}>>>) -> (!fir.tdesc<!fir.type<derived_poly{a:i32,b:i32}>>)
    // CHECK: %[[TDESC:.*]] = fir.box_tdesc %{{.*}} : (!fir.class<!fir.array<10x!fir.type<derived_poly{a:i32,b:i32}>>>) -> !fir.tdesc<!fir.type<derived_poly{a:i32,b:i32}>> 
    %13 = fir.embox %12 tdesc %tdesc : (!fir.ref<!fir.type<derived_poly{a:i32,b:i32}>>, !fir.tdesc<!fir.type<derived_poly{a:i32,b:i32}>>) -> !fir.class<!fir.type<derived_poly{a:i32,b:i32}>>
    // CHECK: %{{.*}} = fir.embox %{{.*}} tdesc %[[TDESC]] : (!fir.ref<!fir.type<derived_poly{a:i32,b:i32}>>, !fir.tdesc<!fir.type<derived_poly{a:i32,b:i32}>>) -> !fir.class<!fir.type<derived_poly{a:i32,b:i32}>>
    %14 = arith.addi %arg2, %c1 : index
    %15 = fir.convert %c1 : (index) -> i32
    %16 = fir.load %0 : !fir.ref<i32>
    %17 = arith.addi %16, %15 : i32
    fir.result %14, %17 : index, i32
  }
  fir.store %4#1 to %0 : !fir.ref<i32>
  return
}
