// Test the FIR operations
// Parse operations and check that we can reparse what we print.
// RUN: fir-opt %s | fir-opt | FileCheck %s

// CHECK-LABEL: func private @it1() -> !fir.int<4>
// CHECK: func private @box1() -> !fir.boxchar<2>
// CHECK: func private @box2() -> !fir.boxproc<(i32, i32) -> i64>
// CHECK: func private @box3() -> !fir.box<!fir.type<derived3{f:f32}>>
func private @it1() -> !fir.int<4>
func private @box1() -> !fir.boxchar<2>
func private @box2() -> !fir.boxproc<(i32, i32) -> i64>
func private @box3() -> !fir.box<!fir.type<derived3{f:f32}>>

// Fortran SUBROUTINE and FUNCTION
// CHECK-LABEL: func private @print_index3(index, index, index)
// CHECK: func private @user_i64(i64)
// CHECK: func private @user_tdesc(!fir.tdesc<!fir.type<x>>)
func private @print_index3(index, index, index)
func private @user_i64(i64)
func private @user_tdesc(!fir.tdesc<!fir.type<x>>)

// expect the void return to be omitted
// CHECK-LABEL: func private @store_tuple(tuple<!fir.type<qq1{f1:i32}>>)
// CHECK: func private @get_method_box() -> !fir.box<!fir.type<derived3{f:f32}>>
// CHECK: func private @method_impl(!fir.box<!fir.type<derived3{f:f32}>>)
func private @store_tuple(tuple<!fir.type<qq1{f1:i32}>>) -> ()
func private @get_method_box() -> !fir.box<!fir.type<derived3{f:f32}>>
func private @method_impl(!fir.box<!fir.type<derived3{f:f32}>>)

// CHECK-LABEL: func private @nop()
// CHECK-LABEL: func private @get_func() -> (() -> ())
func private @nop()
func private @get_func() -> (() -> ())

// CHECK-LABEL:       func @instructions() {
func @instructions() {
// CHECK: [[VAL_0:%.*]] = fir.alloca !fir.array<10xi32>
// CHECK: [[VAL_1:%.*]] = fir.load [[VAL_0]] : !fir.ref<!fir.array<10xi32>>
// CHECK: [[VAL_2:%.*]] = fir.alloca i32
// CHECK: [[VAL_3:%.*]] = constant 22 : i32
  %0 = fir.alloca !fir.array<10xi32>
  %1 = fir.load %0 : !fir.ref<!fir.array<10xi32>>
  %2 = fir.alloca i32
  %3 = constant 22 : i32

// CHECK: fir.store [[VAL_3]] to [[VAL_2]] : !fir.ref<i32>
// CHECK: [[VAL_4:%.*]] = fir.undefined i32
// CHECK: [[VAL_5:%.*]] = fir.allocmem !fir.array<100xf32>
// CHECK: [[VAL_6:%.*]] = fir.embox [[VAL_5]] : (!fir.heap<!fir.array<100xf32>>) -> !fir.box<!fir.array<100xf32>>
  fir.store %3 to %2 : !fir.ref<i32>
  %4 = fir.undefined i32
  %5 = fir.allocmem !fir.array<100xf32>
  %6 = fir.embox %5 : (!fir.heap<!fir.array<100xf32>>) -> !fir.box<!fir.array<100xf32>>

// CHECK: [[VAL_7:%.*]] = fir.box_addr [[VAL_6]] : (!fir.box<!fir.array<100xf32>>) -> !fir.ref<!fir.array<100xf32>>
// CHECK: [[VAL_8:%.*]] = constant 0 : index
// CHECK: [[VAL_9:%.*]]:3 = fir.box_dims [[VAL_6]], [[VAL_8]] : (!fir.box<!fir.array<100xf32>>, index) -> (index, index, index)
// CHECK: fir.call @print_index3([[VAL_9]]#0, [[VAL_9]]#1, [[VAL_9]]#2) : (index, index, index) -> ()
// CHECK: [[VAL_10:%.*]] = fir.call @it1() : () -> !fir.int<4>
  %7 = fir.box_addr %6 : (!fir.box<!fir.array<100xf32>>) -> !fir.ref<!fir.array<100xf32>>
  %c0 = constant 0 : index
  %d1:3 = fir.box_dims %6, %c0 : (!fir.box<!fir.array<100xf32>>, index) -> (index, index, index)
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

// CHECK: [[VAL_21:%.*]] = constant 10 : i32
// CHECK: [[VAL_22:%.*]] = fir.coordinate_of [[VAL_5]], [[VAL_21]] : (!fir.heap<!fir.array<100xf32>>, i32) -> !fir.ref<f32>
// CHECK: [[VAL_23:%.*]] = fir.field_index f, !fir.type<derived{f:f32}>
// CHECK: [[VAL_24:%.*]] = fir.undefined !fir.type<derived{f:f32}>
// CHECK: [[VAL_25:%.*]] = fir.extract_value [[VAL_24]], [[VAL_23]] : (!fir.type<derived{f:f32}>, !fir.field) -> f32
  %19 = constant 10 : i32
  %20 = fir.coordinate_of %5, %19 : (!fir.heap<!fir.array<100xf32>>, i32) -> !fir.ref<f32>
  %21 = fir.field_index f, !fir.type<derived{f:f32}>
  %22 = fir.undefined !fir.type<derived{f:f32}>
  %23 = fir.extract_value %22, %21 : (!fir.type<derived{f:f32}>, !fir.field) -> f32

// CHECK: [[VAL_26:%.*]] = constant 1 : i32
// CHECK: [[VAL_28:%.*]] = constant 1.0
// CHECK: [[VAL_29:%.*]] = fir.insert_value [[VAL_24]], [[VAL_28]], [[VAL_23]] : (!fir.type<derived{f:f32}>, f32, !fir.field) -> !fir.type<derived{f:f32}>
// CHECK: [[VAL_30:%.*]] = fir.len_param_index f, !fir.type<derived3{f:f32}>
  %c1 = constant 1 : i32
  %cf1 = constant 1.0 : f32
  %25 = fir.insert_value %22, %cf1, %21 : (!fir.type<derived{f:f32}>, f32, !fir.field) -> !fir.type<derived{f:f32}>
  %26 = fir.len_param_index f, !fir.type<derived3{f:f32}>

// CHECK: [[VAL_31:%.*]] = fir.call @box3() : () -> !fir.box<!fir.type<derived3{f:f32}>>
// CHECK: [[VAL_32:%.*]] = fir.dispatch "method"([[VAL_31]]) : (!fir.box<!fir.type<derived3{f:f32}>>) -> i32
// CHECK: [[VAL_33:%.*]] = fir.convert [[VAL_32]] : (i32) -> i64
// CHECK: [[VAL_34:%.*]] = fir.gentypedesc !fir.type<x>
// CHECK: fir.call @user_tdesc([[VAL_34]]) : (!fir.tdesc<!fir.type<x>>) -> ()
// CHECK: [[VAL_35:%.*]] = fir.no_reassoc [[VAL_33]] : i64
  %27 = fir.call @box3() : () -> !fir.box<!fir.type<derived3{f:f32}>>
  %28 = fir.dispatch "method"(%27) : (!fir.box<!fir.type<derived3{f:f32}>>) -> i32
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
func @boxing_match() {
// CHECK: [[VAL_38:%.*]] = fir.alloca i32
// CHECK: [[VAL_39:%.*]] = fir.alloca !fir.type<qq2{f1:i32,f2:f64}>
// CHECK: [[VAL_40:%.*]] = fir.alloca !fir.char<1>
// CHECK: [[VAL_41:%.*]] = fir.alloca tuple<i32, f64>
// CHECK: [[VAL_42:%.*]] = fir.embox [[VAL_38]] : (!fir.ref<i32>) -> !fir.box<i32>
// CHECK: [[VAL_43:%.*]]:6 = fir.unbox [[VAL_42]] : (!fir.box<i32>) -> (!fir.ref<i32>, i32, i32, !fir.tdesc<i32>, i32, !fir.array<3x?xindex>)
// CHECK: [[VAL_44:%.*]] = constant 8 : i32
// CHECK: [[VAL_45:%.*]] = fir.undefined !fir.char<1>
// CHECK: [[VAL_46:%.*]] = fir.emboxchar [[VAL_40]], [[VAL_44]] : (!fir.ref<!fir.char<1>>, i32) -> !fir.boxchar<1>
// CHECK: [[VAL_47:%.*]]:2 = fir.unboxchar [[VAL_46]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1>>, i32)
// CHECK: [[VAL_48:%.*]] = fir.undefined !fir.type<qq2{f1:i32,f2:f64}>
// CHECK: [[VAL_49:%.*]] = constant 0 : i32
// CHECK: [[VAL_50:%.*]] = constant 12 : i32
// CHECK: [[VAL_51:%.*]] = fir.insert_value [[VAL_48]], [[VAL_50]], [[VAL_49]] : (!fir.type<qq2{f1:i32,f2:f64}>, i32, i32) -> !fir.type<qq2{f1:i32,f2:f64}>
// CHECK: [[VAL_52:%.*]] = constant 1 : i32
// CHECK: [[VAL_53:%.*]] = constant 4.213000e+01 : f64
// CHECK: [[VAL_54:%.*]] = fir.insert_value [[VAL_48]], [[VAL_53]], [[VAL_52]] : (!fir.type<qq2{f1:i32,f2:f64}>, f64, i32) -> !fir.type<qq2{f1:i32,f2:f64}>
// CHECK: fir.store [[VAL_54]] to [[VAL_39]] : !fir.ref<!fir.type<qq2{f1:i32,f2:f64}>>
// CHECK: [[VAL_55:%.*]] = fir.emboxproc @method_impl, [[VAL_41]] : ((!fir.box<!fir.type<derived3{f:f32}>>) -> (), !fir.ref<tuple<i32, f64>>) -> !fir.boxproc<(!fir.box<!fir.type<derived3{f:f32}>>) -> ()>
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
  %2:6 = fir.unbox %1 : (!fir.box<i32>) -> (!fir.ref<i32>,i32,i32,!fir.tdesc<i32>,i32,!fir.array<3x?xindex>)
  %c8 = constant 8 : i32
  %3 = fir.undefined !fir.char<1>
  %4 = fir.emboxchar %d3, %c8 : (!fir.ref<!fir.char<1>>, i32) -> !fir.boxchar<1>
  %5:2 = fir.unboxchar %4 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1>>, i32)
  %6 = fir.undefined !fir.type<qq2{f1:i32,f2:f64}>
  %z = constant 0 : i32
  %c12 = constant 12 : i32
  %a2 = fir.insert_value %6, %c12, %z : (!fir.type<qq2{f1:i32,f2:f64}>, i32, i32) -> !fir.type<qq2{f1:i32,f2:f64}>
  %z1 = constant 1 : i32
  %c42 = constant 42.13 : f64
  %a3 = fir.insert_value %6, %c42, %z1 : (!fir.type<qq2{f1:i32,f2:f64}>, f64, i32) -> !fir.type<qq2{f1:i32,f2:f64}>
  fir.store %a3 to %d6 : !fir.ref<!fir.type<qq2{f1:i32,f2:f64}>>
  %7 = fir.emboxproc @method_impl, %e6 : ((!fir.box<!fir.type<derived3{f:f32}>>) -> (), !fir.ref<tuple<i32,f64>>) -> !fir.boxproc<(!fir.box<!fir.type<derived3{f:f32}>>) -> ()>
  %8:2 = fir.unboxproc %7 : (!fir.boxproc<(!fir.box<!fir.type<derived3{f:f32}>>) -> ()>) -> ((!fir.box<!fir.type<derived3{f:f32}>>) -> (), !fir.ref<tuple<!fir.type<qq2{f1:i32,f2:f64}>>>)
  %9 = fir.call @box2() : () -> !fir.boxproc<(i32, i32) -> i64>
  %10:2 = fir.unboxproc %9 : (!fir.boxproc<(i32, i32) -> i64>) -> ((i32, i32) -> i64, !fir.ref<tuple<!fir.type<qq1{f1:i32}>>>)
  %11 = fir.load %10#1 : !fir.ref<tuple<!fir.type<qq1{f1:i32}>>>
  fir.call @store_tuple(%11) : (tuple<!fir.type<qq1{f1:i32}>>) -> ()
  return
}

// CHECK-LABEL: func @loop() {
func @loop() {
// CHECK: [[VAL_62:%.*]] = constant 1 : index
// CHECK: [[VAL_63:%.*]] = constant 10 : index
// CHECK: [[VAL_64:%.*]] = constant true
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  %ct = constant true

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
func @bar_select(%arg : i32, %arg2 : i32) -> i32 {
// CHECK: [[VAL_68:%.*]] = constant 1 : i32
// CHECK: [[VAL_69:%.*]] = constant 2 : i32
// CHECK: [[VAL_70:%.*]] = constant 3 : i32
// CHECK: [[VAL_71:%.*]] = constant 4 : i32
  %0 = constant 1 : i32
  %1 = constant 2 : i32
  %2 = constant 3 : i32
  %3 = constant 4 : i32

// CHECK: fir.select [[VAL_66]] : i32 [1, ^bb1([[VAL_68]] : i32), 2, ^bb2([[VAL_70]], [[VAL_66]], [[VAL_67]] : i32, i32, i32), -3, ^bb3([[VAL_67]], [[VAL_70]] : i32, i32), 4, ^bb4([[VAL_69]] : i32), unit, ^bb5]
// CHECK: ^bb1([[VAL_72:%.*]]: i32):
// CHECK: return [[VAL_72]] : i32
// CHECK: ^bb2([[VAL_73:%.*]]: i32, [[VAL_74:%.*]]: i32, [[VAL_75:%.*]]: i32):
// CHECK: [[VAL_76:%.*]] = addi [[VAL_73]], [[VAL_74]] : i32
// CHECK: [[VAL_77:%.*]] = addi [[VAL_76]], [[VAL_75]] : i32
// CHECK: return [[VAL_77]] : i32
// CHECK: ^bb3([[VAL_78:%.*]]: i32, [[VAL_79:%.*]]: i32):
// CHECK: [[VAL_80:%.*]] = addi [[VAL_78]], [[VAL_79]] : i32
// CHECK: return [[VAL_80]] : i32
// CHECK: ^bb4([[VAL_81:%.*]]: i32):
// CHECK: return [[VAL_81]] : i32
// CHECK: ^bb5:
// CHECK: [[VAL_82:%.*]] = constant 0 : i32
// CHECK: return [[VAL_82]] : i32
// CHECK: }
  fir.select %arg:i32 [ 1,^bb1(%0:i32), 2,^bb2(%2,%arg,%arg2:i32,i32,i32), -3,^bb3(%arg2,%2:i32,i32), 4,^bb4(%1:i32), unit,^bb5 ]
^bb1(%a : i32) :
  return %a : i32
^bb2(%b : i32, %b2 : i32, %b3:i32) :
  %4 = addi %b, %b2 : i32
  %5 = addi %4, %b3 : i32
  return %5 : i32
^bb3(%c:i32, %c2:i32) :
  %6 = addi %c, %c2 : i32
  return %6 : i32
^bb4(%d : i32) :
  return %d : i32
^bb5 :
  %zero = constant 0 : i32
  return %zero : i32
}

// CHECK-LABEL: func @bar_select_rank(
// CHECK-SAME: [[VAL_83:%.*]]: i32, [[VAL_84:%.*]]: i32) -> i32 {
func @bar_select_rank(%arg : i32, %arg2 : i32) -> i32 {
// CHECK: [[VAL_85:%.*]] = constant 1 : i32
// CHECK: [[VAL_86:%.*]] = constant 2 : i32
// CHECK: [[VAL_87:%.*]] = constant 3 : i32
// CHECK: [[VAL_88:%.*]] = constant 4 : i32
  %0 = constant 1 : i32
  %1 = constant 2 : i32
  %2 = constant 3 : i32
  %3 = constant 4 : i32

// CHECK: fir.select_rank [[VAL_83]] : i32 [1, ^bb1([[VAL_85]] : i32), 2, ^bb2([[VAL_87]], [[VAL_83]], [[VAL_84]] : i32, i32, i32), 3, ^bb3([[VAL_84]], [[VAL_87]] : i32, i32), -1, ^bb4([[VAL_86]] : i32), unit, ^bb5]
// CHECK: ^bb1([[VAL_89:%.*]]: i32):
// CHECK: return [[VAL_89]] : i32
// CHECK: ^bb2([[VAL_90:%.*]]: i32, [[VAL_91:%.*]]: i32, [[VAL_92:%.*]]: i32):
// CHECK: [[VAL_93:%.*]] = addi [[VAL_90]], [[VAL_91]] : i32
// CHECK: [[VAL_94:%.*]] = addi [[VAL_93]], [[VAL_92]] : i32
// CHECK: return [[VAL_94]] : i32
  fir.select_rank %arg:i32 [ 1,^bb1(%0:i32), 2,^bb2(%2,%arg,%arg2:i32,i32,i32), 3,^bb3(%arg2,%2:i32,i32), -1,^bb4(%1:i32), unit,^bb5 ]
^bb1(%a : i32) :
  return %a : i32
^bb2(%b : i32, %b2 : i32, %b3:i32) :
  %4 = addi %b, %b2 : i32
  %5 = addi %4, %b3 : i32
  return %5 : i32

// CHECK: ^bb3([[VAL_95:%.*]]: i32, [[VAL_96:%.*]]: i32):
// CHECK: [[VAL_97:%.*]] = addi [[VAL_95]], [[VAL_96]] : i32
// CHECK: return [[VAL_97]] : i32
// CHECK: ^bb4([[VAL_98:%.*]]: i32):
// CHECK: return [[VAL_98]] : i32
^bb3(%c:i32, %c2:i32) :
  %6 = addi %c, %c2 : i32
  return %6 : i32
^bb4(%d : i32) :
  return %d : i32

// CHECK: ^bb5:
// CHECK: [[VAL_99:%.*]] = constant 0 : i32
// CHECK: [[VAL_100:%.*]] = fir.call @get_method_box() : () -> !fir.box<!fir.type<derived3{f:f32}>>
// CHECK: fir.dispatch "method"([[VAL_100]]) : (!fir.box<!fir.type<derived3{f:f32}>>) -> ()
^bb5 :
  %zero = constant 0 : i32
  %7 = fir.call @get_method_box() : () -> !fir.box<!fir.type<derived3{f:f32}>>
  fir.dispatch method(%7) : (!fir.box<!fir.type<derived3{f:f32}>>) -> ()

// CHECK: return [[VAL_99]] : i32
// CHECK: }
  return %zero : i32
}

// CHECK-LABEL: func @bar_select_type(
// CHECK-SAME: [[VAL_101:%.*]]: !fir.box<!fir.type<name(param1:i32){fld:!fir.char<1>}>>) -> i32 {
func @bar_select_type(%arg : !fir.box<!fir.type<name(param1:i32){fld:!fir.char<1>}>>) -> i32 {

// CHECK: [[VAL_102:%.*]] = constant 1 : i32
// CHECK: [[VAL_103:%.*]] = constant 2 : i32
// CHECK: [[VAL_104:%.*]] = constant 3 : i32
// CHECK: [[VAL_105:%.*]] = constant 4 : i32
  %0 = constant 1 : i32
  %1 = constant 2 : i32
  %2 = constant 3 : i32
  %3 = constant 4 : i32

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
// CHECK: [[VAL_110:%.*]] = constant 0 : i32
// CHECK: return [[VAL_110]] : i32
// CHECK: }
^bb5 :
  %zero = constant 0 : i32
  return %zero : i32
}

// CHECK-LABEL: func @bar_select_case(
// CHECK-SAME: [[VAL_111:%.*]]: i32, [[VAL_112:%.*]]: i32) -> i32 {
// CHECK: [[VAL_113:%.*]] = constant 1 : i32
// CHECK: [[VAL_114:%.*]] = constant 2 : i32
// CHECK: [[VAL_115:%.*]] = constant 3 : i32
// CHECK: [[VAL_116:%.*]] = constant 4 : i32
func @bar_select_case(%arg : i32, %arg2 : i32) -> i32 {
  %0 = constant 1 : i32
  %1 = constant 2 : i32
  %2 = constant 3 : i32
  %3 = constant 4 : i32

// CHECK: fir.select_case [[VAL_111]] : i32 [#fir.point, [[VAL_113]], ^bb1([[VAL_113]] : i32), #fir.lower, [[VAL_114]], ^bb2([[VAL_115]], [[VAL_111]], [[VAL_112]], [[VAL_114]] : i32, i32, i32, i32), #fir.interval, [[VAL_115]], [[VAL_116]], ^bb3([[VAL_115]], [[VAL_112]] : i32, i32), #fir.upper, [[VAL_111]], ^bb4([[VAL_114]] : i32), unit, ^bb5]
  fir.select_case %arg : i32 [#fir.point, %0, ^bb1(%0:i32), #fir.lower, %1, ^bb2(%2,%arg,%arg2,%1:i32,i32,i32,i32), #fir.interval, %2, %3, ^bb3(%2,%arg2:i32,i32), #fir.upper, %arg, ^bb4(%1:i32), unit, ^bb5]

// CHECK: ^bb1([[VAL_117:%.*]]: i32):
// CHECK: return [[VAL_117]] : i32
// CHECK: ^bb2([[VAL_118:%.*]]: i32, [[VAL_119:%.*]]: i32, [[VAL_120:%.*]]: i32, [[VAL_121:%.*]]: i32):
// CHECK: [[VAL_122:%.*]] = addi [[VAL_118]], [[VAL_119]] : i32
// CHECK: [[VAL_123:%.*]] = muli [[VAL_122]], [[VAL_120]] : i32
// CHECK: [[VAL_124:%.*]] = addi [[VAL_123]], [[VAL_121]] : i32
// CHECK: return [[VAL_124]] : i32
// CHECK: ^bb3([[VAL_125:%.*]]: i32, [[VAL_126:%.*]]: i32):
// CHECK: [[VAL_127:%.*]] = addi [[VAL_125]], [[VAL_126]] : i32
// CHECK: return [[VAL_127]] : i32
// CHECK: ^bb4([[VAL_128:%.*]]: i32):
// CHECK: return [[VAL_128]] : i32
^bb1(%a : i32) :
  return %a : i32
^bb2(%b : i32, %b2:i32, %b3:i32, %b4:i32) :
  %4 = addi %b, %b2 : i32
  %5 = muli %4, %b3 : i32
  %6 = addi %5, %b4 : i32
  return %6 : i32
^bb3(%c : i32, %c2 : i32) :
  %7 = addi %c, %c2 : i32
  return %7 : i32
^bb4(%d : i32) :
  return %d : i32

// CHECK: ^bb5:
// CHECK: [[VAL_129:%.*]] = constant 0 : i32
// CHECK: return [[VAL_129]] : i32
// CHECK: }
^bb5 :
  %zero = constant 0 : i32
  return %zero : i32
}

// CHECK-LABEL: fir.global @global_var : i32 {
// CHECK: [[VAL_130:%.*]] = constant 1 : i32
// CHECK: fir.has_value [[VAL_130]] : i32
// CHECK: }
fir.global @global_var : i32 {
  %0 = constant 1 : i32
  fir.has_value %0 : i32
}

// CHECK-LABEL: fir.global @global_constant constant : i32 {
// CHECK: [[VAL_131:%.*]] = constant 934 : i32
// CHECK: fir.has_value [[VAL_131]] : i32
// CHECK: }
fir.global @global_constant constant : i32 {
  %0 = constant 934 : i32
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

// CHECK-LABEL: fir.dispatch_table @dispatch_tbl {
// CHECK: fir.dt_entry "method", @method_impl
// CHECK: }
fir.dispatch_table @dispatch_tbl {
  fir.dt_entry "method", @method_impl
}

// CHECK-LABEL: func @compare_real(
// CHECK-SAME: [[VAL_133:%.*]]: !fir.real<16>, [[VAL_134:%.*]]: !fir.real<16>) {
func @compare_real(%a : !fir.real<16>, %b : !fir.real<16>) {

// CHECK: [[VAL_135:%.*]] = fir.cmpf "false", [[VAL_133]], [[VAL_134]] : !fir.real<16>
// CHECK: [[VAL_136:%.*]] = fir.cmpf "oeq", [[VAL_133]], [[VAL_134]] : !fir.real<16>
// CHECK: [[VAL_137:%.*]] = fir.cmpf "ogt", [[VAL_133]], [[VAL_134]] : !fir.real<16>
// CHECK: [[VAL_138:%.*]] = fir.cmpf "oge", [[VAL_133]], [[VAL_134]] : !fir.real<16>
  %d0 = fir.cmpf "false", %a, %b : !fir.real<16>
  %d1 = fir.cmpf "oeq", %a, %b : !fir.real<16>
  %d2 = fir.cmpf "ogt", %a, %b : !fir.real<16>
  %d3 = fir.cmpf "oge", %a, %b : !fir.real<16>

// CHECK: [[VAL_139:%.*]] = fir.cmpf "olt", [[VAL_133]], [[VAL_134]] : !fir.real<16>
// CHECK: [[VAL_140:%.*]] = fir.cmpf "ole", [[VAL_133]], [[VAL_134]] : !fir.real<16>
// CHECK: [[VAL_141:%.*]] = fir.cmpf "one", [[VAL_133]], [[VAL_134]] : !fir.real<16>
// CHECK: [[VAL_142:%.*]] = fir.cmpf "ord", [[VAL_133]], [[VAL_134]] : !fir.real<16>
  %a0 = fir.cmpf "olt", %a, %b : !fir.real<16>
  %a1 = fir.cmpf "ole", %a, %b : !fir.real<16>
  %a2 = fir.cmpf "one", %a, %b : !fir.real<16>
  %a3 = fir.cmpf "ord", %a, %b : !fir.real<16>

// CHECK: [[VAL_143:%.*]] = fir.cmpf "ueq", [[VAL_133]], [[VAL_134]] : !fir.real<16>
// CHECK: [[VAL_144:%.*]] = fir.cmpf "ugt", [[VAL_133]], [[VAL_134]] : !fir.real<16>
// CHECK: [[VAL_145:%.*]] = fir.cmpf "uge", [[VAL_133]], [[VAL_134]] : !fir.real<16>
// CHECK: [[VAL_146:%.*]] = fir.cmpf "ult", [[VAL_133]], [[VAL_134]] : !fir.real<16>
  %b0 = fir.cmpf "ueq", %a, %b : !fir.real<16>
  %b1 = fir.cmpf "ugt", %a, %b : !fir.real<16>
  %b2 = fir.cmpf "uge", %a, %b : !fir.real<16>
  %b3 = fir.cmpf "ult", %a, %b : !fir.real<16>

// CHECK: [[VAL_147:%.*]] = fir.cmpf "ule", [[VAL_133]], [[VAL_134]] : !fir.real<16>
// CHECK: [[VAL_148:%.*]] = fir.cmpf "une", [[VAL_133]], [[VAL_134]] : !fir.real<16>
// CHECK: [[VAL_149:%.*]] = fir.cmpf "uno", [[VAL_133]], [[VAL_134]] : !fir.real<16>
// CHECK: [[VAL_150:%.*]] = fir.cmpf "true", [[VAL_133]], [[VAL_134]] : !fir.real<16>
  %c0 = fir.cmpf "ule", %a, %b : !fir.real<16>
  %c1 = fir.cmpf "une", %a, %b : !fir.real<16>
  %c2 = fir.cmpf "uno", %a, %b : !fir.real<16>
  %c3 = fir.cmpf "true", %a, %b : !fir.real<16>

// CHECK: return
// CHECK: }
  return
}

// CHECK-LABEL: func @compare_complex(
// CHECK-SAME: [[VAL_151:%.*]]: !fir.complex<16>, [[VAL_152:%.*]]: !fir.complex<16>) {
func @compare_complex(%a : !fir.complex<16>, %b : !fir.complex<16>) {

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
// CHECK-SAME: [[VAL_169:%.*]]: !fir.real<16>, [[VAL_170:%.*]]: !fir.real<16>) -> !fir.real<16> {
func @arith_real(%a : !fir.real<16>, %b : !fir.real<16>) -> !fir.real<16> {

// CHECK: [[VAL_171:%.*]] = constant 1.0
// CHECK: [[VAL_172:%.*]] = fir.convert [[VAL_171]] : (f32) -> !fir.real<16>
// CHECK: [[VAL_173:%.*]] = fir.negf [[VAL_169]] : !fir.real<16>
// CHECK: [[VAL_174:%.*]] = fir.addf [[VAL_172]], [[VAL_173]] : !fir.real<16>
// CHECK: [[VAL_175:%.*]] = fir.subf [[VAL_174]], [[VAL_170]] : !fir.real<16>
// CHECK: [[VAL_176:%.*]] = fir.mulf [[VAL_173]], [[VAL_175]] : !fir.real<16>
// CHECK: [[VAL_177:%.*]] = fir.divf [[VAL_176]], [[VAL_169]] : !fir.real<16>
  %c1 = constant 1.0 : f32
  %0 = fir.convert %c1 : (f32) -> !fir.real<16>
  %1 = fir.negf %a : !fir.real<16>
  %2 = fir.addf %0, %1 : !fir.real<16>
  %3 = fir.subf %2, %b : !fir.real<16>
  %4 = fir.mulf %1, %3 : !fir.real<16>
  %5 = fir.divf %4, %a : !fir.real<16>
// CHECK: return [[VAL_177]] : !fir.real<16>
// CHECK: }
  return %5 : !fir.real<16>
}

// CHECK-LABEL: func @arith_complex(
// CHECK-SAME: [[VAL_178:%.*]]: !fir.complex<16>, [[VAL_179:%.*]]: !fir.complex<16>) -> !fir.complex<16> {
func @arith_complex(%a : !fir.complex<16>, %b : !fir.complex<16>) -> !fir.complex<16> {
// CHECK: [[VAL_180:%.*]] = fir.negc [[VAL_178]] : !fir.complex<16>
// CHECK: [[VAL_181:%.*]] = fir.addc [[VAL_179]], [[VAL_180]] : !fir.complex<16>
// CHECK: [[VAL_182:%.*]] = fir.subc [[VAL_181]], [[VAL_179]] : !fir.complex<16>
// CHECK: [[VAL_183:%.*]] = fir.mulc [[VAL_180]], [[VAL_182]] : !fir.complex<16>
// CHECK: [[VAL_184:%.*]] = fir.divc [[VAL_183]], [[VAL_178]] : !fir.complex<16>
  %1 = fir.negc %a : !fir.complex<16>
  %2 = fir.addc %b, %1 : !fir.complex<16>
  %3 = fir.subc %2, %b : !fir.complex<16>
  %4 = fir.mulc %1, %3 : !fir.complex<16>
  %5 = fir.divc %4, %a : !fir.complex<16>
// CHECK: return [[VAL_184]] : !fir.complex<16>
// CHECK: }
  return %5 : !fir.complex<16>
}

// CHECK-LABEL: func @character_literal() -> !fir.array<13x!fir.char<1>> {
func @character_literal() -> !fir.array<13 x !fir.char<1>> {
// CHECK: [[VAL_185:%.*]] = fir.string_lit "Hello, World!"(13) : !fir.char<1>
  %0 = fir.string_lit "Hello, World!"(13) : !fir.char<1>
// CHECK: return [[VAL_185]] : !fir.array<13x!fir.char<1>>
  return %0 : !fir.array<13 x !fir.char<1>>
// CHECK: }
}

// CHECK-LABEL: func private @earlyexit2(i32) -> i1
func private @earlyexit2(%a : i32) -> i1

// CHECK-LABEL: func @early_exit(
// CHECK-SAME: [[VAL_186:%.*]]: i1, [[VAL_187:%.*]]: i32) -> i1 {
func @early_exit(%ok : i1, %k : i32) -> i1 {
// CHECK: [[VAL_188:%.*]] = constant 1 : index
// CHECK: [[VAL_189:%.*]] = constant 100 : index
  %c1 = constant 1 : index
  %c100 = constant 100 : index

// CHECK: [[VAL_190:%.*]], [[VAL_191:%.*]] = fir.iterate_while ([[VAL_192:%.*]] = [[VAL_188]] to [[VAL_189]] step [[VAL_188]]) and ([[VAL_193:%.*]] = [[VAL_186]]) iter_args([[VAL_194:%.*]] = [[VAL_187]]) -> (i32) {
// CHECK: [[VAL_195:%.*]] = call @earlyexit2([[VAL_194]]) : (i32) -> i1
// CHECK: fir.result [[VAL_195]], [[VAL_194]] : i1, i32
// CHECK: }
  %newOk:2 = fir.iterate_while (%i = %c1 to %c100 step %c1) and (%ok_ = %ok) iter_args(%v = %k) -> (i32) {
    %stop = call @earlyexit2(%v) : (i32) -> i1
    fir.result %stop, %v : i1, i32
  }
// CHECK: return [[VAL_190]] : i1
// CHECK: }
  return %newOk#0 : i1
}

// CHECK-LABEL: @test_misc_ops(
// CHECK-SAME: [[ARR1:%.*]]: !fir.ref<!fir.array<?x?xf32>>, [[INDXM:%.*]]: index, [[INDXN:%.*]]: index, [[INDXO:%.*]]: index, [[INDXP:%.*]]: index)
func @test_misc_ops(%arr1 : !fir.ref<!fir.array<?x?xf32>>, %m : index, %n : index, %o : index, %p : index) {
  // CHECK: [[I10:%.*]] = constant 10 : index
  // CHECK: [[J20:%.*]] = constant 20 : index
  // CHECK: [[C2:%.*]] = constant 2 : index
  // CHECK: [[C9:%.*]] = constant 9 : index
  // CHECK: [[C1_I32:%.*]] = constant 9 : i32
  %i10 = constant 10 : index
  %j20 = constant 20 : index
  %c2 = constant 2 : index
  %c9 = constant 9 : index
  %c1_i32 = constant 9 : i32

  // CHECK: [[ARR2:%.*]] = fir.zero_bits !fir.array<10xi32>
  %arr2 = fir.zero_bits !fir.array<10xi32>
  return
}
