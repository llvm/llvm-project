// Test the FIR operations

// RUN: tco -emit-fir %s | tco -emit-fir | FileCheck %s
// UNSUPPORTED: !fir

// CHECK-LABEL: func @it1() -> !fir.int<4>
func @it1() -> !fir.int<4>
// CHECK-LABEL: func @box1() -> !fir.boxchar<2>
func @box1() -> !fir.boxchar<2>
// CHECK-LABEL: func @box2() -> !fir.boxproc<(i32, i32) -> i64>
func @box2() -> !fir.boxproc<(i32, i32) -> i64>
// CHECK-LABEL: func @box3() -> !fir.box<!fir.type<derived3{f:f32}>>
func @box3() -> !fir.box<!fir.type<derived3{f:f32}>>

// Fortran SUBROUTINE and FUNCTION
// CHECK-LABEL: func @print_index3(index, index, index)
// CHECK-LABEL: func @user_i64(i64)
// CHECK-LABEL: func @user_tdesc(!fir.tdesc<!fir.type<x>>)
func @print_index3(index, index, index)
func @user_i64(i64)
func @user_tdesc(!fir.tdesc<!fir.type<x>>)

// expect the void return to be omitted
// CHECK-LABEL: func @store_tuple(tuple<!fir.type<qq1{f1:i32}>>)
func @store_tuple(tuple<!fir.type<qq1{f1:i32}>>) -> ()

// CHECK-LABEL: func @get_method_box() -> !fir.box<!fir.type<derived3{f:f32}>>
// CHECK-LABEL: func @method_impl(!fir.box<!fir.type<derived3{f:f32}>>)
func @get_method_box() -> !fir.box<!fir.type<derived3{f:f32}>>
func @method_impl(!fir.box<!fir.type<derived3{f:f32}>>)

// CHECK-LABEL: func @nop()
func @nop()

// CHECK-LABEL: func @get_func() -> (() -> ())
func @get_func() -> (() -> ())

// CHECK-LABEL: @instructions
func @instructions() {
  // CHECK: %[[A0:.*]] = fir.alloca !fir.array<10xi32>
  %0 = fir.alloca !fir.array<10xi32>
  // CHECK: fir.load %[[A0]] : !fir.ref<!fir.array<10xi32>>
  %1 = fir.load %0 : !fir.ref<!fir.array<10xi32>>
  %2 = fir.alloca i32
  %3 = constant 22 : i32
  // CHECK: fir.store %{{.*}} to %{{.*}} : !fir.ref<i32>
  fir.store %3 to %2 : !fir.ref<i32>
  // CHECK: fir.undefined i32
  %4 = fir.undefined i32
  // CHECK: %[[A5:.*]] = fir.allocmem !fir.array<100xf32>
  %5 = fir.allocmem !fir.array<100xf32>
  // CHECK: %[[A6:.*]] = fir.embox %[[A5]] : (!fir.heap<!fir.array<100xf32>>) -> !fir.box<!fir.array<100xf32>>
  %6 = fir.embox %5 : (!fir.heap<!fir.array<100xf32>>) -> !fir.box<!fir.array<100xf32>>
  // CHECK: fir.box_addr %{{.*}} : (!fir.box<!fir.array<100xf32>>) -> !fir.ref<!fir.array<100xf32>>
  %7 = fir.box_addr %6 : (!fir.box<!fir.array<100xf32>>) -> !fir.ref<!fir.array<100xf32>>
  %c0 = constant 0 : index
  // CHECK: %[[A8:.*]]:3 = fir.box_dims %{{.*}}, %{{.*}} : (!fir.box<!fir.array<100xf32>>, index) -> (index, index, index)
  %d1:3 = fir.box_dims %6, %c0 : (!fir.box<!fir.array<100xf32>>, index) -> (index, index, index)
  // CHECK: fir.call @print_index3(%[[A8]]#0, %[[A8]]#1, %[[A8]]#2) : (index, index, index)
  fir.call @print_index3(%d1#0, %d1#1, %d1#2) : (index, index, index) -> ()
  %8 = fir.call @it1() : () -> !fir.int<4>
  // CHECK: fir.box_elesize %[[A6]] : (!fir.box<!fir.array<100xf32>>) -> i64
  %9 = fir.box_elesize %6 : (!fir.box<!fir.array<100xf32>>) -> i64
  // CHECK: fir.box_isalloc %[[A6]] : (!fir.box<!fir.array<100xf32>>) -> i1
  %10 = fir.box_isalloc %6 : (!fir.box<!fir.array<100xf32>>) -> i1
  // CHECK: fir.box_isarray %[[A6]] : (!fir.box<!fir.array<100xf32>>) -> i1
  %11 = fir.box_isarray %6 : (!fir.box<!fir.array<100xf32>>) -> i1
  // CHECK: fir.box_isptr %[[A6]] : (!fir.box<!fir.array<100xf32>>) -> i1
  %12 = fir.box_isptr %6 : (!fir.box<!fir.array<100xf32>>) -> i1
  // CHECK: fir.box_rank %[[A6]] : (!fir.box<!fir.array<100xf32>>) -> i64
  %13 = fir.box_rank %6 : (!fir.box<!fir.array<100xf32>>) -> i64
  // CHECK: fir.box_tdesc %[[A6]] : (!fir.box<!fir.array<100xf32>>) -> !fir.tdesc<!fir.array<100xf32>>
  %14 = fir.box_tdesc %6 : (!fir.box<!fir.array<100xf32>>) -> !fir.tdesc<!fir.array<100xf32>>
  %15 = fir.call @box1() : () -> !fir.boxchar<2>
  // CHECK: fir.boxchar_len %{{.*}} : (!fir.boxchar<2>) -> i32
  %16 = fir.boxchar_len %15 : (!fir.boxchar<2>) -> i32
  %17 = fir.call @box2() : () -> !fir.boxproc<(i32, i32) -> i64>
  // CHECK: fir.boxproc_host %{{.*}} : (!fir.boxproc<(i32, i32) -> i64>) -> !fir.ref<i32>
  %18 = fir.boxproc_host %17 : (!fir.boxproc<(i32, i32) -> i64>) -> !fir.ref<i32>
  %19 = constant 10 : i32
  // CHECK: fir.coordinate_of %{{.*}}, %{{.*}} : (!fir.heap<!fir.array<100xf32>>, i32) -> !fir.ref<f32>
  %20 = fir.coordinate_of %5, %19 : (!fir.heap<!fir.array<100xf32>>, i32) -> !fir.ref<f32>
  // CHECK: fir.field_index f, !fir.type<derived{f:f32}>
  %21 = fir.field_index f, !fir.type<derived{f:f32}>
  // CHECK: fir.undefined !fir.type<derived{f:f32}>
  %22 = fir.undefined !fir.type<derived{f:f32}>
  // CHECK: fir.extract_value %{{.*}}, %{{.*}} : (!fir.type<derived{f:f32}>, !fir.field) -> f32
  %23 = fir.extract_value %22, %21 : (!fir.type<derived{f:f32}>, !fir.field) -> f32
  %c1 = constant 1 : i32
  // CHECK: fir.gendims %{{.*}}, %{{.*}}, %{{.*}} : (i32, i32, i32) -> !fir.dims<1>
  %24 = fir.gendims %c1, %19, %c1 : (i32, i32, i32) -> !fir.dims<1>
  %cf1 = constant 1.0 : f32
  // CHECK: fir.insert_value %{{.*}}, %{{.*}}, %{{.*}} : (!fir.type<derived{f:f32}>, f32, !fir.field) -> !fir.type<derived{f:f32}>
  %25 = fir.insert_value %22, %cf1, %21 : (!fir.type<derived{f:f32}>, f32, !fir.field) -> !fir.type<derived{f:f32}>
  // CHECK: fir.len_param_index f, !fir.type<derived3{f:f32}>
  %26 = fir.len_param_index f, !fir.type<derived3{f:f32}>
  %27 = fir.call @box3() : () -> !fir.box<!fir.type<derived3{f:f32}>>
  // CHECK: fir.dispatch "method"(%{{.*}}) : (!fir.box<!fir.type<derived3{f:f32}>>) -> i32
  %28 = fir.dispatch "method"(%27) : (!fir.box<!fir.type<derived3{f:f32}>>) -> i32
  // CHECK: fir.convert %{{.*}} : (i32) -> i64
  %29 = fir.convert %28 : (i32) -> i64
  // CHECK: fir.gentypedesc !fir.type<x>
  %30 = fir.gentypedesc !fir.type<x>
  fir.call @user_tdesc(%30) : (!fir.tdesc<!fir.type<x>>) -> ()
  // CHECK: fir.no_reassoc %{{.*}} : i64
  %31 = fir.no_reassoc %29 : i64
  fir.call @user_i64(%31) : (i64) -> ()
  // CHECK: fir.freemem %{{.*}} : !fir.heap<!fir.array<100xf32>>
  fir.freemem %5 : !fir.heap<!fir.array<100xf32>>
  %32 = fir.call @get_func() : () -> (() -> ())
  fir.call %32() : () -> ()
  // CHECK: fir.address_of(@it1) : !fir.ref<() -> !fir.int<4>>
  %33 = fir.address_of (@it1) : !fir.ref<() -> !fir.int<4>>
  return
}

// CHECK-LABEL: @boxing_match
func @boxing_match() {
  %0 = fir.alloca i32
  %d6 = fir.alloca !fir.type<qq2{f1:i32,f2:f64}>
  %d3 = fir.alloca !fir.char<1>
  %e6 = fir.alloca tuple<i32,f64>
  %1 = fir.embox %0 : (!fir.ref<i32>) -> !fir.box<i32>
  // CHECK: fir.unbox %{{.*}} : (!fir.box<i32>) -> (!fir.ref<i32>, i32, i32, !fir.tdesc<i32>, i32, !fir.dims<0>)
  %2:6 = fir.unbox %1 : (!fir.box<i32>) -> (!fir.ref<i32>,i32,i32,!fir.tdesc<i32>,i32,!fir.dims<0>)
  %c8 = constant 8 : i32
  %3 = fir.undefined !fir.char<1>
  // CHECK: fir.emboxchar %{{.*}}, %{{.*}} : (!fir.ref<!fir.char<1>>, i32) -> !fir.boxchar<1>
  // CHECK: fir.unboxchar %{{.*}} : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1>>, i32)
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
  // CHECK: fir.emboxproc @method_impl, %{{.*}} : ((!fir.box<!fir.type<derived3{f:f32}>>) -> (), !fir.ref<tuple<i32, f64>>) -> !fir.boxproc<(!fir.box<!fir.type<derived3{f:f32}>>) -> ()>
  // CHECK: fir.unboxproc %{{.*}} : (!fir.boxproc<(!fir.box<!fir.type<derived3{f:f32}>>) -> ()>) -> ((!fir.box<!fir.type<derived3{f:f32}>>) -> (), !fir.ref<tuple<!fir.type<qq2{f1:i32,f2:f64}>>>)
  %9 = fir.call @box2() : () -> !fir.boxproc<(i32, i32) -> i64>
  %10:2 = fir.unboxproc %9 : (!fir.boxproc<(i32, i32) -> i64>) -> ((i32, i32) -> i64, !fir.ref<tuple<!fir.type<qq1{f1:i32}>>>)
  %11 = fir.load %10#1 : !fir.ref<tuple<!fir.type<qq1{f1:i32}>>>
  fir.call @store_tuple(%11) : (tuple<!fir.type<qq1{f1:i32}>>) -> ()
  return
}

// CHECK-LABEL: @loop
func @loop() {
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  %ct = constant true
  // CHECK: fir.loop %{{.*}} = %{{.*}} to %{{.*}} {
  %i = fir.loop %i = %c1 to %c10 {
    // CHECK: fir.where %{{.*}} {
    fir.where %ct {
      fir.call @nop() : () -> ()
    // CHECK: } otherwise {
    } otherwise {
      fir.call @nop() : () -> ()
    }
  }
  // CHECK: fir.unreachable
  fir.unreachable
}

// CHECK-LABEL: @bar_select
func @bar_select(%arg : i32, %arg2 : i32) -> i32 {
  %0 = constant 1 : i32
  %1 = constant 2 : i32
  %2 = constant 3 : i32
  %3 = constant 4 : i32
  // CHECK: fir.select %{{.*}} : i32 [1, ^bb1(%{{.*}} : i32), 2, ^bb2(%{{.*}}, %{{.*}}, %{{.*}} : i32, i32, i32), -3, ^bb3(%{{.*}}, %{{.*}} : i32, i32), 4, ^bb4(%{{.*}} : i32), unit, ^bb5]
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

// CHECK-LABEL: @bar_select_rank
func @bar_select_rank(%arg : i32, %arg2 : i32) -> i32 {
  %0 = constant 1 : i32
  %1 = constant 2 : i32
  %2 = constant 3 : i32
  %3 = constant 4 : i32
  // CHECK: fir.select_rank %{{.*}} : i32 [1, ^bb1(%{{.*}} : i32), 2, ^bb2(%{{.*}}, %{{.*}}, %{{.*}} : i32, i32, i32), 3, ^bb3(%{{.*}}, %{{.*}} : i32, i32), -1, ^bb4(%{{.*}} : i32), unit, ^bb5]
  fir.select_rank %arg:i32 [ 1,^bb1(%0:i32), 2,^bb2(%2,%arg,%arg2:i32,i32,i32), 3,^bb3(%arg2,%2:i32,i32), -1,^bb4(%1:i32), unit,^bb5 ]
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
  %7 = fir.call @get_method_box() : () -> !fir.box<!fir.type<derived3{f:f32}>>
  fir.dispatch method(%7) : (!fir.box<!fir.type<derived3{f:f32}>>) -> ()
  return %zero : i32
}

// CHECK-LABEL: @bar_select_type
func @bar_select_type(%arg : !fir.box<!fir.type<name(param1:i32){fld:!fir.char<1>}>>) -> i32 {
  %0 = constant 1 : i32
  %1 = constant 2 : i32
  %2 = constant 3 : i32
  %3 = constant 4 : i32
  // CHECK: fir.select_type %{{.*}} : !fir.box<!fir.type<name(param1:i32){fld:!fir.char<1>}>> [#fir.instance<!fir.int<4>>, ^bb1(%{{.*}} : i32), #fir.instance<!fir.int<8>>, ^bb2(%{{.*}} : i32), #fir.subsumed<!fir.int<2>>, ^bb3(%{{.*}} : i32), #fir.instance<!fir.int<1>>, ^bb4(%{{.*}} : i32), unit, ^bb5]
  fir.select_type %arg : !fir.box<!fir.type<name(param1:i32){fld:!fir.char<1>}>> [ #fir.instance<!fir.int<4>>,^bb1(%0:i32), #fir.instance<!fir.int<8>>,^bb2(%2:i32), #fir.subsumed<!fir.int<2>>,^bb3(%2:i32), #fir.instance<!fir.int<1>>,^bb4(%1:i32), unit,^bb5 ]
^bb1(%a : i32) :
  return %a : i32
^bb2(%b : i32) :
  return %b : i32
^bb3(%c : i32) :
  return %c : i32
^bb4(%d : i32) :
  return %d : i32
^bb5 :
  %zero = constant 0 : i32
  return %zero : i32
}

// CHECK-LABEL: @bar_select_case
func @bar_select_case(%arg : i32, %arg2 : i32) -> i32 {
  %0 = constant 1 : i32
  %1 = constant 2 : i32
  %2 = constant 3 : i32
  %3 = constant 4 : i32
  // CHECK: fir.select_case %{{.*}} : i32 [#fir.point, %{{.*}}, ^bb1(%{{.*}} : i32), #fir.lower, %{{.*}}, ^bb2(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i32, i32, i32, i32), #fir.interval, %{{.*}}, %{{.*}}, ^bb3(%{{.*}}, %{{.*}} : i32, i32), #fir.upper, %{{.*}}, ^bb4(%{{.*}} : i32), unit, ^bb5]
  fir.select_case %arg : i32 [#fir.point, %0, ^bb1(%0:i32), #fir.lower, %1, ^bb2(%2,%arg,%arg2,%1:i32,i32,i32,i32), #fir.interval, %2, %3, ^bb3(%2,%arg2:i32,i32), #fir.upper, %arg, ^bb4(%1:i32), unit, ^bb5]
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
^bb5 :
  %zero = constant 0 : i32
  return %zero : i32
}

// CHECK-LABEL: @global_var
fir.global @global_var : i32 {
  %0 = constant 1 : i32
  fir.has_value %0 : i32
}

// CHECK-LABEL: @global_constant
fir.global @global_constant constant : i32 {
  %0 = constant 934 : i32
  fir.has_value %0 : i32
}

// CHECK-LABEL: @global_derived
fir.global @global_derived : !fir.type<minez(f:i32)> {
  // CHECK: fir.global_len "f", 1 : i32
  fir.global_len f, 1 : i32
  %0 = fir.undefined !fir.type<minez>
  fir.has_value %0 : !fir.type<minez>
}

// CHECK-LABEL: @dispatch_tbl
fir.dispatch_table @dispatch_tbl {
  // CHECK: fir.dt_entry "method", @method_impl
  fir.dt_entry "method", @method_impl
}

// CHECK-LABEL: @compare_real
func @compare_real(%a : !fir.real<16>, %b : !fir.real<16>) {
  // CHECK: fir.cmpf "false", %{{.*}}, %{{.*}} : !fir.real<16>
  // CHECK: fir.cmpf "oeq", %{{.*}}, %{{.*}} : !fir.real<16>
  // CHECK: fir.cmpf "ogt", %{{.*}}, %{{.*}} : !fir.real<16>
  // CHECK: fir.cmpf "oge", %{{.*}}, %{{.*}} : !fir.real<16>
  %d0 = fir.cmpf "false", %a, %b : !fir.real<16>
  %d1 = fir.cmpf "oeq", %a, %b : !fir.real<16>
  %d2 = fir.cmpf "ogt", %a, %b : !fir.real<16>
  %d3 = fir.cmpf "oge", %a, %b : !fir.real<16>
  // CHECK: fir.cmpf "olt", %{{.*}}, %{{.*}} : !fir.real<16>
  // CHECK: fir.cmpf "ole", %{{.*}}, %{{.*}} : !fir.real<16>
  // CHECK: fir.cmpf "one", %{{.*}}, %{{.*}} : !fir.real<16>
  // CHECK: fir.cmpf "ord", %{{.*}}, %{{.*}} : !fir.real<16>
  %a0 = fir.cmpf "olt", %a, %b : !fir.real<16>
  %a1 = fir.cmpf "ole", %a, %b : !fir.real<16>
  %a2 = fir.cmpf "one", %a, %b : !fir.real<16>
  %a3 = fir.cmpf "ord", %a, %b : !fir.real<16>
  // CHECK: fir.cmpf "ueq", %{{.*}}, %{{.*}} : !fir.real<16>
  // CHECK: fir.cmpf "ugt", %{{.*}}, %{{.*}} : !fir.real<16>
  // CHECK: fir.cmpf "uge", %{{.*}}, %{{.*}} : !fir.real<16>
  // CHECK: fir.cmpf "ult", %{{.*}}, %{{.*}} : !fir.real<16>
  %b0 = fir.cmpf "ueq", %a, %b : !fir.real<16>
  %b1 = fir.cmpf "ugt", %a, %b : !fir.real<16>
  %b2 = fir.cmpf "uge", %a, %b : !fir.real<16>
  %b3 = fir.cmpf "ult", %a, %b : !fir.real<16>
  // CHECK: fir.cmpf "ule", %{{.*}}, %{{.*}} : !fir.real<16>
  // CHECK: fir.cmpf "une", %{{.*}}, %{{.*}} : !fir.real<16>
  // CHECK: fir.cmpf "uno", %{{.*}}, %{{.*}} : !fir.real<16>
  // CHECK: fir.cmpf "true", %{{.*}}, %{{.*}} : !fir.real<16>
  %c0 = fir.cmpf "ule", %a, %b : !fir.real<16>
  %c1 = fir.cmpf "une", %a, %b : !fir.real<16>
  %c2 = fir.cmpf "uno", %a, %b : !fir.real<16>
  %c3 = fir.cmpf "true", %a, %b : !fir.real<16>
  return
}

// CHECK-LABEL: @compare_complex
func @compare_complex(%a : !fir.complex<16>, %b : !fir.complex<16>) {
  // CHECK: fir.cmpc "false", %{{.*}}, %{{.*}} : !fir.complex<16>
  // CHECK: fir.cmpc "oeq", %{{.*}}, %{{.*}} : !fir.complex<16>
  // CHECK: fir.cmpc "ogt", %{{.*}}, %{{.*}} : !fir.complex<16>
  // CHECK: fir.cmpc "oge", %{{.*}}, %{{.*}} : !fir.complex<16>
  %d0 = fir.cmpc "false", %a, %b : !fir.complex<16>
  %d1 = fir.cmpc "oeq", %a, %b : !fir.complex<16>
  %d2 = fir.cmpc "ogt", %a, %b : !fir.complex<16>
  %d3 = fir.cmpc "oge", %a, %b : !fir.complex<16>
  // CHECK: fir.cmpc "olt", %{{.*}}, %{{.*}} : !fir.complex<16>
  // CHECK: fir.cmpc "ole", %{{.*}}, %{{.*}} : !fir.complex<16>
  // CHECK: fir.cmpc "one", %{{.*}}, %{{.*}} : !fir.complex<16>
  // CHECK: fir.cmpc "ord", %{{.*}}, %{{.*}} : !fir.complex<16>
  %a0 = fir.cmpc "olt", %a, %b : !fir.complex<16>
  %a1 = fir.cmpc "ole", %a, %b : !fir.complex<16>
  %a2 = fir.cmpc "one", %a, %b : !fir.complex<16>
  %a3 = fir.cmpc "ord", %a, %b : !fir.complex<16>
  // CHECK: fir.cmpc "ueq", %{{.*}}, %{{.*}} : !fir.complex<16>
  // CHECK: fir.cmpc "ugt", %{{.*}}, %{{.*}} : !fir.complex<16>
  // CHECK: fir.cmpc "uge", %{{.*}}, %{{.*}} : !fir.complex<16>
  // CHECK: fir.cmpc "ult", %{{.*}}, %{{.*}} : !fir.complex<16>
  %b0 = fir.cmpc "ueq", %a, %b : !fir.complex<16>
  %b1 = fir.cmpc "ugt", %a, %b : !fir.complex<16>
  %b2 = fir.cmpc "uge", %a, %b : !fir.complex<16>
  %b3 = fir.cmpc "ult", %a, %b : !fir.complex<16>
  // CHECK: fir.cmpc "ule", %{{.*}}, %{{.*}} : !fir.complex<16>
  // CHECK: fir.cmpc "une", %{{.*}}, %{{.*}} : !fir.complex<16>
  // CHECK: fir.cmpc "uno", %{{.*}}, %{{.*}} : !fir.complex<16>
  // CHECK: fir.cmpc "true", %{{.*}}, %{{.*}} : !fir.complex<16>
  %c0 = fir.cmpc "ule", %a, %b : !fir.complex<16>
  %c1 = fir.cmpc "une", %a, %b : !fir.complex<16>
  %c2 = fir.cmpc "uno", %a, %b : !fir.complex<16>
  %c3 = fir.cmpc "true", %a, %b : !fir.complex<16>
  return
}

// CHECK-LABEL: @arith_real
func @arith_real(%a : !fir.real<16>, %b : !fir.real<16>) -> !fir.real<16> {
  %c1 = constant 1.0 : f32
  %0 = fir.convert %c1 : (f32) -> !fir.real<16>
  // CHECK: %[[R1:.*]] = fir.negf %{{.*}} : !fir.real<16>
  // CHECK: fir.addf %{{.*}}, %{{.*}} : !fir.real<16>
  // CHECK: %[[R3:.*]] = fir.subf %{{.*}}, %{{.*}} : !fir.real<16>
  // CHECK: fir.mulf %[[R1]], %[[R3]] : !fir.real<16>
  // CHECK: fir.divf %{{.*}}, %{{.*}} : !fir.real<16>
  %1 = fir.negf %a : !fir.real<16>
  %2 = fir.addf %0, %1 : !fir.real<16>
  %3 = fir.subf %2, %b : !fir.real<16>
  %4 = fir.mulf %1, %3 : !fir.real<16>
  %5 = fir.divf %4, %a : !fir.real<16>
  return %5 : !fir.real<16>
}

// CHECK-LABEL: @arith_complex
func @arith_complex(%a : !fir.complex<16>, %b : !fir.complex<16>) -> !fir.complex<16> {
  // CHECK: fir.negc %{{.*}} : !fir.complex<16>
  // CHECK: fir.addc %{{.*}}, %{{.*}} : !fir.complex<16>
  // CHECK: fir.subc %{{.*}}, %{{.*}} : !fir.complex<16>
  // CHECK: fir.mulc %{{.*}}, %{{.*}} : !fir.complex<16>
  // CHECK: fir.divc %{{.*}}, %{{.*}} : !fir.complex<16>
  %1 = fir.negc %a : !fir.complex<16>
  %2 = fir.addc %b, %1 : !fir.complex<16>
  %3 = fir.subc %2, %b : !fir.complex<16>
  %4 = fir.mulc %1, %3 : !fir.complex<16>
  %5 = fir.divc %4, %a : !fir.complex<16>
  return %5 : !fir.complex<16>
}

// CHECK-LABEL: @character_literal
func @character_literal() -> !fir.array<13 x !fir.char<1>> {
  // CHECK: fir.string_lit "Hello, World!"(13) : !fir.char<1>
  %0 = fir.string_lit "Hello, World!"(13) : !fir.char<1>
  return %0 : !fir.array<13 x !fir.char<1>>
}
