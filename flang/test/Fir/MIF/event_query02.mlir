// RUN: fir-opt --mif-convert %s | FileCheck %s

func.func @_QPquery_array_element() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 <{bindc_name = "count", uniq_name = "_QFquery_array_elementEcount"}>
  %2:2 = hlfir.declare %1 uniq_name("_QFquery_array_elementEcount") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.address_of(@_QFquery_array_elementEev) : !fir.ref<!fir.box<!fir.heap<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>>
  %4:2 = hlfir.declare %3 uniq_name("_QFquery_array_elementEev") : (!fir.ref<!fir.box<!fir.heap<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>>, !fir.ref<!fir.box<!fir.heap<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>>)
  %5 = fir.load %4#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>>
  %c2 = arith.constant 2 : index
  %6 = hlfir.designate %5 (%c2)  : (!fir.box<!fir.heap<!fir.array<3x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
  %7 = mif.event_query %6 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>) -> i32
  fir.store %7 to %2#0 : !fir.ref<i32>
  return
}

func.func @_QPquery_array_in_loop() {
  %0 = fir.dummy_scope : !fir.dscope
  %c4 = arith.constant 4 : index
  %1 = fir.alloca !fir.array<4xi32> <{bindc_name = "counts", uniq_name = "_QFquery_array_in_loopEcounts"}>
  %2 = fir.shape %c4 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %1(%2) uniq_name("_QFquery_array_in_loopEcounts") : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xi32>>, !fir.ref<!fir.array<4xi32>>)
  %4 = fir.address_of(@_QFquery_array_in_loopEev) : !fir.ref<!fir.box<!fir.heap<!fir.array<4x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>>
  %5:2 = hlfir.declare %4 uniq_name("_QFquery_array_in_loopEev") : (!fir.ref<!fir.box<!fir.heap<!fir.array<4x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<4x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>>, !fir.ref<!fir.box<!fir.heap<!fir.array<4x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>>)
  %6 = fir.alloca i32 <{bindc_name = "i", uniq_name = "_QFquery_array_in_loopEi"}>
  %7:2 = hlfir.declare %6 uniq_name("_QFquery_array_in_loopEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %c4_0 = arith.constant 4 : index
  %8 = fir.alloca !fir.array<4xi32> <{bindc_name = "stats", uniq_name = "_QFquery_array_in_loopEstats"}>
  %9 = fir.shape %c4_0 : (index) -> !fir.shape<1>
  %10:2 = hlfir.declare %8(%9) uniq_name("_QFquery_array_in_loopEstats") : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xi32>>, !fir.ref<!fir.array<4xi32>>)
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %c1_i32_1 = arith.constant 1 : i32
  fir.do_loop %arg0 = %c1_i32 to %c4_i32 step %c1_i32_1 : i32 {
    fir.store %arg0 to %7#0 : !fir.ref<i32>
    %22 = fir.load %5#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<4x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>>
    %23 = fir.load %7#0 : !fir.ref<i32>
    %24 = fir.convert %23 : (i32) -> i64
    %25 = hlfir.designate %22 (%24)  : (!fir.box<!fir.heap<!fir.array<4x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>, i64) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
    %26 = fir.load %7#0 : !fir.ref<i32>
    %27 = fir.convert %26 : (i32) -> i64
    %28 = hlfir.designate %3#0 (%27)  : (!fir.ref<!fir.array<4xi32>>, i64) -> !fir.ref<i32>
    %29 = fir.load %7#0 : !fir.ref<i32>
    %30 = fir.convert %29 : (i32) -> i64
    %31 = hlfir.designate %10#0 (%30)  : (!fir.ref<!fir.array<4xi32>>, i64) -> !fir.ref<i32>
    %32 = mif.event_query %25 stat %31 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, !fir.ref<i32>) -> i32
    fir.store %32 to %28 : !fir.ref<i32>
  }
  %11 = fir.convert %c1_i32 : (i32) -> index
  %12 = fir.convert %c4_i32 : (i32) -> index
  %13 = fir.convert %c1_i32_1 : (i32) -> index
  %c0 = arith.constant 0 : index
  %14 = arith.subi %12, %11 : index
  %15 = arith.addi %14, %13 : index
  %16 = arith.divsi %15, %13 : index
  %17 = arith.cmpi slt, %16, %c0 : index
  %18 = arith.select %17, %c0, %16 : index
  %19 = arith.muli %18, %13 : index
  %20 = arith.addi %11, %19 : index
  %21 = fir.convert %20 : (index) -> i32
  fir.store %21 to %7#0 : !fir.ref<i32>
  return
}

func.func @_QPquery_allocatable() {
  %0 = fir.alloca !fir.array<0xi64>
  %1 = fir.alloca !fir.array<1xi64>
  %2 = fir.dummy_scope : !fir.dscope
  %3 = fir.alloca i32 <{bindc_name = "count", uniq_name = "_QFquery_allocatableEcount"}>
  %4:2 = hlfir.declare %3 uniq_name("_QFquery_allocatableEcount") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5 = fir.alloca !fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1> <{bindc_name = "ev", uniq_name = "_QFquery_allocatableEev"}>
  %6 = fir.zero_bits !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
  %7 = fir.embox %6 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>) -> !fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>
  fir.store %7 to %5 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %8:2 = hlfir.declare %5 uniq_name("_QFquery_allocatableEev") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>, !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>)
  %9 = fir.alloca i32 <{bindc_name = "stat", uniq_name = "_QFquery_allocatableEstat"}>
  %10:2 = hlfir.declare %9 uniq_name("_QFquery_allocatableEstat") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %11 = fir.absent !fir.box<none>
  %c1_i64 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %12 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
  fir.store %c1_i64 to %12 : !fir.ref<i64>
  %13 = fir.embox %1 : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
  %14 = fir.embox %0 : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
  mif.alloc_coarray %8#0 lcobounds %13 ucobounds %14 errmsg %11 uniq_name("_QFquery_allocatableEev") : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>, !fir.box<none>) -> ()
  %15 = fir.load %8#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %16 = fir.box_addr %15 : (!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
  %17 = mif.event_query %16 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>) -> i32
  fir.store %17 to %4#0 : !fir.ref<i32>
  %18 = fir.load %8#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %19 = fir.box_addr %18 : (!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
  %20 = mif.event_query %19 stat %10#0 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, !fir.ref<i32>) -> i32
  fir.store %20 to %4#0 : !fir.ref<i32>
  %21 = fir.absent !fir.box<none>
  %22 = fir.absent !fir.ref<i32>
  mif.dealloc_coarray %8#0 stat %22 errmsg %21 : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>, !fir.ref<i32>, !fir.box<none>) -> ()
  %23 = fir.load %8#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %24 = fir.box_addr %23 : (!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
  %25 = fir.convert %24 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>) -> i64
  %c0_i64 = arith.constant 0 : i64
  %26 = arith.cmpi ne, %25, %c0_i64 : i64
  fir.if %26 {
    %27 = fir.absent !fir.box<none>
    %28 = fir.absent !fir.ref<i32>
    mif.dealloc_coarray %8#0 stat %28 errmsg %27 : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>, !fir.ref<i32>, !fir.box<none>) -> ()
  }
  return
}

func.func @_QPquery_allocatable_array(%arg0: !fir.ref<i32> {fir.bindc_name = "n", fir.fortran_attrs = #fir.var_attrs<intent_in>, fir.read_only}) {
  %0 = fir.alloca !fir.array<0xi64>
  %1 = fir.alloca !fir.array<1xi64>
  %2 = fir.dummy_scope : !fir.dscope
  %3 = fir.alloca i32 <{bindc_name = "count", uniq_name = "_QFquery_allocatable_arrayEcount"}>
  %4:2 = hlfir.declare %3 uniq_name("_QFquery_allocatable_arrayEcount") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5 = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1> <{bindc_name = "ev", uniq_name = "_QFquery_allocatable_arrayEev"}>
  %6 = fir.zero_bits !fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>
  %c0 = arith.constant 0 : index
  %7 = fir.shape %c0 : (index) -> !fir.shape<1>
  %8 = fir.embox %6(%7) : (!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>
  fir.store %8 to %5 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>>
  %9:2 = hlfir.declare %5 uniq_name("_QFquery_allocatable_arrayEev") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>>)
  %10 = fir.alloca i32 <{bindc_name = "i", uniq_name = "_QFquery_allocatable_arrayEi"}>
  %11:2 = hlfir.declare %10 uniq_name("_QFquery_allocatable_arrayEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %12:2 = hlfir.declare %arg0 dummy_scope %2 arg 1 uniq_name("_QFquery_allocatable_arrayEn") fortran_attrs<intent_in> : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %13 = fir.absent !fir.box<none>
  %c1 = arith.constant 1 : index
  %14 = fir.load %12#0 : !fir.ref<i32>
  %c0_i32 = arith.constant 0 : i32
  %15 = fir.convert %9#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>>) -> !fir.ref<!fir.box<none>>
  %16 = fir.convert %c1 : (index) -> i64
  %17 = fir.convert %14 : (i32) -> i64
  fir.call @_FortranAAllocatableSetBounds(%15, %c0_i32, %16, %17) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
  %c1_i64 = arith.constant 1 : i64
  %c0_0 = arith.constant 0 : index
  %18 = fir.coordinate_of %1, %c0_0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
  fir.store %c1_i64 to %18 : !fir.ref<i64>
  %19 = fir.embox %1 : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
  %20 = fir.embox %0 : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
  mif.alloc_coarray %9#0 lcobounds %19 ucobounds %20 errmsg %13 uniq_name("_QFquery_allocatable_arrayEev") : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>, !fir.box<none>) -> ()
  %c1_i32 = arith.constant 1 : i32
  %21 = fir.load %12#0 : !fir.ref<i32>
  %c1_i32_1 = arith.constant 1 : i32
  fir.do_loop %arg1 = %c1_i32 to %21 step %c1_i32_1 : i32 {
    fir.store %arg1 to %11#0 : !fir.ref<i32>
    %39 = fir.load %9#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>>
    %40 = fir.load %11#0 : !fir.ref<i32>
    %41 = fir.convert %40 : (i32) -> i64
    %42 = hlfir.designate %39 (%41)  : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>, i64) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
    %43 = mif.event_query %42 : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>) -> i32
    fir.store %43 to %4#0 : !fir.ref<i32>
  }
  %22 = fir.convert %c1_i32 : (i32) -> index
  %23 = fir.convert %21 : (i32) -> index
  %24 = fir.convert %c1_i32_1 : (i32) -> index
  %c0_2 = arith.constant 0 : index
  %25 = arith.subi %23, %22 : index
  %26 = arith.addi %25, %24 : index
  %27 = arith.divsi %26, %24 : index
  %28 = arith.cmpi slt, %27, %c0_2 : index
  %29 = arith.select %28, %c0_2, %27 : index
  %30 = arith.muli %29, %24 : index
  %31 = arith.addi %22, %30 : index
  %32 = fir.convert %31 : (index) -> i32
  fir.store %32 to %11#0 : !fir.ref<i32>
  %33 = fir.absent !fir.box<none>
  %34 = fir.absent !fir.ref<i32>
  mif.dealloc_coarray %9#0 stat %34 errmsg %33 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>>, !fir.ref<i32>, !fir.box<none>) -> ()
  %35 = fir.load %9#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>>
  %36 = fir.box_addr %35 : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>) -> !fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>
  %37 = fir.convert %36 : (!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>) -> i64
  %c0_i64 = arith.constant 0 : i64
  %38 = arith.cmpi ne, %37, %c0_i64 : i64
  fir.if %38 {
    %39 = fir.absent !fir.box<none>
    %40 = fir.absent !fir.ref<i32>
    mif.dealloc_coarray %9#0 stat %40 errmsg %39 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>>, corank:1>>, !fir.ref<i32>, !fir.box<none>) -> ()
  }
  return
}

func.func @_QPquery_class_allocatable() {
  %0 = fir.alloca !fir.array<0xi64>
  %1 = fir.alloca !fir.array<1xi64>
  %2 = fir.dummy_scope : !fir.dscope
  %3 = fir.alloca i32 <{bindc_name = "count", uniq_name = "_QFquery_class_allocatableEcount"}>
  %4:2 = hlfir.declare %3 uniq_name("_QFquery_class_allocatableEcount") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5 = fir.alloca !fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1> <{bindc_name = "ev", uniq_name = "_QFquery_class_allocatableEev"}>
  %6 = fir.zero_bits !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
  %7 = fir.embox %6 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>) -> !fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>
  fir.store %7 to %5 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %8:2 = hlfir.declare %5 uniq_name("_QFquery_class_allocatableEev") fortran_attrs<allocatable> : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>, !fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>)
  %9 = fir.absent !fir.box<none>
  %10 = fir.type_desc !fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>
  %c0_i32 = arith.constant 0 : i32
  %c0_i32_0 = arith.constant 0 : i32
  %11 = fir.convert %8#0 : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>) -> !fir.ref<!fir.box<none>>
  %12 = fir.convert %10 : (!fir.tdesc<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>) -> !fir.ref<none>
  fir.call @_FortranAAllocatableInitDerivedForAllocate(%11, %12, %c0_i32, %c0_i32_0) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> ()
  %c1_i64 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %13 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
  fir.store %c1_i64 to %13 : !fir.ref<i64>
  %14 = fir.embox %1 : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
  %15 = fir.embox %0 : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
  mif.alloc_coarray %8#0 lcobounds %14 ucobounds %15 errmsg %9 uniq_name("_QFquery_class_allocatableEev") : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>, !fir.box<none>) -> ()
  %16 = fir.load %8#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %17 = mif.event_query %16 : (!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>) -> i32
  fir.store %17 to %4#0 : !fir.ref<i32>
  %18 = fir.load %8#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %19 = fir.box_addr %18 : (!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
  %20 = fir.convert %19 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>) -> i64
  %c0_i64 = arith.constant 0 : i64
  %21 = arith.cmpi ne, %20, %c0_i64 : i64
  fir.if %21 {
    %22 = fir.absent !fir.box<none>
    %23 = fir.absent !fir.ref<i32>
    mif.dealloc_coarray %8#0 stat %23 errmsg %22 : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>, !fir.ref<i32>, !fir.box<none>) -> ()
  }
  return
}

// CHECK-LABEL: func.func @_QPquery_array_element() {
// CHECK: %[[VAL_0:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_1:.*]] = fir.alloca i64
// CHECK: %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
// CHECK: %[[VAL_3:.*]] = fir.alloca i32 <{bindc_name = "count", uniq_name = "_QFquery_array_elementEcount"}>
// CHECK: %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] uniq_name("_QFquery_array_elementEcount") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK: %[[VAL_5:.*]] = fir.address_of(@_QFquery_array_elementEev) : {{.*}}
// CHECK: %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] uniq_name("_QFquery_array_elementEev") : {{.*}}
// CHECK: %[[VAL_7:.*]] = fir.load %[[VAL_6]]#0 : {{.*}}
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[VAL_8:.*]] = hlfir.designate %[[VAL_7]] (%[[C2]]) : ({{.*}}, index) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK: %[[VAL_9:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_10:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i64
// CHECK: %[[VAL_11:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_12:.*]] = fir.coordinate_of %[[VAL_0]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
// CHECK: fir.store %[[VAL_10]] to %[[VAL_12]] : !fir.ref<i64>
// CHECK: fir.call @_QMprifPprif_event_query(%[[VAL_0]], %[[VAL_1]], %[[VAL_9]]) : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<i64>, !fir.ref<i32>) -> ()
// CHECK: %[[VAL_13:.*]] = fir.load %[[VAL_1]] : !fir.ref<i64>
// CHECK: %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> i32
// CHECK: fir.store %[[VAL_14]] to %[[VAL_4]]#0 : !fir.ref<i32>
// CHECK: return
// CHECK: }

// CHECK-LABEL: func.func @_QPquery_array_in_loop() {
// CHECK: %[[VAL_0:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_1:.*]] = fir.alloca i64
// CHECK: %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
// CHECK: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[VAL_3:.*]] = fir.alloca !fir.array<4xi32> <{bindc_name = "counts", uniq_name = "_QFquery_array_in_loopEcounts"}>
// CHECK: %[[VAL_4:.*]] = fir.shape %[[C4]] : (index) -> !fir.shape<1>
// CHECK: %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_4]]) uniq_name("_QFquery_array_in_loopEcounts") : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xi32>>, !fir.ref<!fir.array<4xi32>>)
// CHECK: %[[VAL_6:.*]] = fir.address_of(@_QFquery_array_in_loopEev) : {{.*}}
// CHECK: %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] uniq_name("_QFquery_array_in_loopEev") : {{.*}}
// CHECK: %[[VAL_8:.*]] = fir.alloca i32 <{bindc_name = "i", uniq_name = "_QFquery_array_in_loopEi"}>
// CHECK: %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] uniq_name("_QFquery_array_in_loopEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK: %[[C4_0:.*]] = arith.constant 4 : index
// CHECK: %[[VAL_10:.*]] = fir.alloca !fir.array<4xi32> <{bindc_name = "stats", uniq_name = "_QFquery_array_in_loopEstats"}>
// CHECK: %[[VAL_11:.*]] = fir.shape %[[C4_0]] : (index) -> !fir.shape<1>
// CHECK: %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_10]](%[[VAL_11]]) uniq_name("_QFquery_array_in_loopEstats") : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xi32>>, !fir.ref<!fir.array<4xi32>>)
// CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK: %[[C4_I32:.*]] = arith.constant 4 : i32
// CHECK: %[[C1_I32_1:.*]] = arith.constant 1 : i32
// CHECK: fir.do_loop %[[ARG0:.*]] = %[[C1_I32]] to %[[C4_I32]] step %[[C1_I32_1]] : i32 {
// CHECK: fir.store %[[ARG0]] to %[[VAL_9]]#0 : !fir.ref<i32>
// CHECK: %[[VAL_24:.*]] = fir.load %[[VAL_7]]#0 : {{.*}}
// CHECK: %[[VAL_25:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i32>
// CHECK: %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i32) -> i64
// CHECK: %[[VAL_27:.*]] = hlfir.designate %[[VAL_24]] (%[[VAL_26]]) : ({{.*}}, i64) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK: %[[VAL_28:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i32>
// CHECK: %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i32) -> i64
// CHECK: %[[VAL_30:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_29]]) : (!fir.ref<!fir.array<4xi32>>, i64) -> !fir.ref<i32>
// CHECK: %[[VAL_31:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<i32>
// CHECK: %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i32) -> i64
// CHECK: %[[VAL_33:.*]] = hlfir.designate %[[VAL_12]]#0 (%[[VAL_32]]) : (!fir.ref<!fir.array<4xi32>>, i64) -> !fir.ref<i32>
// CHECK: %[[VAL_34:.*]] = fir.convert %[[VAL_27]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i64
// CHECK: %[[VAL_35:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_36:.*]] = fir.coordinate_of %[[VAL_0]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
// CHECK: fir.store %[[VAL_34]] to %[[VAL_36]] : !fir.ref<i64>
// CHECK: fir.call @_QMprifPprif_event_query(%[[VAL_0]], %[[VAL_1]], %[[VAL_33]]) : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<i64>, !fir.ref<i32>) -> ()
// CHECK: %[[VAL_37:.*]] = fir.load %[[VAL_1]] : !fir.ref<i64>
// CHECK: %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (i64) -> i32
// CHECK: fir.store %[[VAL_38]] to %[[VAL_30]] : !fir.ref<i32>
// CHECK: }
// CHECK: %[[VAL_13:.*]] = fir.convert %[[C1_I32]] : (i32) -> index
// CHECK: %[[VAL_14:.*]] = fir.convert %[[C4_I32]] : (i32) -> index
// CHECK: %[[VAL_15:.*]] = fir.convert %[[C1_I32_1]] : (i32) -> index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[VAL_16:.*]] = arith.subi %[[VAL_14]], %[[VAL_13]] : index
// CHECK: %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_15]] : index
// CHECK: %[[VAL_18:.*]] = arith.divsi %[[VAL_17]], %[[VAL_15]] : index
// CHECK: %[[VAL_19:.*]] = arith.cmpi slt, %[[VAL_18]], %[[C0]] : index
// CHECK: %[[VAL_20:.*]] = arith.select %[[VAL_19]], %[[C0]], %[[VAL_18]] : index
// CHECK: %[[VAL_21:.*]] = arith.muli %[[VAL_20]], %[[VAL_15]] : index
// CHECK: %[[VAL_22:.*]] = arith.addi %[[VAL_13]], %[[VAL_21]] : index
// CHECK: %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (index) -> i32
// CHECK: fir.store %[[VAL_23]] to %[[VAL_9]]#0 : !fir.ref<i32>
// CHECK: return
// CHECK: }

// CHECK-LABEL: func.func @_QPquery_allocatable() {
// CHECK: %[[VAL_0:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_1:.*]] = fir.alloca i64
// CHECK: %[[VAL_2:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_3:.*]] = fir.alloca i64
// CHECK: %[[VAL_4:.*]] = fir.alloca i64
// CHECK: %[[VAL_5:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_6:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{{.*}}>
// CHECK: %[[VAL_7:.*]] = fir.alloca !fir.boxproc<(!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()>
// CHECK: %[[VAL_8:.*]] = fir.alloca !fir.array<0xi64>
// CHECK: %[[VAL_9:.*]] = fir.alloca !fir.array<1xi64>
// CHECK: %[[VAL_10:.*]] = fir.dummy_scope : !fir.dscope
// CHECK: %[[VAL_11:.*]] = fir.alloca i32 <{bindc_name = "count", uniq_name = "_QFquery_allocatableEcount"}>
// CHECK: %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] uniq_name("_QFquery_allocatableEcount") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK: %[[VAL_13:.*]] = fir.alloca {{.*}} <{bindc_name = "ev", uniq_name = "_QFquery_allocatableEev"}>
// CHECK: %[[VAL_14:.*]] = fir.zero_bits !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK: %[[VAL_15:.*]] = fir.embox %[[VAL_14]] : {{.*}}
// CHECK: fir.store %[[VAL_15]] to %[[VAL_13]] : {{.*}}
// CHECK: %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_13]] uniq_name("_QFquery_allocatableEev") fortran_attrs<allocatable> : {{.*}}
// CHECK: %[[VAL_17:.*]] = fir.alloca i32 <{bindc_name = "stat", uniq_name = "_QFquery_allocatableEstat"}>
// CHECK: %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_17]] uniq_name("_QFquery_allocatableEstat") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK: %[[VAL_19:.*]] = fir.absent !fir.box<none>
// CHECK: %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[VAL_20:.*]] = fir.coordinate_of %[[VAL_9]], %[[C0]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
// CHECK: fir.store %[[C1_I64]] to %[[VAL_20]] : !fir.ref<i64>
// CHECK: %[[VAL_21:.*]] = fir.embox %[[VAL_9]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
// CHECK: %[[VAL_22:.*]] = fir.embox %[[VAL_8]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
// CHECK: %[[VAL_23:.*]] = fir.zero_bits (!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK: %[[VAL_24:.*]] = fir.emboxproc %[[VAL_23]] : {{.*}}
// CHECK: fir.store %[[VAL_24]] to %[[VAL_7]] : {{.*}}
// CHECK: %[[VAL_25:.*]] = fir.load %[[VAL_16]]#0 : {{.*}}
// CHECK: %[[VAL_26:.*]] = fir.box_elesize %[[VAL_25]] : ({{.*}}) -> i64
// CHECK: fir.store %[[VAL_26]] to %[[VAL_4]] : !fir.ref<i64>
// CHECK: %[[VAL_27:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_28:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_29:.*]] = fir.convert %[[VAL_21]] : (!fir.box<!fir.array<1xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK: %[[VAL_30:.*]] = fir.convert %[[VAL_22]] : (!fir.box<!fir.array<0xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK: %[[VAL_31:.*]] = fir.convert %[[VAL_7]] : {{.*}} -> !fir.ref<none>
// CHECK: %[[VAL_32:.*]] = fir.convert %[[VAL_6]] : {{.*}} -> !fir.ref<none>
// CHECK: %[[VAL_33:.*]] = fir.convert %[[VAL_19]] : (!fir.box<none>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_allocate_coarray(%[[VAL_29]], %[[VAL_30]], %[[VAL_4]], %[[VAL_31]], %[[VAL_32]], %[[VAL_5]], %[[VAL_27]], %[[VAL_33]], %[[VAL_28]]) : {{.*}}
// CHECK: %[[VAL_34:.*]] = fir.address_of(@_QFquery_allocatableEev_coarray_handle) : {{.*}}
// CHECK: fir.copy %[[VAL_6]] to %[[VAL_34]] : {{.*}}
// CHECK: %[[VAL_35:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_36:.*]] = fir.coordinate_of %[[VAL_5]], __address : {{.*}}
// CHECK: %[[VAL_37:.*]] = fir.load %[[VAL_36]] : !fir.ref<i64>
// CHECK: %[[VAL_38:.*]] = fir.convert %[[VAL_16]]#0 : ({{.*}}) -> !fir.ref<!fir.box<none>>
// CHECK: %[[VAL_39:.*]] = fir.convert %[[VAL_37]] : (i64) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranAAllocatableSetBaseAddr(%[[VAL_38]], %[[VAL_39]]) : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
// CHECK: %[[VAL_40:.*]] = fir.address_of(@_QQcl{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]+}}>>
// CHECK: %[[LINE:.*]] = arith.constant {{[0-9]+}} : i32
// CHECK: %[[VAL_41:.*]] = fir.convert %[[VAL_16]]#0 : ({{.*}}) -> !fir.box<none>
// CHECK: %[[VAL_42:.*]] = fir.convert %[[VAL_40]] : (!fir.ref<!fir.char<1,{{[0-9]+}}>>) -> !fir.ref<i8>
// CHECK: fir.call @_FortranAInitialize(%[[VAL_41]], %[[VAL_42]], %[[LINE]]) : (!fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK: %[[VAL_43:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_44:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_45:.*]] = fir.convert %[[VAL_19]] : (!fir.box<none>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_sync_all(%[[VAL_44]], %[[VAL_45]], %[[VAL_43]]) : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK: %[[VAL_46:.*]] = fir.load %[[VAL_16]]#0 : {{.*}}
// CHECK: %[[VAL_47:.*]] = fir.box_addr %[[VAL_46]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK: %[[VAL_48:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_49:.*]] = fir.convert %[[VAL_47]] : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i64
// CHECK: %[[VAL_50:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_51:.*]] = fir.coordinate_of %[[VAL_2]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
// CHECK: fir.store %[[VAL_49]] to %[[VAL_51]] : !fir.ref<i64>
// CHECK: fir.call @_QMprifPprif_event_query(%[[VAL_2]], %[[VAL_3]], %[[VAL_48]]) : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<i64>, !fir.ref<i32>) -> ()
// CHECK: %[[VAL_52:.*]] = fir.load %[[VAL_3]] : !fir.ref<i64>
// CHECK: %[[VAL_53:.*]] = fir.convert %[[VAL_52]] : (i64) -> i32
// CHECK: fir.store %[[VAL_53]] to %[[VAL_12]]#0 : !fir.ref<i32>
// CHECK: %[[VAL_54:.*]] = fir.load %[[VAL_16]]#0 : {{.*}}
// CHECK: %[[VAL_55:.*]] = fir.box_addr %[[VAL_54]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK: %[[VAL_56:.*]] = fir.convert %[[VAL_55]] : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i64
// CHECK: %[[VAL_57:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_58:.*]] = fir.coordinate_of %[[VAL_0]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
// CHECK: fir.store %[[VAL_56]] to %[[VAL_58]] : !fir.ref<i64>
// CHECK: fir.call @_QMprifPprif_event_query(%[[VAL_0]], %[[VAL_1]], %[[VAL_18]]#0) : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<i64>, !fir.ref<i32>) -> ()
// CHECK: %[[VAL_59:.*]] = fir.load %[[VAL_1]] : !fir.ref<i64>
// CHECK: %[[VAL_60:.*]] = fir.convert %[[VAL_59]] : (i64) -> i32
// CHECK: fir.store %[[VAL_60]] to %[[VAL_12]]#0 : !fir.ref<i32>
// CHECK: %[[VAL_61:.*]] = fir.absent !fir.box<none>
// CHECK: %[[VAL_62:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_63:.*]] = fir.address_of(@_QFquery_allocatableEev_coarray_handle) : {{.*}}
// CHECK: %[[VAL_64:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_65:.*]] = fir.convert %[[VAL_63]] : {{.*}} -> !fir.ref<none>
// CHECK: %[[VAL_66:.*]] = fir.convert %[[VAL_61]] : (!fir.box<none>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_deallocate_coarray(%[[VAL_65]], %[[VAL_62]], %[[VAL_66]], %[[VAL_64]]) : (!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK: %[[VAL_67:.*]] = fir.absent !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
// CHECK: %[[VAL_68:.*]] = fir.convert %[[VAL_16]]#0 : ({{.*}}) -> !fir.ref<!fir.box<none>>
// CHECK: %[[VAL_69:.*]] = fir.convert %[[VAL_67]] : ({{.*}}) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranAPointerAssociateScalar(%[[VAL_68]], %[[VAL_69]]) : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
// CHECK: %[[VAL_70:.*]] = fir.load %[[VAL_16]]#0 : {{.*}}
// CHECK: %[[VAL_71:.*]] = fir.box_addr %[[VAL_70]] : ({{.*}}) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK: %[[VAL_72:.*]] = fir.convert %[[VAL_71]] : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i64
// CHECK: %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK: %[[VAL_73:.*]] = arith.cmpi ne, %[[VAL_72]], %[[C0_I64]] : i64
// CHECK: fir.if %[[VAL_73]] {
// CHECK: %[[VAL_74:.*]] = fir.absent !fir.box<none>
// CHECK: %[[VAL_75:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_76:.*]] = fir.address_of(@_QFquery_allocatableEev_coarray_handle) : {{.*}}
// CHECK: %[[VAL_77:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_78:.*]] = fir.convert %[[VAL_76]] : {{.*}} -> !fir.ref<none>
// CHECK: %[[VAL_79:.*]] = fir.convert %[[VAL_74]] : (!fir.box<none>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_deallocate_coarray(%[[VAL_78]], %[[VAL_75]], %[[VAL_79]], %[[VAL_77]]) : (!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK: %[[VAL_80:.*]] = fir.absent !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
// CHECK: %[[VAL_81:.*]] = fir.convert %[[VAL_16]]#0 : ({{.*}}) -> !fir.ref<!fir.box<none>>
// CHECK: %[[VAL_82:.*]] = fir.convert %[[VAL_80]] : ({{.*}}) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranAPointerAssociateScalar(%[[VAL_81]], %[[VAL_82]]) : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
// CHECK: }
// CHECK: return
// CHECK: }

// CHECK-LABEL: func.func @_QPquery_allocatable_array
// CHECK: %[[VAL_0:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_1:.*]] = fir.alloca i64
// CHECK: %[[VAL_2:.*]] = fir.alloca i64
// CHECK: %[[VAL_3:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_4:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{{.*}}>
// CHECK: %[[VAL_5:.*]] = fir.alloca !fir.boxproc<(!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()>
// CHECK: %[[VAL_6:.*]] = fir.alloca !fir.array<0xi64>
// CHECK: %[[VAL_7:.*]] = fir.alloca !fir.array<1xi64>
// CHECK: %[[VAL_8:.*]] = fir.dummy_scope : !fir.dscope
// CHECK: %[[VAL_9:.*]] = fir.alloca i32 <{bindc_name = "count", uniq_name = "_QFquery_allocatable_arrayEcount"}>
// CHECK: %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] uniq_name("_QFquery_allocatable_arrayEcount") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK: %[[VAL_11:.*]] = fir.alloca {{.*}} <{bindc_name = "ev", uniq_name = "_QFquery_allocatable_arrayEev"}>
// CHECK: %[[VAL_12:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[VAL_13:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
// CHECK: %[[VAL_14:.*]] = fir.embox %[[VAL_12]](%[[VAL_13]]) : {{.*}}
// CHECK: fir.store %[[VAL_14]] to %[[VAL_11]] : {{.*}}
// CHECK: %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_11]] uniq_name("_QFquery_allocatable_arrayEev") fortran_attrs<allocatable> : {{.*}}
// CHECK: %[[VAL_16:.*]] = fir.alloca i32 <{bindc_name = "i", uniq_name = "_QFquery_allocatable_arrayEi"}>
// CHECK: %[[VAL_17:.*]]:2 = hlfir.declare %[[VAL_16]] uniq_name("_QFquery_allocatable_arrayEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK: %[[VAL_18:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_8]] arg 1 uniq_name("_QFquery_allocatable_arrayEn") fortran_attrs<intent_in> : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK: %[[VAL_19:.*]] = fir.absent !fir.box<none>
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[VAL_20:.*]] = fir.load %[[VAL_18]]#0 : !fir.ref<i32>
// CHECK: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK: %[[VAL_21:.*]] = fir.convert %[[VAL_15]]#0 : ({{.*}}) -> !fir.ref<!fir.box<none>>
// CHECK: %[[VAL_22:.*]] = fir.convert %[[C1]] : (index) -> i64
// CHECK: %[[VAL_23:.*]] = fir.convert %[[VAL_20]] : (i32) -> i64
// CHECK: fir.call @_FortranAAllocatableSetBounds(%[[VAL_21]], %[[C0_I32]], %[[VAL_22]], %[[VAL_23]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
// CHECK: %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK: %[[C0_0:.*]] = arith.constant 0 : index
// CHECK: %[[VAL_24:.*]] = fir.coordinate_of %[[VAL_7]], %[[C0_0]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
// CHECK: fir.store %[[C1_I64]] to %[[VAL_24]] : !fir.ref<i64>
// CHECK: %[[VAL_25:.*]] = fir.embox %[[VAL_7]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
// CHECK: %[[VAL_26:.*]] = fir.embox %[[VAL_6]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
// CHECK: %[[VAL_27:.*]] = fir.zero_bits (!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK: %[[VAL_28:.*]] = fir.emboxproc %[[VAL_27]] : {{.*}}
// CHECK: fir.store %[[VAL_28]] to %[[VAL_5]] : {{.*}}
// CHECK: %[[VAL_29:.*]] = fir.load %[[VAL_15]]#0 : {{.*}}
// CHECK: %[[VAL_30:.*]] = fir.box_elesize %[[VAL_29]] : ({{.*}}) -> i64
// CHECK: %[[VAL_31:.*]] = fir.address_of(@_QQcl{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]+}}>>
// CHECK: %[[LINE:.*]] = arith.constant {{[0-9]+}} : i32
// CHECK: %[[VAL_32:.*]] = fir.convert %[[VAL_29]] : ({{.*}}) -> !fir.box<none>
// CHECK: %[[VAL_33:.*]] = fir.convert %[[VAL_31]] : (!fir.ref<!fir.char<1,{{[0-9]+}}>>) -> !fir.ref<i8>
// CHECK: %[[VAL_34:.*]] = fir.call @_FortranASize(%[[VAL_32]], %[[VAL_33]], %[[LINE]]) : (!fir.box<none>, !fir.ref<i8>, i32) -> i64
// CHECK: %[[VAL_35:.*]] = arith.muli %[[VAL_30]], %[[VAL_34]] : i64
// CHECK: fir.store %[[VAL_35]] to %[[VAL_2]] : !fir.ref<i64>
// CHECK: %[[VAL_36:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_37:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_38:.*]] = fir.convert %[[VAL_25]] : (!fir.box<!fir.array<1xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK: %[[VAL_39:.*]] = fir.convert %[[VAL_26]] : (!fir.box<!fir.array<0xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK: %[[VAL_40:.*]] = fir.convert %[[VAL_5]] : {{.*}} -> !fir.ref<none>
// CHECK: %[[VAL_41:.*]] = fir.convert %[[VAL_4]] : {{.*}} -> !fir.ref<none>
// CHECK: %[[VAL_42:.*]] = fir.convert %[[VAL_19]] : (!fir.box<none>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_allocate_coarray(%[[VAL_38]], %[[VAL_39]], %[[VAL_2]], %[[VAL_40]], %[[VAL_41]], %[[VAL_3]], %[[VAL_36]], %[[VAL_42]], %[[VAL_37]]) : {{.*}}
// CHECK: %[[VAL_43:.*]] = fir.address_of(@_QFquery_allocatable_arrayEev_coarray_handle) : {{.*}}
// CHECK: fir.copy %[[VAL_4]] to %[[VAL_43]] : {{.*}}
// CHECK: %[[VAL_44:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_45:.*]] = fir.coordinate_of %[[VAL_3]], __address : {{.*}}
// CHECK: %[[VAL_46:.*]] = fir.load %[[VAL_45]] : !fir.ref<i64>
// CHECK: %[[VAL_47:.*]] = fir.convert %[[VAL_15]]#0 : ({{.*}}) -> !fir.ref<!fir.box<none>>
// CHECK: %[[VAL_48:.*]] = fir.convert %[[VAL_46]] : (i64) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranAAllocatableSetBaseAddr(%[[VAL_47]], %[[VAL_48]]) : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
// CHECK: %[[VAL_49:.*]] = fir.address_of(@_QQcl{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]+}}>>
// CHECK: %[[LINE_2:.*]] = arith.constant {{[0-9]+}} : i32
// CHECK: %[[VAL_50:.*]] = fir.convert %[[VAL_15]]#0 : ({{.*}}) -> !fir.box<none>
// CHECK: %[[VAL_51:.*]] = fir.convert %[[VAL_49]] : (!fir.ref<!fir.char<1,{{[0-9]+}}>>) -> !fir.ref<i8>
// CHECK: fir.call @_FortranAInitialize(%[[VAL_50]], %[[VAL_51]], %[[LINE_2]]) : (!fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK: %[[VAL_52:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_53:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_54:.*]] = fir.convert %[[VAL_19]] : (!fir.box<none>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_sync_all(%[[VAL_53]], %[[VAL_54]], %[[VAL_52]]) : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK: %[[VAL_55:.*]] = fir.load %[[VAL_18]]#0 : !fir.ref<i32>
// CHECK: %[[C1_I32_3:.*]] = arith.constant 1 : i32
// CHECK: fir.do_loop %[[ARG1:.*]] = %[[C1_I32]] to %[[VAL_55]] step %[[C1_I32_3]] : i32 {
// CHECK: fir.store %[[ARG1]] to %[[VAL_17]]#0 : !fir.ref<i32>
// CHECK: %[[VAL_80:.*]] = fir.load %[[VAL_15]]#0 : {{.*}}
// CHECK: %[[VAL_81:.*]] = fir.load %[[VAL_17]]#0 : !fir.ref<i32>
// CHECK: %[[VAL_82:.*]] = fir.convert %[[VAL_81]] : (i32) -> i64
// CHECK: %[[VAL_83:.*]] = hlfir.designate %[[VAL_80]] (%[[VAL_82]]) : ({{.*}}, i64) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK: %[[VAL_84:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_85:.*]] = fir.convert %[[VAL_83]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i64
// CHECK: %[[VAL_86:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_87:.*]] = fir.coordinate_of %[[VAL_0]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
// CHECK: fir.store %[[VAL_85]] to %[[VAL_87]] : !fir.ref<i64>
// CHECK: fir.call @_QMprifPprif_event_query(%[[VAL_0]], %[[VAL_1]], %[[VAL_84]]) : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<i64>, !fir.ref<i32>) -> ()
// CHECK: %[[VAL_88:.*]] = fir.load %[[VAL_1]] : !fir.ref<i64>
// CHECK: %[[VAL_89:.*]] = fir.convert %[[VAL_88]] : (i64) -> i32
// CHECK: fir.store %[[VAL_89]] to %[[VAL_10]]#0 : !fir.ref<i32>
// CHECK: }
// CHECK: %[[VAL_56:.*]] = fir.convert %[[C1_I32]] : (i32) -> index
// CHECK: %[[VAL_57:.*]] = fir.convert %[[VAL_55]] : (i32) -> index
// CHECK: %[[VAL_58:.*]] = fir.convert %[[C1_I32_3]] : (i32) -> index
// CHECK: %[[C0_4:.*]] = arith.constant 0 : index
// CHECK: %[[VAL_59:.*]] = arith.subi %[[VAL_57]], %[[VAL_56]] : index
// CHECK: %[[VAL_60:.*]] = arith.addi %[[VAL_59]], %[[VAL_58]] : index
// CHECK: %[[VAL_61:.*]] = arith.divsi %[[VAL_60]], %[[VAL_58]] : index
// CHECK: %[[VAL_62:.*]] = arith.cmpi slt, %[[VAL_61]], %[[C0_4]] : index
// CHECK: %[[VAL_63:.*]] = arith.select %[[VAL_62]], %[[C0_4]], %[[VAL_61]] : index
// CHECK: %[[VAL_64:.*]] = arith.muli %[[VAL_63]], %[[VAL_58]] : index
// CHECK: %[[VAL_65:.*]] = arith.addi %[[VAL_56]], %[[VAL_64]] : index
// CHECK: %[[VAL_66:.*]] = fir.convert %[[VAL_65]] : (index) -> i32
// CHECK: fir.store %[[VAL_66]] to %[[VAL_17]]#0 : !fir.ref<i32>
// CHECK: %[[VAL_67:.*]] = fir.absent !fir.box<none>
// CHECK: %[[VAL_68:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_69:.*]] = fir.address_of(@_QFquery_allocatable_arrayEev_coarray_handle) : {{.*}}
// CHECK: %[[VAL_70:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_71:.*]] = fir.convert %[[VAL_69]] : {{.*}} -> !fir.ref<none>
// CHECK: %[[VAL_72:.*]] = fir.convert %[[VAL_67]] : (!fir.box<none>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_deallocate_coarray(%[[VAL_71]], %[[VAL_68]], %[[VAL_72]], %[[VAL_70]]) : (!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK: %[[VAL_73:.*]] = fir.absent !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>>, corank:1>>
// CHECK: %[[VAL_74:.*]] = fir.convert %[[VAL_15]]#0 : ({{.*}}) -> !fir.ref<!fir.box<none>>
// CHECK: %[[VAL_75:.*]] = fir.convert %[[VAL_73]] : ({{.*}}) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranAPointerAssociateScalar(%[[VAL_74]], %[[VAL_75]]) : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
// CHECK: %[[VAL_76:.*]] = fir.load %[[VAL_15]]#0 : {{.*}}
// CHECK: %[[VAL_77:.*]] = fir.box_addr %[[VAL_76]] : ({{.*}}) -> !fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>>
// CHECK: %[[VAL_78:.*]] = fir.convert %[[VAL_77]] : ({{.*}}) -> i64
// CHECK: %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK: %[[VAL_79:.*]] = arith.cmpi ne, %[[VAL_78]], %[[C0_I64]] : i64
// CHECK: fir.if %[[VAL_79]] {
// CHECK: %[[VAL_80:.*]] = fir.absent !fir.box<none>
// CHECK: %[[VAL_81:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_82:.*]] = fir.address_of(@_QFquery_allocatable_arrayEev_coarray_handle) : {{.*}}
// CHECK: %[[VAL_83:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_84:.*]] = fir.convert %[[VAL_82]] : {{.*}} -> !fir.ref<none>
// CHECK: %[[VAL_85:.*]] = fir.convert %[[VAL_80]] : (!fir.box<none>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_deallocate_coarray(%[[VAL_84]], %[[VAL_81]], %[[VAL_85]], %[[VAL_83]]) : (!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK: %[[VAL_86:.*]] = fir.absent !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>>, corank:1>>
// CHECK: %[[VAL_87:.*]] = fir.convert %[[VAL_15]]#0 : ({{.*}}) -> !fir.ref<!fir.box<none>>
// CHECK: %[[VAL_88:.*]] = fir.convert %[[VAL_86]] : ({{.*}}) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranAPointerAssociateScalar(%[[VAL_87]], %[[VAL_88]]) : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
// CHECK: }
// CHECK: return
// CHECK: }

// CHECK-LABEL: func.func @_QPquery_class_allocatable() {
// CHECK: %[[VAL_0:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_1:.*]] = fir.alloca i64
// CHECK: %[[VAL_2:.*]] = fir.alloca i64
// CHECK: %[[VAL_3:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_4:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{{.*}}>
// CHECK: %[[VAL_5:.*]] = fir.alloca !fir.boxproc<(!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()>
// CHECK: %[[VAL_6:.*]] = fir.alloca !fir.array<0xi64>
// CHECK: %[[VAL_7:.*]] = fir.alloca !fir.array<1xi64>
// CHECK: %[[VAL_8:.*]] = fir.dummy_scope : !fir.dscope
// CHECK: %[[VAL_9:.*]] = fir.alloca i32 <{bindc_name = "count", uniq_name = "_QFquery_class_allocatableEcount"}>
// CHECK: %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] uniq_name("_QFquery_class_allocatableEcount") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK: %[[VAL_11:.*]] = fir.alloca !fir.class<{{.*}}> <{bindc_name = "ev", uniq_name = "_QFquery_class_allocatableEev"}>
// CHECK: %[[VAL_12:.*]] = fir.zero_bits !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK: %[[VAL_13:.*]] = fir.embox %[[VAL_12]] : ({{.*}}) -> !fir.class<{{.*}}>
// CHECK: fir.store %[[VAL_13]] to %[[VAL_11]] : {{.*}}
// CHECK: %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_11]] uniq_name("_QFquery_class_allocatableEev") fortran_attrs<allocatable> : {{.*}}
// CHECK: %[[VAL_15:.*]] = fir.absent !fir.box<none>
// CHECK: %[[VAL_16:.*]] = fir.type_desc !fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>
// CHECK: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK: %[[C0_I32_0:.*]] = arith.constant 0 : i32
// CHECK: %[[VAL_17:.*]] = fir.convert %[[VAL_14]]#0 : ({{.*}}) -> !fir.ref<!fir.box<none>>
// CHECK: %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : ({{.*}}) -> !fir.ref<none>
// CHECK: fir.call @_FortranAAllocatableInitDerivedForAllocate(%[[VAL_17]], %[[VAL_18]], %[[C0_I32]], %[[C0_I32_0]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> ()
// CHECK: %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[VAL_19:.*]] = fir.coordinate_of %[[VAL_7]], %[[C0]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
// CHECK: fir.store %[[C1_I64]] to %[[VAL_19]] : !fir.ref<i64>
// CHECK: %[[VAL_20:.*]] = fir.embox %[[VAL_7]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
// CHECK: %[[VAL_21:.*]] = fir.embox %[[VAL_6]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
// CHECK: %[[VAL_22:.*]] = fir.zero_bits (!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK: %[[VAL_23:.*]] = fir.emboxproc %[[VAL_22]] : {{.*}}
// CHECK: fir.store %[[VAL_23]] to %[[VAL_5]] : {{.*}}
// CHECK: %[[VAL_24:.*]] = fir.load %[[VAL_14]]#0 : {{.*}}
// CHECK: %[[VAL_25:.*]] = fir.box_elesize %[[VAL_24]] : ({{.*}}) -> i64
// CHECK: fir.store %[[VAL_25]] to %[[VAL_2]] : !fir.ref<i64>
// CHECK: %[[VAL_26:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_27:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_28:.*]] = fir.convert %[[VAL_20]] : (!fir.box<!fir.array<1xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK: %[[VAL_29:.*]] = fir.convert %[[VAL_21]] : (!fir.box<!fir.array<0xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK: %[[VAL_30:.*]] = fir.convert %[[VAL_5]] : {{.*}} -> !fir.ref<none>
// CHECK: %[[VAL_31:.*]] = fir.convert %[[VAL_4]] : {{.*}} -> !fir.ref<none>
// CHECK: %[[VAL_32:.*]] = fir.convert %[[VAL_15]] : (!fir.box<none>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_allocate_coarray(%[[VAL_28]], %[[VAL_29]], %[[VAL_2]], %[[VAL_30]], %[[VAL_31]], %[[VAL_3]], %[[VAL_26]], %[[VAL_32]], %[[VAL_27]]) : {{.*}}
// CHECK: %[[VAL_33:.*]] = fir.address_of(@_QFquery_class_allocatableEev_coarray_handle) : {{.*}}
// CHECK: fir.copy %[[VAL_4]] to %[[VAL_33]] : {{.*}}
// CHECK: %[[VAL_34:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_35:.*]] = fir.coordinate_of %[[VAL_3]], __address : {{.*}}
// CHECK: %[[VAL_36:.*]] = fir.load %[[VAL_35]] : !fir.ref<i64>
// CHECK: %[[VAL_37:.*]] = fir.convert %[[VAL_14]]#0 : ({{.*}}) -> !fir.ref<!fir.box<none>>
// CHECK: %[[VAL_38:.*]] = fir.convert %[[VAL_36]] : (i64) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranAAllocatableSetBaseAddr(%[[VAL_37]], %[[VAL_38]]) : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
// CHECK: %[[VAL_39:.*]] = fir.address_of(@_QQcl{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]+}}>>
// CHECK: %[[LINE:.*]] = arith.constant {{[0-9]+}} : i32
// CHECK: %[[VAL_40:.*]] = fir.convert %[[VAL_14]]#0 : ({{.*}}) -> !fir.box<none>
// CHECK: %[[VAL_41:.*]] = fir.convert %[[VAL_39]] : (!fir.ref<!fir.char<1,{{[0-9]+}}>>) -> !fir.ref<i8>
// CHECK: fir.call @_FortranAInitialize(%[[VAL_40]], %[[VAL_41]], %[[LINE]]) : (!fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK: %[[VAL_42:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_43:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_44:.*]] = fir.convert %[[VAL_15]] : (!fir.box<none>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_sync_all(%[[VAL_43]], %[[VAL_44]], %[[VAL_42]]) : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK: %[[VAL_45:.*]] = fir.load %[[VAL_14]]#0 : {{.*}}
// CHECK: %[[VAL_46:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_47:.*]] = fir.box_addr %[[VAL_45]] : (!fir.class<{{.*}}>) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK: %[[VAL_48:.*]] = fir.convert %[[VAL_47]] : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i64
// CHECK: %[[VAL_49:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK: %[[VAL_50:.*]] = fir.coordinate_of %[[VAL_0]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
// CHECK: fir.store %[[VAL_48]] to %[[VAL_50]] : !fir.ref<i64>
// CHECK: fir.call @_QMprifPprif_event_query(%[[VAL_0]], %[[VAL_1]], %[[VAL_46]]) : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<i64>, !fir.ref<i32>) -> ()
// CHECK: %[[VAL_51:.*]] = fir.load %[[VAL_1]] : !fir.ref<i64>
// CHECK: %[[VAL_52:.*]] = fir.convert %[[VAL_51]] : (i64) -> i32
// CHECK: fir.store %[[VAL_52]] to %[[VAL_10]]#0 : !fir.ref<i32>
// CHECK: %[[VAL_53:.*]] = fir.load %[[VAL_14]]#0 : {{.*}}
// CHECK: %[[VAL_54:.*]] = fir.box_addr %[[VAL_53]] : (!fir.class<{{.*}}>) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK: %[[VAL_55:.*]] = fir.convert %[[VAL_54]] : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>) -> i64
// CHECK: %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK: %[[VAL_56:.*]] = arith.cmpi ne, %[[VAL_55]], %[[C0_I64]] : i64
// CHECK: fir.if %[[VAL_56]] {
// CHECK: %[[VAL_57:.*]] = fir.absent !fir.box<none>
// CHECK: %[[VAL_58:.*]] = fir.absent !fir.ref<i32>
// CHECK: %[[VAL_59:.*]] = fir.address_of(@_QFquery_class_allocatableEev_coarray_handle) : {{.*}}
// CHECK: %[[VAL_60:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK: %[[VAL_61:.*]] = fir.convert %[[VAL_59]] : {{.*}} -> !fir.ref<none>
// CHECK: %[[VAL_62:.*]] = fir.convert %[[VAL_57]] : (!fir.box<none>) -> !fir.box<!fir.char<1,?>>
// CHECK: fir.call @_QMprifPprif_deallocate_coarray(%[[VAL_61]], %[[VAL_58]], %[[VAL_62]], %[[VAL_60]]) : (!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK: %[[VAL_63:.*]] = fir.absent !fir.ref<!fir.class<{{.*}}>>
// CHECK: %[[VAL_64:.*]] = fir.convert %[[VAL_14]]#0 : ({{.*}}) -> !fir.ref<!fir.box<none>>
// CHECK: %[[VAL_65:.*]] = fir.convert %[[VAL_63]] : ({{.*}}) -> !fir.llvm_ptr<i8>
// CHECK: fir.call @_FortranAPointerAssociateScalar(%[[VAL_64]], %[[VAL_65]]) : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
// CHECK: }
// CHECK: return
// CHECK: }
