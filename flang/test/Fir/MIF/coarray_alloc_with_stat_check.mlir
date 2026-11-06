// RUN: fir-opt --mif-convert %s | FileCheck %s

func.func @_QPalloc_stat(%arg0: !fir.ref<i32> {fir.bindc_name = "s"}) {
  %0 = fir.alloca !fir.array<0xi64>
  %1 = fir.alloca !fir.array<1xi64>
  %2 = fir.alloca i32
  %3 = fir.dummy_scope : !fir.dscope
  %4 = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>, corank:1> <{bindc_name = "a", uniq_name = "_QFalloc_statEa"}>
  %5 = fir.zero_bits !fir.heap<!fir.array<?xi32>>
  %c0 = arith.constant 0 : index
  %6 = fir.shape %c0 : (index) -> !fir.shape<1>
  %7 = fir.embox %5(%6) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>
  fir.store %7 to %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>>
  %8:2 = hlfir.declare %4 uniq_name("_QFalloc_statEa") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>>)
  %9:2 = hlfir.declare %arg0 dummy_scope %3 arg 1 uniq_name("_QFalloc_statEs") : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %10 = fir.absent !fir.box<none>
  %c1 = arith.constant 1 : index
  %c10_i32 = arith.constant 10 : i32
  %c0_i32 = arith.constant 0 : i32
  %11 = fir.convert %8#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>>) -> !fir.ref<!fir.box<none>>
  %12 = fir.convert %c1 : (index) -> i64
  %13 = fir.convert %c10_i32 : (i32) -> i64
  fir.call @_FortranAAllocatableSetBounds(%11, %c0_i32, %12, %13) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
  %c1_i64 = arith.constant 1 : i64
  %c0_0 = arith.constant 0 : index
  %14 = fir.coordinate_of %1, %c0_0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
  fir.store %c1_i64 to %14 : !fir.ref<i64>
  %15 = fir.embox %1 : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
  %16 = fir.embox %0 : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
  mif.alloc_coarray %8#0 lcobounds %15 ucobounds %16 stat %2 errmsg %10 uniq_name("_QFalloc_statEa") : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>, !fir.ref<i32>, !fir.box<none>) -> ()
  %17 = fir.load %2 : !fir.ref<i32>
  fir.store %17 to %9#0 : !fir.ref<i32>
  %c6_i32 = arith.constant 6 : i32
  %18 = fir.address_of(@_QQclX7b2453781d5ba1e8d2b9b4520250e325) : !fir.ref<!fir.char<1,48>>
  %19 = fir.convert %18 : (!fir.ref<!fir.char<1,48>>) -> !fir.ref<i8>
  %c5_i32 = arith.constant 5 : i32
  %20 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %19, %c5_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  %21 = fir.load %8#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>>
  %22 = fir.box_addr %21 : (!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>) -> !fir.heap<!fir.array<?xi32>>
  %23 = fir.convert %22 : (!fir.heap<!fir.array<?xi32>>) -> i64
  %c0_i64 = arith.constant 0 : i64
  %24 = arith.cmpi ne, %23, %c0_i64 : i64
  %25 = fir.call @_FortranAioOutputLogical(%20, %24) fastmath<contract> : (!fir.ref<i8>, i1) -> i1
  %26 = fir.call @_FortranAioEndIoStatement(%20) fastmath<contract> : (!fir.ref<i8>) -> i32
  %27 = fir.load %8#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>>
  %28 = fir.box_addr %27 : (!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>) -> !fir.heap<!fir.array<?xi32>>
  %29 = fir.convert %28 : (!fir.heap<!fir.array<?xi32>>) -> i64
  %c0_i64_1 = arith.constant 0 : i64
  %30 = arith.cmpi ne, %29, %c0_i64_1 : i64
  fir.if %30 {
    %31 = fir.absent !fir.box<none>
    %32 = fir.absent !fir.ref<i32>
    mif.dealloc_coarray %8#0 stat %32 errmsg %31 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>>, !fir.ref<i32>, !fir.box<none>) -> ()
  }
  return
}

func.func @_QPalloc_event_stat(%arg0: !fir.ref<i32> {fir.bindc_name = "s"}) {
  %0 = fir.alloca !fir.array<0xi64>
  %1 = fir.alloca !fir.array<1xi64>
  %2 = fir.alloca i32
  %3 = fir.dummy_scope : !fir.dscope
  %4 = fir.alloca !fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1> <{bindc_name = "ev", uniq_name = "_QFalloc_event_statEev"}>
  %5 = fir.zero_bits !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
  %6 = fir.embox %5 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>) -> !fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>
  fir.store %6 to %4 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %7:2 = hlfir.declare %4 uniq_name("_QFalloc_event_statEev") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>, !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>)
  %8:2 = hlfir.declare %arg0 dummy_scope %3 arg 1 uniq_name("_QFalloc_event_statEs") : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %9 = fir.absent !fir.box<none>
  %c1_i64 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %10 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
  fir.store %c1_i64 to %10 : !fir.ref<i64>
  %11 = fir.embox %1 : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
  %12 = fir.embox %0 : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
  mif.alloc_coarray %7#0 lcobounds %11 ucobounds %12 stat %2 errmsg %9 uniq_name("_QFalloc_event_statEev") : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>, !fir.ref<i32>, !fir.box<none>) -> ()
  %13 = fir.load %2 : !fir.ref<i32>
  fir.store %13 to %8#0 : !fir.ref<i32>
  %14 = fir.load %7#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>
  %15 = fir.box_addr %14 : (!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>) -> !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>
  %16 = fir.convert %15 : (!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>) -> i64
  %c0_i64 = arith.constant 0 : i64
  %17 = arith.cmpi ne, %16, %c0_i64 : i64
  fir.if %17 {
    %18 = fir.absent !fir.box<none>
    %19 = fir.absent !fir.ref<i32>
    mif.dealloc_coarray %7#0 stat %19 errmsg %18 : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{_QM__fortran_builtinsT__builtin_event_type.__m1:i64,_QM__fortran_builtinsT__builtin_event_type.__m2:i64,_QM__fortran_builtinsT__builtin_event_type.__m3:i64,_QM__fortran_builtinsT__builtin_event_type.__m4:i64,_QM__fortran_builtinsT__builtin_event_type.__m5:i64,_QM__fortran_builtinsT__builtin_event_type.__m6:i64,_QM__fortran_builtinsT__builtin_event_type.__m7:i64,_QM__fortran_builtinsT__builtin_event_type.__m8:i64}>>, corank:1>>, !fir.ref<i32>, !fir.box<none>) -> ()
  }
  return
}

// Default allocation without STAT
func.func @_QPdefault_init() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QFdefault_initE.n.pad) : !fir.ref<!fir.char<1,3>>
  %c3 = arith.constant 3 : index
  %2:2 = hlfir.declare %1 typeparams %c3 uniq_name("_QFdefault_initE.n.pad") fortran_attrs<target> : (!fir.ref<!fir.char<1,3>>, index) -> (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>)
  %3 = fir.address_of(@_QFdefault_initE.di.t.pad) : !fir.ref<i64>
  %4:2 = hlfir.declare %3 uniq_name("_QFdefault_initE.di.t.pad") fortran_attrs<target> : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
  %5 = fir.address_of(@_QFdefault_initE.n.x) : !fir.ref<!fir.char<1>>
  %c1 = arith.constant 1 : index
  %6:2 = hlfir.declare %5 typeparams %c1 uniq_name("_QFdefault_initE.n.x") fortran_attrs<target> : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
  %7 = fir.address_of(@_QFdefault_initE.di.t.x) : !fir.ref<i32>
  %8:2 = hlfir.declare %7 uniq_name("_QFdefault_initE.di.t.x") fortran_attrs<target> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %9 = fir.address_of(@_QFdefault_initE.n.t) : !fir.ref<!fir.char<1>>
  %c1_0 = arith.constant 1 : index
  %10:2 = hlfir.declare %9 typeparams %c1_0 uniq_name("_QFdefault_initE.n.t") fortran_attrs<target> : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
  %11 = fir.address_of(@_QFdefault_initEc) : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFdefault_initTt{pad:i64,x:i32}>>, corank:1>>
  %12:2 = hlfir.declare %11 uniq_name("_QFdefault_initEc") : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QFdefault_initTt{pad:i64,x:i32}>>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<_QFdefault_initTt{pad:i64,x:i32}>>, corank:1>>, !fir.ref<!fir.box<!fir.heap<!fir.type<_QFdefault_initTt{pad:i64,x:i32}>>, corank:1>>)
  %13 = fir.address_of(@_QFdefault_initE.c.t) : !fir.ref<!fir.array<2x!fir.type<_QM__fortran_type_infoTcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,genre:i8,category:i8,kind:i8,rank:i8,memoryspace:i8,__padding0:!fir.array<3xi8>,offset:i64,characterlen:!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>,derived:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>>>>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>,sizeinbytes:i64,uninstantiated:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,kindparameter:!fir.box<!fir.ptr<!fir.array<?xi64>>>,lenparameterkind:!fir.box<!fir.ptr<!fir.array<?xi8>>>,component:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTcomponent>>>>,procptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTprocptrcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,offset:i64,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,special:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTspecialbinding{which:i8,isargdescriptorset:i8,istypebound:i8,specialcaseflag:i8,__padding0:!fir.array<4xi8>,proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,specialbitset:i32,hasparent:i8,noinitializationneeded:i8,nodestructionneeded:i8,nofinalizationneeded:i8,nodefinedassignment:i8,__padding0:!fir.array<3xi8>}>>>,lenvalue:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,bounds:!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>>
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %14 = fir.shape_shift %c0, %c2 : (index, index) -> !fir.shapeshift<1>
  %15:2 = hlfir.declare %13(%14) uniq_name("_QFdefault_initE.c.t") fortran_attrs<target> : (!fir.ref<!fir.array<2x!fir.type<_QM__fortran_type_infoTcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,genre:i8,category:i8,kind:i8,rank:i8,memoryspace:i8,__padding0:!fir.array<3xi8>,offset:i64,characterlen:!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>,derived:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>>>>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>,sizeinbytes:i64,uninstantiated:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,kindparameter:!fir.box<!fir.ptr<!fir.array<?xi64>>>,lenparameterkind:!fir.box<!fir.ptr<!fir.array<?xi8>>>,component:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTcomponent>>>>,procptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTprocptrcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,offset:i64,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,special:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTspecialbinding{which:i8,isargdescriptorset:i8,istypebound:i8,specialcaseflag:i8,__padding0:!fir.array<4xi8>,proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,specialbitset:i32,hasparent:i8,noinitializationneeded:i8,nodestructionneeded:i8,nofinalizationneeded:i8,nodefinedassignment:i8,__padding0:!fir.array<3xi8>}>>>,lenvalue:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,bounds:!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<2x!fir.type<_QM__fortran_type_infoTcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,genre:i8,category:i8,kind:i8,rank:i8,memoryspace:i8,__padding0:!fir.array<3xi8>,offset:i64,characterlen:!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>,derived:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>>>>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>,sizeinbytes:i64,uninstantiated:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,kindparameter:!fir.box<!fir.ptr<!fir.array<?xi64>>>,lenparameterkind:!fir.box<!fir.ptr<!fir.array<?xi8>>>,component:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTcomponent>>>>,procptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTprocptrcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,offset:i64,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,special:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTspecialbinding{which:i8,isargdescriptorset:i8,istypebound:i8,specialcaseflag:i8,__padding0:!fir.array<4xi8>,proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,specialbitset:i32,hasparent:i8,noinitializationneeded:i8,nodestructionneeded:i8,nofinalizationneeded:i8,nodefinedassignment:i8,__padding0:!fir.array<3xi8>}>>>,lenvalue:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,bounds:!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>>, !fir.ref<!fir.array<2x!fir.type<_QM__fortran_type_infoTcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,genre:i8,category:i8,kind:i8,rank:i8,memoryspace:i8,__padding0:!fir.array<3xi8>,offset:i64,characterlen:!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>,derived:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>>>>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>,sizeinbytes:i64,uninstantiated:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,kindparameter:!fir.box<!fir.ptr<!fir.array<?xi64>>>,lenparameterkind:!fir.box<!fir.ptr<!fir.array<?xi8>>>,component:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTcomponent>>>>,procptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTprocptrcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,offset:i64,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,special:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTspecialbinding{which:i8,isargdescriptorset:i8,istypebound:i8,specialcaseflag:i8,__padding0:!fir.array<4xi8>,proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,specialbitset:i32,hasparent:i8,noinitializationneeded:i8,nodestructionneeded:i8,nofinalizationneeded:i8,nodefinedassignment:i8,__padding0:!fir.array<3xi8>}>>>,lenvalue:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,bounds:!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>>)
  %16 = fir.address_of(@_QFdefault_initE.dt.t) : !fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>>>>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>,sizeinbytes:i64,uninstantiated:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,kindparameter:!fir.box<!fir.ptr<!fir.array<?xi64>>>,lenparameterkind:!fir.box<!fir.ptr<!fir.array<?xi8>>>,component:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,genre:i8,category:i8,kind:i8,rank:i8,memoryspace:i8,__padding0:!fir.array<3xi8>,offset:i64,characterlen:!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>,derived:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,lenvalue:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,bounds:!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>>>,procptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTprocptrcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,offset:i64,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,special:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTspecialbinding{which:i8,isargdescriptorset:i8,istypebound:i8,specialcaseflag:i8,__padding0:!fir.array<4xi8>,proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,specialbitset:i32,hasparent:i8,noinitializationneeded:i8,nodestructionneeded:i8,nofinalizationneeded:i8,nodefinedassignment:i8,__padding0:!fir.array<3xi8>}>>
  %17:2 = hlfir.declare %16 uniq_name("_QFdefault_initE.dt.t") fortran_attrs<target> : (!fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>>>>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>,sizeinbytes:i64,uninstantiated:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,kindparameter:!fir.box<!fir.ptr<!fir.array<?xi64>>>,lenparameterkind:!fir.box<!fir.ptr<!fir.array<?xi8>>>,component:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,genre:i8,category:i8,kind:i8,rank:i8,memoryspace:i8,__padding0:!fir.array<3xi8>,offset:i64,characterlen:!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>,derived:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,lenvalue:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,bounds:!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>>>,procptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTprocptrcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,offset:i64,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,special:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTspecialbinding{which:i8,isargdescriptorset:i8,istypebound:i8,specialcaseflag:i8,__padding0:!fir.array<4xi8>,proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,specialbitset:i32,hasparent:i8,noinitializationneeded:i8,nodestructionneeded:i8,nofinalizationneeded:i8,nodefinedassignment:i8,__padding0:!fir.array<3xi8>}>>) -> (!fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>>>>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>,sizeinbytes:i64,uninstantiated:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,kindparameter:!fir.box<!fir.ptr<!fir.array<?xi64>>>,lenparameterkind:!fir.box<!fir.ptr<!fir.array<?xi8>>>,component:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,genre:i8,category:i8,kind:i8,rank:i8,memoryspace:i8,__padding0:!fir.array<3xi8>,offset:i64,characterlen:!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>,derived:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,lenvalue:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,bounds:!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>>>,procptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTprocptrcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,offset:i64,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,special:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTspecialbinding{which:i8,isargdescriptorset:i8,istypebound:i8,specialcaseflag:i8,__padding0:!fir.array<4xi8>,proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,specialbitset:i32,hasparent:i8,noinitializationneeded:i8,nodestructionneeded:i8,nofinalizationneeded:i8,nodefinedassignment:i8,__padding0:!fir.array<3xi8>}>>, !fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>>>>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>,sizeinbytes:i64,uninstantiated:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,kindparameter:!fir.box<!fir.ptr<!fir.array<?xi64>>>,lenparameterkind:!fir.box<!fir.ptr<!fir.array<?xi8>>>,component:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,genre:i8,category:i8,kind:i8,rank:i8,memoryspace:i8,__padding0:!fir.array<3xi8>,offset:i64,characterlen:!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>,derived:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,lenvalue:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,bounds:!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>>>,procptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTprocptrcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,offset:i64,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,special:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTspecialbinding{which:i8,isargdescriptorset:i8,istypebound:i8,specialcaseflag:i8,__padding0:!fir.array<4xi8>,proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,specialbitset:i32,hasparent:i8,noinitializationneeded:i8,nodestructionneeded:i8,nofinalizationneeded:i8,nodefinedassignment:i8,__padding0:!fir.array<3xi8>}>>)
  return
}

// CHECK-LABEL: func.func @_QPalloc_stat(%arg0: !fir.ref<i32> {fir.bindc_name = "s"}) {
// CHECK:   %[[VAL_0:.*]] = fir.alloca i64
// CHECK:   %[[VAL_1:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK:   %[[VAL_2:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{{.*}}>
// CHECK:   %[[VAL_3:.*]] = fir.alloca !fir.boxproc<{{.*}}>
// CHECK:   %[[VAL_4:.*]] = fir.alloca !fir.array<0xi64>
// CHECK:   %[[VAL_5:.*]] = fir.alloca !fir.array<1xi64>
// CHECK:   %[[VAL_6:.*]] = fir.alloca i32
// CHECK:   %[[VAL_7:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:   %[[VAL_8:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>, corank:1> {{.*}}
// CHECK:   %[[VAL_9:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
// CHECK:   %c0 = arith.constant 0 : index
// CHECK:   %[[VAL_10:.*]] = fir.shape %c0 : (index) -> !fir.shape<1>
// CHECK:   %[[VAL_11:.*]] = fir.embox %[[VAL_9]](%[[VAL_10]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>
// CHECK:   fir.store %[[VAL_11]] to %[[VAL_8]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>>
// CHECK:   %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_8]] uniq_name("_QFalloc_statEa") fortran_attrs<allocatable> : {{.*}}
// CHECK:   %[[VAL_13:.*]]:2 = hlfir.declare %arg0 dummy_scope %[[VAL_7]] arg 1 uniq_name("_QFalloc_statEs") : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:   %[[VAL_14:.*]] = fir.absent !fir.box<none>
// CHECK:   %c1 = arith.constant 1 : index
// CHECK:   %c10_i32 = arith.constant 10 : i32
// CHECK:   %c0_i32 = arith.constant 0 : i32
// CHECK:   %[[VAL_15:.*]] = fir.convert %[[VAL_12]]#0 : {{.*}} -> !fir.ref<!fir.box<none>>
// CHECK:   %[[VAL_16:.*]] = fir.convert %c1 : (index) -> i64
// CHECK:   %[[VAL_17:.*]] = fir.convert %c10_i32 : (i32) -> i64
// CHECK:   fir.call @_FortranAAllocatableSetBounds(%[[VAL_15]], %c0_i32, %[[VAL_16]], %[[VAL_17]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
// CHECK:   %c1_i64 = arith.constant 1 : i64
// CHECK:   %c0_0 = arith.constant 0 : index
// CHECK:   %[[VAL_18:.*]] = fir.coordinate_of %[[VAL_5]], %c0_0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
// CHECK:   fir.store %c1_i64 to %[[VAL_18]] : !fir.ref<i64>
// CHECK:   %[[VAL_19:.*]] = fir.embox %[[VAL_5]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
// CHECK:   %[[VAL_20:.*]] = fir.embox %[[VAL_4]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
// CHECK:   %[[VAL_21:.*]] = fir.zero_bits (!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK:   %[[VAL_22:.*]] = fir.emboxproc %[[VAL_21]] : {{.*}}
// CHECK:   fir.store %[[VAL_22]] to %[[VAL_3]] : {{.*}}
// CHECK:   %[[VAL_23:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>>
// CHECK:   %[[VAL_24:.*]] = fir.box_elesize %[[VAL_23]] : (!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>) -> i64
// CHECK:   %[[VAL_25:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]+}}>>
// CHECK:   %c30_i32 = arith.constant 30 : i32
// CHECK:   %[[VAL_26:.*]] = fir.convert %[[VAL_23]] : (!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>) -> !fir.box<none>
// CHECK:   %[[VAL_27:.*]] = fir.convert %[[VAL_25]] : (!fir.ref<!fir.char<1,{{[0-9]+}}>>) -> !fir.ref<i8>
// CHECK:   %[[VAL_28:.*]] = fir.call @_FortranASize(%[[VAL_26]], %[[VAL_27]], %c30_i32) : (!fir.box<none>, !fir.ref<i8>, i32) -> i64
// CHECK:   %[[VAL_29:.*]] = arith.muli %[[VAL_24]], %[[VAL_28]] : i64
// CHECK:   fir.store %[[VAL_29]] to %[[VAL_0]] : !fir.ref<i64>
// CHECK:   %[[VAL_30:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK:   %[[VAL_31:.*]] = fir.convert %[[VAL_19]] : (!fir.box<!fir.array<1xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK:   %[[VAL_32:.*]] = fir.convert %[[VAL_20]] : (!fir.box<!fir.array<0xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK:   %[[VAL_33:.*]] = fir.convert %[[VAL_3]] : {{.*}} -> !fir.ref<none>
// CHECK:   %[[VAL_34:.*]] = fir.convert %[[VAL_2]] : {{.*}} -> !fir.ref<none>
// CHECK:   %[[VAL_35:.*]] = fir.convert %[[VAL_14]] : (!fir.box<none>) -> !fir.box<!fir.char<1,?>>
// CHECK:   fir.call @_QMprifPprif_allocate_coarray(%[[VAL_31]], %[[VAL_32]], %[[VAL_0]], %[[VAL_33]], %[[VAL_34]], %[[VAL_1]], %[[VAL_6]], %[[VAL_35]], %[[VAL_30]]) : {{.*}} -> ()
// CHECK:   %[[VAL_36:.*]] = fir.address_of(@_QFalloc_statEa_coarray_handle) : {{.*}}
// CHECK:   fir.copy %[[VAL_2]] to %[[VAL_36]] : {{.*}}
// CHECK:   %c0_i32_1 = arith.constant 0 : i32
// CHECK:   %[[VAL_37:.*]] = fir.load %[[VAL_6]] : !fir.ref<i32>
// CHECK:   %[[VAL_38:.*]] = arith.cmpi eq, %[[VAL_37]], %c0_i32_1 : i32
// CHECK:   cf.cond_br %[[VAL_38]], ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK:   %[[VAL_39:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK:   %[[VAL_40:.*]] = fir.coordinate_of %[[VAL_1]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
// CHECK:   %[[VAL_41:.*]] = fir.load %[[VAL_40]] : !fir.ref<i64>
// CHECK:   %[[VAL_42:.*]] = fir.convert %[[VAL_12]]#0 : {{.*}} -> !fir.ref<!fir.box<none>>
// CHECK:   %[[VAL_43:.*]] = fir.convert %[[VAL_41]] : (i64) -> !fir.llvm_ptr<i8>
// CHECK:   fir.call @_FortranAAllocatableSetBaseAddr(%[[VAL_42]], %[[VAL_43]]) : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
// CHECK:   cf.br ^bb2
// CHECK: ^bb2:
// CHECK:   %[[VAL_44:.*]] = fir.load %[[VAL_6]] : !fir.ref<i32>
// CHECK:   fir.store %[[VAL_44]] to %[[VAL_13]]#0 : !fir.ref<i32>
// CHECK:   %c6_i32 = arith.constant 6 : i32
// CHECK:   %[[VAL_45:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,48>>
// CHECK:   %[[VAL_46:.*]] = fir.convert %[[VAL_45]] : (!fir.ref<!fir.char<1,48>>) -> !fir.ref<i8>
// CHECK:   %c5_i32 = arith.constant 5 : i32
// CHECK:   %[[VAL_47:.*]] = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %[[VAL_46]], %c5_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
// CHECK:   %[[VAL_48:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>>
// CHECK:   %[[VAL_49:.*]] = fir.box_addr %[[VAL_48]] : (!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>) -> !fir.heap<!fir.array<?xi32>>
// CHECK:   %[[VAL_50:.*]] = fir.convert %[[VAL_49]] : (!fir.heap<!fir.array<?xi32>>) -> i64
// CHECK:   %c0_i64 = arith.constant 0 : i64
// CHECK:   %[[VAL_51:.*]] = arith.cmpi ne, %[[VAL_50]], %c0_i64 : i64
// CHECK:   %[[VAL_52:.*]] = fir.call @_FortranAioOutputLogical(%[[VAL_47]], %[[VAL_51]]) fastmath<contract> : (!fir.ref<i8>, i1) -> i1
// CHECK:   %[[VAL_53:.*]] = fir.call @_FortranAioEndIoStatement(%[[VAL_47]]) fastmath<contract> : (!fir.ref<i8>) -> i32
// CHECK:   %[[VAL_54:.*]] = fir.load %[[VAL_12]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>>
// CHECK:   %[[VAL_55:.*]] = fir.box_addr %[[VAL_54]] : (!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>) -> !fir.heap<!fir.array<?xi32>>
// CHECK:   %[[VAL_56:.*]] = fir.convert %[[VAL_55]] : (!fir.heap<!fir.array<?xi32>>) -> i64
// CHECK:   %c0_i64_2 = arith.constant 0 : i64
// CHECK:   %[[VAL_57:.*]] = arith.cmpi ne, %[[VAL_56]], %c0_i64_2 : i64
// CHECK:   fir.if %[[VAL_57]] {
// CHECK:     %[[VAL_58:.*]] = fir.absent !fir.box<none>
// CHECK:     %[[VAL_59:.*]] = fir.absent !fir.ref<i32>
// CHECK:     %[[VAL_60:.*]] = fir.address_of(@_QFalloc_statEa_coarray_handle) : {{.*}}
// CHECK:     %[[VAL_61:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK:     %[[VAL_62:.*]] = fir.convert %[[VAL_60]] : {{.*}} -> !fir.ref<none>
// CHECK:     %[[VAL_63:.*]] = fir.convert %[[VAL_58]] : (!fir.box<none>) -> !fir.box<!fir.char<1,?>>
// CHECK:     fir.call @_QMprifPprif_deallocate_coarray(%[[VAL_62]], %[[VAL_59]], %[[VAL_63]], %[[VAL_61]]) : {{.*}} -> ()
// CHECK:     %[[VAL_64:.*]] = fir.absent !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>, corank:1>>
// CHECK:     %[[VAL_65:.*]] = fir.convert %[[VAL_12]]#0 : {{.*}} -> !fir.ref<!fir.box<none>>
// CHECK:     %[[VAL_66:.*]] = fir.convert %[[VAL_64]] : {{.*}} -> !fir.llvm_ptr<i8>
// CHECK:     fir.call @_FortranAPointerAssociateScalar(%[[VAL_65]], %[[VAL_66]]) : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
// CHECK:   }
// CHECK:   return
// CHECK: }

// CHECK-LABEL: func.func @_QPalloc_event_stat(%arg0: !fir.ref<i32> {fir.bindc_name = "s"}) {
// CHECK:   %[[VAL_0:.*]] = fir.alloca i64
// CHECK:   %[[VAL_1:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK:   %[[VAL_2:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_prif_coarray_handle_type{{.*}}>
// CHECK:   %[[VAL_3:.*]] = fir.alloca !fir.boxproc<{{.*}}>
// CHECK:   %[[VAL_4:.*]] = fir.alloca !fir.array<0xi64>
// CHECK:   %[[VAL_5:.*]] = fir.alloca !fir.array<1xi64>
// CHECK:   %[[VAL_6:.*]] = fir.alloca i32
// CHECK:   %[[VAL_7:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:   %[[VAL_8:.*]] = fir.alloca !fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1> {{.*}}
// CHECK:   %[[VAL_9:.*]] = fir.zero_bits !fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>
// CHECK:   %[[VAL_10:.*]] = fir.embox %[[VAL_9]] : {{.*}} -> !fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>
// CHECK:   fir.store %[[VAL_10]] to %[[VAL_8]] : {{.*}}
// CHECK:   %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_8]] uniq_name("_QFalloc_event_statEev") fortran_attrs<allocatable> : {{.*}}
// CHECK:   %[[VAL_12:.*]]:2 = hlfir.declare %arg0 dummy_scope %[[VAL_7]] arg 1 uniq_name("_QFalloc_event_statEs") : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:   %[[VAL_13:.*]] = fir.absent !fir.box<none>
// CHECK:   %c1_i64 = arith.constant 1 : i64
// CHECK:   %c0 = arith.constant 0 : index
// CHECK:   %[[VAL_14:.*]] = fir.coordinate_of %[[VAL_5]], %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
// CHECK:   fir.store %c1_i64 to %[[VAL_14]] : !fir.ref<i64>
// CHECK:   %[[VAL_15:.*]] = fir.embox %[[VAL_5]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
// CHECK:   %[[VAL_16:.*]] = fir.embox %[[VAL_4]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
// CHECK:   %[[VAL_17:.*]] = fir.zero_bits (!fir.ref<none>, !fir.ref<i32>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK:   %[[VAL_18:.*]] = fir.emboxproc %[[VAL_17]] : {{.*}}
// CHECK:   fir.store %[[VAL_18]] to %[[VAL_3]] : {{.*}}
// CHECK:   %[[VAL_19:.*]] = fir.load %[[VAL_11]]#0 : {{.*}}
// CHECK:   %[[VAL_20:.*]] = fir.box_elesize %[[VAL_19]] : {{.*}} -> i64
// CHECK:   fir.store %[[VAL_20]] to %[[VAL_0]] : !fir.ref<i64>
// CHECK:   %[[VAL_21:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK:   %[[VAL_22:.*]] = fir.convert %[[VAL_15]] : (!fir.box<!fir.array<1xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK:   %[[VAL_23:.*]] = fir.convert %[[VAL_16]] : (!fir.box<!fir.array<0xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK:   %[[VAL_24:.*]] = fir.convert %[[VAL_3]] : {{.*}} -> !fir.ref<none>
// CHECK:   %[[VAL_25:.*]] = fir.convert %[[VAL_2]] : {{.*}} -> !fir.ref<none>
// CHECK:   %[[VAL_26:.*]] = fir.convert %[[VAL_13]] : (!fir.box<none>) -> !fir.box<!fir.char<1,?>>
// CHECK:   fir.call @_QMprifPprif_allocate_coarray(%[[VAL_22]], %[[VAL_23]], %[[VAL_0]], %[[VAL_24]], %[[VAL_25]], %[[VAL_1]], %[[VAL_6]], %[[VAL_26]], %[[VAL_21]]) : {{.*}} -> ()
// CHECK:   %[[VAL_27:.*]] = fir.address_of(@_QFalloc_event_statEev_coarray_handle) : {{.*}}
// CHECK:   fir.copy %[[VAL_2]] to %[[VAL_27]] : {{.*}}
// CHECK:   %c0_i32 = arith.constant 0 : i32
// CHECK:   %[[VAL_28:.*]] = fir.load %[[VAL_6]] : !fir.ref<i32>
// CHECK:   %[[VAL_29:.*]] = arith.cmpi eq, %[[VAL_28]], %c0_i32 : i32
// CHECK:   cf.cond_br %[[VAL_29]], ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK:   %[[VAL_30:.*]] = fir.field_index __address, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>
// CHECK:   %[[VAL_31:.*]] = fir.coordinate_of %[[VAL_1]], __address : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> !fir.ref<i64>
// CHECK:   %[[VAL_32:.*]] = fir.load %[[VAL_31]] : !fir.ref<i64>
// CHECK:   %[[VAL_33:.*]] = fir.convert %[[VAL_11]]#0 : {{.*}} -> !fir.ref<!fir.box<none>>
// CHECK:   %[[VAL_34:.*]] = fir.convert %[[VAL_32]] : (i64) -> !fir.llvm_ptr<i8>
// CHECK:   fir.call @_FortranAAllocatableSetBaseAddr(%[[VAL_33]], %[[VAL_34]]) : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
// CHECK:   %[[VAL_35:.*]] = fir.address_of(@_QQclX{{.*}}) : !fir.ref<!fir.char<1,{{[0-9]+}}>>
// CHECK:   %c76_i32 = arith.constant 76 : i32
// CHECK:   %[[VAL_36:.*]] = fir.convert %[[VAL_11]]#0 : {{.*}} -> !fir.box<none>
// CHECK:   %[[VAL_37:.*]] = fir.convert %[[VAL_35]] : (!fir.ref<!fir.char<1,{{[0-9]+}}>>) -> !fir.ref<i8>
// CHECK:   fir.call @_FortranAInitialize(%[[VAL_36]], %[[VAL_37]], %c76_i32) : (!fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:   %[[VAL_38:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK:   %[[VAL_39:.*]] = fir.convert %[[VAL_13]] : (!fir.box<none>) -> !fir.box<!fir.char<1,?>>
// CHECK:   fir.call @_QMprifPprif_sync_all(%[[VAL_6]], %[[VAL_39]], %[[VAL_38]]) : (!fir.ref<i32>, !fir.box<!fir.char<1,?>>, !fir.box<!fir.char<1,?>>) -> ()
// CHECK:   cf.br ^bb2
// CHECK: ^bb2:
// CHECK:   %[[VAL_40:.*]] = fir.load %[[VAL_6]] : !fir.ref<i32>
// CHECK:   fir.store %[[VAL_40]] to %[[VAL_12]]#0 : !fir.ref<i32>
// CHECK:   %[[VAL_41:.*]] = fir.load %[[VAL_11]]#0 : {{.*}}
// CHECK:   %[[VAL_42:.*]] = fir.box_addr %[[VAL_41]] : {{.*}}
// CHECK:   %[[VAL_43:.*]] = fir.convert %[[VAL_42]] : {{.*}} -> i64
// CHECK:   %c0_i64 = arith.constant 0 : i64
// CHECK:   %[[VAL_44:.*]] = arith.cmpi ne, %[[VAL_43]], %c0_i64 : i64
// CHECK:   fir.if %[[VAL_44]] {
// CHECK:     %[[VAL_45:.*]] = fir.absent !fir.box<none>
// CHECK:     %[[VAL_46:.*]] = fir.absent !fir.ref<i32>
// CHECK:     %[[VAL_47:.*]] = fir.address_of(@_QFalloc_event_statEev_coarray_handle) : {{.*}}
// CHECK:     %[[VAL_48:.*]] = fir.absent !fir.box<!fir.char<1,?>>
// CHECK:     %[[VAL_49:.*]] = fir.convert %[[VAL_47]] : {{.*}} -> !fir.ref<none>
// CHECK:     %[[VAL_50:.*]] = fir.convert %[[VAL_45]] : (!fir.box<none>) -> !fir.box<!fir.char<1,?>>
// CHECK:     fir.call @_QMprifPprif_deallocate_coarray(%[[VAL_49]], %[[VAL_46]], %[[VAL_50]], %[[VAL_48]]) : {{.*}} -> ()
// CHECK:     %[[VAL_51:.*]] = fir.absent !fir.ref<!fir.box<!fir.heap<!fir.type<_QM__fortran_builtinsT__builtin_event_type{{.*}}>>, corank:1>>
// CHECK:     %[[VAL_52:.*]] = fir.convert %[[VAL_11]]#0 : {{.*}} -> !fir.ref<!fir.box<none>>
// CHECK:     %[[VAL_53:.*]] = fir.convert %[[VAL_51]] : {{.*}} -> !fir.llvm_ptr<i8>
// CHECK:     fir.call @_FortranAPointerAssociateScalar(%[[VAL_52]], %[[VAL_53]]) : (!fir.ref<!fir.box<none>>, !fir.llvm_ptr<i8>) -> ()
// CHECK:   }
// CHECK:   return
// CHECK: }


// CHECK-LABEL: func.func @_QPdefault_init() {
// CHECK-NEXT:   %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
// CHECK-NEXT:   %[[VAL_1:.*]] = fir.address_of(@_QFdefault_initE.n.pad) : !fir.ref<!fir.char<1,3>>
// CHECK-NEXT:   %c3 = arith.constant 3 : index
// CHECK-NEXT:   %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]] typeparams %c3 uniq_name("_QFdefault_initE.n.pad") fortran_attrs<target> : (!fir.ref<!fir.char<1,3>>, index) -> (!fir.ref<!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>)
// CHECK-NEXT:   %[[VAL_3:.*]] = fir.address_of(@_QFdefault_initE.di.t.pad) : !fir.ref<i64>
// CHECK-NEXT:   %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]] uniq_name("_QFdefault_initE.di.t.pad") fortran_attrs<target> : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
// CHECK-NEXT:   %[[VAL_5:.*]] = fir.address_of(@_QFdefault_initE.n.x) : !fir.ref<!fir.char<1>>
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_5]] typeparams %c1 uniq_name("_QFdefault_initE.n.x") fortran_attrs<target> : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
// CHECK-NEXT:   %[[VAL_7:.*]] = fir.address_of(@_QFdefault_initE.di.t.x) : !fir.ref<i32>
// CHECK-NEXT:   %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] uniq_name("_QFdefault_initE.di.t.x") fortran_attrs<target> : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK-NEXT:   %[[VAL_9:.*]] = fir.address_of(@_QFdefault_initE.n.t) : !fir.ref<!fir.char<1>>
// CHECK-NEXT:   %c1_0 = arith.constant 1 : index
// CHECK-NEXT:   %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_9]] typeparams %c1_0 uniq_name("_QFdefault_initE.n.t") fortran_attrs<target> : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
// CHECK-NEXT:   %[[VAL_11:.*]] = fir.address_of(@_QFdefault_initEc) : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFdefault_initTt{pad:i64,x:i32}>>, corank:1>>
// CHECK-NEXT:   %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] uniq_name("_QFdefault_initEc") : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QFdefault_initTt{pad:i64,x:i32}>>, corank:1>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<_QFdefault_initTt{pad:i64,x:i32}>>, corank:1>>, !fir.ref<!fir.box<!fir.heap<!fir.type<_QFdefault_initTt{pad:i64,x:i32}>>, corank:1>>)
// CHECK-NEXT:   %[[VAL_13:.*]] = fir.address_of(@_QFdefault_initE.c.t) : !fir.ref<!fir.array<2x!fir.type<_QM__fortran_type_infoTcomponent{{.*}}>>>
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c2 = arith.constant 2 : index
// CHECK-NEXT:   %[[VAL_14:.*]] = fir.shape_shift %c0, %c2 : (index, index) -> !fir.shapeshift<1>
// CHECK-NEXT:   %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_13]](%[[VAL_14]]) uniq_name("_QFdefault_initE.c.t") fortran_attrs<target> : {{.*}}
// CHECK-NEXT:   %[[VAL_16:.*]] = fir.address_of(@_QFdefault_initE.dt.t) : !fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype{{.*}}>>
// CHECK-NEXT:   %[[VAL_17:.*]]:2 = hlfir.declare %[[VAL_16]] uniq_name("_QFdefault_initE.dt.t") fortran_attrs<target> : {{.*}}
// CHECK-NEXT:   return
// CHECK-NEXT: }
