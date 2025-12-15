! RUN: bbc -emit-fir -I nowhere %s -o - | FileCheck %s

! Test allocatable return.
! Allocatable arrays must have default runtime lbounds after the return.

function test_alloc_return_scalar
  real, allocatable :: test_alloc_return_scalar
  allocate(test_alloc_return_scalar)
end function test_alloc_return_scalar
! CHECK-LABEL:   func.func @_QPtest_alloc_return_scalar() -> !fir.box<!fir.heap<f32>> {
! CHECK:           %[[VAL_DUMMY_SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "test_alloc_return_scalar", uniq_name = "_QFtest_alloc_return_scalarEtest_alloc_return_scalar"}
! CHECK:           %[[VAL_ZERO_BITS:.*]] = fir.zero_bits !fir.heap<f32>
! CHECK:           %[[VAL_EMBOX:.*]] = fir.embox %[[VAL_ZERO_BITS]] : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
! CHECK:           fir.store %[[VAL_EMBOX]] to %[[VAL_ALLOCA]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:           %[[VAL_DECLARE:.*]] = fir.declare %[[VAL_ALLOCA]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_alloc_return_scalarEtest_alloc_return_scalar"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:           %[[VAL_ALLOCMEM:.*]] = fir.allocmem f32 {fir.must_be_heap = true, uniq_name = "_QFtest_alloc_return_scalarEtest_alloc_return_scalar.alloc"}
! CHECK:           %[[VAL_EMBOX_ALLOC:.*]] = fir.embox %[[VAL_ALLOCMEM]] : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
! CHECK:           fir.store %[[VAL_EMBOX_ALLOC]] to %[[VAL_DECLARE]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:           %[[VAL_LOAD:.*]] = fir.load %[[VAL_DECLARE]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:           return %[[VAL_LOAD]] : !fir.box<!fir.heap<f32>>
! CHECK:         }

function test_alloc_return_array
  real, allocatable :: test_alloc_return_array(:)
  allocate(test_alloc_return_array(7:8))
end function test_alloc_return_array
! CHECK-LABEL:   func.func @_QPtest_alloc_return_array() -> !fir.box<!fir.heap<!fir.array<?xf32>>> {
! CHECK:           %[[C2:.*]] = arith.constant 2 : index
! CHECK:           %[[C7:.*]] = arith.constant 7 : index
! CHECK:           %[[C1:.*]] = arith.constant 1 : index
! CHECK:           %[[C0:.*]] = arith.constant 0 : index
! CHECK:           %[[DUMMY_SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "test_alloc_return_array", uniq_name = "_QFtest_alloc_return_arrayEtest_alloc_return_array"}
! CHECK:           %[[ZERO_BITS:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
! CHECK:           %[[SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
! CHECK:           %[[EMBOX:.*]] = fir.embox %[[ZERO_BITS]](%[[SHAPE]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:           fir.store %[[EMBOX]] to %[[ALLOCA]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           %[[DECLARE:.*]] = fir.declare %[[ALLOCA]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_alloc_return_arrayEtest_alloc_return_array"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           %[[ALLOCMEM:.*]] = fir.allocmem !fir.array<?xf32>, %[[C2]] {fir.must_be_heap = true, uniq_name = "_QFtest_alloc_return_arrayEtest_alloc_return_array.alloc"}
! CHECK:           %[[SHAPE_SHIFT:.*]] = fir.shape_shift %[[C7]], %[[C2]] : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[EMBOX2:.*]] = fir.embox %[[ALLOCMEM]](%[[SHAPE_SHIFT]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:           fir.store %[[EMBOX2]] to %[[DECLARE]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           %[[LOAD:.*]] = fir.load %[[DECLARE]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           %[[SHIFT:.*]] = fir.shift %[[C1]] : (index) -> !fir.shift<1>
! CHECK:           %[[REBOX:.*]] = fir.rebox %[[LOAD]](%[[SHIFT]]) : (!fir.box<!fir.heap<!fir.array<?xf32>>>, !fir.shift<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:           return %[[REBOX]] : !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:         }

function test_alloc_return_char_scalar
  character(3), allocatable :: test_alloc_return_char_scalar
  allocate(test_alloc_return_char_scalar)
end function test_alloc_return_char_scalar
! CHECK-LABEL:   func.func @_QPtest_alloc_return_char_scalar() -> !fir.box<!fir.heap<!fir.char<1,3>>> {
! CHECK:           %[[C3:.*]] = arith.constant 3 : index
! CHECK:           %[[DUMMY_SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,3>>> {bindc_name = "test_alloc_return_char_scalar", uniq_name = "_QFtest_alloc_return_char_scalarEtest_alloc_return_char_scalar"}
! CHECK:           %[[ZERO_BITS:.*]] = fir.zero_bits !fir.heap<!fir.char<1,3>>
! CHECK:           %[[EMBOX:.*]] = fir.embox %[[ZERO_BITS]] : (!fir.heap<!fir.char<1,3>>) -> !fir.box<!fir.heap<!fir.char<1,3>>>
! CHECK:           fir.store %[[EMBOX]] to %[[ALLOCA]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,3>>>>
! CHECK:           %[[DECLARE:.*]] = fir.declare %[[ALLOCA]] typeparams %[[C3]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_alloc_return_char_scalarEtest_alloc_return_char_scalar"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,3>>>>, index) -> !fir.ref<!fir.box<!fir.heap<!fir.char<1,3>>>>
! CHECK:           %[[ALLOCMEM:.*]] = fir.allocmem !fir.char<1,3> {fir.must_be_heap = true, uniq_name = "_QFtest_alloc_return_char_scalarEtest_alloc_return_char_scalar.alloc"}
! CHECK:           %[[EMBOX2:.*]] = fir.embox %[[ALLOCMEM]] : (!fir.heap<!fir.char<1,3>>) -> !fir.box<!fir.heap<!fir.char<1,3>>>
! CHECK:           fir.store %[[EMBOX2]] to %[[DECLARE]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,3>>>>
! CHECK:           %[[LOAD:.*]] = fir.load %[[DECLARE]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,3>>>>
! CHECK:           return %[[LOAD]] : !fir.box<!fir.heap<!fir.char<1,3>>>
! CHECK:         }

function test_alloc_return_char_array
  character(3), allocatable :: test_alloc_return_char_array(:)
  allocate(test_alloc_return_char_array(7:8))
end function test_alloc_return_char_array
! CHECK-LABEL:   func.func @_QPtest_alloc_return_char_array() -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>> {
! CHECK:           %[[C2:.*]] = arith.constant 2 : index
! CHECK:           %[[C7:.*]] = arith.constant 7 : index
! CHECK:           %[[C1:.*]] = arith.constant 1 : index
! CHECK:           %[[C0:.*]] = arith.constant 0 : index
! CHECK:           %[[C3:.*]] = arith.constant 3 : index
! CHECK:           %[[DUMMY_SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>> {bindc_name = "test_alloc_return_char_array", uniq_name = "_QFtest_alloc_return_char_arrayEtest_alloc_return_char_array"}
! CHECK:           %[[ZERO_BITS:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.char<1,3>>>
! CHECK:           %[[SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
! CHECK:           %[[EMBOX:.*]] = fir.embox %[[ZERO_BITS]](%[[SHAPE]]) : (!fir.heap<!fir.array<?x!fir.char<1,3>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>>
! CHECK:           fir.store %[[EMBOX]] to %[[ALLOCA]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>>>
! CHECK:           %[[DECLARE:.*]] = fir.declare %[[ALLOCA]] typeparams %[[C3]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_alloc_return_char_arrayEtest_alloc_return_char_array"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>>>, index) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>>>
! CHECK:           %[[ALLOCMEM:.*]] = fir.allocmem !fir.array<?x!fir.char<1,3>>, %[[C2]] {fir.must_be_heap = true, uniq_name = "_QFtest_alloc_return_char_arrayEtest_alloc_return_char_array.alloc"}
! CHECK:           %[[SHAPE_SHIFT:.*]] = fir.shape_shift %[[C7]], %[[C2]] : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[EMBOX2:.*]] = fir.embox %[[ALLOCMEM]](%[[SHAPE_SHIFT]]) : (!fir.heap<!fir.array<?x!fir.char<1,3>>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>>
! CHECK:           fir.store %[[EMBOX2]] to %[[DECLARE]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>>>
! CHECK:           %[[LOAD:.*]] = fir.load %[[DECLARE]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>>>
! CHECK:           %[[SHIFT:.*]] = fir.shift %[[C1]] : (index) -> !fir.shift<1>
! CHECK:           %[[REBOX:.*]] = fir.rebox %[[LOAD]](%[[SHIFT]]) : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>>, !fir.shift<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>>
! CHECK:           return %[[REBOX]] : !fir.box<!fir.heap<!fir.array<?x!fir.char<1,3>>>>
! CHECK:         }

function test_alloc_return_poly_scalar
  type t
  end type t
  class(*), allocatable :: test_alloc_return_poly_scalar
  allocate(t :: test_alloc_return_poly_scalar)
end function test_alloc_return_poly_scalar
! CHECK-LABEL:   func.func @_QPtest_alloc_return_poly_scalar() -> !fir.class<!fir.heap<none>> {
! CHECK:           %[[C_NEG1_I64:.*]] = arith.constant -1 : i64
! CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
! CHECK:           %[[C100_I32:.*]] = arith.constant 100 : i32
! CHECK:           %[[FALSE:.*]] = arith.constant false
! CHECK:           %[[C1:.*]] = arith.constant 1 : index
! CHECK:           %[[DUMMY_SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[ADDRESS_OF_N_T:.*]] = fir.address_of(@_QFtest_alloc_return_poly_scalarE.n.t) : !fir.ref<!fir.char<1>>
! CHECK:           %[[DECLARE_N_T:.*]] = fir.declare %[[ADDRESS_OF_N_T]] typeparams %[[C1]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtest_alloc_return_poly_scalarE.n.t"} : (!fir.ref<!fir.char<1>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:           %[[ALLOCA:.*]] = fir.alloca !fir.class<!fir.heap<none>> {bindc_name = "test_alloc_return_poly_scalar", uniq_name = "_QFtest_alloc_return_poly_scalarEtest_alloc_return_poly_scalar"}
! CHECK:           %[[ZERO_BITS:.*]] = fir.zero_bits !fir.heap<none>
! CHECK:           %[[EMBOX:.*]] = fir.embox %[[ZERO_BITS]] : (!fir.heap<none>) -> !fir.class<!fir.heap<none>>
! CHECK:           fir.store %[[EMBOX]] to %[[ALLOCA]] : !fir.ref<!fir.class<!fir.heap<none>>>
! CHECK:           %[[DECLARE:.*]] = fir.declare %[[ALLOCA]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_alloc_return_poly_scalarEtest_alloc_return_poly_scalar"} : (!fir.ref<!fir.class<!fir.heap<none>>>) -> !fir.ref<!fir.class<!fir.heap<none>>>
! CHECK:           %[[ABSENT:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[ADDRESS_OF_DT_T:.*]] = fir.address_of(@_QFtest_alloc_return_poly_scalarE.dt.t) : !fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>>>>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>,sizeinbytes:i64,uninstantiated:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,kindparameter:!fir.box<!fir.ptr<!fir.array<?xi64>>>,lenparameterkind:!fir.box<!fir.ptr<!fir.array<?xi8>>>,component:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,genre:i8,category:i8,kind:i8,rank:i8,__padding0:!fir.array<4xi8>,offset:i64,characterlen:!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>,derived:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,lenvalue:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,bounds:!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>>>,procptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTprocptrcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,offset:i64,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,special:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTspecialbinding{which:i8,isargdescriptorset:i8,istypebound:i8,specialcaseflag:i8,__padding0:!fir.array<4xi8>,proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,specialbitset:i32,hasparent:i8,noinitializationneeded:i8,nodestructionneeded:i8,nofinalizationneeded:i8,nodefinedassignment:i8,__padding0:!fir.array<3xi8>}>>
! CHECK:           %[[TYPE_DESC:.*]] = fir.type_desc !fir.type<_QFtest_alloc_return_poly_scalarTt>
! CHECK:           %[[CONVERT_DECLARE:.*]] = fir.convert %[[DECLARE]] : (!fir.ref<!fir.class<!fir.heap<none>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[CONVERT_TYPE_DESC:.*]] = fir.convert %[[TYPE_DESC]] : (!fir.tdesc<!fir.type<_QFtest_alloc_return_poly_scalarTt>>) -> !fir.ref<none>
! CHECK:           fir.call @_FortranAAllocatableInitDerivedForAllocate(%[[CONVERT_DECLARE]], %[[CONVERT_TYPE_DESC]], %[[C0_I32]], %[[C0_I32]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> ()
! CHECK:           %[[CONVERT_C_NEG1_I64:.*]] = fir.convert %[[C_NEG1_I64]] : (i64) -> !fir.ref<i64>
! CHECK:           %[[ADDRESS_OF_STRING:.*]] = fir.address_of(@_QQclXcda6f2a3409966289c088d50bcd8bfed) : !fir.ref<!fir.char<1,88>>
! CHECK:           %[[CONVERT_STRING:.*]] = fir.convert %[[ADDRESS_OF_STRING]] : (!fir.ref<!fir.char<1,88>>) -> !fir.ref<i8>
! CHECK:           %[[CALL_ALLOCATE:.*]] = fir.call @_FortranAAllocatableAllocate(%[[CONVERT_DECLARE]], %[[CONVERT_C_NEG1_I64]], %[[FALSE]], %[[ABSENT]], %[[CONVERT_STRING]], %[[C100_I32]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[LOAD:.*]] = fir.load %[[DECLARE]] : !fir.ref<!fir.class<!fir.heap<none>>>
! CHECK:           return %[[LOAD]] : !fir.class<!fir.heap<none>>>
! CHECK:         }

function test_alloc_return_poly_array
  type t
  end type t
  class(*), allocatable :: test_alloc_return_poly_array(:)
  allocate(t :: test_alloc_return_poly_array(7:8))
end function test_alloc_return_poly_array
! CHECK-LABEL:   func.func @_QPtest_alloc_return_poly_array() -> !fir.class<!fir.heap<!fir.array<?xnone>>> {
! CHECK:           %[[C_NEG1_I64:.*]] = arith.constant -1 : i64
! CHECK:           %[[C8_I32:.*]] = arith.constant 8 : i32
! CHECK:           %[[C7_I32:.*]] = arith.constant 7 : i32
! CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
! CHECK:           %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK:           %[[C134_I32:.*]] = arith.constant 134 : i32
! CHECK:           %[[FALSE:.*]] = arith.constant false
! CHECK:           %[[C0:.*]] = arith.constant 0 : index
! CHECK:           %[[C1:.*]] = arith.constant 1 : index
! CHECK:           %[[DUMMY_SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[ADDRESS_OF_N_T:.*]] = fir.address_of(@_QFtest_alloc_return_poly_arrayE.n.t) : !fir.ref<!fir.char<1>>
! CHECK:           %[[DECLARE_N_T:.*]] = fir.declare %[[ADDRESS_OF_N_T]] typeparams %[[C1]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtest_alloc_return_poly_arrayE.n.t"} : (!fir.ref<!fir.char<1>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:           %[[ALLOCA:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?xnone>>> {bindc_name = "test_alloc_return_poly_array", uniq_name = "_QFtest_alloc_return_poly_arrayEtest_alloc_return_poly_array"}
! CHECK:           %[[ZERO_BITS:.*]] = fir.zero_bits !fir.heap<!fir.array<?xnone>>
! CHECK:           %[[SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
! CHECK:           %[[EMBOX:.*]] = fir.embox %[[ZERO_BITS]](%[[SHAPE]]) : (!fir.heap<!fir.array<?xnone>>, !fir.shape<1>) -> !fir.class<!fir.heap<!fir.array<?xnone>>>
! CHECK:           fir.store %[[EMBOX]] to %[[ALLOCA]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>
! CHECK:           %[[DECLARE:.*]] = fir.declare %[[ALLOCA]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_alloc_return_poly_arrayEtest_alloc_return_poly_array"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>) -> !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>
! CHECK:           %[[ADDRESS_OF_DT_T:.*]] = fir.address_of(@_QFtest_alloc_return_poly_arrayE.dt.t) : !fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype{{.*}}>>
! CHECK:           %[[ABSENT:.*]] = fir.absent !fir.box<none>
! CHECK:           %[[TYPE_DESC:.*]] = fir.type_desc !fir.type<_QFtest_alloc_return_poly_arrayTt>
! CHECK:           %[[CONVERT_DECLARE:.*]] = fir.convert %[[DECLARE]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>) -> !fir.ref<!fir.box<none>>
! CHECK:           %[[CONVERT_TYPE_DESC:.*]] = fir.convert %[[TYPE_DESC]] : (!fir.tdesc<!fir.type<_QFtest_alloc_return_poly_arrayTt>>) -> !fir.ref<none>
! CHECK:           fir.call @_FortranAAllocatableInitDerivedForAllocate(%[[CONVERT_DECLARE]], %[[CONVERT_TYPE_DESC]], %[[C1_I32]], %[[C0_I32]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<none>, i32, i32) -> ()
! CHECK:           %[[CONVERT_C7_I32:.*]] = fir.convert %[[C7_I32]] : (i32) -> i64
! CHECK:           %[[CONVERT_C8_I32:.*]] = fir.convert %[[C8_I32]] : (i32) -> i64
! CHECK:           fir.call @_FortranAAllocatableSetBounds(%[[CONVERT_DECLARE]], %[[C0_I32]], %[[CONVERT_C7_I32]], %[[CONVERT_C8_I32]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
! CHECK:           %[[CONVERT_C_NEG1_I64:.*]] = fir.convert %[[C_NEG1_I64]] : (i64) -> !fir.ref<i64>
! CHECK:           %[[ADDRESS_OF_STRING:.*]] = fir.address_of(@_QQclXcda6f2a3409966289c088d50bcd8bfed) : !fir.ref<!fir.char<1,88>>
! CHECK:           %[[CONVERT_STRING:.*]] = fir.convert %[[ADDRESS_OF_STRING]] : (!fir.ref<!fir.char<1,88>>) -> !fir.ref<i8>
! CHECK:           %[[CALL_ALLOCATE:.*]] = fir.call @_FortranAAllocatableAllocate(%[[CONVERT_DECLARE]], %[[CONVERT_C_NEG1_I64]], %[[FALSE]], %[[ABSENT]], %[[CONVERT_STRING]], %[[C134_I32]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.ref<i64>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
! CHECK:           %[[LOAD:.*]] = fir.load %[[DECLARE]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?xnone>>>>
! CHECK:           %[[SHIFT:.*]] = fir.shift %[[C1]] : (index) -> !fir.shift<1>
! CHECK:           %[[REBOX:.*]] = fir.rebox %[[LOAD]](%[[SHIFT]]) : (!fir.class<!fir.heap<!fir.array<?xnone>>>, !fir.shift<1>) -> !fir.class<!fir.heap<!fir.array<?xnone>>>
! CHECK:           return %[[REBOX]] : !fir.class<!fir.heap<!fir.array<?xnone>>>
! CHECK:         }