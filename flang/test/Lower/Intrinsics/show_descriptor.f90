! RUN: bbc -emit-fir %s -o - | FileCheck %s

module test_show_descriptor
use flang_debug
contains
subroutine test_int
! CHECK-LABEL:   func.func @_QMtest_show_descriptorPtest_int() {
  implicit none
  integer :: n
  integer,allocatable :: a(:)
  n = 5
  allocate(a(n))
! CHECK:           %[[C0_I64:.*]] = arith.constant 0 : i64
! CHECK:           %[[C5:.*]] = arith.constant 5 : i32
! CHECK:           %[[C0:.*]] = arith.constant 0 : index
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "a", uniq_name = "_QMtest_show_descriptorFtest_intEa"}
! CHECK:           %[[ZERO_BITS_0:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[ZERO_BITS_0]](%[[SHAPE_0]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[EMBOX_0]] to %[[ALLOCA_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ALLOCA_0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMtest_show_descriptorFtest_intEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[ALLOCA_1:.*]] = fir.alloca i32 {bindc_name = "n", uniq_name = "_QMtest_show_descriptorFtest_intEn"}
! CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ALLOCA_1]] {uniq_name = "_QMtest_show_descriptorFtest_intEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
! CHECK:           fir.store %[[C5]] to %[[DECLARE_1]] : !fir.ref<i32>
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[DECLARE_1]] : !fir.ref<i32>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_0]] : (i32) -> index
! CHECK:           %[[CMPI_0:.*]] = arith.cmpi sgt, %[[CONVERT_0]], %[[C0]] : index
! CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[CONVERT_0]], %[[C0]] : index
! CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.array<?xi32>, %[[SELECT_0]] {fir.must_be_heap = true, uniq_name = "_QMtest_show_descriptorFtest_intEa.alloc"}

  call show_descriptor(a)
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[SELECT_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[EMBOX_1:.*]] = fir.embox %[[ALLOCMEM_0]](%[[SHAPE_1]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[EMBOX_1]] to %[[DECLARE_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           fir.call @_FortranAShowDescriptor(%[[DECLARE_0]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> ()

  call show_descriptor(a(1:3))
! CHECK:           %[[ZERO_BITS_1:.*]] = fir.zero_bits !fir.box<none>
! CHECK:           fir.call @_FortranAShowDescriptor(%[[ZERO_BITS_1]]) fastmath<contract> : (!fir.box<none>) -> ()
  deallocate(a)
end subroutine test_int

subroutine test_char
! CHECK-LABEL:   func.func @_QMtest_show_descriptorPtest_char() {
  implicit none
  character(len=9) :: c = 'Hey buddy'
  call show_descriptor(c)
! CHECK:           %[[C9:.*]] = arith.constant 9 : index
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QMtest_show_descriptorFtest_charEc) : !fir.ref<!fir.char<1,9>>
! CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ADDRESS_OF_0]] typeparams %[[C9]] {uniq_name = "_QMtest_show_descriptorFtest_charEc"} : (!fir.ref<!fir.char<1,9>>, index) -> !fir.ref<!fir.char<1,9>>
! CHECK:           %[[ZERO_BITS_2:.*]] = fir.zero_bits !fir.box<none>
! CHECK:           fir.call @_FortranAShowDescriptor(%[[ZERO_BITS_2]]) fastmath<contract> : (!fir.box<none>) -> ()

  call show_descriptor(c(1:3))
! CHECK:           fir.call @_FortranAShowDescriptor(%[[ZERO_BITS_2]]) fastmath<contract> : (!fir.box<none>) -> ()
! CHECK:           return
end subroutine test_char

subroutine test_logical
! CHECK-LABEL:   func.func @_QMtest_show_descriptorPtest_logical() {
  implicit none
  logical(kind=1) :: l1 = .false.
  logical(kind=2) :: l2 = .true.
  logical(kind=2), dimension(2), target :: la2 = (/ .true., .false. /)
  logical(kind=2), dimension(:), pointer :: pla2
! CHECK:           %[[C0:.*]] = arith.constant 0 : index
! CHECK:           %[[C2:.*]] = arith.constant 2 : index
! CHECK:           %[[DUMMY_SCOPE_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QMtest_show_descriptorFtest_logicalEl1) : !fir.ref<!fir.logical<1>>
! CHECK:           %[[DECLARE_0:.*]] = fir.declare %[[ADDRESS_OF_0]] {uniq_name = "_QMtest_show_descriptorFtest_logicalEl1"} : (!fir.ref<!fir.logical<1>>) -> !fir.ref<!fir.logical<1>>
! CHECK:           %[[ADDRESS_OF_1:.*]] = fir.address_of(@_QMtest_show_descriptorFtest_logicalEl2) : !fir.ref<!fir.logical<2>>
! CHECK:           %[[DECLARE_1:.*]] = fir.declare %[[ADDRESS_OF_1]] {uniq_name = "_QMtest_show_descriptorFtest_logicalEl2"} : (!fir.ref<!fir.logical<2>>) -> !fir.ref<!fir.logical<2>>
! CHECK:           %[[ADDRESS_OF_2:.*]] = fir.address_of(@_QMtest_show_descriptorFtest_logicalEla2) : !fir.ref<!fir.array<2x!fir.logical<2>>>
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[C2]] : (index) -> !fir.shape<1>
! CHECK:           %[[DECLARE_2:.*]] = fir.declare %[[ADDRESS_OF_2]](%[[SHAPE_0]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMtest_show_descriptorFtest_logicalEla2"} : (!fir.ref<!fir.array<2x!fir.logical<2>>>, !fir.shape<1>) -> !fir.ref<!fir.array<2x!fir.logical<2>>>
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x!fir.logical<2>>>> {bindc_name = "pla2", uniq_name = "_QMtest_show_descriptorFtest_logicalEpla2"}
! CHECK:           %[[ZERO_BITS_0:.*]] = fir.zero_bits !fir.ptr<!fir.array<?x!fir.logical<2>>>
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[ZERO_BITS_0]](%[[SHAPE_1]]) : (!fir.ptr<!fir.array<?x!fir.logical<2>>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.logical<2>>>>
! CHECK:           fir.store %[[EMBOX_0]] to %[[ALLOCA_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<2>>>>>

  call show_descriptor(l1)
  call show_descriptor(l2)
  pla2 => la2
! CHECK:           %[[DECLARE_3:.*]] = fir.declare %[[ALLOCA_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMtest_show_descriptorFtest_logicalEpla2"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<2>>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<2>>>>>
! CHECK:           %[[ZERO_BITS_3:.*]] = fir.zero_bits !fir.box<none>
! CHECK:           fir.call @_FortranAShowDescriptor(%[[ZERO_BITS_3]]) fastmath<contract> : (!fir.box<none>) -> ()
! CHECK:           fir.call @_FortranAShowDescriptor(%[[ZERO_BITS_3]]) fastmath<contract> : (!fir.box<none>) -> ()

  call show_descriptor(la2)
  call show_descriptor(pla2)
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[DECLARE_2]] : (!fir.ref<!fir.array<2x!fir.logical<2>>>) -> !fir.ref<!fir.array<?x!fir.logical<2>>>
! CHECK:           %[[EMBOX_3:.*]] = fir.embox %[[CONVERT_0]](%[[SHAPE_0]]) : (!fir.ref<!fir.array<?x!fir.logical<2>>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?x!fir.logical<2>>>>
! CHECK:           fir.store %[[EMBOX_3]] to %[[DECLARE_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<2>>>>>
! CHECK:           fir.call @_FortranAShowDescriptor(%[[ZERO_BITS_3]]) fastmath<contract> : (!fir.box<none>) -> ()
! CHECK:           fir.call @_FortranAShowDescriptor(%[[DECLARE_3]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<2>>>>>) -> ()
! CHECK:           return
end subroutine test_logical

subroutine test_real
! CHECK-LABEL:   func.func @_QMtest_show_descriptorPtest_real() {
  implicit none
  real :: half = 0.5
  real :: row(3) = (/ 1 , 2, 3 /)
  real(kind=8) :: w(4) = (/ .00011_8 , .00012_8, .00013_8, .00014_8 /)
! CHECK:           %[[C2:.*]] = arith.constant 2 : index
! CHECK:           %[[C1:.*]] = arith.constant 1 : index
! CHECK:           %[[C4:.*]] = arith.constant 4 : index
! CHECK:           %[[C3:.*]] = arith.constant 3 : index
! CHECK:           %[[ALLOCA_BOX:.*]] = fir.alloca !fir.box<!fir.array<2xf64>>
! CHECK:           %[[DUMMY_SCOPE_2:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[ADDRESS_OF_4:.*]] = fir.address_of(@_QMtest_show_descriptorFtest_realEhalf) : !fir.ref<f32>
! CHECK:           %[[DECLARE_5:.*]] = fir.declare %[[ADDRESS_OF_4]] {uniq_name = "_QMtest_show_descriptorFtest_realEhalf"} : (!fir.ref<f32>) -> !fir.ref<f32>
! CHECK:           %[[ADDRESS_OF_5:.*]] = fir.address_of(@_QMtest_show_descriptorFtest_realErow) : !fir.ref<!fir.array<3xf32>>
! CHECK:           %[[SHAPE_2:.*]] = fir.shape %[[C3]] : (index) -> !fir.shape<1>
! CHECK:           %[[DECLARE_6:.*]] = fir.declare %[[ADDRESS_OF_5]](%[[SHAPE_2]]) {uniq_name = "_QMtest_show_descriptorFtest_realErow"} : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<3xf32>>
! CHECK:           %[[ADDRESS_OF_6:.*]] = fir.address_of(@_QMtest_show_descriptorFtest_realEw) : !fir.ref<!fir.array<4xf64>>
! CHECK:           %[[SHAPE_3:.*]] = fir.shape %[[C4]] : (index) -> !fir.shape<1>
! CHECK:           %[[DECLARE_7:.*]] = fir.declare %[[ADDRESS_OF_6]](%[[SHAPE_3]]) {uniq_name = "_QMtest_show_descriptorFtest_realEw"} : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>) -> !fir.ref<!fir.array<4xf64>>
! CHECK:           %[[ZERO_BITS_4:.*]] = fir.zero_bits !fir.box<none>

  call show_descriptor(half)
  call show_descriptor(row)
  call show_descriptor(w)
  call show_descriptor(w(1:4:2))
! CHECK:           fir.call @_FortranAShowDescriptor(%[[ZERO_BITS_4]]) fastmath<contract> : (!fir.box<none>) -> ()
! CHECK:           fir.call @_FortranAShowDescriptor(%[[ZERO_BITS_4]]) fastmath<contract> : (!fir.box<none>) -> ()
! CHECK:           fir.call @_FortranAShowDescriptor(%[[ZERO_BITS_4]]) fastmath<contract> : (!fir.box<none>) -> ()
! CHECK:           %[[SHAPE_4:.*]] = fir.shape %[[C2]] : (index) -> !fir.shape<1>
! CHECK:           %[[UNDEFINED_0:.*]] = fir.undefined index
! CHECK:           %[[SLICE_0:.*]] = fir.slice %[[C1]], %[[C4]], %[[C2]] : (index, index, index) -> !fir.slice<1>
! CHECK:           %[[EMBOX_10:.*]] = fir.embox %[[DECLARE_7]](%[[SHAPE_3]]) {{\[}}%[[SLICE_0]]] : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<2xf64>>
! CHECK:           fir.store %[[EMBOX_10]] to %[[ALLOCA_BOX]] : !fir.ref<!fir.box<!fir.array<2xf64>>>
! CHECK:           fir.call @_FortranAShowDescriptor(%[[ALLOCA_BOX]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.array<2xf64>>>) -> ()
! CHECK:           return
end subroutine test_real

subroutine test_complex
! CHECK-LABEL:   func.func @_QMtest_show_descriptorPtest_complex() {
  implicit none
  complex, parameter :: hr = 0.5
  complex, parameter :: hi = (0, 0.5)
  complex :: c1 = hr
  complex :: c2 = hi
  complex :: a2(2) = (/ hr, hi /)
! CHECK:           %[[C2:.*]] = arith.constant 2 : index
! CHECK:           %[[DUMMY_SCOPE_3:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[ADDRESS_OF_7:.*]] = fir.address_of(@_QMtest_show_descriptorFtest_complexEa2) : !fir.ref<!fir.array<2xcomplex<f32>>>
! CHECK:           %[[SHAPE_5:.*]] = fir.shape %[[C2]] : (index) -> !fir.shape<1>
! CHECK:           %[[DECLARE_8:.*]] = fir.declare %[[ADDRESS_OF_7]](%[[SHAPE_5]]) {uniq_name = "_QMtest_show_descriptorFtest_complexEa2"} : (!fir.ref<!fir.array<2xcomplex<f32>>>, !fir.shape<1>) -> !fir.ref<!fir.array<2xcomplex<f32>>>
! CHECK:           %[[ADDRESS_OF_8:.*]] = fir.address_of(@_QMtest_show_descriptorFtest_complexEc1) : !fir.ref<complex<f32>>
! CHECK:           %[[DECLARE_9:.*]] = fir.declare %[[ADDRESS_OF_8]] {uniq_name = "_QMtest_show_descriptorFtest_complexEc1"} : (!fir.ref<complex<f32>>) -> !fir.ref<complex<f32>>
! CHECK:           %[[ADDRESS_OF_9:.*]] = fir.address_of(@_QMtest_show_descriptorFtest_complexEc2) : !fir.ref<complex<f32>>
! CHECK:           %[[DECLARE_10:.*]] = fir.declare %[[ADDRESS_OF_9]] {uniq_name = "_QMtest_show_descriptorFtest_complexEc2"} : (!fir.ref<complex<f32>>) -> !fir.ref<complex<f32>>
! CHECK:           %[[ADDRESS_OF_10:.*]] = fir.address_of(@_QMtest_show_descriptorFtest_complexEChi) : !fir.ref<complex<f32>>
! CHECK:           %[[DECLARE_11:.*]] = fir.declare %[[ADDRESS_OF_10]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMtest_show_descriptorFtest_complexEChi"} : (!fir.ref<complex<f32>>) -> !fir.ref<complex<f32>>
! CHECK:           %[[ADDRESS_OF_11:.*]] = fir.address_of(@_QMtest_show_descriptorFtest_complexEChr) : !fir.ref<complex<f32>>
! CHECK:           %[[DECLARE_12:.*]] = fir.declare %[[ADDRESS_OF_11]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMtest_show_descriptorFtest_complexEChr"} : (!fir.ref<complex<f32>>) -> !fir.ref<complex<f32>>
! CHECK:           %[[ZERO_BITS_5:.*]] = fir.zero_bits !fir.box<none>

  call show_descriptor(hr)
! CHECK:           fir.call @_FortranAShowDescriptor(%[[ZERO_BITS_5]]) fastmath<contract> : (!fir.box<none>) -> ()

  call show_descriptor(hi)
! CHECK:           fir.call @_FortranAShowDescriptor(%[[ZERO_BITS_5]]) fastmath<contract> : (!fir.box<none>) -> ()

  call show_descriptor(a2)
! CHECK:           fir.call @_FortranAShowDescriptor(%[[ZERO_BITS_5]]) fastmath<contract> : (!fir.box<none>) -> ()
! CHECK:           return
end subroutine test_complex

subroutine test_derived
! CHECK-LABEL:   func.func @_QMtest_show_descriptorPtest_derived() {
  implicit none
  type :: t1
     integer :: a
     integer :: b
  end type t1
  type, extends (t1) :: t2
     integer :: c
  end type t2
  type(t2) :: vt2 = t2(7,5,3)
  class(t1), allocatable :: c_t1
  class(*), allocatable :: c_unlimited
  allocate(t2 :: c_t1)
  c_t1 = vt2
  allocate(c_unlimited, source=vt2)

! CHECK:           %[[C0:.*]] = arith.constant 0 : index
! CHECK:           %[[C2:.*]] = arith.constant 2 : index
! CHECK:           %[[C1:.*]] = arith.constant 1 : index
! CHECK:           %[[DUMMY_SCOPE_4:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[ADDRESS_OF_12:.*]] = fir.address_of(@_QMtest_show_descriptorFtest_derivedE.n.a) : !fir.ref<!fir.char<1>>
! CHECK:           %[[DECLARE_13:.*]] = fir.declare %[[ADDRESS_OF_12]] typeparams %[[C1]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMtest_show_descriptorFtest_derivedE.n.a"} : (!fir.ref<!fir.char<1>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:           %[[ADDRESS_OF_13:.*]] = fir.address_of(@_QMtest_show_descriptorFtest_derivedE.n.b) : !fir.ref<!fir.char<1>>
! CHECK:           %[[DECLARE_14:.*]] = fir.declare %[[ADDRESS_OF_13]] typeparams %[[C1]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMtest_show_descriptorFtest_derivedE.n.b"} : (!fir.ref<!fir.char<1>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:           %[[ADDRESS_OF_14:.*]] = fir.address_of(@_QMtest_show_descriptorFtest_derivedE.n.t1) : !fir.ref<!fir.char<1,2>>
! CHECK:           %[[DECLARE_15:.*]] = fir.declare %[[ADDRESS_OF_14]] typeparams %[[C2]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMtest_show_descriptorFtest_derivedE.n.t1"} : (!fir.ref<!fir.char<1,2>>, index) -> !fir.ref<!fir.char<1,2>>
! CHECK:           %[[ADDRESS_OF_15:.*]] = fir.address_of(@_QMtest_show_descriptorFtest_derivedE.n.c) : !fir.ref<!fir.char<1>>
! CHECK:           %[[DECLARE_16:.*]] = fir.declare %[[ADDRESS_OF_15]] typeparams %[[C1]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMtest_show_descriptorFtest_derivedE.n.c"} : (!fir.ref<!fir.char<1>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:           %[[ADDRESS_OF_16:.*]] = fir.address_of(@_QMtest_show_descriptorFtest_derivedE.n.t2) : !fir.ref<!fir.char<1,2>>
! CHECK:           %[[DECLARE_17:.*]] = fir.declare %[[ADDRESS_OF_16]] typeparams %[[C2]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMtest_show_descriptorFtest_derivedE.n.t2"} : (!fir.ref<!fir.char<1,2>>, index) -> !fir.ref<!fir.char<1,2>>
! CHECK:           %[[ALLOCA_CT1:.*]] = fir.alloca !fir.class<!fir.heap<!fir.type<_QMtest_show_descriptorFtest_derivedTt1{a:i32,b:i32}>>> {bindc_name = "c_t1", uniq_name = "_QMtest_show_descriptorFtest_derivedEc_t1"}
! CHECK:           %[[ZERO_BITS_CT1:.*]] = fir.zero_bits !fir.heap<!fir.type<_QMtest_show_descriptorFtest_derivedTt1{a:i32,b:i32}>>
! CHECK:           %[[EMBOX_CT1:.*]] = fir.embox %[[ZERO_BITS_CT1]] : (!fir.heap<!fir.type<_QMtest_show_descriptorFtest_derivedTt1{a:i32,b:i32}>>) -> !fir.class<!fir.heap<!fir.type<_QMtest_show_descriptorFtest_derivedTt1{a:i32,b:i32}>>>
! CHECK:           fir.store %[[EMBOX_CT1]] to %[[ALLOCA_CT1]] : !fir.ref<!fir.class<!fir.heap<!fir.type<_QMtest_show_descriptorFtest_derivedTt1{a:i32,b:i32}>>>>
! CHECK:           %[[DECLARE_CT1:.*]] = fir.declare %[[ALLOCA_CT1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMtest_show_descriptorFtest_derivedEc_t1"} : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMtest_show_descriptorFtest_derivedTt1{a:i32,b:i32}>>>>) -> !fir.ref<!fir.class<!fir.heap<!fir.type<_QMtest_show_descriptorFtest_derivedTt1{a:i32,b:i32}>>>>
! CHECK:           %[[ALLOCA_CU:.*]] = fir.alloca !fir.class<!fir.heap<none>> {bindc_name = "c_unlimited", uniq_name = "_QMtest_show_descriptorFtest_derivedEc_unlimited"}
! CHECK:           %[[ZERO_BITS_CU:.*]] = fir.zero_bits !fir.heap<none>
! CHECK:           %[[EMBOX_CU:.*]] = fir.embox %[[ZERO_BITS_CU]] : (!fir.heap<none>) -> !fir.class<!fir.heap<none>>
! CHECK:           fir.store %[[EMBOX_CU]] to %[[ALLOCA_CU]] : !fir.ref<!fir.class<!fir.heap<none>>>
! CHECK:           %[[DECLARE_CU:.*]] = fir.declare %[[ALLOCA_CU]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMtest_show_descriptorFtest_derivedEc_unlimited"} : (!fir.ref<!fir.class<!fir.heap<none>>>) -> !fir.ref<!fir.class<!fir.heap<none>>>
! CHECK:           %[[ADDRESS_OF_17:.*]] = fir.address_of(@_QMtest_show_descriptorFtest_derivedEvt2) : !fir.ref<!fir.type<_QMtest_show_descriptorFtest_derivedTt2{t1:!fir.type<_QMtest_show_descriptorFtest_derivedTt1{a:i32,b:i32}>,c:i32}>>
! CHECK:           %[[DECLARE_18:.*]] = fir.declare %[[ADDRESS_OF_17]] {uniq_name = "_QMtest_show_descriptorFtest_derivedEvt2"} : (!fir.ref<!fir.type<_QMtest_show_descriptorFtest_derivedTt2{t1:!fir.type<_QMtest_show_descriptorFtest_derivedTt1{a:i32,b:i32}>,c:i32}>>) -> !fir.ref<!fir.type<_QMtest_show_descriptorFtest_derivedTt2{t1:!fir.type<_QMtest_show_descriptorFtest_derivedTt1{a:i32,b:i32}>,c:i32}>>
! CHECK:           %[[ZERO_BITS_6:.*]] = fir.zero_bits !fir.box<none>

  call show_descriptor(vt2)
! CHECK:           fir.call @_FortranAShowDescriptor(%[[ZERO_BITS_6]]) fastmath<contract> : (!fir.box<none>) -> ()

  call show_descriptor(c_t1)
! CHECK:           fir.call @_FortranAShowDescriptor(%[[DECLARE_CT1]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.heap<!fir.type<_QMtest_show_descriptorFtest_derivedTt1{a:i32,b:i32}>>>>) -> ()

  call show_descriptor(c_unlimited)
! CHECK:           fir.call @_FortranAShowDescriptor(%[[DECLARE_CU]]) fastmath<contract> : (!fir.ref<!fir.class<!fir.heap<none>>>) -> ()
! CHECK:           return
end subroutine test_derived

subroutine test_derived_member
! CHECK-LABEL:   func.func @_QMtest_show_descriptorPtest_derived_member() {
  implicit none
  type :: t3
     integer, allocatable :: a(:)
  end type t3
  type(t3) :: vt3
! CHECK:           %[[VT3:.*]] = fir.alloca !fir.type<_QMtest_show_descriptorFtest_derived_memberTt3{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>
! CHECK:           %[[VT3_DECL:.*]] = fir.declare %[[VT3]] {uniq_name = "_QMtest_show_descriptorFtest_derived_memberEvt3"} : (!fir.ref<!fir.type<_QMtest_show_descriptorFtest_derived_memberTt3{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.type<_QMtest_show_descriptorFtest_derived_memberTt3{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
  allocate(vt3%a(5))
! CHECK:           %[[A_FIELD:.*]] = fir.field_index a, !fir.type<_QMtest_show_descriptorFtest_derived_memberTt3{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>
! CHECK:           %[[A_COORD:.*]] = fir.coordinate_of %[[VT3_DECL]], a : (!fir.ref<!fir.type<_QMtest_show_descriptorFtest_derived_memberTt3{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  call show_descriptor(vt3%a)
! CHECK:           fir.call @_FortranAShowDescriptor(%[[A_COORD]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> ()
  deallocate(vt3%a)
end subroutine test_derived_member
end module test_show_descriptor
