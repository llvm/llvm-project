! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

module reduce_mod

type :: t1
  integer :: a
end type

  abstract interface
    pure function red_int1_interface(a, b)
      integer(1), intent(in) :: a, b
      integer(1) :: red_int1_interface
    end function
    pure function red_int1_interface_value(a, b)
      integer(1), value, intent(in) :: a, b
      integer(1) :: red_int1_interface_value
    end function
  end interface

contains

pure function red_int1(a,b)
  integer(1), intent(in) :: a, b
  integer(1) :: red_int1
  red_int1 = a + b
end function

pure function red_int1_value(a,b)
  integer(1), value, intent(in) :: a, b
  integer(1) :: red_int1_value
  red_int1_value = a + b
end function

subroutine integer1(a, id, d1, d2)
  integer(1), intent(in) :: a(:)
  integer(1) :: res, id
  procedure(red_int1_interface), pointer :: fptr
  procedure(red_int1_interface_value), pointer :: fptr_value
  procedure(red_int1_interface) :: d1
  procedure(red_int1_interface_value) :: d2

  res = reduce(a, red_int1)

  res = reduce(a, red_int1, identity=id)
  
  res = reduce(a, red_int1, identity=id, ordered = .true.)

  res = reduce(a, red_int1, [.true., .true., .false.])
  
  res = reduce(a, red_int1_value)

  fptr => red_int1
  res = reduce(a, fptr)

  fptr_value => red_int1_value
  res = reduce(a, fptr_value)

  !res = reduce(a, d1)
  !res = reduce(a, d2)
end subroutine

! CHECK-LABEL: func.func @_QMreduce_modPinteger1(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xi8>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.ref<i8> {fir.bindc_name = "id"}
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{.*}} {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMreduce_modFinteger1Ea"} : (!fir.box<!fir.array<?xi8>>, !fir.dscope) -> (!fir.box<!fir.array<?xi8>>, !fir.box<!fir.array<?xi8>>)
! CHECK: %[[ID:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{.*}} {uniq_name = "_QMreduce_modFinteger1Eid"} : (!fir.ref<i8>, !fir.dscope) -> (!fir.ref<i8>, !fir.ref<i8>)
! CHECK: %[[ALLOC_RES:.*]] = fir.alloca i8 {bindc_name = "res", uniq_name = "_QMreduce_modFinteger1Eres"}
! CHECK: %[[RES:.*]]:2 = hlfir.declare %[[ALLOC_RES]] {uniq_name = "_QMreduce_modFinteger1Eres"} : (!fir.ref<i8>) -> (!fir.ref<i8>, !fir.ref<i8>)
! CHECK: %[[ADDR_OP:.*]] = fir.address_of(@_QMreduce_modPred_int1) : (!fir.ref<i8>, !fir.ref<i8>) -> i8
! CHECK: %[[BOX_PROC:.*]] = fir.emboxproc %[[ADDR_OP]] : ((!fir.ref<i8>, !fir.ref<i8>) -> i8) -> !fir.boxproc<() -> ()>
! CHECK: %[[MASK:.*]] = fir.absent !fir.box<i1>
! CHECK: %[[IDENTITY:.*]] = fir.absent !fir.ref<i8>
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX_PROC]] : (!fir.boxproc<() -> ()>) -> ((!fir.ref<i8>, !fir.ref<i8>) -> !fir.ref<i8>)
! CHECK: %[[A_NONE:.*]] = fir.convert %[[A]]#1 : (!fir.box<!fir.array<?xi8>>) -> !fir.box<none>
! CHECK: %[[MASK_NONE:.*]] = fir.convert %[[MASK]] : (!fir.box<i1>) -> !fir.box<none>
! CHECK: %[[REDUCE_RES:.*]] = fir.call @_FortranAReduceInteger1Ref(%[[A_NONE]], %[[BOX_ADDR]], %{{.*}}, %{{.*}}, %c1{{.*}}, %[[MASK_NONE]], %[[IDENTITY]], %false) fastmath<contract> : (!fir.box<none>, (!fir.ref<i8>, !fir.ref<i8>) -> !fir.ref<i8>, !fir.ref<i8>, i32, i32, !fir.box<none>, !fir.ref<i8>, i1) -> i8
! CHECK: hlfir.assign %[[REDUCE_RES]] to %[[RES]]#0 : i8, !fir.ref<i8>
! CHECK: %[[ADDR_OP:.*]] = fir.address_of(@_QMreduce_modPred_int1) : (!fir.ref<i8>, !fir.ref<i8>) -> i8
! CHECK: %[[BOX_PROC:.*]] = fir.emboxproc %[[ADDR_OP]] : ((!fir.ref<i8>, !fir.ref<i8>) -> i8) -> !fir.boxproc<() -> ()>
! CHECK: %[[MASK:.*]] = fir.absent !fir.box<i1>
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX_PROC]] : (!fir.boxproc<() -> ()>) -> ((!fir.ref<i8>, !fir.ref<i8>) -> !fir.ref<i8>)
! CHECK: %[[A_NONE:.*]] = fir.convert %[[A]]#1 : (!fir.box<!fir.array<?xi8>>) -> !fir.box<none>
! CHECK: %[[MASK_NONE:.*]] = fir.convert %[[MASK]] : (!fir.box<i1>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAReduceInteger1Ref(%[[A_NONE]], %[[BOX_ADDR]], %{{.*}}, %{{.*}}, %c1{{.*}}, %[[MASK_NONE]], %[[ID]]#1, %false{{.*}}) fastmath<contract> : (!fir.box<none>, (!fir.ref<i8>, !fir.ref<i8>) -> !fir.ref<i8>, !fir.ref<i8>, i32, i32, !fir.box<none>, !fir.ref<i8>, i1) -> i8
! CHECK: fir.call @_FortranAReduceInteger1Ref(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}#1, %true)
! CHECK: %[[MASK:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro.3xl4.0"} : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.ref<!fir.array<3x!fir.logical<4>>>)
! CHECK: %[[SHAPE_C3:.*]] = fir.shape %c3{{.*}} : (index) -> !fir.shape<1>
! CHECK: %[[BOXED_MASK:.*]] = fir.embox %[[MASK]]#1(%[[SHAPE_C3]]) : (!fir.ref<!fir.array<3x!fir.logical<4>>>, !fir.shape<1>) -> !fir.box<!fir.array<3x!fir.logical<4>>>
! CHECK: %[[CONV_MASK:.*]] = fir.convert %[[BOXED_MASK]] : (!fir.box<!fir.array<3x!fir.logical<4>>>) -> !fir.box<none>
! CHECK: fir.call @_FortranAReduceInteger1Ref(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[CONV_MASK]], %{{.*}}, %false{{.*}})
! CHECK: fir.call @_FortranAReduceInteger1Value
! CHECK: fir.call @_FortranAReduceInteger1Ref
! CHECK: fir.call @_FortranAReduceInteger1Value
! TODO fir.call @_FortranAReduceInteger1Ref
! TODO fir.call @_FortranAReduceInteger1Value

pure function red_int2(a,b)
  integer(2), intent(in) :: a, b
  integer(2) :: red_int2
  red_int2 = a + b
end function

pure function red_int2_value(a,b)
  integer(2), value, intent(in) :: a, b
  integer(2) :: red_int2_value
  red_int2_value = a + b
end function

subroutine integer2(a)
  integer(2), intent(in) :: a(:)
  integer(2) :: res
  res = reduce(a, red_int2)
  res = reduce(a, red_int2_value)
end subroutine

! CHECK: fir.call @_FortranAReduceInteger2Ref
! CHECK: fir.call @_FortranAReduceInteger2Value

pure function red_int4(a,b)
  integer(4), intent(in) :: a, b
  integer(4) :: red_int4
  red_int4 = a + b
end function

pure function red_int4_value(a,b)
  integer(4), value, intent(in) :: a, b
  integer(4) :: red_int4_value
  red_int4_value = a + b
end function

subroutine integer4(a)
  integer(4), intent(in) :: a(:)
  integer(4) :: res
  res = reduce(a, red_int4)
  res = reduce(a, red_int4_value)
end subroutine

! CHECK: fir.call @_FortranAReduceInteger4Ref
! CHECK: fir.call @_FortranAReduceInteger4Value

pure function red_int8(a,b)
  integer(8), intent(in) :: a, b
  integer(8) :: red_int8
  red_int8 = a + b
end function

pure function red_int8_value(a,b)
  integer(8), value, intent(in) :: a, b
  integer(8) :: red_int8_value
  red_int8_value = a + b
end function

subroutine integer8(a)
  integer(8), intent(in) :: a(:)
  integer(8) :: res
  res = reduce(a, red_int8)
  res = reduce(a, red_int8_value)
end subroutine

! CHECK: fir.call @_FortranAReduceInteger8Ref
! CHECK: fir.call @_FortranAReduceInteger8Value

pure function red_int16(a,b)
  integer(16), intent(in) :: a, b
  integer(16) :: red_int16
  red_int16 = a + b
end function

pure function red_int16_value(a,b)
  integer(16), value, intent(in) :: a, b
  integer(16) :: red_int16_value
  red_int16_value = a + b
end function

subroutine integer16(a)
  integer(16), intent(in) :: a(:)
  integer(16) :: res
  res = reduce(a, red_int16)
  res = reduce(a, red_int16_value)
end subroutine

! CHECK: fir.call @_FortranAReduceInteger16Ref
! CHECK: fir.call @_FortranAReduceInteger16Value

pure function red_real2(a,b)
  real(2), intent(in) :: a, b
  real(2) :: red_real2
  red_real2 = a + b
end function

pure function red_real2_value(a,b)
  real(2), value, intent(in) :: a, b
  real(2) :: red_real2_value
  red_real2_value = a + b
end function

subroutine real2(a)
  real(2), intent(in) :: a(:)
  real(2) :: res
  res = reduce(a, red_real2)
  res = reduce(a, red_real2_value)
end subroutine

! CHECK: fir.call @_FortranAReduceReal2Ref
! CHECK: fir.call @_FortranAReduceReal2Value

pure function red_real3(a,b)
  real(3), intent(in) :: a, b
  real(3) :: red_real3
  red_real3 = a + b
end function

pure function red_real3_value(a,b)
  real(3), value, intent(in) :: a, b
  real(3) :: red_real3_value
  red_real3_value = a + b
end function

subroutine real3(a)
  real(3), intent(in) :: a(:)
  real(3) :: res
  res = reduce(a, red_real3)
  res = reduce(a, red_real3_value)
end subroutine

! CHECK: fir.call @_FortranAReduceReal3Ref
! CHECK: fir.call @_FortranAReduceReal3Value

pure function red_real4(a,b)
  real(4), intent(in) :: a, b
  real(4) :: red_real4
  red_real4 = a + b
end function

pure function red_real4_value(a,b)
  real(4), value, intent(in) :: a, b
  real(4) :: red_real4_value
  red_real4_value = a + b
end function

subroutine real4(a)
  real(4), intent(in) :: a(:)
  real(4) :: res
  res = reduce(a, red_real4)
  res = reduce(a, red_real4_value)
end subroutine

! CHECK: fir.call @_FortranAReduceReal4Ref
! CHECK: fir.call @_FortranAReduceReal4Value

pure function red_real8(a,b)
  real(8), intent(in) :: a, b
  real(8) :: red_real8
  red_real8 = a + b
end function

pure function red_real8_value(a,b)
  real(8), value, intent(in) :: a, b
  real(8) :: red_real8_value
  red_real8_value = a + b
end function

subroutine real8(a)
  real(8), intent(in) :: a(:)
  real(8) :: res
  res = reduce(a, red_real8)
  res = reduce(a, red_real8_value)
end subroutine

! CHECK: fir.call @_FortranAReduceReal8Ref
! CHECK: fir.call @_FortranAReduceReal8Value

pure function red_real10(a,b)
  real(10), intent(in) :: a, b
  real(10) :: red_real10
  red_real10 = a + b
end function

pure function red_real10_value(a,b)
  real(10), value, intent(in) :: a, b
  real(10) :: red_real10_value
  red_real10_value = a + b
end function

subroutine real10(a)
  real(10), intent(in) :: a(:)
  real(10) :: res
  res = reduce(a, red_real10)
  res = reduce(a, red_real10_value)
end subroutine

! CHECK: fir.call @_FortranAReduceReal10Ref
! CHECK: fir.call @_FortranAReduceReal10Value

pure function red_real16(a,b)
  real(16), intent(in) :: a, b
  real(16) :: red_real16
  red_real16 = a + b
end function

pure function red_real16_value(a,b)
  real(16), value, intent(in) :: a, b
  real(16) :: red_real16_value
  red_real16_value = a + b
end function

subroutine real16(a)
  real(16), intent(in) :: a(:)
  real(16) :: res
  res = reduce(a, red_real16)
  res = reduce(a, red_real16_value)
end subroutine

! CHECK: fir.call @_FortranAReduceReal16Ref
! CHECK: fir.call @_FortranAReduceReal16Value

pure function red_complex2(a,b)
  complex(2), intent(in) :: a, b
  complex(2) :: red_complex2
  red_complex2 = a + b
end function

pure function red_complex2_value(a,b)
  complex(2), value, intent(in) :: a, b
  complex(2) :: red_complex2_value
  red_complex2_value = a + b
end function

subroutine complex2(a)
  complex(2), intent(in) :: a(:)
  complex(2) :: res
  res = reduce(a, red_complex2)
  res = reduce(a, red_complex2_value)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex2Ref
! CHECK: fir.call @_FortranACppReduceComplex2Value

pure function red_complex3(a,b)
  complex(3), intent(in) :: a, b
  complex(3) :: red_complex3
  red_complex3 = a + b
end function

pure function red_complex3_value(a,b)
  complex(3), value, intent(in) :: a, b
  complex(3) :: red_complex3_value
  red_complex3_value = a + b
end function

subroutine complex3(a)
  complex(3), intent(in) :: a(:)
  complex(3) :: res
  res = reduce(a, red_complex3)
  res = reduce(a, red_complex3_value)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex3Ref
! CHECK: fir.call @_FortranACppReduceComplex3Value

pure function red_complex4(a,b)
  complex(4), intent(in) :: a, b
  complex(4) :: red_complex4
  red_complex4 = a + b
end function

pure function red_complex4_value(a,b)
  complex(4), value, intent(in) :: a, b
  complex(4) :: red_complex4_value
  red_complex4_value = a + b
end function

subroutine complex4(a)
  complex(4), intent(in) :: a(:)
  complex(4) :: res
  res = reduce(a, red_complex4)
  res = reduce(a, red_complex4_value)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex4Ref
! CHECK: fir.call @_FortranACppReduceComplex4Value

pure function red_complex8(a,b)
  complex(8), intent(in) :: a, b
  complex(8) :: red_complex8
  red_complex8 = a + b
end function

pure function red_complex8_value(a,b)
  complex(8), value, intent(in) :: a, b
  complex(8) :: red_complex8_value
  red_complex8_value = a + b
end function

subroutine complex8(a)
  complex(8), intent(in) :: a(:)
  complex(8) :: res
  res = reduce(a, red_complex8)
  res = reduce(a, red_complex8_value)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex8Ref
! CHECK: fir.call @_FortranACppReduceComplex8Value

pure function red_complex10(a,b)
  complex(10), intent(in) :: a, b
  complex(10) :: red_complex10
  red_complex10 = a + b
end function

pure function red_complex10_value(a,b)
  complex(10), value, intent(in) :: a, b
  complex(10) :: red_complex10_value
  red_complex10_value = a + b
end function

subroutine complex10(a)
  complex(10), intent(in) :: a(:)
  complex(10) :: res
  res = reduce(a, red_complex10)
  res = reduce(a, red_complex10_value)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex10Ref
! CHECK: fir.call @_FortranACppReduceComplex10Value

pure function red_complex16(a,b)
  complex(16), intent(in) :: a, b
  complex(16) :: red_complex16
  red_complex16 = a + b
end function

pure function red_complex16_value(a,b)
  complex(16), value, intent(in) :: a, b
  complex(16) :: red_complex16_value
  red_complex16_value = a + b
end function

subroutine complex16(a)
  complex(16), intent(in) :: a(:)
  complex(16) :: res
  res = reduce(a, red_complex16)
  res = reduce(a, red_complex16_value)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex16Ref
! CHECK: fir.call @_FortranACppReduceComplex16Value

pure function red_log1(a,b)
  logical(1), intent(in) :: a, b
  logical(1) :: red_log1
  red_log1 = a .and. b
end function

pure function red_log1_value(a,b)
  logical(1), value, intent(in) :: a, b
  logical(1) :: red_log1_value
  red_log1_value = a .and. b
end function

subroutine log1(a)
  logical(1), intent(in) :: a(:)
  logical(1) :: res
  res = reduce(a, red_log1)
  res = reduce(a, red_log1_value)
end subroutine

! CHECK: fir.call @_FortranAReduceLogical1Ref
! CHECK: fir.call @_FortranAReduceLogical1Value

pure function red_log2(a,b)
  logical(2), intent(in) :: a, b
  logical(2) :: red_log2
  red_log2 = a .and. b
end function

pure function red_log2_value(a,b)
  logical(2), value, intent(in) :: a, b
  logical(2) :: red_log2_value
  red_log2_value = a .and. b
end function

subroutine log2(a)
  logical(2), intent(in) :: a(:)
  logical(2) :: res
  res = reduce(a, red_log2)
  res = reduce(a, red_log2_value)
end subroutine

! CHECK: fir.call @_FortranAReduceLogical2Ref
! CHECK: fir.call @_FortranAReduceLogical2Value

pure function red_log4(a,b)
  logical(4), intent(in) :: a, b
  logical(4) :: red_log4
  red_log4 = a .and. b
end function

pure function red_log4_value(a,b)
  logical(4), value, intent(in) :: a, b
  logical(4) :: red_log4_value
  red_log4_value = a .and. b
end function

subroutine log4(a)
  logical(4), intent(in) :: a(:)
  logical(4) :: res
  res = reduce(a, red_log4)
  res = reduce(a, red_log4_value)
end subroutine

! CHECK: fir.call @_FortranAReduceLogical4Ref
! CHECK: fir.call @_FortranAReduceLogical4Value

pure function red_log8(a,b)
  logical(8), intent(in) :: a, b
  logical(8) :: red_log8
  red_log8 = a .and. b
end function

pure function red_log8_value(a,b)
  logical(8), value, intent(in) :: a, b
  logical(8) :: red_log8_value
  red_log8_value = a .and. b
end function

subroutine log8(a)
  logical(8), intent(in) :: a(:)
  logical(8) :: res
  res = reduce(a, red_log8)
  res = reduce(a, red_log8_value)
end subroutine

! CHECK: fir.call @_FortranAReduceLogical8Ref
! CHECK: fir.call @_FortranAReduceLogical8Value

pure function red_char1(a,b)
  character(1), intent(in) :: a, b
  character(1) :: red_char1
  red_char1 = a // b
end function

subroutine char1(a)
  character(1), intent(in) :: a(:)
  character(1) :: res
  res = reduce(a, red_char1)
end subroutine

! CHECK: %[[CHRTMP:.*]] = fir.alloca !fir.char<1> {bindc_name = ".chrtmp"}
! CHECK: %[[RESULT:.*]] = fir.convert %[[CHRTMP]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
! CHECK: fir.call @_FortranAReduceChar1(%[[RESULT]], {{.*}})

pure function red_char2(a,b)
  character(kind=2, len=10), intent(in) :: a, b
  character(kind=2, len=10) :: red_char2
  red_char2 = a // b
end function

subroutine char2(a)
  character(kind=2, len=10), intent(in) :: a(:)
  character(kind=2, len=10) :: res
  res = reduce(a, red_char2)
end subroutine

! CHECK: %[[CHRTMP:.*]] = fir.alloca !fir.char<2,10> {bindc_name = ".chrtmp"}
! CHECK: %[[RESULT:.*]] = fir.convert %[[CHRTMP]] : (!fir.ref<!fir.char<2,10>>) -> !fir.ref<i16>
! CHECK: fir.call @_FortranAReduceChar2(%[[RESULT]], {{.*}})

pure function red_char4(a,b)
  character(kind=4), intent(in) :: a, b
  character(kind=4) :: red_char4
  red_char4 = a // b
end function

subroutine char4(a)
  character(kind=4), intent(in) :: a(:)
  character(kind=4) :: res
  res = reduce(a, red_char4)
end subroutine

! CHECK: fir.call @_FortranAReduceChar4

pure function red_type(a,b)
  type(t1), intent(in) :: a, b
  type(t1) :: red_type
  red_type%a = a%a + b%a
end function

subroutine testtype(a)
  type(t1), intent(in) :: a(:)
  type(t1) :: res
  res = reduce(a, red_type)
end subroutine

! CHECK: fir.call @_FortranAReduceDerivedType

subroutine integer1dim(a, id)
  integer(1), intent(in) :: a(:,:)
  integer(1), allocatable :: res(:)

  res = reduce(a, red_int1, 2)
  res = reduce(a, red_int1_value, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceInteger1DimRef
! CHECK: fir.call @_FortranAReduceInteger1DimValue

subroutine integer2dim(a, id)
  integer(2), intent(in) :: a(:,:)
  integer(2), allocatable :: res(:)

  res = reduce(a, red_int2, 2)
  res = reduce(a, red_int2_value, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceInteger2DimRef
! CHECK: fir.call @_FortranAReduceInteger2DimValue

subroutine integer4dim(a, id)
  integer(4), intent(in) :: a(:,:)
  integer(4), allocatable :: res(:)

  res = reduce(a, red_int4, 2)
  res = reduce(a, red_int4_value, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceInteger4DimRef
! CHECK: fir.call @_FortranAReduceInteger4DimValue

subroutine integer8dim(a, id)
  integer(8), intent(in) :: a(:,:)
  integer(8), allocatable :: res(:)

  res = reduce(a, red_int8, 2)
  res = reduce(a, red_int8_value, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceInteger8DimRef
! CHECK: fir.call @_FortranAReduceInteger8DimValue

subroutine integer16dim(a, id)
  integer(16), intent(in) :: a(:,:)
  integer(16), allocatable :: res(:)

  res = reduce(a, red_int16, 2)
  res = reduce(a, red_int16_value, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceInteger16DimRef
! CHECK: fir.call @_FortranAReduceInteger16DimValue

subroutine real2dim(a, id)
  real(2), intent(in) :: a(:,:)
  real(2), allocatable :: res(:)

  res = reduce(a, red_real2, 2)
  res = reduce(a, red_real2_value, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceReal2DimRef
! CHECK: fir.call @_FortranAReduceReal2DimValue

subroutine real3dim(a, id)
  real(3), intent(in) :: a(:,:)
  real(3), allocatable :: res(:)

  res = reduce(a, red_real3, 2)
  res = reduce(a, red_real3_value, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceReal3DimRef
! CHECK: fir.call @_FortranAReduceReal3DimValue

subroutine real4dim(a, id)
  real(4), intent(in) :: a(:,:)
  real(4), allocatable :: res(:)

  res = reduce(a, red_real4, 2)
  res = reduce(a, red_real4_value, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceReal4DimRef
! CHECK: fir.call @_FortranAReduceReal4DimValue

subroutine real8dim(a, id)
  real(8), intent(in) :: a(:,:)
  real(8), allocatable :: res(:)

  res = reduce(a, red_real8, 2)
  res = reduce(a, red_real8_value, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceReal8DimRef
! CHECK: fir.call @_FortranAReduceReal8DimValue

subroutine real10dim(a, id)
  real(10), intent(in) :: a(:,:)
  real(10), allocatable :: res(:)

  res = reduce(a, red_real10, 2)
  res = reduce(a, red_real10_value, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceReal10DimRef
! CHECK: fir.call @_FortranAReduceReal10DimValue

subroutine real16dim(a, id)
  real(16), intent(in) :: a(:,:)
  real(16), allocatable :: res(:)

  res = reduce(a, red_real16, 2)
  res = reduce(a, red_real16_value, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceReal16DimRef
! CHECK: fir.call @_FortranAReduceReal16DimValue

subroutine complex2dim(a, id)
  complex(2), intent(in) :: a(:,:)
  complex(2), allocatable :: res(:)

  res = reduce(a, red_complex2, 2)
  res = reduce(a, red_complex2_value, 2)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex2DimRef
! CHECK: fir.call @_FortranACppReduceComplex2DimValue

subroutine complex3dim(a, id)
  complex(3), intent(in) :: a(:,:)
  complex(3), allocatable :: res(:)

  res = reduce(a, red_complex3, 2)
  res = reduce(a, red_complex3_value, 2)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex3DimRef
! CHECK: fir.call @_FortranACppReduceComplex3DimValue

subroutine complex4dim(a, id)
  complex(4), intent(in) :: a(:,:)
  complex(4), allocatable :: res(:)

  res = reduce(a, red_complex4, 2)
  res = reduce(a, red_complex4_value, 2)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex4DimRef
! CHECK: fir.call @_FortranACppReduceComplex4DimValue

subroutine complex8dim(a, id)
  complex(8), intent(in) :: a(:,:)
  complex(8), allocatable :: res(:)

  res = reduce(a, red_complex8, 2)
  res = reduce(a, red_complex8_value, 2)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex8DimRef
! CHECK: fir.call @_FortranACppReduceComplex8DimValue

subroutine complex10dim(a, id)
  complex(10), intent(in) :: a(:,:)
  complex(10), allocatable :: res(:)

  res = reduce(a, red_complex10, 2)
  res = reduce(a, red_complex10_value, 2)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex10DimRef
! CHECK: fir.call @_FortranACppReduceComplex10DimValue

subroutine complex16dim(a, id)
  complex(16), intent(in) :: a(:,:)
  complex(16), allocatable :: res(:)

  res = reduce(a, red_complex16, 2)
  res = reduce(a, red_complex16_value, 2)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex16DimRef
! CHECK: fir.call @_FortranACppReduceComplex16DimValue

subroutine logical1dim(a, id)
  logical(1), intent(in) :: a(:,:)
  logical(1), allocatable :: res(:)

  res = reduce(a, red_log1, 2)
  res = reduce(a, red_log1_value, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceLogical1DimRef
! CHECK: fir.call @_FortranAReduceLogical1DimValue

subroutine logical2dim(a, id)
  logical(2), intent(in) :: a(:,:)
  logical(2), allocatable :: res(:)

  res = reduce(a, red_log2, 2)
  res = reduce(a, red_log2_value, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceLogical2DimRef
! CHECK: fir.call @_FortranAReduceLogical2DimValue

subroutine logical4dim(a, id)
  logical(4), intent(in) :: a(:,:)
  logical(4), allocatable :: res(:)

  res = reduce(a, red_log4, 2)
  res = reduce(a, red_log4_value, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceLogical4DimRef
! CHECK: fir.call @_FortranAReduceLogical4DimValue

subroutine logical8dim(a, id)
  logical(8), intent(in) :: a(:,:)
  logical(8), allocatable :: res(:)

  res = reduce(a, red_log8, 2)
  res = reduce(a, red_log8_value, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceLogical8DimRef
! CHECK: fir.call @_FortranAReduceLogical8DimValue

subroutine testtypeDim(a)
  type(t1), intent(in) :: a(:,:)
  type(t1), allocatable :: res(:)
  res = reduce(a, red_type, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceDerivedTypeDim

subroutine char1dim(a)
  character(1), intent(in) :: a(:, :)
  character(1), allocatable :: res(:)
  res = reduce(a, red_char1, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceCharacter1Dim

subroutine char2dim(a)
  character(kind=2, len=10), intent(in) :: a(:, :)
  character(kind=2, len=10), allocatable :: res(:)
  res = reduce(a, red_char2, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceCharacter2Dim

subroutine char4dim(a)
  character(kind=4), intent(in) :: a(:, :)
  character(kind=4), allocatable :: res(:)
  res = reduce(a, red_char4, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceCharacter4Dim

pure function red_char_dyn(a, b)
  character(*), intent(In) :: a, b
  character(max(len(a),len(b))) :: red_char_dyn
  red_char_dyn = max(a, b)
end function

subroutine charDyn()
  character(5) :: res
  character(:), allocatable :: a(:)
  allocate(character(10)::a(10))
  res = reduce(a, red_char_dyn)
end subroutine

! CHECK: %[[BOX_ELESIZE:.*]] = fir.box_elesize %{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> index
! CHECK: %[[CHRTMP:.*]] = fir.alloca !fir.char<1,?>(%[[BOX_ELESIZE]] : index) {bindc_name = ".chrtmp"}
! CHECK: %[[RESULT:.*]] = fir.convert %[[CHRTMP]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK: fir.call @_FortranAReduceChar1(%[[RESULT]], {{.*}})

end module
