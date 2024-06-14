! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

module reduce_mod

type :: t1
  integer :: a
end type

contains

pure function red_int1(a,b)
  integer(1), intent(in) :: a, b
  integer(1) :: red_int1
  red_int1 = a + b
end function

subroutine integer1(a, id)
  integer(1), intent(in) :: a(:)
  integer(1) :: res, id

  res = reduce(a, red_int1)

  res = reduce(a, red_int1, identity=id)
  
  res = reduce(a, red_int1, identity=id, ordered = .true.)

  res = reduce(a, red_int1, [.true., .true., .false.])
end subroutine

! CHECK-LABEL: func.func @_QMreduce_modPinteger1(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xi8>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.ref<i8> {fir.bindc_name = "id"})
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

pure function red_int2(a,b)
  integer(2), intent(in) :: a, b
  integer(2) :: red_int2
  red_int2 = a + b
end function

subroutine integer2(a)
  integer(2), intent(in) :: a(:)
  integer(2) :: res
  res = reduce(a, red_int2)
end subroutine

! CHECK: fir.call @_FortranAReduceInteger2Ref

pure function red_int4(a,b)
  integer(4), intent(in) :: a, b
  integer(4) :: red_int4
  red_int4 = a + b
end function

subroutine integer4(a)
  integer(4), intent(in) :: a(:)
  integer(4) :: res
  res = reduce(a, red_int4)
end subroutine

! CHECK: fir.call @_FortranAReduceInteger4Ref

pure function red_int8(a,b)
  integer(8), intent(in) :: a, b
  integer(8) :: red_int8
  red_int8 = a + b
end function

subroutine integer8(a)
  integer(8), intent(in) :: a(:)
  integer(8) :: res
  res = reduce(a, red_int8)
end subroutine

! CHECK: fir.call @_FortranAReduceInteger8Ref

pure function red_int16(a,b)
  integer(16), intent(in) :: a, b
  integer(16) :: red_int16
  red_int16 = a + b
end function

subroutine integer16(a)
  integer(16), intent(in) :: a(:)
  integer(16) :: res
  res = reduce(a, red_int16)
end subroutine

! CHECK: fir.call @_FortranAReduceInteger16Ref

pure function red_real2(a,b)
  real(2), intent(in) :: a, b
  real(2) :: red_real2
  red_real2 = a + b
end function

subroutine real2(a)
  real(2), intent(in) :: a(:)
  real(2) :: res
  res = reduce(a, red_real2)
end subroutine

! CHECK: fir.call @_FortranAReduceReal2Ref

pure function red_real3(a,b)
  real(3), intent(in) :: a, b
  real(3) :: red_real3
  red_real3 = a + b
end function

subroutine real3(a)
  real(3), intent(in) :: a(:)
  real(3) :: res
  res = reduce(a, red_real3)
end subroutine

! CHECK: fir.call @_FortranAReduceReal3Ref

pure function red_real4(a,b)
  real(4), intent(in) :: a, b
  real(4) :: red_real4
  red_real4 = a + b
end function

subroutine real4(a)
  real(4), intent(in) :: a(:)
  real(4) :: res
  res = reduce(a, red_real4)
end subroutine

! CHECK: fir.call @_FortranAReduceReal4Ref

pure function red_real8(a,b)
  real(8), intent(in) :: a, b
  real(8) :: red_real8
  red_real8 = a + b
end function

subroutine real8(a)
  real(8), intent(in) :: a(:)
  real(8) :: res
  res = reduce(a, red_real8)
end subroutine

! CHECK: fir.call @_FortranAReduceReal8Ref

pure function red_real10(a,b)
  real(10), intent(in) :: a, b
  real(10) :: red_real10
  red_real10 = a + b
end function

subroutine real10(a)
  real(10), intent(in) :: a(:)
  real(10) :: res
  res = reduce(a, red_real10)
end subroutine

! CHECK: fir.call @_FortranAReduceReal10Ref

pure function red_real16(a,b)
  real(16), intent(in) :: a, b
  real(16) :: red_real16
  red_real16 = a + b
end function

subroutine real16(a)
  real(16), intent(in) :: a(:)
  real(16) :: res
  res = reduce(a, red_real16)
end subroutine

! CHECK: fir.call @_FortranAReduceReal16Ref

pure function red_complex2(a,b)
  complex(2), intent(in) :: a, b
  complex(2) :: red_complex2
  red_complex2 = a + b
end function

subroutine complex2(a)
  complex(2), intent(in) :: a(:)
  complex(2) :: res
  res = reduce(a, red_complex2)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex2

pure function red_complex3(a,b)
  complex(3), intent(in) :: a, b
  complex(3) :: red_complex3
  red_complex3 = a + b
end function

subroutine complex3(a)
  complex(3), intent(in) :: a(:)
  complex(3) :: res
  res = reduce(a, red_complex3)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex3

pure function red_complex4(a,b)
  complex(4), intent(in) :: a, b
  complex(4) :: red_complex4
  red_complex4 = a + b
end function

subroutine complex4(a)
  complex(4), intent(in) :: a(:)
  complex(4) :: res
  res = reduce(a, red_complex4)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex4

pure function red_complex8(a,b)
  complex(8), intent(in) :: a, b
  complex(8) :: red_complex8
  red_complex8 = a + b
end function

subroutine complex8(a)
  complex(8), intent(in) :: a(:)
  complex(8) :: res
  res = reduce(a, red_complex8)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex8

pure function red_complex10(a,b)
  complex(10), intent(in) :: a, b
  complex(10) :: red_complex10
  red_complex10 = a + b
end function

subroutine complex10(a)
  complex(10), intent(in) :: a(:)
  complex(10) :: res
  res = reduce(a, red_complex10)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex10

pure function red_complex16(a,b)
  complex(16), intent(in) :: a, b
  complex(16) :: red_complex16
  red_complex16 = a + b
end function

subroutine complex16(a)
  complex(16), intent(in) :: a(:)
  complex(16) :: res
  res = reduce(a, red_complex16)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex16

pure function red_log1(a,b)
  logical(1), intent(in) :: a, b
  logical(1) :: red_log1
  red_log1 = a .and. b
end function

subroutine log1(a)
  logical(1), intent(in) :: a(:)
  logical(1) :: res
  res = reduce(a, red_log1)
end subroutine

! CHECK: fir.call @_FortranAReduceLogical1Ref

pure function red_log2(a,b)
  logical(2), intent(in) :: a, b
  logical(2) :: red_log2
  red_log2 = a .and. b
end function

subroutine log2(a)
  logical(2), intent(in) :: a(:)
  logical(2) :: res
  res = reduce(a, red_log2)
end subroutine

! CHECK: fir.call @_FortranAReduceLogical2Ref

pure function red_log4(a,b)
  logical(4), intent(in) :: a, b
  logical(4) :: red_log4
  red_log4 = a .and. b
end function

subroutine log4(a)
  logical(4), intent(in) :: a(:)
  logical(4) :: res
  res = reduce(a, red_log4)
end subroutine

! CHECK: fir.call @_FortranAReduceLogical4Ref

pure function red_log8(a,b)
  logical(8), intent(in) :: a, b
  logical(8) :: red_log8
  red_log8 = a .and. b
end function

subroutine log8(a)
  logical(8), intent(in) :: a(:)
  logical(8) :: res
  res = reduce(a, red_log8)
end subroutine

! CHECK: fir.call @_FortranAReduceLogical8Ref

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
end subroutine

! CHECK: fir.call @_FortranAReduceInteger1DimRef

subroutine integer2dim(a, id)
  integer(2), intent(in) :: a(:,:)
  integer(2), allocatable :: res(:)

  res = reduce(a, red_int2, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceInteger2DimRef

subroutine integer4dim(a, id)
  integer(4), intent(in) :: a(:,:)
  integer(4), allocatable :: res(:)

  res = reduce(a, red_int4, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceInteger4DimRef

subroutine integer8dim(a, id)
  integer(8), intent(in) :: a(:,:)
  integer(8), allocatable :: res(:)

  res = reduce(a, red_int8, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceInteger8DimRef

subroutine integer16dim(a, id)
  integer(16), intent(in) :: a(:,:)
  integer(16), allocatable :: res(:)

  res = reduce(a, red_int16, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceInteger16DimRef

subroutine real2dim(a, id)
  real(2), intent(in) :: a(:,:)
  real(2), allocatable :: res(:)

  res = reduce(a, red_real2, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceReal2DimRef

subroutine real3dim(a, id)
  real(3), intent(in) :: a(:,:)
  real(3), allocatable :: res(:)

  res = reduce(a, red_real3, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceReal3DimRef

subroutine real4dim(a, id)
  real(4), intent(in) :: a(:,:)
  real(4), allocatable :: res(:)

  res = reduce(a, red_real4, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceReal4DimRef

subroutine real8dim(a, id)
  real(8), intent(in) :: a(:,:)
  real(8), allocatable :: res(:)

  res = reduce(a, red_real8, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceReal8DimRef

subroutine real10dim(a, id)
  real(10), intent(in) :: a(:,:)
  real(10), allocatable :: res(:)

  res = reduce(a, red_real10, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceReal10DimRef

subroutine real16dim(a, id)
  real(16), intent(in) :: a(:,:)
  real(16), allocatable :: res(:)

  res = reduce(a, red_real16, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceReal16DimRef

subroutine complex2dim(a, id)
  complex(2), intent(in) :: a(:,:)
  complex(2), allocatable :: res(:)

  res = reduce(a, red_complex2, 2)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex2Dim

subroutine complex3dim(a, id)
  complex(3), intent(in) :: a(:,:)
  complex(3), allocatable :: res(:)

  res = reduce(a, red_complex3, 2)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex3Dim

subroutine complex4dim(a, id)
  complex(4), intent(in) :: a(:,:)
  complex(4), allocatable :: res(:)

  res = reduce(a, red_complex4, 2)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex4Dim

subroutine complex8dim(a, id)
  complex(8), intent(in) :: a(:,:)
  complex(8), allocatable :: res(:)

  res = reduce(a, red_complex8, 2)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex8Dim

subroutine complex10dim(a, id)
  complex(10), intent(in) :: a(:,:)
  complex(10), allocatable :: res(:)

  res = reduce(a, red_complex10, 2)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex10Dim

subroutine complex16dim(a, id)
  complex(16), intent(in) :: a(:,:)
  complex(16), allocatable :: res(:)

  res = reduce(a, red_complex16, 2)
end subroutine

! CHECK: fir.call @_FortranACppReduceComplex16Dim

subroutine logical1dim(a, id)
  logical(1), intent(in) :: a(:,:)
  logical(1), allocatable :: res(:)

  res = reduce(a, red_log1, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceLogical1DimRef

subroutine logical2dim(a, id)
  logical(2), intent(in) :: a(:,:)
  logical(2), allocatable :: res(:)

  res = reduce(a, red_log2, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceLogical2DimRef

subroutine logical4dim(a, id)
  logical(4), intent(in) :: a(:,:)
  logical(4), allocatable :: res(:)

  res = reduce(a, red_log4, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceLogical4DimRef

subroutine logical8dim(a, id)
  logical(8), intent(in) :: a(:,:)
  logical(8), allocatable :: res(:)

  res = reduce(a, red_log8, 2)
end subroutine

! CHECK: fir.call @_FortranAReduceLogical8DimRef

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
