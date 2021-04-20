! RUN: bbc -emit-fir %s -o - | FileCheck %s

! ABS
! CHECK-LABEL: abs_testi
subroutine abs_testi(a, b)
  integer :: a, b
  ! CHECK: shift_right_signed
  ! CHECK: xor
  ! CHECK: subi
  b = abs(a)
end subroutine

! CHECK-LABEL: abs_testr
subroutine abs_testr(a, b)
  real :: a, b
  ! CHECK: fir.call @llvm.fabs.f32
  b = abs(a)
end subroutine

! CHECK-LABEL: abs_testz
subroutine abs_testz(a, b)
  complex :: a
  real :: b
  ! CHECK: fir.extract_value
  ! CHECK: fir.extract_value
  ! CHECK: fir.call @{{.*}}hypot
  b = abs(a)
end subroutine

! AIMAG
! CHECK-LABEL: aimag_test
subroutine aimag_test(a, b)
  complex :: a
  real :: b
  ! CHECK: fir.extract_value
  b = aimag(a)
end subroutine

! AINT
! CHECK-LABEL: aint_test
subroutine aint_test(a, b)
  real :: a, b
  ! CHECK: fir.call @llvm.trunc.f32
  b = aint(a)
end subroutine

! ALL
! CHECK-LABEL: all_test
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.logical<4>
logical function all_test(mask)
  logical :: mask(:)
! CHECK: %[[c1:.*]] = constant 1 : index
! CHECK: %[[a1:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
! CHECK: %[[a2:.*]] = fir.convert %[[c1]] : (index) -> i32
  all_test = all(mask)
! CHECK:  %[[a3:.*]] = fir.call @_FortranAAll(%[[a1]], %{{.*}}, %{{.*}}, %[[a2]]) : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> i1
end function all_test 

! ALL
! CHECK-LABEL: all_test2
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>
! CHECK-SAME: %[[arg1:.*]]: !fir.ref<i32>
! CHECK-SAME: %[[arg2:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
subroutine all_test2(mask, d, rslt)
  logical :: mask(:,:)
  integer :: d
  logical :: rslt(:)
! CHECK-DAG:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
! CHECK-DAG:  %[[a1:.*]] = fir.load %[[arg1:.*]] : !fir.ref<i32>
! CHECK-DAG:  %[[a6:.*]] = fir.convert %[[a0:.*]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG:  %[[a7:.*]] = fir.convert %[[arg0:.*]]: (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.box<none>
  rslt = all(mask, d)
! CHECK:  %[[r1:.*]] = fir.call @_FortranAAllDim(%[[a6:.*]], %[[a7:.*]], %[[a1:.*]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
! CHECK-DAG:  %[[a10:.*]] = fir.load %[[a0:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
! CHECK-DAG:  %[[a12:.*]] = fir.box_addr %[[a10:.*]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>) -> !fir.heap<!fir.array<?x!fir.logical<4>>>
! CHECK-DAG  fir.freemem %[[a12:.*]] : !fir.heap<!fir.array<?x!fir.logical<4>>>
end subroutine

! ALLOCATED
! CHECK-LABEL: allocated_test
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.heap<f32>>>,
! CHECK-SAME: %[[arg1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
subroutine allocated_test(scalar, array)
  real, allocatable  :: scalar, array(:)
  ! CHECK: %[[scalar:.*]] = fir.load %[[arg0]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! CHECK: %[[addr0:.*]] = fir.box_addr %[[scalar]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  ! CHECK: %[[addrToInt0:.*]] = fir.convert %[[addr0]]
  ! CHECK: cmpi ne, %[[addrToInt0]], %c0{{.*}}
  print *, allocated(scalar)
  ! CHECK: %[[array:.*]] = fir.load %[[arg1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  ! CHECK: %[[addr1:.*]] = fir.box_addr %[[array]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
  ! CHECK: %[[addrToInt1:.*]] = fir.convert %[[addr1]]
  ! CHECK: cmpi ne, %[[addrToInt1]], %c0{{.*}}
  print *, allocated(array)
end subroutine

! ANINT
! CHECK-LABEL: anint_test
subroutine anint_test(a, b)
  real :: a, b
  ! CHECK: fir.call @llvm.round.f32
  b = anint(a)
end subroutine

! ANY
! CHECK-LABEL: any_test
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.logical<4>
logical function any_test(mask)
  logical :: mask(:)
! CHECK: %[[c1:.*]] = constant 1 : index
! CHECK: %[[a1:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
! CHECK: %[[a2:.*]] = fir.convert %[[c1]] : (index) -> i32
  any_test = any(mask)
! CHECK:  %[[a3:.*]] = fir.call @_FortranAAny(%[[a1]], %{{.*}},  %{{.*}}, %[[a2]]) : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> i1
end function any_test 

! ANY
! CHECK-LABEL: any_test2
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>
! CHECK-SAME: %[[arg1:.*]]: !fir.ref<i32>
! CHECK-SAME: %[[arg2:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
subroutine any_test2(mask, d, rslt)
  logical :: mask(:,:)
  integer :: d
  logical :: rslt(:)
! CHECK-DAG:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
! CHECK-DAG:  %[[a1:.*]] = fir.load %[[arg1:.*]] : !fir.ref<i32>
! CHECK-DAG:  %[[a6:.*]] = fir.convert %[[a0:.*]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG:  %[[a7:.*]] = fir.convert %[[arg0:.*]]: (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.box<none>
  rslt = any(mask, d)
! CHECK:  %[[r1:.*]] = fir.call @_FortranAAnyDim(%[[a6:.*]], %[[a7:.*]], %[[a1:.*]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
! CHECK-DAG:  %[[a10:.*]] = fir.load %[[a0:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>>
! CHECK-DAG:  %[[a12:.*]] = fir.box_addr %[[a10:.*]] : (!fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>) -> !fir.heap<!fir.array<?x!fir.logical<4>>>
! CHECK-DAG  fir.freemem %[[a12:.*]] : !fir.heap<!fir.array<?x!fir.logical<4>>>
end subroutine

! ASSOCIATED
! CHECK-LABEL: associated_test
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.box<!fir.ptr<f32>>>,
! CHECK-SAME: %[[arg1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
subroutine associated_test(scalar, array)
  real, pointer  :: scalar, array(:)
  ! CHECK: %[[scalar:.*]] = fir.load %[[arg0]] : !fir.ref<!fir.box<!fir.ptr<f32>>>
  ! CHECK: %[[addr0:.*]] = fir.box_addr %[[scalar]] : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  ! CHECK: %[[addrToInt0:.*]] = fir.convert %[[addr0]]
  ! CHECK: cmpi ne, %[[addrToInt0]], %c0{{.*}}
  print *, associated(scalar)
  ! CHECK: %[[array:.*]] = fir.load %[[arg1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK: %[[addr1:.*]] = fir.box_addr %[[array]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
  ! CHECK: %[[addrToInt1:.*]] = fir.convert %[[addr1]]
  ! CHECK: cmpi ne, %[[addrToInt1]], %c0{{.*}}
  print *, associated(array)
end subroutine

! DBLE
! CHECK-LABEL: dble_test
subroutine dble_test(a)
  real :: a
  ! CHECK: fir.convert {{.*}} : (f32) -> f64
  print *, dble(a)
end subroutine

! DIM
! CHECK-LABEL: dim_testr
subroutine dim_testr(x, y, z)
  real :: x, y, z
  ! CHECK-DAG: %[[x:.*]] = fir.load %arg0
  ! CHECK-DAG: %[[y:.*]] = fir.load %arg1
  ! CHECK-DAG: %[[zero:.*]] = constant 0.0
  ! CHECK-DAG: %[[diff:.*]] = subf %[[x]], %[[y]]
  ! CHECK: %[[cmp:.*]] = fir.cmpf "ogt", %[[diff]], %[[zero]]
  ! CHECK: %[[res:.*]] = select %[[cmp]], %[[diff]], %[[zero]]
  ! CHECK: fir.store %[[res]] to %arg2
  z = dim(x, y)
end subroutine
! CHECK-LABEL: dim_testi
subroutine dim_testi(i, j, k)
  integer :: i, j, k
  ! CHECK-DAG: %[[i:.*]] = fir.load %arg0
  ! CHECK-DAG: %[[j:.*]] = fir.load %arg1
  ! CHECK-DAG: %[[zero:.*]] = constant 0
  ! CHECK-DAG: %[[diff:.*]] = subi %[[i]], %[[j]]
  ! CHECK: %[[cmp:.*]] = cmpi sgt, %[[diff]], %[[zero]]
  ! CHECK: %[[res:.*]] = select %[[cmp]], %[[diff]], %[[zero]]
  ! CHECK: fir.store %[[res]] to %arg2
  k = dim(i, j)
end subroutine

! DPROD
! CHECK-LABEL: dprod_test
subroutine dprod_test (x, y, z)
  real :: x,y
  double precision :: z
  z = dprod(x,y)
  ! CHECK-DAG: %[[x:.*]] = fir.load %arg0
  ! CHECK-DAG: %[[y:.*]] = fir.load %arg1
  ! CHECK-DAG: %[[a:.*]] = fir.convert %[[x]] : (f32) -> f64
  ! CHECK-DAG: %[[b:.*]] = fir.convert %[[y]] : (f32) -> f64
  ! CHECK: %[[res:.*]] = mulf %[[a]], %[[b]]
  ! CHECK: fir.store %[[res]] to %arg2
end subroutine

! CEILING
! CHECK-LABEL: ceiling_test1
subroutine ceiling_test1(i, a)
  integer :: i
  real :: a
  i = ceiling(a)
  ! CHECK: %[[f:.*]] = fir.call @llvm.ceil.f32
  ! CHECK: fir.convert %[[f]] : (f32) -> i32
end subroutine
! CHECK-LABEL: ceiling_test2
subroutine ceiling_test2(i, a)
  integer(8) :: i
  real :: a
  i = ceiling(a, 8)
  ! CHECK: %[[f:.*]] = fir.call @llvm.ceil.f32
  ! CHECK: fir.convert %[[f]] : (f32) -> i64
end subroutine


! CONJG
! CHECK-LABEL: conjg_test
subroutine conjg_test(z1, z2)
  complex :: z1, z2
  ! CHECK: fir.extract_value
  ! CHECK: fir.negf
  ! CHECK: fir.insert_value
  z2 = conjg(z1)
end subroutine

! FLOOR
! CHECK-LABEL: floor_test1
subroutine floor_test1(i, a)
  integer :: i
  real :: a
  i = floor(a)
  ! CHECK: %[[f:.*]] = fir.call @llvm.floor.f32
  ! CHECK: fir.convert %[[f]] : (f32) -> i32
end subroutine
! CHECK-LABEL: floor_test2
subroutine floor_test2(i, a)
  integer(8) :: i
  real :: a
  i = floor(a, 8)
  ! CHECK: %[[f:.*]] = fir.call @llvm.floor.f32
  ! CHECK: fir.convert %[[f]] : (f32) -> i64
end subroutine

! IABS
! CHECK-LABEL: iabs_test
subroutine iabs_test(a, b)
  integer :: a, b
  ! CHECK: shift_right_signed
  ! CHECK: xor
  ! CHECK: subi
  b = iabs(a)
end subroutine

! IABS - Check if the return type (RT) has default kind.
! CHECK-LABEL: iabs_test
subroutine iabs_testRT(a, b)
  integer(KIND=4) :: a
  integer(KIND=16) :: b
  ! CHECK: shift_right_signed
  ! CHECK: xor
  ! CHECK: %[[RT:.*]] =  subi
  ! CHECK: fir.convert %[[RT]] : (i32)
  b = iabs(a)
end subroutine

! IAND
! CHECK-LABEL: iand_test
subroutine iand_test(a, b)
  integer :: a, b
  print *, iand(a, b)
  ! CHECK: %{{[0-9]+}} = and %{{[0-9]+}}, %{{[0-9]+}} : i{{(8|16|32|64|128)}}
end subroutine iand_test

! ICHAR
! CHECK-LABEL: ichar_test
subroutine ichar_test(c)
  character(1) :: c
  character :: str(10)
  ! CHECK-DAG: %[[unbox:.*]]:2 = fir.unboxchar
  ! CHECK-DAG: %[[J:.*]] = fir.alloca i32 {name = "{{.*}}Ej"}
  ! CHECK-DAG: %[[STR:.*]] = fir.alloca !fir.array{{.*}} {name = "{{.*}}Estr"}
  ! CHECK: %[[BOX:.*]] = fir.convert %[[unbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1>>
  ! CHECK: %[[CHAR:.*]] = fir.load %[[BOX]] : !fir.ref<!fir.char<1>>
  ! CHECK: fir.extract_value %[[CHAR]], %c0{{.*}}
  print *, ichar(c)
  ! CHECK: fir.call @{{.*}}EndIoStatement

  ! CHECK-DAG: %{{.*}} = fir.load %[[J]] : !fir.ref<i32>
  ! CHECK: %[[ptr:.*]] = fir.coordinate_of %[[STR]], %
  ! CHECK: %[[VAL:.*]] = fir.load %[[ptr]] : !fir.ref<!fir.char<1>>
  ! CHECK: fir.extract_value %[[VAL]], %c0{{.*}}
  print *, ichar(str(J))
  ! CHECK: fir.call @{{.*}}EndIoStatement
end subroutine

! IEOR
! CHECK-LABEL: ieor_test
subroutine ieor_test(a, b)
  integer :: a, b
  print *, ieor(a, b)
  ! CHECK: %{{[0-9]+}} = xor %{{[0-9]+}}, %{{[0-9]+}} : i{{(8|16|32|64|128)}}
end subroutine ieor_test

! INDEX
! CHECK-LABEL: func @_QPindex_test(%
! CHECK-SAME: [[s:[^:]+]]: !fir.boxchar<1>, %
! CHECK-SAME: [[ss:[^:]+]]: !fir.boxchar<1>) -> i32
integer function index_test(s1, s2)
  character(*) :: s1, s2
  ! CHECK: %[[st:[^:]*]]:2 = fir.unboxchar %[[s]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[sst:[^:]*]]:2 = fir.unboxchar %[[ss]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[a1:.*]] = fir.convert %[[st]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[a2:.*]] = fir.convert %[[st]]#1 : (index) -> i64
  ! CHECK: %[[a3:.*]] = fir.convert %[[sst]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[a4:.*]] = fir.convert %[[sst]]#1 : (index) -> i64
  ! CHECK: = fir.call @_FortranAIndex1(%[[a1]], %[[a2]], %[[a3]], %[[a4]], %{{.*}}) : (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
  index_test = index(s1, s2)
end function index_test

! CHECK-LABEL: func @_QPindex_test2(%
! CHECK-SAME: [[s:[^:]+]]: !fir.boxchar<1>, %
! CHECK-SAME: [[ss:[^:]+]]: !fir.boxchar<1>) -> i32
integer function index_test2(s1, s2)
  character(*) :: s1, s2
  ! CHECK: %[[mut:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
  ! CHECK: %[[st:[^:]*]]:2 = fir.unboxchar %[[s]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[sst:[^:]*]]:2 = fir.unboxchar %[[ss]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[sb:.*]] = fir.embox %[[st]]#0 typeparams %[[st]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
  ! CHECK: %[[ssb:.*]] = fir.embox %[[sst]]#0 typeparams %[[sst]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
  ! CHECK: %[[back:.*]] = fir.embox %{{.*}} : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
  ! CHECK: %[[hb:.*]] = fir.embox %{{.*}} : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  ! CHECK: %[[a0:.*]] = fir.convert %[[mut]] : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: %[[a1:.*]] = fir.convert %[[sb]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
  ! CHECK: %[[a2:.*]] = fir.convert %[[ssb]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
  ! CHECK: %[[a3:.*]] = fir.convert %[[back]] : (!fir.box<!fir.logical<4>>) -> !fir.box<none>
  ! CHECK: %[[a5:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK:  fir.call @_FortranAIndex(%[[a0]], %[[a1]], %[[a2]], %[[a3]], %{{.*}}, %[[a5]], %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
  index_test2 = index(s1, s2, .true., 4)
  ! CHECK: %[[ld1:.*]] = fir.load %[[mut]] : !fir.ref<!fir.box<!fir.heap<i32>>>
  ! CHECK: %[[ad1:.*]] = fir.box_addr %[[ld1]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
  ! CHECK: %[[ld2:.*]] = fir.load %[[ad1]] : !fir.heap<i32>
  ! CHECK: fir.freemem %[[ad1]]
end function index_test2

! IOR
! CHECK-LABEL: ior_test
subroutine ior_test(a, b)
  integer :: a, b
  print *, ior(a, b)
  ! CHECK: %{{[0-9]+}} = or %{{[0-9]+}}, %{{[0-9]+}} : i{{(8|16|32|64|128)}}
end subroutine ior_test

! LEN
! CHECK-LABEL: len_test
subroutine len_test(i, c)
  integer :: i
  character(*) :: c
  ! CHECK: %[[c:.*]]:2 = fir.unboxchar %arg1
  ! CHECK: %[[xx:.*]] = fir.convert %[[c]]#1 : (index) -> i64
  ! CHECK: %[[x:.*]] = fir.convert %[[xx]] : (i64) -> i32
  ! CHECK: fir.store %[[x]] to %arg0
  i = len(c)
end subroutine

! LEN_TRIM
! CHECK-LABEL: len_trim_test
integer function len_trim_test(c)
  character(*) :: c
  ltrim = len_trim(c)
  ! CHECK-DAG: %[[c0:.*]] = constant 0 : index
  ! CHECK-DAG: %[[c1:.*]] = constant 1 : index
  ! CHECK-DAG: %[[cm1:.*]] = constant -1 : index
  ! CHECK-DAG: %[[lastChar:.*]] = subi {{.*}}, %[[c1]]
  ! CHECK: %[[iterateResult:.*]]:2 = fir.iterate_while (%[[index:.*]] = %[[lastChar]] to %[[c0]] step %[[cm1]]) and ({{.*}}) iter_args({{.*}}) {
    ! CHECK: %[[addr:.*]] = fir.coordinate_of {{.*}}, %[[index]]
    ! CHECK: %[[codeAddr:.*]] = fir.convert %[[addr]]
    ! CHECK: %[[code:.*]] = fir.load %[[codeAddr]]
    ! CHECK: %[[bool:.*]] = cmpi eq
    ! CHECK: fir.result %[[bool]], %[[index]]
  ! CHECK: }
  ! CHECK: %[[len:.*]] = addi %[[iterateResult]]#1, %[[c1]]
  ! CHECK: select %[[iterateResult]]#0, %[[c0]], %[[len]]
end function

! MAXLOC
! CHECK-LABEL: maxloc_test
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi32>>,
! CHECK-SAME: %[[arg1:.*]]: !fir.box<!fir.array<?xi32>>
subroutine maxloc_test(arr,res)
  integer :: arr(:)
  integer :: res(:)
! CHECK-DAG:  %[[c4:.*]] = constant 4 : index
! CHECK-DAG: %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK-DAG: %[[a1:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG:  %[[a6:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG:  %[[a7:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK-DAG:  %[[a8:.*]] = fir.convert %[[c4]] : (index) -> i32
! CHECK-DAG:  %[[a10:.*]] = fir.convert %[[a1]] : (!fir.box<i1>) -> !fir.box<none>
  res = maxloc(arr)
! CHECK:  %{{.*}} = fir.call @_FortranAMaxloc(%[[a6]], %[[a7]], %[[a8]], %{{.*}}, %{{.*}}, %[[a10]], %false) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
! CHECK-DAG:  %[[a12:.*]] = fir.load %[[a0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK-DAG:  %[[a14:.*]] = fir.box_addr %[[a12]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK-DAG:  fir.freemem %[[a14]] : !fir.heap<!fir.array<?xi32>>
end subroutine

! MAXLOC
! CHECK-LABEL: maxloc_test2
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi32>>, %[[arg1:.*]]: !fir.box<!fir.array<?xi32>>, %[[arg2:.*]]: !fir.ref<i32>
subroutine maxloc_test2(arr,res,d)
  integer :: arr(:) 
  integer :: res(:)
  integer :: d
! CHECK-DAG:  %[[c4:.*]] = constant 4 : index
! CHECK-DAG:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
! CHECK-DAG:  %[[a1:.*]] = fir.load %arg2 : !fir.ref<i32>
! CHECK-DAG:  %[[a2:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG:  %[[a6:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG:  %[[a7:.*]] = fir.convert %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK-DAG:  %[[a8:.*]] = fir.convert %[[c4]] : (index) -> i32
! CHECK-DAG:  %[[a10:.*]] = fir.convert %[[a2]] : (!fir.box<i1>) -> !fir.box<none>
  res = maxloc(arr, dim=d)
! CHECK:  %{{.*}} = fir.call @_FortranAMaxlocDim(%[[a6]], %[[a7]], %[[a8]], %[[a1]], %{{.*}}, %{{.*}}, %[[a10]], %false) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
! CHECK:  %[[a12:.*]] = fir.load %0 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:  %[[a13:.*]] = fir.box_addr %[[a12]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:  fir.freemem %[[a13]] : !fir.heap<i32>
end subroutine

! MINLOC
! CHECK-LABEL: minloc_test
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi32>>,
! CHECK-SAME: %[[arg1:.*]]: !fir.box<!fir.array<?xi32>>
subroutine minloc_test(arr,res)
  integer :: arr(:)
  integer :: res(:)
! CHECK-DAG:  %[[c4:.*]] = constant 4 : index
! CHECK-DAG: %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK-DAG: %[[a1:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG:  %[[a6:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG:  %[[a7:.*]] = fir.convert %[[arg0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK-DAG:  %[[a8:.*]] = fir.convert %[[c4]] : (index) -> i32
! CHECK-DAG:  %[[a10:.*]] = fir.convert %[[a1]] : (!fir.box<i1>) -> !fir.box<none>
  res = minloc(arr)
! CHECK:  %{{.*}} = fir.call @_FortranAMinloc(%[[a6]], %[[a7]], %[[a8]], %{{.*}}, %{{.*}}, %[[a10]], %false) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
! CHECK-DAG:  %[[a12:.*]] = fir.load %[[a0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK-DAG:  %[[a14:.*]] = fir.box_addr %[[a12]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK-DAG:  fir.freemem %[[a14]] : !fir.heap<!fir.array<?xi32>>
end subroutine

! MINLOC
! CHECK-LABEL: minloc_test2
! CHECK-SAME: %[[arg0:.*]]: !fir.box<!fir.array<?xi32>>, %[[arg1:.*]]: !fir.box<!fir.array<?xi32>>, %[[arg2:.*]]: !fir.ref<i32>
subroutine minloc_test2(arr,res,d)
  integer :: arr(:) 
  integer :: res(:)
  integer :: d
! CHECK-DAG:  %[[c4:.*]] = constant 4 : index
! CHECK-DAG:  %[[a0:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
! CHECK-DAG:  %[[a1:.*]] = fir.load %arg2 : !fir.ref<i32>
! CHECK-DAG:  %[[a2:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG:  %[[a6:.*]] = fir.convert %[[a0]] : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK-DAG:  %[[a7:.*]] = fir.convert %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
! CHECK-DAG:  %[[a8:.*]] = fir.convert %[[c4]] : (index) -> i32
! CHECK-DAG:  %[[a10:.*]] = fir.convert %[[a2]] : (!fir.box<i1>) -> !fir.box<none>
  res = minloc(arr, dim=d)
! CHECK:  %{{.*}} = fir.call @_FortranAMinlocDim(%[[a6]], %[[a7]], %[[a8]], %[[a1]], %{{.*}}, %{{.*}}, %[[a10]], %false) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
! CHECK:  %[[a12:.*]] = fir.load %0 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:  %[[a13:.*]] = fir.box_addr %[[a12]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:  fir.freemem %[[a13]] : !fir.heap<i32>
end subroutine

! NINT
! CHECK-LABEL: nint_test1
subroutine nint_test1(i, a)
  integer :: i
  real :: a
  i = nint(a)
  ! CHECK: fir.call @llvm.lround.i32.f32
end subroutine
! CHECK-LABEL: nint_test2
subroutine nint_test2(i, a)
  integer(8) :: i
  real(8) :: a
  i = nint(a, 8)
  ! CHECK: fir.call @llvm.lround.i64.f64
end subroutine

! MODULO
! CHECK-LABEL: modulo_testr
! CHECK-SAME: (%[[arg0:.*]]: !fir.ref<f64>, %[[arg1:.*]]: !fir.ref<f64>, %[[arg2:.*]]: !fir.ref<f64>)
subroutine modulo_testr(r, a, p)
  real(8) :: r, a, p
  ! CHECK-DAG: %[[a:.*]] = fir.load %[[arg1]] : !fir.ref<f64>
  ! CHECK-DAG: %[[p:.*]] = fir.load %[[arg2]] : !fir.ref<f64>
  ! CHECK-DAG: %[[rem:.*]] = remf %[[a]], %[[p]] : f64
  ! CHECK-DAG: %[[zero:.*]] = constant 0.000000e+00 : f64
  ! CHECK-DAG: %[[remNotZero:.*]] = cmpf une, %[[rem]], %[[zero]] : f64
  ! CHECK-DAG: %[[aNeg:.*]] = cmpf olt, %[[a]], %[[zero]] : f64
  ! CHECK-DAG: %[[pNeg:.*]] = cmpf olt, %[[p]], %[[zero]] : f64
  ! CHECK-DAG: %[[signDifferent:.*]] = xor %[[aNeg]], %[[pNeg]] : i1
  ! CHECK-DAG: %[[mustAddP:.*]] = and %[[remNotZero]], %[[signDifferent]] : i1
  ! CHECK-DAG: %[[remPlusP:.*]] = addf %[[rem]], %[[p]] : f64
  ! CHECK: %[[res:.*]] = select %[[mustAddP]], %[[remPlusP]], %[[rem]] : f64
  ! CHECK: fir.store %[[res]] to %[[arg0]] : !fir.ref<f64>
  r = modulo(a, p)
end subroutine

! CHECK-LABEL: modulo_testi
! CHECK-SAME: (%[[arg0:.*]]: !fir.ref<i64>, %[[arg1:.*]]: !fir.ref<i64>, %[[arg2:.*]]: !fir.ref<i64>)
subroutine modulo_testi(r, a, p)
  integer(8) :: r, a, p
  ! CHECK-DAG: %[[a:.*]] = fir.load %[[arg1]] : !fir.ref<i64>
  ! CHECK-DAG: %[[p:.*]] = fir.load %[[arg2]] : !fir.ref<i64>
  ! CHECK-DAG: %[[rem:.*]] = remi_signed %[[a]], %[[p]] : i64
  ! CHECK-DAG: %[[argXor:.*]] = xor %[[a]], %[[p]] : i64
  ! CHECK-DAG: %[[signDifferent:.*]] = cmpi slt, %[[argXor]], %c0{{.*}} : i64
  ! CHECK-DAG: %[[remNotZero:.*]] = cmpi ne, %[[rem]], %c0{{.*}} : i64
  ! CHECK-DAG: %[[mustAddP:.*]] = and %[[remNotZero]], %[[signDifferent]] : i1
  ! CHECK-DAG: %[[remPlusP:.*]] = addi %[[rem]], %[[p]] : i64
  ! CHECK: %[[res:.*]] = select %[[mustAddP]], %[[remPlusP]], %[[rem]] : i64
  ! CHECK: fir.store %[[res]] to %[[arg0]] : !fir.ref<i64>
  r = modulo(a, p)
end subroutine

! RRSPACING
! CHECK-LABEL: rrspacing_test
! CHECK-SAME: [[x:[^:]+]]: !fir.ref<f32>) -> f32
real*4 function rrspacing_test(x)
  real*4 :: x
  rrspacing_test = rrspacing(x)
!CHECK %[[a1:.*]] = fir.load %[[x]] : !fir.ref<f32>
!CHECK %{{.*}} = fir.call @_FortranARRSpacing4(%[[a1]]) : (f32) -> f32
end function

! RRSPACING
! CHECK-LABEL: rrspacing_test2
! CHECK-SAME: [[x:[^:]+]]: !fir.ref<f128>) -> f128
real*16 function rrspacing_test2(x)
  real*16 :: x
  rrspacing_test2 = spacing(x)
!CHECK %[[a1:.*]] = fir.load %[[x]] : !fir.ref<f128>
!CHECK %{{.*}} = fir.call @_FortranARRSpacing16(%[[a1]]) : (f128) -> f128
end function

! SCAN 
! CHECK-LABEL: func @_QPscan_test(%
! CHECK-SAME: [[s:[^:]+]]: !fir.boxchar<1>, %
! CHECK-SAME: [[ss:[^:]+]]: !fir.boxchar<1>) -> i32 
integer function scan_test(s1, s2)
  character(*) :: s1, s2
! CHECK: %[[tmpBox:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
! CHECK-DAG: %[[c:.*]]:2 = fir.unboxchar %[[s]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-DAG: %[[cBox:.*]] = fir.embox %[[c]]#0 typeparams %[[c]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-DAG: %[[cBoxNone:.*]] = fir.convert %[[cBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-DAG: %[[c2:.*]]:2 = fir.unboxchar %[[ss]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-DAG: %[[cBox2:.*]] = fir.embox %[[c2]]#0 typeparams %[[c2]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-DAG: %[[cBoxNone2:.*]] = fir.convert %[[cBox2]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-DAG: %[[backOptBox:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG: %[[backBox:.*]] = fir.convert %[[backOptBox]] : (!fir.box<i1>) -> !fir.box<none>
! CHECK-DAG: %[[kindConstant:.*]] = constant 4 : i32
! CHECK-DAG: %[[resBox:.*]] = fir.convert %[[tmpBox:.*]] : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK: fir.call @{{.*}}Scan(%[[resBox]], %[[cBoxNone]], %[[cBoxNone2]], %[[backBox]], %[[kindConstant]], {{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
  scan_test = scan(s1, s2, kind=4)
! CHECK-DAG: %[[tmpAddr:.*]] = fir.box_addr
! CHECK: fir.freemem %[[tmpAddr]] : !fir.heap<i32>
end function scan_test

! SCAN 
! CHECK-LABEL: func @_QPscan_test2(%
! CHECK-SAME: [[s:[^:]+]]: !fir.boxchar<1>, %
! CHECK-SAME: [[ss:[^:]+]]: !fir.boxchar<1>) -> i32
integer function scan_test2(s1, s2)
  character(*) :: s1, s2
  ! CHECK: %[[st:[^:]*]]:2 = fir.unboxchar %[[s]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[sst:[^:]*]]:2 = fir.unboxchar %[[ss]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[a1:.*]] = fir.convert %[[st]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[a2:.*]] = fir.convert %[[st]]#1 : (index) -> i64
  ! CHECK: %[[a3:.*]] = fir.convert %[[sst]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[a4:.*]] = fir.convert %[[sst]]#1 : (index) -> i64
  ! CHECK: = fir.call @_FortranAScan1(%[[a1]], %[[a2]], %[[a3]], %[[a4]], %{{.*}}) : (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
  scan_test2 = scan(s1, s2, .true.)
end function scan_test2

! SIGN
! CHECK-LABEL: sign_testi
subroutine sign_testi(a, b, c)
  integer a, b, c
  ! CHECK: shift_right_signed
  ! CHECK: xor
  ! CHECK: subi
  ! CHECK-DAG: subi
  ! CHECK-DAG: cmpi slt
  ! CHECK: select
  c = sign(a, b)
end subroutine

! CHECK-LABEL: sign_testr
subroutine sign_testr(a, b, c)
  real a, b, c
  ! CHECK-DAG: fir.call {{.*}}fabs
  ! CHECK-DAG: fir.negf
  ! CHECK-DAG: fir.cmpf "olt"
  ! CHECK: select
  c = sign(a, b)
end subroutine

! SPACING
! CHECK-LABEL: spacing_test
! CHECK-SAME: [[x:[^:]+]]: !fir.ref<f32>) -> f32
real*4 function spacing_test(x)
  real*4 :: x
  spacing_test = spacing(x)
!CHECK %[[a1:.*]] = fir.load %[[x]] : !fir.ref<f32>
!CHECK %{{.*}} = fir.call @_FortranASpacing4(%[[a1]]) : (f32) -> f32
end function

! SPACING
! CHECK-LABEL: spacing_test2
! CHECK-SAME: [[x:[^:]+]]: !fir.ref<f80>) -> f80
real*10 function spacing_test2(x)
  real*10 :: x
  spacing_test2 = spacing(x)
!CHECK %[[a1:.*]] = fir.load %[[x]] : !fir.ref<f80>
!CHECK %{{.*}} = fir.call @_FortranASpacing10(%[[a1]]) : (f80) -> f80
end function

! SQRT
! CHECK-LABEL: sqrt_testr
subroutine sqrt_testr(a, b)
  real :: a, b
  ! CHECK: fir.call {{.*}}sqrt
  b = sqrt(a)
end subroutine

! TRIM
! CHECK-LABEL: trim_test
! CHECK-SAME: (%[[arg0:.*]]: !fir.boxchar<1>)
subroutine trim_test(c)
  character(*) :: c
  ! CHECK: %[[tmpBox:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
  ! CHECK-DAG: %[[c:.*]]:2 = fir.unboxchar %[[arg0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK-DAG: %[[cBox:.*]] = fir.embox %[[c]]#0 typeparams %[[c]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
  ! CHECK-DAG: %[[cBoxNone:.*]] = fir.convert %[[cBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
  ! CHECK-DAG: %[[resBox:.*]] = fir.convert %[[tmpBox]] : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @{{.*}}Trim(%[[resBox]], %[[cBoxNone]], {{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  ! CHECK-DAG: %[[tmpAddr:.*]] = fir.box_addr
  ! CHECK-DAG: fir.box_elesize
  ! CHECK: fir.call @{{.*}}bar_trim_test
  call bar_trim_test(trim(c))
  ! CHECK: fir.freemem %[[tmpAddr]] : !fir.heap<!fir.char<1,?>>
  return
end subroutine

! VERIFY 
! CHECK-LABEL: func @_QPverify_test(%
! CHECK-SAME: [[s:[^:]+]]: !fir.boxchar<1>, %
! CHECK-SAME: [[ss:[^:]+]]: !fir.boxchar<1>) -> i32 
integer function verify_test(s1, s2)
  character(*) :: s1, s2
! CHECK: %[[tmpBox:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
! CHECK-DAG: %[[c:.*]]:2 = fir.unboxchar %[[s]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-DAG: %[[cBox:.*]] = fir.embox %[[c]]#0 typeparams %[[c]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-DAG: %[[cBoxNone:.*]] = fir.convert %[[cBox]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-DAG: %[[c2:.*]]:2 = fir.unboxchar %[[ss]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK-DAG: %[[cBox2:.*]] = fir.embox %[[c2]]#0 typeparams %[[c2]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
! CHECK-DAG: %[[cBoxNone2:.*]] = fir.convert %[[cBox2]] : (!fir.box<!fir.char<1,?>>) -> !fir.box<none>
! CHECK-DAG: %[[backOptBox:.*]] = fir.absent !fir.box<i1>
! CHECK-DAG: %[[backBox:.*]] = fir.convert %[[backOptBox]] : (!fir.box<i1>) -> !fir.box<none>
! CHECK-DAG: %[[kindConstant:.*]] = constant 4 : i32
! CHECK-DAG: %[[resBox:.*]] = fir.convert %[[tmpBox:.*]] : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
! CHECK: fir.call @{{.*}}Verify(%[[resBox]], %[[cBoxNone]], %[[cBoxNone2]], %[[backBox]], %[[kindConstant]], {{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
  verify_test = verify(s1, s2, kind=4)
! CHECK-DAG: %[[tmpAddr:.*]] = fir.box_addr
! CHECK: fir.freemem %[[tmpAddr]] : !fir.heap<i32>
end function verify_test

! VERIFY 
! CHECK-LABEL: func @_QPverify_test2(%
! CHECK-SAME: [[s:[^:]+]]: !fir.boxchar<1>, %
! CHECK-SAME: [[ss:[^:]+]]: !fir.boxchar<1>) -> i32
integer function verify_test2(s1, s2)
  character(*) :: s1, s2
  ! CHECK: %[[st:[^:]*]]:2 = fir.unboxchar %[[s]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[sst:[^:]*]]:2 = fir.unboxchar %[[ss]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[a1:.*]] = fir.convert %[[st]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[a2:.*]] = fir.convert %[[st]]#1 : (index) -> i64
  ! CHECK: %[[a3:.*]] = fir.convert %[[sst]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: %[[a4:.*]] = fir.convert %[[sst]]#1 : (index) -> i64
  ! CHECK: = fir.call @_FortranAVerify1(%[[a1]], %[[a2]], %[[a3]], %[[a4]], %{{.*}}) : (!fir.ref<i8>, i64, !fir.ref<i8>, i64, i1) -> i64
  verify_test2 = verify(s1, s2, .true.)
end function verify_test2

