! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: ichar_test
subroutine ichar_test(c)
  character(1) :: c
  character :: str(10)
  ! CHECK-DAG: %[[unbox:.*]]:2 = fir.unboxchar
  ! CHECK-DAG: %[[J:.*]] = fir.alloca i32 {{{.*}}uniq_name = "{{.*}}Ej"}
  ! CHECK-DAG: %[[STR:.*]] = fir.alloca !fir.array{{.*}} {{{.*}}uniq_name = "{{.*}}Estr"}
  ! CHECK: %[[BOX:.*]] = fir.convert %[[unbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1>>
  ! CHECK: %[[PTR:.*]] = fir.load %[[BOX]] : !fir.ref<!fir.char<1>>
  ! CHECK: %[[CHAR:.*]] = fir.extract_value %[[PTR]], [0 : index] :
  ! CHECK: %[[ARG:.*]] = arith.extui %[[CHAR]] : i8 to i32
  ! CHECK: fir.call @{{.*}}OutputInteger32{{.*}}%[[ARG]]
  ! CHECK: fir.call @{{.*}}EndIoStatement
  print *, ichar(c)

  ! CHECK-DAG: %{{.*}} = fir.load %[[J]] : !fir.ref<i32>
  ! CHECK: %[[PTR1:.*]] = fir.coordinate_of %[[STR]], %
  ! CHECK: %[[PTR2:.*]] = fir.load %[[PTR1]] : !fir.ref<!fir.char<1>>
  ! CHECK: %[[CHAR:.*]] = fir.extract_value %[[PTR2]], [0 : index] :
  ! CHECK: %[[ARG:.*]] = arith.extui %[[CHAR]] : i8 to i32
  ! CHECK: fir.call @{{.*}}OutputInteger32{{.*}}%[[ARG]]
  ! CHECK: fir.call @{{.*}}EndIoStatement
  print *, ichar(str(J))

  ! "Magic" 88 below is the ASCII code for `X` and the value returned by IACHAR (’X’)
  ! CHECK: %[[c88:.*]] = arith.constant 88 : i32
  ! CHECK-NEXT: fir.call @{{.*}}OutputInteger32({{.*}}, %[[c88]])
  ! CHECK-NEXT: fir.call @{{.*}}EndIoStatement
  print *, iachar('X')
end subroutine

! Check that 'arith.extui' op is not generated if type are matching.
 ! CHECK-LABEL: no_extui
subroutine no_extui(ch)
  integer, parameter :: kind = selected_char_kind('ISO_10646')
  character(*, kind), intent(in) :: ch(:)
  integer :: i, j 
  ! CHECK-NOT: arith.extui
  j = ichar(ch(i)(i:i))
end subroutine
