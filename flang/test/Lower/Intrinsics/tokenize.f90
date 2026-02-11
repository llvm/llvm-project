! RUN: bbc --use-desc-for-alloc=false -emit-fir -hlfir=false %s -o - | FileCheck %s
! RUN: %flang_fc1 -mllvm --use-desc-for-alloc=false -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: tok_form1
subroutine tok_form1()
  implicit none
  character(:), allocatable :: tokens(:)
  call tokenize("a,b", ",", tokens)
  ! CHECK-DAG: %[[TOKENS:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
  ! CHECK-DAG: %[[TOKENS_NONE:.*]] = fir.convert %[[TOKENS]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[SEP_NONE:.*]] = fir.zero_bits !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @_FortranATokenize(%[[TOKENS_NONE]], %[[SEP_NONE]],
end subroutine tok_form1

! CHECK-LABEL: tok_form2
subroutine tok_form2()
  implicit none
  integer, allocatable :: first(:), last(:)
  call tokenize("a,,b", ",", first, last)
  ! CHECK-DAG: %[[FIRST:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[LAST:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
  ! CHECK-DAG: %[[FIRST_NONE:.*]] = fir.convert %[[FIRST]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK-DAG: %[[LAST_NONE:.*]] = fir.convert %[[LAST]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK: fir.call @_FortranATokenizePositions(%[[FIRST_NONE]], %[[LAST_NONE]],
end subroutine tok_form2
