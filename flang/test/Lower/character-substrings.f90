! Test character substring lowering
! RUN: bbc %s -o - -emit-fir | FileCheck %s

! Test substring lower where the parent is a scalar-char-literal-constant
! CHECK-LABEL: func @_QPfoo(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<i64>,
! CHECK-SAME: %[[arg1:.*]]: !fir.ref<i64>)
subroutine foo(i, j)
  integer(8) :: i, j
  call bar("abcHello World!dfg"(i:j))
  ! CHECK: %[[baseAddr:.*]] = fir.address_of(@[[fooGlobalLiteral:.*]]) : !fir.ref<!fir.char<1,18>>
  ! CHECK-DAG: %[[i64:.*]] = fir.load %[[arg0]] : !fir.ref<i64>
  ! CHECK-DAG: %[[j64:.*]] = fir.load %[[arg1]] : !fir.ref<i64>
  ! CHECK-DAG: %[[i:.*]] = fir.convert %[[i64]] : (i64) -> index
  ! CHECK-DAG: %[[j:.*]] = fir.convert %[[j64]] : (i64) -> index
  ! CHECK-DAG: %[[startIdx:.*]] = subi %[[i]], %c1{{.*}} : index
  ! CHECK-DAG: %[[charArrayView:.*]] = fir.convert %[[baseAddr]] : (!fir.ref<!fir.char<1,18>>) -> !fir.ref<!fir.array<18x!fir.char<1>>>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[charArrayView]], %[[startIdx]] : (!fir.ref<!fir.array<18x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK: %[[substringBase:.*]] = fir.convert %[[coor]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[boundsDiff:.*]] = subi %[[j]], %[[i]] : index
  ! CHECK: %[[len:.*]] = addi %[[boundsDiff]], %c1 : index
  ! CHECK: %[[slt:.*]] = cmpi slt, %[[len]], %c0{{.*}} : index
  ! CHECK: %[[safeLen:.*]] = select %[[slt]], %c0{{.*}}, %[[len]] : index
  ! CHECK: %[[substring:.*]] = fir.emboxchar %[[substringBase]], %[[safeLen]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPbar(%[[substring]]) : (!fir.boxchar<1>) -> ()
end subroutine


! CHECK: fir.global linkonce @[[fooGlobalLiteral]] constant : !fir.char<1,18>
  ! CHECK: fir.string_lit "abcHello World!dfg"(18) : !fir.char<1,18>
