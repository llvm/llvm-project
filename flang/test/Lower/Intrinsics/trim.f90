! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtrim_test(
! CHECK-SAME: %[[arg0:.*]]: !fir.boxchar<1>{{.*}})
subroutine trim_test(c)
  character(*) :: c
  ! CHECK-DAG: %[[c:.*]]:2 = fir.unboxchar %[[arg0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK-DAG: %[[cDecl:.*]]:2 = hlfir.declare %[[c]]#0 typeparams %[[c]]#1 {{.*}}{uniq_name = "_QFtrim_testEc"}
  ! CHECK: %[[trimmed:.*]] = hlfir.char_trim %[[cDecl]]#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
  ! CHECK: %[[trimLen:.*]] = hlfir.get_length %[[trimmed]] : (!hlfir.expr<!fir.char<1,?>>) -> index
  ! CHECK: %[[assoc:.*]]:3 = hlfir.associate %[[trimmed]] typeparams %[[trimLen]] {adapt.valuebyref}
  ! CHECK: fir.call @{{.*}}bar_trim_test(%[[assoc]]#0)
  ! CHECK: hlfir.end_associate %[[assoc]]#1, %[[assoc]]#2
  ! CHECK: hlfir.destroy %[[trimmed]] : !hlfir.expr<!fir.char<1,?>>
  call bar_trim_test(trim(c))
  return
end subroutine
