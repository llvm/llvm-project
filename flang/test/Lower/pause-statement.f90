! RUN: bbc %s -emit-fir --canonicalize -o - | FileCheck %s

! CHECK-LABEL: pause_test
subroutine pause_test()
  pause
  ! CHECK: fir.call @_FortranA{{.*}}PauseStatement()
  ! CHECK-NEXT: return
end subroutine

! CHECK-LABEL: pause_code
subroutine pause_code()
  pause 42
  ! CHECK: %[[c42:.*]] = arith.constant 42 : i32
  ! CHECK: fir.call @_FortranA{{.*}}PauseStatementInt(%[[c42]])
  ! CHECK-NEXT: return
end subroutine

! CHECK-LABEL: pause_msg
subroutine pause_msg()
  pause "hello"
  ! CHECK-DAG: %[[five:.*]] = arith.constant 5 : index
  ! CHECK-DAG: %[[addr:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,5>>
  ! CHECK-DAG: %[[str:.*]]:2 = hlfir.declare %[[addr]] typeparams %[[five]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQ{{.*}}"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
  ! CHECK-DAG: %[[buff:.*]] = fir.convert %[[str]]#0 : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
  ! CHECK-DAG: %[[len:.*]] = fir.convert %[[five]] : (index) -> i64
  ! CHECK: fir.call @_FortranA{{.*}}PauseStatementText(%[[buff]], %[[len]])
  ! CHECK-NEXT: return
end subroutine

! CHECK-DAG: func private @_FortranA{{.*}}PauseStatement
! CHECK-DAG: func private @_FortranA{{.*}}PauseStatementInt
! CHECK-DAG: func private @_FortranA{{.*}}PauseStatementText
