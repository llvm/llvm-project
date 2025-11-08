! RUN: bbc %s -emit-fir -hlfir=false --canonicalize -o - | FileCheck %s

! CHECK-LABEL: pause_test
subroutine pause_test()
  ! CHECK: fir.call @_Fortran{{.*}}PauseStatement()
  ! CHECK-NEXT: return
  pause
end subroutine

! CHECK-LABEL: pause_code
subroutine pause_code()
  pause 42
 ! CHECK: fir.call @_Fortran{{.*}}PauseStatement
 ! CHECK-NEXT: return
end subroutine

! CHECK-LABEL: pause_msg
subroutine pause_msg()
  pause "hello"
  ! CHECK-DAG: %[[five:.*]] = arith.constant 5 : index
  ! CHECK-DAG: %[[lit:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,5>>
  ! CHECK-DAG: %[[buff:.*]] = fir.convert %[[lit]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
  ! CHECK-DAG: %[[len:.*]] = fir.convert %[[five]] : (index) -> i64
  ! CHECK: fir.call @_Fortran{{.*}}PauseStatementText(%[[buff]], %[[len]])
  ! CHECK-NEXT: return
end subroutine

! CHECK-DAG: func private @_Fortran{{.*}}PauseStatement
! CHECK-DAG: func private @_Fortran{{.*}}PauseStatementText