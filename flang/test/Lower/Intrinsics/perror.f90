! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes=CHECK %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck --check-prefixes=CHECK %s

! CHECK-LABEL: func @_QPtest_perror(
subroutine test_perror()
  character(len=10) :: string
  call perror(string)
  ! CHECK: %[[C10:.*]] = arith.constant 10 : index
  ! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.char<1,10> {bindc_name = "string", uniq_name = "_QFtest_perrorEstring"}
  ! CHECK: %[[VAL_1:.*]] = fir.declare %[[VAL_0]] typeparams %[[C10]] {uniq_name = "_QFtest_perrorEstring"} : (!fir.ref<!fir.char<1,10>>, index) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[VAL_2:.*]] = fir.emboxchar %[[VAL_1]], %[[C10]] : (!fir.ref<!fir.char<1,10>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPperror(%[[VAL_2]]) fastmath<contract> : (!fir.boxchar<1>) -> () 
  ! CHECK: return
end subroutine test_perror
