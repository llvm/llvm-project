! RUN: bbc -emit-hlfir %s -o - | FileCheck --check-prefixes=CHECK %s
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck --check-prefixes=CHECK %s

! CHECK-LABEL: func @_QPtest_perror(
subroutine test_perror()
  character(len=10) :: string
  character(len=1) :: one
  ! CHECK: %[[C1:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_0:.*]] = fir.alloca !fir.char<1> {bindc_name = "one", uniq_name = "_QFtest_perrorEone"}
  ! CHECK: %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[C1]] {uniq_name = "_QFtest_perrorEone"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
  ! CHECK: %[[C10:.*]] = arith.constant 10 : index
  ! CHECK: %[[VAL_2:.*]] = fir.alloca !fir.char<1,10> {bindc_name = "string", uniq_name = "_QFtest_perrorEstring"}
  ! CHECK: %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] typeparams %[[C10]] {uniq_name = "_QFtest_perrorEstring"} : (!fir.ref<!fir.char<1,10>>, index) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)

  call perror(string)
  ! CHECK: %[[VAL_4:.*]] = fir.embox %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,10>>) -> !fir.box<!fir.char<1,10>>
  ! CHECK: %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]] : (!fir.box<!fir.char<1,10>>) -> !fir.ref<!fir.char<1,10>>
  ! CHECK: %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  ! CHECK: fir.call @_FortranAPerror(%[[VAL_6]]) fastmath<contract> : (!fir.ref<i8>) -> ()

  call perror("prefix")
  ! CHECK: %[[VAL_7:.*]] = fir.address_of(@{{.*}}) : !fir.ref<!fir.char<1,6>>
  ! CHECK: %[[C6:.*]] = arith.constant 6 : index
  ! CHECK: %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] typeparams %[[C6]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = {{.*}}} : (!fir.ref<!fir.char<1,6>>, index) -> (!fir.ref<!fir.char<1,6>>, !fir.ref<!fir.char<1,6>>)
  ! CHECK: %[[VAL_9:.*]] = fir.embox %[[VAL_8]]#0 : (!fir.ref<!fir.char<1,6>>) -> !fir.box<!fir.char<1,6>>
  ! CHECK: %[[VAL_10:.*]] = fir.box_addr %[[VAL_9]] : (!fir.box<!fir.char<1,6>>) -> !fir.ref<!fir.char<1,6>>
  ! CHECK: %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<!fir.char<1,6>>) -> !fir.ref<i8>
  ! CHECK: fir.call @_FortranAPerror(%[[VAL_11]]) fastmath<contract> : (!fir.ref<i8>) -> ()

  call perror(one)
  ! CHECK: %[[VAL_12:.*]] = fir.embox %[[VAL_1]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.box<!fir.char<1>>
  ! CHECK: %[[VAL_13:.*]] = fir.box_addr %[[VAL_12]] : (!fir.box<!fir.char<1>>) -> !fir.ref<!fir.char<1>>
  ! CHECK: %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
  ! CHECK: fir.call @_FortranAPerror(%[[VAL_14]]) fastmath<contract> : (!fir.ref<i8>) -> ()
end subroutine test_perror

! CHECK-LABEL: func @_QPtest_perror_unknown_length(
! CHECK-SAME: %[[ARG0:.*]]:  !fir.boxchar<1> {fir.bindc_name = "str"}
subroutine test_perror_unknown_length(str)
    implicit none
    character(len=*), intent(in) :: str

  call perror(str)
  ! CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK: %[[VAL_1:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_1]]#0 typeparams %[[VAL_1]]#1 dummy_scope %[[VAL_0]] arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest_perror_unknown_lengthEstr"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  ! CHECK: %[[VAL_3:.*]] = fir.embox %[[VAL_2]]#1 typeparams %[[VAL_1]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.box<!fir.char<1,?>>
  ! CHECK: %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK: %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK: fir.call @_FortranAPerror(%[[VAL_5]]) fastmath<contract> : (!fir.ref<i8>) -> ()
  ! CHECK: return
end subroutine test_perror_unknown_length
