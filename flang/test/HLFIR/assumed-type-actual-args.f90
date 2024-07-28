! Test lowering to FIR of actual arguments that are assumed type
! variables (Fortran 2018 7.3.2.2 point 3).
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine test1(x)
  interface
    subroutine s1(x)
      type(*) :: x
    end subroutine
  end interface
  type(*) :: x
  call s1(x)
end subroutine

subroutine test2(x)
  interface
    subroutine s2(x)
      type(*) :: x(*)
    end subroutine
  end interface
  type(*) :: x(*)
  call s2(x)
end subroutine

subroutine test3(x)
  interface
    subroutine s3(x)
      type(*) :: x(:)
    end subroutine
  end interface
  type(*) :: x(:)
  call s3(x)
end subroutine

subroutine test4(x)
  interface
    subroutine s4(x)
      type(*) :: x(*)
    end subroutine
  end interface
  type(*) :: x(:)
  call s4(x)
end subroutine

subroutine test3b(x)
  interface
    subroutine s3b(x)
      type(*), optional, contiguous :: x(:)
    end subroutine
  end interface
  type(*), optional :: x(:)
  call s3b(x)
end subroutine

subroutine test4b(x)
  interface
    subroutine s4b(x)
      type(*), optional :: x(*)
    end subroutine
  end interface
  type(*), optional :: x(:)
  call s4b(x)
end subroutine

subroutine test4c(x)
  interface
    subroutine s4c(x)
      type(*), optional :: x(*)
    end subroutine
  end interface
  type(*), contiguous, optional :: x(:)
  call s4c(x)
end subroutine

subroutine test4d(x)
  interface
    subroutine s4d(x)
      type(*) :: x(*)
    end subroutine
  end interface
  type(*), contiguous :: x(:)
  call s4d(x)
end subroutine

subroutine test5(x)
  interface
    subroutine s5(x)
      type(*) :: x(..)
    end subroutine
  end interface
  type(*) :: x(:)
  call s5(x)
end subroutine

subroutine test5b(x)
  interface
    subroutine s5b(x)
      type(*), optional, contiguous :: x(..)
    end subroutine
  end interface
  type(*), optional :: x(:)
  call s5b(x)
end subroutine

! CHECK-LABEL:   func.func @_QPtest1(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<none> {fir.bindc_name = "x"}) {
! CHECK:           %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[DSCOPE]] {uniq_name = "_QFtest1Ex"} : (!fir.ref<none>, !fir.dscope) -> (!fir.ref<none>, !fir.ref<none>)
! CHECK:           fir.call @_QPs1(%[[VAL_1]]#1) fastmath<contract> : (!fir.ref<none>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest2(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.array<?xnone>> {fir.bindc_name = "x"}) {
! CHECK:           %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]] = arith.constant -1 : index
! CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_2]]) dummy_scope %[[DSCOPE]] {uniq_name = "_QFtest2Ex"} : (!fir.ref<!fir.array<?xnone>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xnone>>, !fir.ref<!fir.array<?xnone>>)
! CHECK:           fir.call @_QPs2(%[[VAL_3]]#1) fastmath<contract> : (!fir.ref<!fir.array<?xnone>>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest3(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.box<!fir.array<?xnone>> {fir.bindc_name = "x"}) {
! CHECK:           %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[DSCOPE]] {uniq_name = "_QFtest3Ex"} : (!fir.box<!fir.array<?xnone>>, !fir.dscope) -> (!fir.box<!fir.array<?xnone>>, !fir.box<!fir.array<?xnone>>)
! CHECK:           fir.call @_QPs3(%[[VAL_1]]#0) fastmath<contract> : (!fir.box<!fir.array<?xnone>>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest4(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.box<!fir.array<?xnone>> {fir.bindc_name = "x"}) {
! CHECK:           %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[DSCOPE]] {uniq_name = "_QFtest4Ex"} : (!fir.box<!fir.array<?xnone>>, !fir.dscope) -> (!fir.box<!fir.array<?xnone>>, !fir.box<!fir.array<?xnone>>)
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.copy_in %[[VAL_1]]#0 to %[[TMP_BOX:.*]] : (!fir.box<!fir.array<?xnone>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xnone>>>>) -> (!fir.box<!fir.array<?xnone>>, i1)
! CHECK:           %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]]#0 : (!fir.box<!fir.array<?xnone>>) -> !fir.ref<!fir.array<?xnone>>
! CHECK:           fir.call @_QPs4(%[[VAL_3]]) fastmath<contract> : (!fir.ref<!fir.array<?xnone>>) -> ()
! CHECK:           hlfir.copy_out %[[TMP_BOX]], %[[VAL_2]]#1 to %[[VAL_1]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xnone>>>>, i1, !fir.box<!fir.array<?xnone>>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest3b(
! CHECK-SAME:                         %[[VAL_0:.*]]: !fir.box<!fir.array<?xnone>> {fir.bindc_name = "x", fir.optional}) {
! CHECK:           %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[DSCOPE]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFtest3bEx"} : (!fir.box<!fir.array<?xnone>>, !fir.dscope) -> (!fir.box<!fir.array<?xnone>>, !fir.box<!fir.array<?xnone>>)
! CHECK:           %[[VAL_2:.*]] = fir.is_present %[[VAL_1]]#0 : (!fir.box<!fir.array<?xnone>>) -> i1
! CHECK:           %[[VAL_3:.*]]:3 = fir.if %[[VAL_2]] -> (!fir.box<!fir.array<?xnone>>, i1, !fir.box<!fir.array<?xnone>>) {
! CHECK:             %[[VAL_4:.*]]:2 = hlfir.copy_in %[[VAL_1]]#0 to %[[TMP_BOX:.*]] : (!fir.box<!fir.array<?xnone>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xnone>>>>) -> (!fir.box<!fir.array<?xnone>>, i1)
! CHECK:             fir.result %[[VAL_4]]#0, %[[VAL_4]]#1, %[[VAL_1]]#0 : !fir.box<!fir.array<?xnone>>, i1, !fir.box<!fir.array<?xnone>>
! CHECK:           } else {
! CHECK:             %[[VAL_6:.*]] = fir.absent !fir.box<!fir.array<?xnone>>
! CHECK:             %[[VAL_7:.*]] = arith.constant false
! CHECK:             %[[VAL_8:.*]] = fir.absent !fir.box<!fir.array<?xnone>>
! CHECK:             fir.result %[[VAL_6]], %[[VAL_7]], %[[VAL_8]] : !fir.box<!fir.array<?xnone>>, i1, !fir.box<!fir.array<?xnone>>
! CHECK:           }
! CHECK:           fir.call @_QPs3b(%[[VAL_9:.*]]#0) fastmath<contract> : (!fir.box<!fir.array<?xnone>>) -> ()
! CHECK:           hlfir.copy_out %[[TMP_BOX]], %[[VAL_9]]#1 to %[[VAL_9]]#2 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xnone>>>>, i1, !fir.box<!fir.array<?xnone>>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest4b(
! CHECK-SAME:                         %[[VAL_0:.*]]: !fir.box<!fir.array<?xnone>> {fir.bindc_name = "x", fir.optional}) {
! CHECK:           %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[DSCOPE]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFtest4bEx"} : (!fir.box<!fir.array<?xnone>>, !fir.dscope) -> (!fir.box<!fir.array<?xnone>>, !fir.box<!fir.array<?xnone>>)
! CHECK:           %[[VAL_2:.*]] = fir.is_present %[[VAL_1]]#0 : (!fir.box<!fir.array<?xnone>>) -> i1
! CHECK:           %[[VAL_3:.*]]:3 = fir.if %[[VAL_2]] -> (!fir.ref<!fir.array<?xnone>>, i1, !fir.box<!fir.array<?xnone>>) {
! CHECK:             %[[VAL_4:.*]]:2 = hlfir.copy_in %[[VAL_1]]#0 to %[[TMP_BOX:.*]] : (!fir.box<!fir.array<?xnone>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xnone>>>>) -> (!fir.box<!fir.array<?xnone>>, i1)
! CHECK:             %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]]#0 : (!fir.box<!fir.array<?xnone>>) -> !fir.ref<!fir.array<?xnone>>
! CHECK:             fir.result %[[VAL_5]], %[[VAL_4]]#1, %[[VAL_1]]#0 : !fir.ref<!fir.array<?xnone>>, i1, !fir.box<!fir.array<?xnone>>
! CHECK:           } else {
! CHECK:             %[[VAL_7:.*]] = fir.absent !fir.ref<!fir.array<?xnone>>
! CHECK:             %[[VAL_8:.*]] = arith.constant false
! CHECK:             %[[VAL_9:.*]] = fir.absent !fir.box<!fir.array<?xnone>>
! CHECK:             fir.result %[[VAL_7]], %[[VAL_8]], %[[VAL_9]] : !fir.ref<!fir.array<?xnone>>, i1, !fir.box<!fir.array<?xnone>>
! CHECK:           }
! CHECK:           fir.call @_QPs4b(%[[VAL_10:.*]]#0) fastmath<contract> : (!fir.ref<!fir.array<?xnone>>) -> ()
! CHECK:           hlfir.copy_out %[[TMP_BOX]], %[[VAL_10]]#1 to %[[VAL_10]]#2 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xnone>>>>, i1, !fir.box<!fir.array<?xnone>>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest4c(
! CHECK-SAME:                         %[[VAL_0:.*]]: !fir.box<!fir.array<?xnone>> {fir.bindc_name = "x", fir.contiguous, fir.optional}) {
! CHECK:           %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[DSCOPE]] {fortran_attrs = #fir.var_attrs<contiguous, optional>, uniq_name = "_QFtest4cEx"} : (!fir.box<!fir.array<?xnone>>, !fir.dscope) -> (!fir.box<!fir.array<?xnone>>, !fir.box<!fir.array<?xnone>>)
! CHECK:           %[[VAL_2:.*]] = fir.is_present %[[VAL_1]]#0 : (!fir.box<!fir.array<?xnone>>) -> i1
! CHECK:           %[[VAL_3:.*]] = fir.if %[[VAL_2]] -> (!fir.ref<!fir.array<?xnone>>) {
! CHECK:             %[[VAL_4:.*]] = fir.box_addr %[[VAL_1]]#1 : (!fir.box<!fir.array<?xnone>>) -> !fir.ref<!fir.array<?xnone>>
! CHECK:             fir.result %[[VAL_4]] : !fir.ref<!fir.array<?xnone>>
! CHECK:           } else {
! CHECK:             %[[VAL_5:.*]] = fir.absent !fir.ref<!fir.array<?xnone>>
! CHECK:             fir.result %[[VAL_5]] : !fir.ref<!fir.array<?xnone>>
! CHECK:           }
! CHECK:           fir.call @_QPs4c(%[[VAL_3]]) fastmath<contract> : (!fir.ref<!fir.array<?xnone>>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest4d(
! CHECK-SAME:                         %[[VAL_0:.*]]: !fir.box<!fir.array<?xnone>> {fir.bindc_name = "x", fir.contiguous}) {
! CHECK:           %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[DSCOPE]] {fortran_attrs = #fir.var_attrs<contiguous>, uniq_name = "_QFtest4dEx"} : (!fir.box<!fir.array<?xnone>>, !fir.dscope) -> (!fir.box<!fir.array<?xnone>>, !fir.box<!fir.array<?xnone>>)
! CHECK:           %[[VAL_2:.*]] = fir.box_addr %[[VAL_1]]#1 : (!fir.box<!fir.array<?xnone>>) -> !fir.ref<!fir.array<?xnone>>
! CHECK:           fir.call @_QPs4d(%[[VAL_2]]) fastmath<contract> : (!fir.ref<!fir.array<?xnone>>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest5(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.box<!fir.array<?xnone>> {fir.bindc_name = "x"}) {
! CHECK:           %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[DSCOPE]] {uniq_name = "_QFtest5Ex"} : (!fir.box<!fir.array<?xnone>>, !fir.dscope) -> (!fir.box<!fir.array<?xnone>>, !fir.box<!fir.array<?xnone>>)
! CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#0 : (!fir.box<!fir.array<?xnone>>) -> !fir.box<!fir.array<*:none>>
! CHECK:           fir.call @_QPs5(%[[VAL_2]]) fastmath<contract> : (!fir.box<!fir.array<*:none>>) -> ()
! CHECK:           return
! CHECK:         }

! CHECK-LABEL:   func.func @_QPtest5b(
! CHECK-SAME:                         %[[VAL_0:.*]]: !fir.box<!fir.array<?xnone>> {fir.bindc_name = "x", fir.optional}) {
! CHECK:           %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[DSCOPE]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFtest5bEx"} : (!fir.box<!fir.array<?xnone>>, !fir.dscope) -> (!fir.box<!fir.array<?xnone>>, !fir.box<!fir.array<?xnone>>)
! CHECK:           %[[VAL_2:.*]] = fir.is_present %[[VAL_1]]#0 : (!fir.box<!fir.array<?xnone>>) -> i1
! CHECK:           %[[VAL_3:.*]]:3 = fir.if %[[VAL_2]] -> (!fir.box<!fir.array<?xnone>>, i1, !fir.box<!fir.array<?xnone>>) {
! CHECK:             %[[VAL_4:.*]]:2 = hlfir.copy_in %[[VAL_1]]#0 to %[[TMP_BOX:.*]] : (!fir.box<!fir.array<?xnone>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xnone>>>>) -> (!fir.box<!fir.array<?xnone>>, i1)
! CHECK:             fir.result %[[VAL_4]]#0, %[[VAL_4]]#1, %[[VAL_1]]#0 : !fir.box<!fir.array<?xnone>>, i1, !fir.box<!fir.array<?xnone>>
! CHECK:           } else {
! CHECK:             %[[VAL_6:.*]] = fir.absent !fir.box<!fir.array<?xnone>>
! CHECK:             %[[VAL_7:.*]] = arith.constant false
! CHECK:             %[[VAL_8:.*]] = fir.absent !fir.box<!fir.array<?xnone>>
! CHECK:             fir.result %[[VAL_6]], %[[VAL_7]], %[[VAL_8]] : !fir.box<!fir.array<?xnone>>, i1, !fir.box<!fir.array<?xnone>>
! CHECK:           }
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_10:.*]]#0 : (!fir.box<!fir.array<?xnone>>) -> !fir.box<!fir.array<*:none>>
! CHECK:           fir.call @_QPs5b(%[[VAL_9]]) fastmath<contract> : (!fir.box<!fir.array<*:none>>) -> ()
! CHECK:           hlfir.copy_out %[[TMP_BOX]], %[[VAL_10]]#1 to %[[VAL_10]]#2 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xnone>>>>, i1, !fir.box<!fir.array<?xnone>>) -> ()
! CHECK:           return
! CHECK:         }
