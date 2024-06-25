! Test GETCWD with dynamically optional arguments.
! RUN: bbc -emit-fir %s -o - | FileCheck %s


! CHECK-LABEL: func.func @_QPtest(
! CHECK-SAME: %[[cwdArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "cwd"},
! CHECK-SAME: %[[statusArg:.*]]: !fir.ref<i32> {fir.bindc_name = "status", fir.optional}) {
subroutine test(cwd, status)
  CHARACTER(len=255) :: cwd
  INTEGER, OPTIONAL :: status
  call getcwd(cwd, status)
  ! CHECK-NEXT:        %[[c0:.*]] = arith.constant 0 : i64
  ! CHECK-NEXT:        %[[c11:.*]] = arith.constant 11 : i32
  ! CHECK-NEXT:        %[[c255:.*]] = arith.constant 255 : index
  ! CHECK-NEXT:        %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK-NEXT:        %[[cwdUnbox:.*]]:2 = fir.unboxchar %[[cwdArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK-NEXT:        %[[cwdCast:.*]] = fir.convert %[[cwdUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,255>>
  ! CHECK-NEXT:        %[[cwdDeclare:.*]] = fir.declare %[[cwdCast]] typeparams %[[c255]] dummy_scope %[[DSCOPE]] {uniq_name = "_QFtestEcwd"} : (!fir.ref<!fir.char<1,255>>, index, !fir.dscope) -> !fir.ref<!fir.char<1,255>>
  ! CHECK-NEXT:        %[[statusAddr:.*]] = fir.declare %[[statusArg]] dummy_scope %[[DSCOPE]] {fortran_attrs = #fir.var_attrs<optional>, uniq_name = "_QFtestEstatus"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  ! CHECK-NEXT:       %[[cwdBox:.*]] = fir.embox %[[cwdDeclare]] : (!fir.ref<!fir.char<1,255>>) -> !fir.box<!fir.char<1,255>>
  ! CHECK:            %[[cwd:.*]] = fir.convert %[[cwdBox]] : (!fir.box<!fir.char<1,255>>) -> !fir.box<none>
  ! CHECK:            %[[statusValue:.*]] = fir.call @_FortranAGetCwd(%[[cwd]], %[[VAL_8:.*]], %[[c11]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> i32
  ! CHECK-NEXT:       %[[statusCast:.*]] = fir.convert %[[statusAddr]] : (!fir.ref<i32>) -> i64
  ! CHECK-NEXT:       %[[isPresent:.*]] = arith.cmpi ne, %[[statusCast]], %[[c0]] : i64
  ! CHECK-NEXT:       fir.if %[[isPresent]] {
  ! CHECK-NEXT:         fir.store %[[statusValue]] to %[[statusAddr]] : !fir.ref<i32>
  ! CHECK-NEXT:       }
  ! CHECK-NEXT:       return
end subroutine
