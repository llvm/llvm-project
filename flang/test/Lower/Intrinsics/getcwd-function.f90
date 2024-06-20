! Test GETCWD with dynamically optional arguments.
! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtest(
! CHECK-SAME: %[[cwdArg:.*]]: !fir.boxchar<1> {fir.bindc_name = "cwd"}) -> i32 {
integer function test(cwd)
  CHARACTER(len=255) :: cwd
  test = getcwd(cwd)
  ! CHECK-NEXT:        %[[c8:.*]] = arith.constant 8 : i32
  ! CHECK-NEXT:        %[[c255:.*]] = arith.constant 255 : index
  ! CHECK-NEXT:        %[[DSCOPE:.*]] = fir.dummy_scope : !fir.dscope
  ! CHECK-NEXT:        %[[cwdUnbox:.*]]:2 = fir.unboxchar %[[cwdArg]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK-NEXT:        %[[cwdCast:.*]] = fir.convert %[[cwdUnbox]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,255>>
  ! CHECK-NEXT:        %[[cwdDeclare:.*]] = fir.declare %[[cwdCast]] typeparams %[[c255]] dummy_scope %[[DSCOPE]] {uniq_name = "_QFtestEcwd"} : (!fir.ref<!fir.char<1,255>>, index, !fir.dscope) -> !fir.ref<!fir.char<1,255>>
  ! CHECK-NEXT:        %[[test:.*]] = fir.alloca i32 {bindc_name = "test", uniq_name = "_QFtestEtest"}
  ! CHECK-NEXT:        %[[testAddr:.*]] = fir.declare %[[test]] {uniq_name = "_QFtestEtest"} : (!fir.ref<i32>) -> !fir.ref<i32>
  ! CHECK-NEXT:        %[[cwdBox:.*]] = fir.embox %[[cwdDeclare]] : (!fir.ref<!fir.char<1,255>>) -> !fir.box<!fir.char<1,255>>
  ! CHECK:             %[[cwd:.*]] = fir.convert %[[cwdBox]] : (!fir.box<!fir.char<1,255>>) -> !fir.box<none>
  ! CHECK:             %[[statusValue:.*]] = fir.call @_FortranAGetCwd(%[[cwd]], %[[VAL_9:.*]], %[[c8]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> i32
  ! CHECK-NEXT:        fir.store %[[statusValue]] to %[[testAddr]] : !fir.ref<i32>
  ! CHECK-NEXT:        %[[returnValue:.*]] = fir.load %[[testAddr]] : !fir.ref<i32>
  ! CHECK-NEXT:        return %[[returnValue]] : i32
end function
