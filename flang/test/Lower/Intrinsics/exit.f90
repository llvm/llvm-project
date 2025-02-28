! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck --check-prefixes=CHECK,CHECK-32 -DDEFAULT_INTEGER_SIZE=32 %s
! bbc doesn't have a way to set the default kinds so we use flang driver
! RUN: %flang_fc1 -fdefault-integer-8 -emit-fir -flang-deprecated-no-hlfir %s -o - | FileCheck --check-prefixes=CHECK,CHECK-64 -DDEFAULT_INTEGER_SIZE=64 %s

! CHECK-LABEL: func @_QPexit_test1() {
subroutine exit_test1
    call exit()
  ! CHECK: %[[status:.*]] = arith.constant 0 : i[[DEFAULT_INTEGER_SIZE]]
  ! CHECK-64: %[[statusConvert:.*]] = fir.convert %[[status]] : (i64) -> i32
  ! CHECK-32: fir.call @_FortranAExit(%[[status]]) {{.*}}: (i32) -> ()
  ! CHECK-64: fir.call @_FortranAExit(%[[statusConvert]]) {{.*}}: (i32) -> ()
  end subroutine exit_test1
  
  ! CHECK-LABEL: func @_QPexit_test2(
  ! CHECK-SAME: %[[statusArg:.*]]: !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>{{.*}}) {
  subroutine exit_test2(status)
    integer :: status
    call exit(status)
  ! CHECK: %[[status:.*]] = fir.load %[[statusArg]] : !fir.ref<i[[DEFAULT_INTEGER_SIZE]]>
  ! CHECK-64: %[[statusConv:.*]] = fir.convert %[[status]] : (i64) -> i32
  ! CHECK-32: fir.call @_FortranAExit(%[[status]]) {{.*}}: (i32) -> ()
  ! CHECK-64: fir.call @_FortranAExit(%[[statusConv]]) {{.*}}: (i32) -> ()
  end subroutine exit_test2
