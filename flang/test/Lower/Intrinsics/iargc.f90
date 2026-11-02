! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s
! bbc doesn't have a way to set the default kinds so we use flang driver
! RUN: %flang_fc1 -fdefault-integer-8 -emit-hlfir %s -o - | FileCheck --check-prefixes=CHECK %s

! CHECK-LABEL: iargc_test
subroutine iargc_test()
integer(4) :: arg_count_test
arg_count_test = iargc()
! CHECK: %[[argumentCount:.*]] = fir.call @_FortranAArgumentCount() {{.*}}: () -> i32
! CHECK: return
end subroutine iargc_test
