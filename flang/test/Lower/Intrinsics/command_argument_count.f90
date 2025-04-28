! RUN: bbc -emit-fir %s -o - | FileCheck %s
! bbc doesn't have a way to set the default kinds so we use flang-new driver
! RUN: flang-new -fc1 -fdefault-integer-8 -emit-fir %s -o - | FileCheck --check-prefixes=CHECK,CHECK-64  %s

! CHECK-LABEL: argument_count_test
subroutine argument_count_test()
integer :: arg_count_test
arg_count_test = command_argument_count()
! CHECK: %[[argumentCount:.*]] = fir.call @_FortranAArgumentCount() {{.*}}: () -> i32
! CHECK-64: %{{[0-9]+}} = fir.convert %[[argumentCount]] : (i32) -> i64
end subroutine argument_count_test
