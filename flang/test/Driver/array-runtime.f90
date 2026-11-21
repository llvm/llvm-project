! Numeric broadcasts to non-allocatable arrays and SUM reductions are expanded
! even at -O0. Other array operations retain the optimization-level-dependent
! default.
! RUN: %flang_fc1 -emit-fir -O0 %s -o - | FileCheck %s --check-prefixes=ALL,RUNTIME
! RUN: %flang_fc1 -emit-fir -O1 %s -o - | FileCheck %s --check-prefixes=ALL,EXPAND
! RUN: %flang_fc1 -emit-fir -O2 %s -o - | FileCheck %s --check-prefixes=ALL,EXPAND

subroutine broadcast(x)
  real :: x(:)
  x = 1.
end subroutine

! ALL-LABEL: func{{(\.func)?}} @_QPbroadcast(
! ALL-NOT: fir.call @_FortranAAssign
! ALL: fir.do_loop
! ALL-NOT: fir.call @_FortranAAssign
! ALL: return

subroutine allocatable_broadcast(x)
  real, allocatable :: x(:)
  x = 1.
end subroutine

! ALL-LABEL: func{{(\.func)?}} @_QPallocatable_broadcast(
! RUNTIME: fir.call @_FortranAAssign
! EXPAND-NOT: fir.call @_FortranAAssign
! EXPAND: fir.do_loop
! EXPAND-NOT: fir.call @_FortranAAssign
! ALL: return

subroutine array_copy(x, y)
  real :: x(:), y(:)
  x = y
end subroutine

! ALL-LABEL: func{{(\.func)?}} @_QParray_copy(
! RUNTIME: fir.call @_FortranAAssign
! EXPAND-NOT: fir.call @_FortranAAssign
! EXPAND: fir.do_loop
! EXPAND-NOT: fir.call @_FortranAAssign
! ALL: return

subroutine array_expression(x, y, z)
  real :: x(:), y(:), z(:)
  x = y + z
end subroutine

! ALL-LABEL: func{{(\.func)?}} @_QParray_expression(
! RUNTIME: fir.call @_FortranAAssign
! EXPAND-NOT: fir.call @_FortranAAssign
! EXPAND: fir.do_loop
! EXPAND: arith.addf
! EXPAND-NOT: fir.call @_FortranAAssign
! ALL: return

function sum_reduction(x)
  real :: x(:), sum_reduction
  sum_reduction = sum(x)
end function

! ALL-LABEL: func{{(\.func)?}} @_QPsum_reduction(
! ALL-NOT: fir.call @_FortranASum
! ALL: fir.do_loop
! ALL: arith.addf
! ALL-NOT: fir.call @_FortranASum
! ALL: return

function product_reduction(x)
  real :: x(:), product_reduction
  product_reduction = product(x)
end function

! ALL-LABEL: func{{(\.func)?}} @_QPproduct_reduction(
! RUNTIME: fir.call @_FortranAProductReal4
! EXPAND-NOT: fir.call @_FortranAProduct
! EXPAND: fir.do_loop
! EXPAND: arith.mulf
! EXPAND-NOT: fir.call @_FortranAProduct
! ALL: return
