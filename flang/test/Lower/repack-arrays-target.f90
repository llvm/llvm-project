! RUN: bbc -emit-hlfir -frepack-arrays %s -o - | FileCheck --check-prefixes=CHECK %s

! Check that there is no repacking for TARGET dummy argument.

! CHECK-LABEL:   func.func @_QPtest(
! CHECK-NOT: fir.pack_array
! CHECK-NOT: fir.unpack_array
subroutine test(x)
  integer, target :: x(:)
end subroutine test
