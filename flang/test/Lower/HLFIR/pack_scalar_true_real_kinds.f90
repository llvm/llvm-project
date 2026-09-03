! RUN: %flang_fc1 %s -o %t.o
! RUN: %flang_fc1 -O2 %s -o %t2.o

subroutine pack_real_k2
  integer, parameter :: k = 2
  real(k) :: c(3, 3)
  real(k) :: d(6)
  real(k) :: e(9)
  c = reshape((/0, 3, 2, 4, 3, 2, 5, 1, 2/), (/3, 3/))
  d = pack(c, mask=c.ne.2)
  e = pack(c, mask=.true.)
  if (any(d /= (/0, 3, 4, 3, 5, 1/))) print *, 'err1'
  if (any(e /= (/0, 3, 2, 4, 3, 2, 5, 1, 2/))) print *, 'err2'
end subroutine pack_real_k2

subroutine pack_real_k8
  integer, parameter :: k = 8
  real(k) :: c(3, 3)
  real(k) :: d(6)
  real(k) :: e(9)
  c = reshape((/0, 3, 2, 4, 3, 2, 5, 1, 2/), (/3, 3/))
  d = pack(c, mask=c.ne.2)
  e = pack(c, mask=.true.)
  if (any(d /= (/0, 3, 4, 3, 5, 1/))) print *, 'err1'
  if (any(e /= (/0, 3, 2, 4, 3, 2, 5, 1, 2/))) print *, 'err2'
end subroutine pack_real_k8

subroutine pack_real_scalar_true_only
  real(4) :: c(3, 3)
  real(4) :: e(9)
  c = reshape((/1, 2, 3, 4, 5, 6, 7, 8, 9/), (/3, 3/))
  e = pack(c, mask=.true.)
end subroutine pack_real_scalar_true_only

! RUN: %flang_fc1 -O2 -S %s -o - | FileCheck %s --check-prefix=OPT
! OPT-LABEL: pack_real_scalar_true_only_
! OPT-NOT: _FortranAPack
