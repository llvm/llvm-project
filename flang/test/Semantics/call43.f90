! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
subroutine from(a, b, c, d)
  real a(10), b(:), c
  real, contiguous :: d(:)
  call to(a)
  call to(a(1)) ! ok
  call to(b) ! ok, passed via temp
  !WARNING: Reference to the procedure 'to' has an implicit interface that is distinct from another reference: incompatible dummy argument #1: incompatible dummy data object shapes [-Wincompatible-implicit-interfaces]
  call to(b(1))
  !WARNING: Reference to the procedure 'to' has an implicit interface that is distinct from another reference: incompatible dummy argument #1: incompatible dummy data object shapes [-Wincompatible-implicit-interfaces]
  call to(c)
  !WARNING: Reference to the procedure 'to' has an implicit interface that is distinct from another reference: incompatible dummy argument #1: incompatible dummy data object shapes [-Wincompatible-implicit-interfaces]
  call to(1.)
  call to([1., 2.]) ! ok
  call to(d) ! ok
  call to(d(1)) ! ok
end
