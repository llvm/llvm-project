module m1
  integer :: x
  integer :: y
  integer :: z
end
module m2
  real :: y
  real :: z
  real :: w
end

use m1, xx => x, y => z
use m2
volatile w
!ERROR: Cannot change CONTIGUOUS attribute on use-associated 'w'
contiguous w
!ERROR: 'z' is use-associated from module 'm2' and cannot be re-declared
integer z
!ERROR: Reference to 'y' is ambiguous
y = 1
end
