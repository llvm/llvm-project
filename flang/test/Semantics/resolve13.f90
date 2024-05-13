! RUN: %python %S/test_errors.py %s %flang_fc1
module m1
  integer :: x
  integer, private :: y
  interface operator(.foo.)
    module procedure ifoo
  end interface
  interface operator(-)
    module procedure ifoo
  end interface
  interface operator(.priv.)
    module procedure ifoo
  end interface
  interface operator(*)
    module procedure ifoo
  end interface
  private :: operator(.priv.), operator(*)
contains
  integer function ifoo(x, y)
    logical, intent(in) :: x, y
  end
end

use m1, local_x => x
!ERROR: 'y' is PRIVATE in 'm1'
use m1, local_y => y
!ERROR: 'z' not found in module 'm1'
use m1, local_z => z
use m1, operator(.localfoo.) => operator(.foo.)
!ERROR: 'OPERATOR(.bar.)' not found in module 'm1'
use m1, operator(.localbar.) => operator(.bar.)

!ERROR: 'y' is PRIVATE in 'm1'
use m1, only: y
!ERROR: 'OPERATOR(.priv.)' is PRIVATE in 'm1'
use m1, only: operator(.priv.)
!ERROR: 'OPERATOR(*)' is PRIVATE in 'm1'
use m1, only: operator(*)
!ERROR: 'z' not found in module 'm1'
use m1, only: z
!ERROR: 'z' not found in module 'm1'
use m1, only: my_x => z
use m1, only: operator(.foo.)
!ERROR: 'OPERATOR(.bar.)' not found in module 'm1'
use m1, only: operator(.bar.)
use m1, only: operator(-) , ifoo
!ERROR: 'OPERATOR(+)' not found in module 'm1'
use m1, only: operator(+)

end
