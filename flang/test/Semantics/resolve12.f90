! RUN: %python %S/test_errors.py %s %flang_fc1
module m1
end

subroutine sub
end

use m1
!ERROR: Cannot parse module file for module 'm2': Source file 'm2.mod' was not found
use m2
!ERROR: 'sub' is not a module
use sub
end
