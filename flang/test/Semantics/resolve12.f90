! RUN: %S/test_errors.sh %s %t %f18
module m1
end

subroutine sub
end

use m1
!ERROR: Error reading module file for module 'm2'
use m2
!ERROR: 'sub' is not a module
use sub
end
