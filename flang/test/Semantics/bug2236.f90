!RUN: flang -c -Xflang -fdisable-real-10 %s 2>&1 | FileCheck --allow-empty %s
!REQUIRES: x86-registered-target
!CHECK-NOT: error
!Ensure ieee_arithmetic.mod can be used even when REAL(10) is disabled.
use ieee_arithmetic
end
