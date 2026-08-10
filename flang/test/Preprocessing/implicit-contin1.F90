! RUN: %flang -E %s | FileCheck %s
! When there's an object-like macro don't apply implicit continuation.
#define M )
call foo (1 M
end

!CHECK:      call foo(1 )
!CHECK:      end
