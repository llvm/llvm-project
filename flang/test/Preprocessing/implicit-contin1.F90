! RUN: %flang -E %s | FileCheck %s
! When there's a macro definition with unbalanced parentheses,
! don't apply implicit continuation.
#define M )
call foo (1 M
end

!CHECK:      call foo(1 )
!CHECK:      end
