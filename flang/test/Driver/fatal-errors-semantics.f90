! RUN: not %flang_fc1 %s 2>&1 | FileCheck %s --check-prefix=CHECK1
! RUN: not %flang_fc1 -Wfatal-errors %s 2>&1 | FileCheck %s --check-prefix=CHECK2

module m
    contains
     subroutine s0(p)
       real, pointer, intent(in) :: p
     end
     subroutine s1(p)
       real, pointer, intent(in) :: p(:)
     end
     subroutine sa(p)
       real, pointer, intent(in) :: p(..)
     end
     subroutine sao(p)
       real, intent(in), optional, pointer :: p(..)
     end
     subroutine soa(a)
       real, intent(in), optional, allocatable :: a(..)
     end
     subroutine test
       real, pointer :: a0, a1(:)
       !CHECK1: fatal-errors-semantics.f90:{{.*}} error:
       !CHECK2: fatal-errors-semantics.f90:{{.*}} error:
       call s0(null(a1))
       !CHECK1: fatal-errors-semantics.f90:{{.*}} error:
       !CHECK2-NOT: error:
       call s1(null(a0))
       !CHECK1: fatal-errors-semantics.f90:{{.*}} error:
       !CHECK2-NOT: error:
       call sa(null())
       !CHECK1: fatal-errors-semantics.f90:{{.*}} error:
       !CHECK2-NOT: error:
       call sao(null())
       !CHECK1: fatal-errors-semantics.f90:{{.*}} error:
       !CHECK2-NOT: error:
       call soa(null())
     end
   end
   