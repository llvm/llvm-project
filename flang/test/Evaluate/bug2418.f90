!RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s

!CHECK-NOT: error:

program main
  integer, pointer :: p1(:) => NULL()
  !CHECK: warning: Argument of ALLOCATED() should be an allocatable, but is instead an object pointer [-Wallocated-for-associated]
  !CHECK:  PRINT *, associated(p1)
  print *, allocated(p1)
end

subroutine s1
  interface
    logical function allocated(p)
      class(*), pointer, intent(in) :: p(..)
    end
  end interface
  real, pointer :: p2(:) => NULL()
  !CHECK-NOT: error:
  !CHECK-NOT: warning:
  !CHECK: PRINT *, allocated(p2)
  print *, allocated(p2)
end

subroutine s2
  interface allocated
    logical function specificallocated(p)
      class(*), pointer, intent(in) :: p(..)
    end
  end interface
  real, pointer :: p3(:) => NULL()
  !CHECK-NOT: error:
  !CHECK-NOT: warning:
  !CHECK: PRINT *, specificallocated(p3)
  print *, allocated(p3)
end
