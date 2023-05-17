! RUN: %python %S/test_errors.py %s %flang_fc1
! Allow the same external or intrinsic procedure to be use-associated
! by multiple paths when they are unambiguous.
module m1
  intrinsic :: sin
  intrinsic :: iabs
  interface
    subroutine ext1(a, b)
      integer, intent(in) :: a(:)
      real, intent(in) :: b(:)
    end subroutine
    subroutine ext2(a, b)
      real, intent(in) :: a(:)
      integer, intent(in) :: b(:)
    end subroutine
  end interface
end module m1

module m2
  intrinsic :: sin, tan
  intrinsic :: iabs, idim
  interface
    subroutine ext1(a, b)
      integer, intent(in) :: a(:)
      real, intent(in) :: b(:)
    end subroutine
    subroutine ext2(a, b)
      real, intent(in) :: a(:)
      integer, intent(in) :: b(:)
    end subroutine
  end interface
end module m2

subroutine s2a
  use m1
  use m2
  procedure(sin), pointer :: p1 => sin
  procedure(iabs), pointer :: p2 => iabs
  procedure(ext1), pointer :: p3 => ext1
  procedure(ext2), pointer :: p4 => ext2
end subroutine

subroutine s2b
  use m1, only: x1 => sin, x2 => iabs, x3 => ext1, x4 => ext2
  use m2, only: x1 => sin, x2 => iabs, x3 => ext1, x4 => ext2
  use m1, only: iface1 => sin, iface2 => iabs, iface3 => ext1, iface4 => ext2
  procedure(iface1), pointer :: p1 => x1
  procedure(iface2), pointer :: p2 => x2
  procedure(iface3), pointer :: p3 => x3
  procedure(iface4), pointer :: p4 => x4
end subroutine

module m3
  use m1
  use m2
end module
subroutine s3
  use m3
  procedure(sin), pointer :: p1 => sin
  procedure(iabs), pointer :: p2 => iabs
  procedure(ext1), pointer :: p3 => ext1
  procedure(ext2), pointer :: p4 => ext2
end subroutine

module m4
  use m1, only: x1 => sin, x2 => iabs, x3 => ext1, x4 => ext2
  use m2, only: x1 => sin, x2 => iabs, x3 => ext1, x4 => ext2
end module
subroutine s4
  use m4
  use m1, only: iface1 => sin, iface2 => iabs, iface3 => ext1, iface4 => ext2
  procedure(iface1), pointer :: p1 => x1
  procedure(iface2), pointer :: p2 => x2
  procedure(iface3), pointer :: p3 => x3
  procedure(iface4), pointer :: p4 => x4
end subroutine

subroutine s5
  use m1, only: x1 => sin, x2 => iabs, x3 => ext1, x4 => ext2
  use m2, only: x1 => tan, x2 => idim, x3 => ext2, x4 => ext1
  use m1, only: iface1 => sin, iface2 => iabs, iface3 => ext1, iface4 => ext2
  !ERROR: Reference to 'x1' is ambiguous
  procedure(iface1), pointer :: p1 => x1
  !ERROR: Reference to 'x2' is ambiguous
  procedure(iface2), pointer :: p2 => x2
  !ERROR: Reference to 'x3' is ambiguous
  procedure(iface3), pointer :: p3 => x3
  !ERROR: Reference to 'x4' is ambiguous
  procedure(iface4), pointer :: p4 => x4
end subroutine
