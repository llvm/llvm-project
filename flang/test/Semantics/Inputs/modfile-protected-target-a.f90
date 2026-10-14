module pt_mod
  integer, pointer, protected_target :: p
  integer, target :: x
  type holder
    integer, pointer, protected_target :: q
  end type
contains
  function get() result(r)
    integer, pointer, protected_target :: r
    r => x
  end
  subroutine read_pt(a)
    integer, pointer, protected_target, intent(in) :: a
  end
  subroutine read_value(a)
    integer, intent(in) :: a
  end
  subroutine ordinary_pointer(a)
    integer, pointer, intent(in) :: a
  end
  subroutine default_intent(a)
    integer :: a
  end
end module

!Expect: pt_mod.mod
!module pt_mod
!integer(4),pointer,protected_target::p
!integer(4),target::x
!type::holder
!integer(4),pointer,protected_target::q
!end type
!contains
!function get() result(r)
!integer(4),pointer,protected_target::r
!end
!subroutine read_pt(a)
!integer(4),intent(in),pointer,protected_target::a
!end
!subroutine read_value(a)
!integer(4),intent(in)::a
!end
!subroutine ordinary_pointer(a)
!integer(4),intent(in),pointer::a
!end
!subroutine default_intent(a)
!integer(4)::a
!end
!end
