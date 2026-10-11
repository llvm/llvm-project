! RUN: %python %S/test_modfile.py %s %flang_fc1

module modfile_protected_target
  implicit none
  integer, target, save :: x = 1
  integer, pointer, protected_target, save :: p => x
  type holder
    integer, pointer, protected_target :: c => x
  end type
contains
  subroutine takes_plain(a)
    integer, pointer :: a
  end
  subroutine takes_pt(a)
    integer, pointer, protected_target :: a
  end
  function plain_result() result(r)
    integer, pointer :: r
    r => x
  end
  function pt_result() result(r)
    integer, pointer, protected_target :: r
    r => x
  end
end module

!Expect: modfile_protected_target.mod
!module modfile_protected_target
!integer(4),save,target::x
!integer(4),pointer,protected_target,save::p
!type::holder
!integer(4),pointer,protected_target::c=>x
!end type
!contains
!subroutine takes_plain(a)
!integer(4),pointer::a
!end
!subroutine takes_pt(a)
!integer(4),pointer,protected_target::a
!end
!function plain_result() result(r)
!integer(4),pointer::r
!end
!function pt_result() result(r)
!integer(4),pointer,protected_target::r
!end
!end
