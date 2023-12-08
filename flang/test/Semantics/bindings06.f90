! RUN: %flang_fc1 -fdebug-dump-symbols %s 2>&1 | FileCheck %s
module ma
  type a
   contains
    procedure, private, nopass :: tbp_private => sub_a1
    procedure, public, nopass :: tbp_public => sub_a2
    generic, public :: gen => tbp_private, tbp_public
  end type
 contains
  subroutine sub_a1(w)
    character*(*), intent(in) :: w
    print *, w, ' -> a1'
  end
  subroutine sub_a2(w, j)
    character*(*), intent(in) :: w
    integer, intent(in) :: j
    print *, w, ' -> a2'
  end
  subroutine test_mono_a
    type(a) x
    call x%tbp_private('type(a) tbp_private')
    call x%tbp_public('type(a) tbp_public', 0)
    call x%gen('type(a) gen 1')
    call x%gen('type(a) gen 2', 0)
  end
  subroutine test_poly_a(x, w)
    class(a), intent(in) :: x
    character*(*), intent(in) :: w
    call x%tbp_private('class(a) (' // w // ') tbp_private')
    call x%tbp_public('class(a) (' // w // ') tbp_public', 0)
    call x%gen('class(a) (' // w // ') gen 1')
    call x%gen('class(a) (' // w // ') gen 2', 0)
  end
end

module mb
  use ma
  type, extends(a) :: ab
   contains
    procedure, private, nopass :: tbp_private => sub_ab1
    procedure, public, nopass :: tbp_public => sub_ab2
  end type
 contains
  subroutine sub_ab1(w)
    character*(*), intent(in) :: w
    print *, w, ' -> ab1'
  end
  subroutine sub_ab2(w, j)
    character*(*), intent(in) :: w
    integer, intent(in) :: j
    print *, w, ' -> ab2'
  end
  subroutine test_mono_ab
    type(ab) x
    call x%tbp_private('type(ab) tbp_private')
    call x%tbp_public('type(ab) tbp_public', 0)
    call x%gen('type(ab) gen 1')
    call x%gen('type(ab) gen 2', 0)
  end
  subroutine test_poly_ab(x, w)
    class(ab), intent(in) :: x
    character*(*), intent(in) :: w
    call x%tbp_private('class(ab) (' // w // ') tbp_private')
    call x%tbp_public('class(ab) (' // w // ') tbp_public', 0)
    call x%gen('class(ab) (' // w // ') gen 1')
    call x%gen('class(ab) (' // w // ') gen 2', 0)
  end
end

program main
  use mb
  call test_mono_a
  call test_mono_ab
  call test_poly_a(a(), 'a')
  call test_poly_a(ab(), 'ab')
  call test_poly_ab(ab(), 'ab')
end

!CHECK: .v.a, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(binding) shape: 0_8:1_8 init:[binding::binding(proc=sub_a1,name=.n.tbp_private),binding(proc=sub_a2,name=.n.tbp_public)]
!CHECK: .v.ab, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(binding) shape: 0_8:2_8 init:[binding::binding(proc=sub_a1,name=.n.tbp_private),binding(proc=sub_ab2,name=.n.tbp_public),binding(proc=sub_ab1,name=.n.tbp_private)]
!CHECK: tbp_private, NOPASS, PRIVATE: ProcBinding => sub_ab1 numPrivatesNotOverridden: 1
