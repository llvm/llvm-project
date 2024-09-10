!RUN: bbc --dump-symbols %s | FileCheck %s
!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s

module m1
  type base
   contains
    procedure :: wf => wf1
    generic :: write(formatted) => wf
  end type
  type, extends(base) :: extended
   contains
    procedure :: wf => wf2
  end type
 contains
  subroutine wf1(x,u,iot,v,iostat,iomsg)
    class(base), intent(in) :: x
    integer, intent(in) :: u
    character(len=*), intent(in) :: iot
    integer, intent(in) :: v(:)
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg
  end subroutine
  subroutine wf2(x,u,iot,v,iostat,iomsg)
    class(extended), intent(in) :: x
    integer, intent(in) :: u
    character(len=*), intent(in) :: iot
    integer, intent(in) :: v(:)
    integer, intent(out) :: iostat
    character(len=*), intent(inout) :: iomsg
  end subroutine
end module
!CHECK: .s.base, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(specialbinding) shape: 0_8:0_8 init:[specialbinding::specialbinding(which=7_1,isargdescriptorset=1_1,istypebound=1_1,isargcontiguousset=0_1,proc=wf1)]
!CHECK: .s.extended, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(specialbinding) shape: 0_8:0_8 init:[specialbinding::specialbinding(which=7_1,isargdescriptorset=1_1,istypebound=1_1,isargcontiguousset=0_1,proc=wf2)]
