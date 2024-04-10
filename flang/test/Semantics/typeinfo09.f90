!RUN: bbc --dump-symbols %s | FileCheck %s
!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s
! test setting of isargdescriptorset in the runtime type info.

module m
 type :: sometype
 contains
  procedure :: copy => copy_impl
  generic :: assignment(=) => copy
 end type
interface
  subroutine copy_impl(this, x)
    import
    class(sometype), intent(out) :: this
    type(sometype), target, intent(in) :: x
  end subroutine
end interface
end module

!CHECK: .s.sometype, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(specialbinding) shape: 0_8:0_8 init:[specialbinding::specialbinding(which=1_1,isargdescriptorset=1_1,istypebound=1_1,isargcontiguousset=0_1,proc=copy_impl)]
