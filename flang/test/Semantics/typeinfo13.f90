!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s
!Ensure ASSIGNMENT(=) overrides are applied to the special procedures table.
module m
  type base
   contains
    procedure :: baseAssign
    generic :: assignment(=) => baseAssign
  end type
  type, extends(base) :: child
   contains
    procedure :: override
    generic :: assignment(=) => override
  end type
 contains
  impure elemental subroutine baseAssign(to, from)
    class(base), intent(out) :: to
    type(base), intent(in) :: from
  end
  impure elemental subroutine override(to, from)
    class(child), intent(out) :: to
    type(child), intent(in) :: from
  end
end

!CHECK: .s.child, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(specialbinding) shape: 0_8:0_8 init:[specialbinding::specialbinding(which=2_1,isargdescriptorset=1_1,istypebound=1_1,isargcontiguousset=0_1,proc=override)]
!CHECK: .v.child, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(binding) shape: 0_8:1_8 init:[binding::binding(proc=baseassign,name=.n.baseassign),binding(proc=override,name=.n.override)]
