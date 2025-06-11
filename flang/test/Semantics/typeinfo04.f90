!RUN: bbc --dump-symbols %s | FileCheck %s
!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s
! Ensure ABSTRACT types are properly marked for finalization &/or
! destruction.
module m
  type :: finalizable
   contains
    final :: final
  end type
!CHECK: .dt.finalizable, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.finalizable,sizeinbytes=0_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=NULL(),procptr=NULL(),special=.s.finalizable,specialbitset=128_4,hasparent=0_1,noinitializationneeded=1_1,nodestructionneeded=0_1,nofinalizationneeded=0_1)
  type, abstract :: t1
  end type
!CHECK: .dt.t1, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(name=.n.t1,sizeinbytes=0_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=NULL(),procptr=NULL(),specialbitset=0_4,hasparent=0_1,noinitializationneeded=1_1,nodestructionneeded=1_1,nofinalizationneeded=1_1)
  type, abstract :: t2
    real, allocatable :: a(:)
  end type
!CHECK: .dt.t2, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(name=.n.t2,sizeinbytes=48_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.t2,procptr=NULL(),specialbitset=0_4,hasparent=0_1,noinitializationneeded=0_1,nodestructionneeded=0_1,nofinalizationneeded=1_1)
  type, abstract :: t3
    type(finalizable) :: x
  end type
!CHECK: .dt.t3, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(name=.n.t3,sizeinbytes=0_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.t3,procptr=NULL(),specialbitset=0_4,hasparent=0_1,noinitializationneeded=1_1,nodestructionneeded=0_1,nofinalizationneeded=0_1)
 contains
  impure elemental subroutine final(x)
    type(finalizable), intent(in out) :: x
  end
end
