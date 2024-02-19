!RUN: bbc --dump-symbols %s | FileCheck %s
!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s
! Ensure that cycles via POINTERs do not instantiate incomplete derived
! types that would lead to types whose sizeinbytes=0
program main
  implicit none
  type t1
    type(t2), pointer :: b
  end type t1
!CHECK: .dt.t1, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.t1,sizeinbytes=40_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.t1,procptr=NULL(),special=NULL(),specialbitset=0_4,hasparent=0_1,noinitializationneeded=0_1,nodestructionneeded=1_1,nofinalizationneeded=1_1)
  type :: t2
    type(t1) :: a
  end type t2
! CHECK: .dt.t2, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.t2,sizeinbytes=40_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.t2,procptr=NULL(),special=NULL(),specialbitset=0_4,hasparent=0_1,noinitializationneeded=0_1,nodestructionneeded=1_1,nofinalizationneeded=1_1)
end program main

