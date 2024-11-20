!RUN: bbc --dump-symbols %s | FileCheck %s
!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s

!Tests that derived types with polymorphic potential subobject
!components do not have their noFinalizationNeeded flags set, even
!when those components are packaged within another allocatable.

type t1
  class(*), allocatable :: a
end type
type t2
  type(t1), allocatable :: b
end type
type(t2) x
end

!CHECK: .dt.t2, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.t2,sizeinbytes=40_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.t2,procptr=NULL(),special=NULL(),specialbitset=0_4,hasparent=0_1,noinitializationneeded=0_1,nodestructionneeded=0_1,nofinalizationneeded=0_1)
