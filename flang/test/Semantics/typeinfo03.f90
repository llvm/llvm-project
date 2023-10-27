!RUN: bbc --dump-symbols %s | FileCheck %s
!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s
!Ensure that type with pointer component(s) has "noinitializationneeded=0"
module m
  type hasPointer
    class(*), pointer :: sp, ap(:)
  end type
end module
!CHECK: .dt.haspointer, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.haspointer,sizeinbytes=104_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.haspointer,procptr=NULL(),special=NULL(),specialbitset=0_4,hasparent=0_1,noinitializationneeded=0_1,nodestructionneeded=1_1,nofinalizationneeded=1_1)
