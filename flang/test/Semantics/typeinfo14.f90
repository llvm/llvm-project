!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s
!Check the "isenumeration" and "enumeratorcount" settings for F2023
!enumeration types and for a derived type with an enumeration component.
module m
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
  enumeration type :: single
    enumerator :: only
  end enumeration type
  type :: holder
    type(color) :: c
  end type
end module

!CHECK: .dt.color, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.color,sizeinbytes=4_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.color,procptr=NULL(),special=NULL(),specialbitset=0_4,hasparent=0_1,noinitializationneeded=1_1,nodestructionneeded=1_1,nofinalizationneeded=1_1,nodefinedassignment=1_1,isenumeration=1_1,enumeratorcount=3_4)
!CHECK: .dt.holder, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.holder,sizeinbytes=4_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.holder,procptr=NULL(),special=NULL(),specialbitset=0_4,hasparent=0_1,noinitializationneeded=1_1,nodestructionneeded=1_1,nofinalizationneeded=1_1,nodefinedassignment=1_1,isenumeration=0_1,enumeratorcount=0_4)
!CHECK: .dt.single, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.single,sizeinbytes=4_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.single,procptr=NULL(),special=NULL(),specialbitset=0_4,hasparent=0_1,noinitializationneeded=1_1,nodestructionneeded=1_1,nofinalizationneeded=1_1,nodefinedassignment=1_1,isenumeration=1_1,enumeratorcount=1_4)
