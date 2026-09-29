!RUN: bbc --dump-symbols %s | FileCheck %s
!Check "nodefinedassignment" settings.

module m01

  type hasAsst1
   contains
    procedure asst1
    generic :: assignment(=) => asst1
  end type
!CHECK: .dt.hasasst1, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=.v.hasasst1,name=.n.hasasst1,sizeinbytes=0_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=NULL(),procptr=NULL(),special=.s.hasasst1,specialbitset=4_4,hasparent=0_1,noinitializationneeded=1_1,nodestructionneeded=1_1,nofinalizationneeded=1_1,nodefinedassignment=0_1)

  type hasAsst2 ! no defined assignment relevant to the runtime
  end type
  interface assignment(=)
    procedure asst2
  end interface
!CHECK: .dt.hasasst2, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.hasasst2,sizeinbytes=0_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=NULL(),procptr=NULL(),special=NULL(),specialbitset=0_4,hasparent=0_1,noinitializationneeded=1_1,nodestructionneeded=1_1,nofinalizationneeded=1_1,nodefinedassignment=1_1)

  type test1
    type(hasAsst1) c
  end type
!CHECK: .dt.test1, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.test1,sizeinbytes=0_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.test1,procptr=NULL(),special=NULL(),specialbitset=0_4,hasparent=0_1,noinitializationneeded=1_1,nodestructionneeded=1_1,nofinalizationneeded=1_1,nodefinedassignment=0_1)

  type test2
    type(hasAsst2) c
  end type
!CHECK: .dt.test2, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.test2,sizeinbytes=0_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.test2,procptr=NULL(),special=NULL(),specialbitset=0_4,hasparent=0_1,noinitializationneeded=1_1,nodestructionneeded=1_1,nofinalizationneeded=1_1,nodefinedassignment=1_1)

  type test3
    type(hasAsst1), pointer :: p
  end type
!CHECK: .dt.test3, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.test3,sizeinbytes=40_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.test3,procptr=NULL(),special=NULL(),specialbitset=0_4,hasparent=0_1,noinitializationneeded=0_1,nodestructionneeded=1_1,nofinalizationneeded=1_1,nodefinedassignment=1_1)

  type test4
    type(hasAsst2), pointer :: p
  end type
!CHECK: .dt.test4, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.test4,sizeinbytes=40_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.test4,procptr=NULL(),special=NULL(),specialbitset=0_4,hasparent=0_1,noinitializationneeded=0_1,nodestructionneeded=1_1,nofinalizationneeded=1_1,nodefinedassignment=1_1)

  type, extends(hasAsst1) :: test5
  end type
!CHECK: .dt.test5, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=.v.test5,name=.n.test5,sizeinbytes=0_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.test5,procptr=NULL(),special=.s.test5,specialbitset=4_4,hasparent=1_1,noinitializationneeded=1_1,nodestructionneeded=1_1,nofinalizationneeded=1_1,nodefinedassignment=0_1)

  type, extends(hasAsst2) :: test6
  end type
!CHECK: .dt.test6, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.test6,sizeinbytes=0_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.test6,procptr=NULL(),special=NULL(),specialbitset=0_4,hasparent=1_1,noinitializationneeded=1_1,nodestructionneeded=1_1,nofinalizationneeded=1_1,nodefinedassignment=1_1)

  type test7
    type(test7), allocatable :: c
  end type
!CHECK: .dt.test7, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.test7,sizeinbytes=40_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.test7,procptr=NULL(),special=NULL(),specialbitset=0_4,hasparent=0_1,noinitializationneeded=0_1,nodestructionneeded=0_1,nofinalizationneeded=1_1,nodefinedassignment=1_1)

  type test8
    class(test8), allocatable :: c
  end type
!CHECK: .dt.test8, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=NULL(),name=.n.test8,sizeinbytes=40_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.test8,procptr=NULL(),special=NULL(),specialbitset=0_4,hasparent=0_1,noinitializationneeded=0_1,nodestructionneeded=0_1,nofinalizationneeded=0_1,nodefinedassignment=0_1)

 contains
  impure elemental subroutine asst1(left, right)
    class(hasAsst1), intent(out) :: left
    class(hasAsst1), intent(in) :: right
  end
  impure elemental subroutine asst2(left, right)
    class(hasAsst2), intent(out) :: left
    class(hasAsst2), intent(in) :: right
  end
end
