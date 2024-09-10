!RUN: bbc --dump-symbols %s | FileCheck %s
!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s

! Test defined assignment with allocatable / pointer LHS arguments.
! The special bindings for the defined assignments must reflect that
! their LHS arguments are allocatables and pointers.
! (This program is executable and should print 1; 102; 3 204.)

module m
  type :: base
     integer :: i
  contains
     procedure, pass(src) :: ass1, ass2
     generic :: assignment(=) => ass1, ass2
  end type base
  type, extends(base) :: derived
  end type

!CHECK: .dt.base, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=.v.base,name=.n.base,sizeinbytes=4_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.base,procptr=NULL(),special=.s.base,specialbitset=12_4,hasparent=0_1,noinitializationneeded=1_1,nodestructionneeded=1_1,nofinalizationneeded=1_1)
!CHECK: .dt.derived, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(derivedtype) init:derivedtype(binding=.v.derived,name=.n.derived,sizeinbytes=4_8,uninstantiated=NULL(),kindparameter=NULL(),lenparameterkind=NULL(),component=.c.derived,procptr=NULL(),special=.s.derived,specialbitset=12_4,hasparent=1_1,noinitializationneeded=1_1,nodestructionneeded=1_1,nofinalizationneeded=1_1)
!CHECK: .s.base, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(specialbinding) shape: 0_8:1_8 init:[specialbinding::specialbinding(which=2_1,isargdescriptorset=3_1,istypebound=1_1,isargcontiguousset=0_1,proc=ass1),specialbinding(which=3_1,isargdescriptorset=3_1,istypebound=1_1,isargcontiguousset=0_1,proc=ass2)]
!CHECK: .s.derived, SAVE, TARGET (CompilerCreated, ReadOnly): ObjectEntity type: TYPE(specialbinding) shape: 0_8:1_8 init:[specialbinding::specialbinding(which=2_1,isargdescriptorset=3_1,istypebound=1_1,isargcontiguousset=0_1,proc=ass1),specialbinding(which=3_1,isargdescriptorset=3_1,istypebound=1_1,isargcontiguousset=0_1,proc=ass2)]

contains
  subroutine ass1(res, src)
    class(base), allocatable, intent(out) :: res
    class(base), intent(in) :: src
    allocate(res, source=src)
    res%i = res%i + 100
  end subroutine
  subroutine ass2(res, src)
    class(base), pointer, intent(in out) :: res
    class(base), intent(in) :: src
    allocate(res, source=src)
    res%i = src%i + 200
  end subroutine
end
program genext
  use m
  type(derived) :: od1
  class(base), allocatable :: od2
  class(base), pointer :: od3a, od3b
  od1 = derived(1)
  print *, od1%i
  od2 = derived(2)
  print *, od2%i
  allocate(od3a)
  od3a%i = 3
  od3b => od3a
  od3b = derived(4)
  print *, od3a%i, od3b%i
end program genext
