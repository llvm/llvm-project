!RUN: %flang -g -S -emit-llvm %s -o - | FileCheck %s

!CHECK: !DIGlobalVariable(name: "gsubptr"
!CHECK-SAME: type: [[SPTRTYPE:![0-9]+]]
!CHECK: !DIGlobalVariable(name: "gfunptr"
!CHECK-SAME: type: [[FPTRTYPE:![0-9]+]]
!CHECK: [[FPTRTYPE]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: [[FUNTYPE:![0-9]+]]
!CHECK: [[FUNTYPE]] = !DISubroutineType(types: [[FPARLIST:![0-9]+]])
!CHECK: [[FPARLIST]] = !{[[INTTYPE:![0-9]+]], [[INTTYPE]], [[REALTYPE:![0-9]+]]}
!CHECK: [[INTTYPE]] = !DIBasicType(name: "integer"
!CHECK: [[REALTYPE]] = !DIBasicType(name: "real"
!CHECK: [[SPTRTYPE]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: [[SUBTYPE:![0-9]+]]
!CHECK: [[SUBTYPE]] = !DISubroutineType(types: [[SPARLIST:![0-9]+]])
!CHECK: [[SPARLIST]] = !{null, [[REALTYPE]]}
!CHECK: !DILocalVariable(name: "lsubptr"
!CHECK-SAME: type: [[SPTRTYPE]]
!CHECK: !DILocalVariable(name: "lfunptr"
!CHECK-SAME: type: [[FPTRTYPE]]

program test

  interface
    integer function fun (farg1, farg2)
      integer :: farg1
      real :: farg2
    end function fun
    subroutine sub (sarg)
      real :: sarg
    end subroutine
  end interface

  procedure(fun), pointer:: gfunptr => NULL()
  procedure(fun), pointer:: lfunptr
  procedure(sub), pointer:: gsubptr => NULL()
  procedure(sub), pointer:: lsubptr

  gfunptr => fun
  lfunptr => fun
  print *, gfunptr (3, 2.5)
  print *, lfunptr (3, 2.5)

  gsubptr => sub
  lsubptr => sub
  call gsubptr (2.5)
  call lsubptr (2.5)

end program test

subroutine sub (a)
  real :: a, res
  res = 2.1 * a
  print *, res
end subroutine

integer function fun (x, y)
  implicit none
  integer :: x
  real :: y
  fun = x + y + 1
end function fun
