! RUN: %python %S/test_symbols.py %s %flang_fc1
! Forward references in pointer initializers and TBP bindings.

!DEF: /m Module
module m
 implicit none
 abstract interface
  !DEF: /m/iface ABSTRACT, PUBLIC (Subroutine) Subprogram
  subroutine iface
  end subroutine
 end interface
 !DEF: /m/op1 POINTER, PUBLIC ObjectEntity REAL(4)
 real, pointer :: op1
 !DEF: /m/op2 POINTER, PUBLIC ObjectEntity REAL(4)
 !DEF: /m/null INTRINSIC, PUBLIC, PURE (Function) ProcEntity
 real, pointer :: op2 => null()
 !DEF: /m/op3 POINTER, PUBLIC ObjectEntity REAL(4)
 !DEF: /m/x PUBLIC, TARGET ObjectEntity REAL(4)
 real, pointer :: op3 => x
 !DEF: /m/op4 POINTER, PUBLIC ObjectEntity REAL(4)
 !DEF: /m/y PUBLIC, TARGET ObjectEntity REAL(4)
 real, pointer :: op4 => y(1)
 !REF: /m/iface
 !DEF: /m/pp1 EXTERNAL, POINTER, PUBLIC (Subroutine) ProcEntity
 procedure(iface), pointer :: pp1
 !REF: /m/iface
 !DEF: /m/pp2 EXTERNAL, POINTER, PUBLIC (Subroutine) ProcEntity
 !REF: /m/null
 procedure(iface), pointer :: pp2 => null()
 !REF: /m/iface
 !DEF: /m/pp3 EXTERNAL, POINTER, PUBLIC (Subroutine) ProcEntity
 !DEF: /m/ext1 EXTERNAL, PUBLIC (Subroutine) ProcEntity
 procedure(iface), pointer :: pp3 => ext1
 !REF: /m/iface
 !DEF: /m/pp4 EXTERNAL, POINTER, PUBLIC (Subroutine) ProcEntity
 !DEF: /m/ext2 EXTERNAL, PUBLIC (Subroutine) Subprogram
 procedure(iface), pointer :: pp4 => ext2
 !REF: /m/iface
 !DEF: /m/pp5 EXTERNAL, POINTER, PUBLIC (Subroutine) ProcEntity
 !DEF: /m/ext3 EXTERNAL, PUBLIC (Subroutine) ProcEntity
 procedure(iface), pointer :: pp5 => ext3
 !REF: /m/iface
 !DEF: /m/pp6 EXTERNAL, POINTER, PUBLIC (Subroutine) ProcEntity
 !DEF: /m/modproc1 PUBLIC (Subroutine) Subprogram
 procedure(iface), pointer :: pp6 => modproc1
 !DEF: /m/t1 PUBLIC DerivedType
 type :: t1
  !DEF: /m/t1/opc1 POINTER ObjectEntity REAL(4)
  real, pointer :: opc1
  !DEF: /m/t1/opc2 POINTER ObjectEntity REAL(4)
  !REF: /m/null
  real, pointer :: opc2 => null()
  !DEF: /m/t1/opc3 POINTER ObjectEntity REAL(4)
  !REF: /m/x
  real, pointer :: opc3 => x
  !DEF: /m/t1/opc4 POINTER ObjectEntity REAL(4)
  !REF: /m/y
  real, pointer :: opc4 => y(1)
  !REF: /m/iface
  !DEF: /m/t1/ppc1 NOPASS, POINTER (Subroutine) ProcEntity
  procedure(iface), nopass, pointer :: ppc1
  !REF: /m/iface
  !DEF: /m/t1/ppc2 NOPASS, POINTER (Subroutine) ProcEntity
  !REF: /m/null
  procedure(iface), nopass, pointer :: ppc2 => null()
  !REF: /m/iface
  !DEF: /m/t1/ppc3 NOPASS, POINTER (Subroutine) ProcEntity
  !REF: /m/ext1
  procedure(iface), nopass, pointer :: ppc3 => ext1
  !REF: /m/iface
  !DEF: /m/t1/ppc4 NOPASS, POINTER (Subroutine) ProcEntity
  !REF: /m/ext2
  procedure(iface), nopass, pointer :: ppc4 => ext2
  !REF: /m/iface
  !DEF: /m/t1/ppc5 NOPASS, POINTER (Subroutine) ProcEntity
  !REF: /m/ext3
  procedure(iface), nopass, pointer :: ppc5 => ext3
  !REF: /m/iface
  !DEF: /m/t1/ppc6 NOPASS, POINTER (Subroutine) ProcEntity
  !REF: /m/modproc1
  procedure(iface), nopass, pointer :: ppc6 => modproc1
 contains
  !DEF: /m/t1/b2 NOPASS ProcBinding
  !REF: /m/ext2
  procedure, nopass :: b2 => ext2
  !DEF: /m/t1/b3 NOPASS ProcBinding
  !REF: /m/ext3
  procedure, nopass :: b3 => ext3
  !DEF: /m/t1/b4 NOPASS ProcBinding
  !REF: /m/modproc1
  procedure, nopass :: b4 => modproc1
 end type
 !DEF: /m/pdt1 PUBLIC DerivedType
 !DEF: /m/pdt1/k TypeParam INTEGER(4)
 type :: pdt1(k)
  !REF: /m/pdt1/k
  integer, kind :: k
  !DEF: /m/pdt1/opc1 POINTER ObjectEntity REAL(4)
  real, pointer :: opc1
  !DEF: /m/pdt1/opc2 POINTER ObjectEntity REAL(4)
  !REF: /m/null
  real, pointer :: opc2 => null()
  !DEF: /m/pdt1/opc3 POINTER ObjectEntity REAL(4)
  !REF: /m/x
  real, pointer :: opc3 => x
  !DEF: /m/pdt1/opc4 POINTER ObjectEntity REAL(4)
  !REF: /m/y
  !REF: /m/pdt1/k
  real, pointer :: opc4 => y(k)
  !REF: /m/iface
  !DEF: /m/pdt1/ppc1 NOPASS, POINTER (Subroutine) ProcEntity
  procedure(iface), nopass, pointer :: ppc1
  !REF: /m/iface
  !DEF: /m/pdt1/ppc2 NOPASS, POINTER (Subroutine) ProcEntity
  !REF: /m/null
  procedure(iface), nopass, pointer :: ppc2 => null()
  !REF: /m/iface
  !DEF: /m/pdt1/ppc3 NOPASS, POINTER (Subroutine) ProcEntity
  !REF: /m/ext1
  procedure(iface), nopass, pointer :: ppc3 => ext1
  !REF: /m/iface
  !DEF: /m/pdt1/ppc4 NOPASS, POINTER (Subroutine) ProcEntity
  !REF: /m/ext2
  procedure(iface), nopass, pointer :: ppc4 => ext2
  !REF: /m/iface
  !DEF: /m/pdt1/ppc5 NOPASS, POINTER (Subroutine) ProcEntity
  !REF: /m/ext3
  procedure(iface), nopass, pointer :: ppc5 => ext3
  !REF: /m/iface
  !DEF: /m/pdt1/ppc6 NOPASS, POINTER (Subroutine) ProcEntity
  !REF: /m/modproc1
  procedure(iface), nopass, pointer :: ppc6 => modproc1
 contains
  !DEF: /m/pdt1/b2 NOPASS ProcBinding
  !REF: /m/ext2
  procedure, nopass :: b2 => ext2
  !DEF: /m/pdt1/b3 NOPASS ProcBinding
  !REF: /m/ext3
  procedure, nopass :: b3 => ext3
  !DEF: /m/pdt1/b4 NOPASS ProcBinding
  !REF: /m/modproc1
  procedure, nopass :: b4 => modproc1
 end type
 !REF: /m/t1
 !DEF: /m/t1x PUBLIC ObjectEntity TYPE(t1)
 type(t1) :: t1x
 !REF: /m/pdt1
 !DEF: /m/pdt1x PUBLIC ObjectEntity TYPE(pdt1(k=1_4))
 type(pdt1(1)) :: pdt1x
 !REF: /m/x
 !REF: /m/y
 real, target :: x, y(2)
 !REF: /m/ext1
 external :: ext1
 !REF: /m/iface
 !REF: /m/ext3
 procedure(iface) :: ext3
 interface
  !REF: /m/ext2
  subroutine ext2
  end subroutine
 end interface
 !DEF: /m/op10 POINTER, PUBLIC ObjectEntity REAL(4)
 !REF: /m/x
 real, pointer :: op10 => x
 !DEF: /m/op11 POINTER, PUBLIC ObjectEntity REAL(4)
 !REF: /m/y
 real, pointer :: op11 => y(1)
 !REF: /m/iface
 !DEF: /m/pp10 EXTERNAL, POINTER, PUBLIC (Subroutine) ProcEntity
 !REF: /m/ext1
 procedure(iface), pointer :: pp10 => ext1
 !REF: /m/iface
 !DEF: /m/pp11 EXTERNAL, POINTER, PUBLIC (Subroutine) ProcEntity
 !REF: /m/ext2
 procedure(iface), pointer :: pp11 => ext2
 !DEF: /m/t2 PUBLIC DerivedType
 type :: t2
  !DEF: /m/t2/opc10 POINTER ObjectEntity REAL(4)
  !REF: /m/x
  real, pointer :: opc10 => x
  !DEF: /m/t2/opc11 POINTER ObjectEntity REAL(4)
  !REF: /m/y
  real, pointer :: opc11 => y(1)
  !REF: /m/iface
  !DEF: /m/t2/ppc10 NOPASS, POINTER (Subroutine) ProcEntity
  !REF: /m/ext1
  procedure(iface), nopass, pointer :: ppc10 => ext1
  !REF: /m/iface
  !DEF: /m/t2/ppc11 NOPASS, POINTER (Subroutine) ProcEntity
  !REF: /m/ext2
  procedure(iface), nopass, pointer :: ppc11 => ext2
 contains
  !DEF: /m/t2/b10 NOPASS ProcBinding
  !REF: /m/ext2
  procedure, nopass :: b10 => ext2
  !DEF: /m/t2/b11 NOPASS ProcBinding
  !REF: /m/ext3
  procedure, nopass :: b11 => ext3
 end type
 !DEF: /m/pdt2 PUBLIC DerivedType
 !DEF: /m/pdt2/k TypeParam INTEGER(4)
 type :: pdt2(k)
  !REF: /m/pdt2/k
  integer, kind :: k
  !DEF: /m/pdt2/opc10 POINTER ObjectEntity REAL(4)
  !REF: /m/x
  real, pointer :: opc10 => x
  !DEF: /m/pdt2/opc11 POINTER ObjectEntity REAL(4)
  !REF: /m/y
  !REF: /m/pdt2/k
  real, pointer :: opc11 => y(k)
  !REF: /m/iface
  !DEF: /m/pdt2/ppc10 NOPASS, POINTER (Subroutine) ProcEntity
  !REF: /m/ext1
  procedure(iface), nopass, pointer :: ppc10 => ext1
  !REF: /m/iface
  !DEF: /m/pdt2/ppc11 NOPASS, POINTER (Subroutine) ProcEntity
  !REF: /m/ext2
  procedure(iface), nopass, pointer :: ppc11 => ext2
 contains
  !DEF: /m/pdt2/b10 NOPASS ProcBinding
  !REF: /m/ext2
  procedure, nopass :: b10 => ext2
  !DEF: /m/pdt2/b11 NOPASS ProcBinding
  !REF: /m/ext3
  procedure, nopass :: b11 => ext3
 end type
 !REF: /m/t2
 !DEF: /m/t2x PUBLIC ObjectEntity TYPE(t2)
 type(t2) :: t2x
 !REF: /m/pdt2
 !DEF: /m/pdt2x PUBLIC ObjectEntity TYPE(pdt2(k=1_4))
 type(pdt2(1)) :: pdt2x
contains
 !REF: /m/modproc1
 subroutine modproc1
 end subroutine
end module
!DEF: /ext1 (Subroutine) Subprogram
subroutine ext1
end subroutine
!DEF: /ext2 (Subroutine) Subprogram
subroutine ext2
end subroutine
!DEF: /ext3 (Subroutine) Subprogram
subroutine ext3
end subroutine
!DEF: /main MainProgram
program main
 !REF: /m
 use :: m
 !DEF: /main/pdt1 Use
 !DEF: /main/pdt1y ObjectEntity TYPE(pdt1(k=2_4))
 type(pdt1(2)) :: pdt1y
 !DEF: /main/pdt2 Use
 !DEF: /main/pdt2y ObjectEntity TYPE(pdt2(k=2_4))
 type(pdt2(2)) :: pdt2y
 print *, "compiled"
end program
