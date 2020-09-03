! RUN: %S/test_symbols.sh %s %t %f18
! Tests for "proc-interface" semantics.
! These cases are all valid.

!DEF: /module1 Module
module module1
 abstract interface
  !DEF: /module1/abstract1 PUBLIC (Function) Subprogram REAL(4)
  !DEF: /module1/abstract1/x INTENT(IN) ObjectEntity REAL(4)
  real function abstract1(x)
   !REF: /module1/abstract1/x
   real, intent(in) :: x
  end function abstract1
 end interface

 interface
  !DEF: /module1/explicit1 EXTERNAL, PUBLIC (Function) Subprogram REAL(4)
  !DEF: /module1/explicit1/x INTENT(IN) ObjectEntity REAL(4)
  real function explicit1(x)
   !REF: /module1/explicit1/x
   real, intent(in) :: x
  end function explicit1
  !DEF: /module1/logical EXTERNAL, PUBLIC (Function) Subprogram INTEGER(4)
  !DEF: /module1/logical/x INTENT(IN) ObjectEntity REAL(4)
  integer function logical(x)
   !REF: /module1/logical/x
   real, intent(in) :: x
  end function logical
  !DEF: /module1/tan EXTERNAL, PUBLIC (Function) Subprogram CHARACTER(1_4,1)
  !DEF: /module1/tan/x INTENT(IN) ObjectEntity REAL(4)
  character(len=1) function tan(x)
   !REF: /module1/tan/x
   real, intent(in) :: x
  end function tan
 end interface

 !DEF: /module1/derived1 PUBLIC DerivedType
 type :: derived1
  !REF: /module1/abstract1
  !DEF: /module1/derived1/p1 NOPASS, POINTER (Function) ProcEntity REAL(4)
  !DEF: /module1/nested1 PUBLIC (Function) Subprogram REAL(4)
  procedure(abstract1), pointer, nopass :: p1 => nested1
  !REF: /module1/explicit1
  !DEF: /module1/derived1/p2 NOPASS, POINTER (Function) ProcEntity REAL(4)
  !REF: /module1/nested1
  procedure(explicit1), pointer, nopass :: p2 => nested1
  !DEF: /module1/derived1/p3 NOPASS, POINTER (Function) ProcEntity LOGICAL(4)
  !DEF: /module1/nested2 PUBLIC (Function) Subprogram LOGICAL(4)
  procedure(logical), pointer, nopass :: p3 => nested2
  !DEF: /module1/derived1/p4 NOPASS, POINTER (Function) ProcEntity LOGICAL(4)
  !DEF: /module1/nested3 PUBLIC (Function) Subprogram LOGICAL(4)
  procedure(logical(kind=4)), pointer, nopass :: p4 => nested3
  !DEF: /module1/derived1/p5 NOPASS, POINTER (Function) ProcEntity COMPLEX(4)
  !DEF: /module1/nested4 PUBLIC (Function) Subprogram COMPLEX(4)
  procedure(complex), pointer, nopass :: p5 => nested4
  !DEF: /module1/sin ELEMENTAL, INTRINSIC, PUBLIC ProcEntity
  !DEF: /module1/derived1/p6 NOPASS, POINTER ProcEntity
  !REF: /module1/nested1
  procedure(sin), pointer, nopass :: p6 => nested1
  !REF: /module1/sin
  !DEF: /module1/derived1/p7 NOPASS, POINTER ProcEntity
  !DEF: /module1/cos ELEMENTAL, INTRINSIC, PUBLIC ProcEntity
  procedure(sin), pointer, nopass :: p7 => cos
  !REF: /module1/tan
  !DEF: /module1/derived1/p8 NOPASS, POINTER (Function) ProcEntity CHARACTER(1_4,1)
  !DEF: /module1/nested5 PUBLIC (Function) Subprogram CHARACTER(1_8,1)
  procedure(tan), pointer, nopass :: p8 => nested5
 end type derived1

contains

 !REF: /module1/nested1
 !DEF: /module1/nested1/x INTENT(IN) ObjectEntity REAL(4)
 real function nested1(x)
  !REF: /module1/nested1/x
  real, intent(in) :: x
  !DEF: /module1/nested1/nested1 ObjectEntity REAL(4)
  !REF: /module1/nested1/x
  nested1 = x+1.
 end function nested1

 !REF: /module1/nested2
 !DEF: /module1/nested2/x INTENT(IN) ObjectEntity REAL(4)
 logical function nested2(x)
  !REF: /module1/nested2/x
  real, intent(in) :: x
  !DEF: /module1/nested2/nested2 ObjectEntity LOGICAL(4)
  !REF: /module1/nested2/x
  nested2 = x/=0
 end function nested2

 !REF: /module1/nested3
 !DEF: /module1/nested3/x INTENT(IN) ObjectEntity REAL(4)
 logical function nested3(x)
  !REF: /module1/nested3/x
  real, intent(in) :: x
  !DEF: /module1/nested3/nested3 ObjectEntity LOGICAL(4)
  !REF: /module1/nested3/x
  nested3 = x>0
 end function nested3

 !REF: /module1/nested4
 !DEF: /module1/nested4/x INTENT(IN) ObjectEntity REAL(4)
 complex function nested4(x)
  !REF: /module1/nested4/x
  real, intent(in) :: x
  !DEF: /module1/nested4/nested4 ObjectEntity COMPLEX(4)
  !DEF: /module1/nested4/cmplx INTRINSIC (Function) ProcEntity
  !REF: /module1/nested4/x
  nested4 = cmplx(x+4., 6.)
 end function nested4

 !REF: /module1/nested5
 !DEF: /module1/nested5/x INTENT(IN) ObjectEntity REAL(4)
 character function nested5(x)
  !REF: /module1/nested5/x
  real, intent(in) :: x
  !DEF: /module1/nested5/nested5 ObjectEntity CHARACTER(1_8,1)
  nested5 = "a"
 end function nested5
end module module1

!DEF: /explicit1 ELEMENTAL (Function) Subprogram REAL(4)
!DEF: /explicit1/x INTENT(IN) ObjectEntity REAL(4)
real elemental function explicit1(x)
 !REF: /explicit1/x
 real, intent(in) :: x
 !DEF: /explicit1/explicit1 ObjectEntity REAL(4)
 !REF: /explicit1/x
 explicit1 = -x
end function explicit1

!DEF: /logical (Function) Subprogram INTEGER(4)
!DEF: /logical/x INTENT(IN) ObjectEntity REAL(4)
integer function logical(x)
 !REF: /logical/x
 real, intent(in) :: x
 !DEF: /logical/logical ObjectEntity INTEGER(4)
 !REF: /logical/x
 logical = x+3.
end function logical

!DEF: /tan (Function) Subprogram REAL(4)
!DEF: /tan/x INTENT(IN) ObjectEntity REAL(4)
real function tan(x)
 !REF: /tan/x
 real, intent(in) :: x
 !DEF: /tan/tan ObjectEntity REAL(4)
 !REF: /tan/x
 tan = x+5.
end function tan

!DEF: /main MainProgram
program main
 !REF: /module1
 use :: module1
 !DEF: /main/derived1 Use
 !DEF: /main/instance ObjectEntity TYPE(derived1)
 type(derived1) :: instance
 !REF: /main/instance
 !REF: /module1/derived1/p1
 if (instance%p1(1.)/=2.) print *, "p1 failed"
 !REF: /main/instance
 !REF: /module1/derived1/p2
 if (instance%p2(1.)/=2.) print *, "p2 failed"
 !REF: /main/instance
 !REF: /module1/derived1/p3
 if (.not.instance%p3(1.)) print *, "p3 failed"
 !REF: /main/instance
 !REF: /module1/derived1/p4
 if (.not.instance%p4(1.)) print *, "p4 failed"
 !REF: /main/instance
 !REF: /module1/derived1/p5
 if (instance%p5(1.)/=(5.,6.)) print *, "p5 failed"
 !REF: /main/instance
 !REF: /module1/derived1/p6
 if (instance%p6(1.)/=2.) print *, "p6 failed"
 !REF: /main/instance
 !REF: /module1/derived1/p7
 if (instance%p7(0.)/=1.) print *, "p7 failed"
 !REF: /main/instance
 !REF: /module1/derived1/p8
 if (instance%p8(1.)/="a") print *, "p8 failed"
end program main
