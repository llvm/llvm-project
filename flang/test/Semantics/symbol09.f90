! RUN: %python %S/test_symbols.py %s %flang_fc1
!DEF: /s1 (Subroutine) Subprogram
subroutine s1
 !DEF: /s1/a ObjectEntity REAL(4)
 !DEF: /s1/b ObjectEntity REAL(4)
 real a(10), b(10)
 !DEF: /s1/i ObjectEntity INTEGER(8)
 integer(kind=8) i
 !DEF: /s1/Forall1/i ObjectEntity INTEGER(8)
 forall(i=1:10)
  !REF: /s1/a
  !REF: /s1/Forall1/i
  !REF: /s1/b
  a(i) = b(i)
 end forall
 !DEF: /s1/Forall2/i ObjectEntity INTEGER(8)
 !REF: /s1/a
 !REF: /s1/b
 forall(i=1:10)a(i) = b(i)
end subroutine

!DEF: /s2 (Subroutine) Subprogram
subroutine s2
 !DEF: /s2/a ObjectEntity REAL(4)
 real a(10)
 !DEF: /s2/i ObjectEntity INTEGER(4)
 integer i
 !DEF: /s2/Forall1/i ObjectEntity INTEGER(4)
 do concurrent(i=1:10)
  !REF: /s2/a
  !REF: /s2/Forall1/i
  a(i) = i
 end do
 !REF: /s2/i
 do i=1,10
  !REF: /s2/a
  !REF: /s2/i
  a(i) = i
 end do
end subroutine

!DEF: /s3 (Subroutine) Subprogram
subroutine s3
 !DEF: /s3/n PARAMETER ObjectEntity INTEGER(4)
 integer, parameter :: n = 4
 !DEF: /s3/n2 PARAMETER ObjectEntity INTEGER(4)
 !REF: /s3/n
 integer, parameter :: n2 = n*n
 !REF: /s3/n
 !DEF: /s3/x (InDataStmt) ObjectEntity REAL(4)
 real, dimension(n,n) :: x
 !REF: /s3/x
 !DEF: /s3/ImpliedDos1/k (Implicit) ObjectEntity INTEGER(4)
 !DEF: /s3/ImpliedDos1/j ObjectEntity INTEGER(8)
 !REF: /s3/n
 !REF: /s3/n2
 data ((x(k,j),integer(kind=8)::j=1,n),k=1,n)/n2*3.0/
end subroutine

!DEF: /s4 (Subroutine) Subprogram
subroutine s4
 !DEF: /s4/t DerivedType
 !DEF: /s4/t/k TypeParam INTEGER(4)
 type :: t(k)
  !REF: /s4/t/k
  integer, kind :: k
  !DEF: /s4/t/a ObjectEntity INTEGER(4)
  integer :: a
 end type t
 !REF: /s4/t
 !DEF: /s4/x (InDataStmt) ObjectEntity TYPE(t(k=1_4))
 type(t(1)) :: x
 !REF: /s4/x
 !REF: /s4/t
 data x/t(1)(2)/
 !REF: /s4/x
 !REF: /s4/t
 x = t(1)(2)
end subroutine

!DEF: /s5 (Subroutine) Subprogram
subroutine s5
 !DEF: /s5/t DerivedType
 !DEF: /s5/t/l TypeParam INTEGER(4)
 type :: t(l)
  !REF: /s5/t/l
  integer, len :: l
 end type t
 !REF: /s5/t
 !DEF: /s5/x ALLOCATABLE ObjectEntity TYPE(t(l=:))
 type(t(:)), allocatable :: x
 !DEF: /s5/y ALLOCATABLE ObjectEntity REAL(4)
 real, allocatable :: y
 !REF: /s5/t
 !REF: /s5/x
 allocate(t(1)::x)
 !REF: /s5/y
 allocate(real::y)
end subroutine

!DEF: /s6 (Subroutine) Subprogram
subroutine s6
 !DEF: /s6/j ObjectEntity INTEGER(8)
 integer(kind=8) j
 !DEF: /s6/a ObjectEntity INTEGER(4)
 integer :: a(5) = 1
 !DEF: /s6/Forall1/i ObjectEntity INTEGER(4)
 !DEF: /s6/Forall1/j (LocalityLocal) HostAssoc INTEGER(8)
 !DEF: /s6/Forall1/k (Implicit, LocalityLocalInit) HostAssoc INTEGER(4)
 !DEF: /s6/Forall1/a (LocalityShared) HostAssoc INTEGER(4)
 do concurrent(integer::i=1:5)local(j)local_init(k)shared(a)
  !REF: /s6/Forall1/a
  !REF: /s6/Forall1/i
  !REF: /s6/Forall1/j
  a(i) = j+1
 end do
end subroutine

!DEF: /s7 (Subroutine) Subprogram
subroutine s7
 !DEF: /s7/one PARAMETER ObjectEntity REAL(4)
 real, parameter :: one = 1.0
 !DEF: /s7/z ObjectEntity COMPLEX(4)
 !REF: /s7/one
 complex :: z = (one, -one)
end subroutine

!DEF: /s8 (Subroutine) Subprogram
subroutine s8
 !DEF: /s8/one PARAMETER ObjectEntity REAL(4)
 real, parameter :: one = 1.0
 !DEF: /s8/y (InDataStmt) ObjectEntity REAL(4)
 !DEF: /s8/z (InDataStmt) ObjectEntity REAL(4)
 real y(10), z(10)
 !REF: /s8/y
 !DEF: /s8/ImpliedDos1/i (Implicit) ObjectEntity INTEGER(4)
 !REF: /s8/z
 !DEF: /s8/ImpliedDos2/i (Implicit) ObjectEntity INTEGER(4)
 !DEF: /s8/x (Implicit, InDataStmt) ObjectEntity REAL(4)
 !REF: /s8/one
 data (y(i),i=1,10),(z(i),i=1,10),x/21*one/
end subroutine
