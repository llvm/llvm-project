!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s

! Size and alignment of derived types

! Array of derived type with 64-bit alignment
subroutine s1
  type t1
    real(8) :: a
    real(4) :: b
  end type
  !CHECK: x1 size=12 offset=0:
  !CHECK: y1 size=12 offset=16:
  type(t1) :: x1, y1
  !CHECK: z1 size=160 offset=32:
  type(t1) :: z1(10)
end

! Like t1 but t2 does not need to be aligned on 64-bit boundary
subroutine s2
  type t2
    real(4) :: a
    real(4) :: b
    real(4) :: c
  end type
  !CHECK: x2 size=12 offset=0:
  !CHECK: y2 size=12 offset=12:
  type(t2) :: x2, y2
  !CHECK: z2 size=120 offset=24:
  type(t2) :: z2(10)
end

! Parameterized derived types
subroutine s3
  type :: t(k, l)
    integer, kind :: k
    integer, len :: l
    real(k) :: a3
    integer(kind=k) :: b3
    character(kind=k, len=8) :: c3
    character(kind=k, len=l) :: d3
  end type
  !CHECK: DerivedType scope: size=48 alignment=8 instantiation of t(k=2_4,l=10_4)
  !CHECK: a3 size=2 offset=0:
  !CHECK: b3 size=2 offset=2:
  !CHECK: c3 size=16 offset=4:
  !CHECK: d3 size=24 offset=24:
  type(t(2, 10)) :: x3
  !CHECK: DerivedType scope: size=64 alignment=8 instantiation of t(k=4_4,l=20_4)
  !CHECK: a3 size=4 offset=0:
  !CHECK: b3 size=4 offset=4:
  !CHECK: c3 size=32 offset=8:
  !CHECK: d3 size=24 offset=40:
  type(t(4, 20)) :: x4
end

subroutine s4
  type t(k)
    integer, kind :: k
    character(len=k) :: c
  end type
  type(t(7)) :: x4
  !CHECK: DerivedType scope: size=7 alignment=1 instantiation of t(k=7_4)
  !CHECK: c size=7 offset=0: ObjectEntity type: CHARACTER(7_4,1)
end subroutine
