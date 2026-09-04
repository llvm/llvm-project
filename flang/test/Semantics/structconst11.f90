!RUN: %flang_fc1 -fdebug-unparse %s | FileCheck %s
program test

  type t1p
    type(t1p), pointer :: arr(:)
  end type
  type, extends(t1p) :: t1c
  end type
  type t2p
    type(t2p), pointer :: scalar
  end type
  type, extends(t2p) :: t2c
  end type
  type t3p
    type(t3p), allocatable :: arr(:)
  end type
  type, extends(t3p) :: t3c
  end type
  type t4p
    type(t4p), allocatable :: scalar
  end type
  type, extends(t4p) :: t4c
  end type
  type t5p
    class(*), pointer :: arr(:)
  end type
  type, extends(t5p) :: t5c
  end type
  type t6p
    class(*), pointer :: scalar
  end type
  type, extends(t6p) :: t6c
  end type
  type t7p
    class(*), allocatable :: arr(:)
  end type
  type, extends(t7p) :: t7c
  end type
  type t8p
    class(*), allocatable :: scalar
  end type
  type, extends(t8p) :: t8c
  end type

  type(t1p), target :: t1pt(1)
  type(t1p), pointer :: t1pp(:)
  type(t2p), target :: t2pt
  type(t2p), pointer :: t2pp
  type(t3p) t3pa(1)
  type(t4p) t4ps

  type(t1c) x1
  type(t2c) x2
  type(t3c) x3
  type(t4c) x4
  type(t5c) x5
  type(t6c) x6
  type(t7c) x7
  type(t8c) x8

!CHECK: x1=t1c(arr=t1pt)
  x1 = t1c(t1pt)
!CHECK: x1=t1c(arr=t1pp)
  x1 = t1c(t1pp)
!CHECK: x2=t2c(scalar=t2pt)
  x2 = t2c(t2pt)
!CHECK: x2=t2c(scalar=t2pp)
  x2 = t2c(t2pp)
!CHECK: x3=t3c(arr=t3pa)
  x3 = t3c(t3pa)
!CHECK: x4=t4c(scalar=t4ps)
  x4 = t4c(t4ps)
!CHECK: x4=t4c(scalar=t4p(scalar=NULL()))
  x4 = t4c(t4p())
!CHECK: x5=t5c(arr=t1pt)
  x5 = t5c(t1pt)
!CHECK: x5=t5c(arr=t1pp)
  x5 = t5c(t1pp)
!CHECK: x6=t6c(scalar=t2pt)
  x6 = t6c(t2pt)
!CHECK: x6=t6c(scalar=t2pp)
  x6 = t6c(t2pp)
!CHECK: x7=t7c(arr=t3pa)
  x7 = t7c(t3pa)
!CHECK: x8=t8c(scalar=t4ps)
  x8 = t8c(t4ps)
!CHECK: x8=t8c(scalar=t4p(scalar=NULL()))
  x8 = t8c(t4p())
end
