!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s
module m
  type pdt1(k1,l1)
    integer, kind :: k1
    integer, len :: l1
    type(pdt2(k1,l1)), allocatable :: a1
  end type pdt1
  type pdt2(k2,l2)
    integer, kind :: k2
    integer, len :: l2
    integer(k2) :: j2
    type(pdt1(k2,l2)) :: a2(k2)
  end type pdt2
  interface
    module function mf(n,str,x1) result(res)
      integer, intent(in) :: n
      character(n), intent(in) :: str
      type(pdt1(1,n)), intent(in) :: x1
      type(pdt2(2,n)) :: res
    end function
    module subroutine ms(f)
      procedure(mf) :: f
    end subroutine
  end interface
  integer sm
end module
!CHECK:    mf, MODULE, PUBLIC (Function): Subprogram isInterface result:TYPE(pdt2(k2=2_4,l2=n)) res (INTEGER(4) n,CHARACTER(n,1) str,TYPE(pdt1(k1=1_4,l1=n)) x1)
!CHECK:    pdt1, PUBLIC: DerivedType components: a1
!CHECK:    pdt2, PUBLIC: DerivedType components: j2,a2
!CHECK:    sm, PUBLIC size=4 offset=0: ObjectEntity type: INTEGER(4)
!CHECK:    DerivedType scope: pdt1
!CHECK:      a1, ALLOCATABLE: ObjectEntity type: TYPE(pdt2(int(k1,kind=4),int(l1,kind=4)))
!CHECK:      k1: TypeParam type:INTEGER(4) Kind
!CHECK:      l1: TypeParam type:INTEGER(4) Len
!CHECK:    DerivedType scope: pdt2
!CHECK:      a2: ObjectEntity type: TYPE(pdt1(k1=int(k2,kind=4),l1=int(l2,kind=4))) shape: 1_8:k2
!CHECK:      j2: ObjectEntity type: INTEGER(int(int(k2,kind=4),kind=8))
!CHECK:      k2: TypeParam type:INTEGER(4) Kind
!CHECK:      l2: TypeParam type:INTEGER(4) Len
!CHECK:    Subprogram scope: mf size=112 alignment=8
!CHECK:      mf (Function): HostAssoc
!CHECK:      n, INTENT(IN) size=4 offset=0: ObjectEntity dummy type: INTEGER(4)
!CHECK:      res size=40 offset=72: ObjectEntity funcResult type: TYPE(pdt2(k2=2_4,l2=n))
!CHECK:      str, INTENT(IN) size=24 offset=8: ObjectEntity dummy type: CHARACTER(n,1)
!CHECK:      x1, INTENT(IN) size=40 offset=32: ObjectEntity dummy type: TYPE(pdt1(k1=1_4,l1=n))
!CHECK:      DerivedType scope: size=40 alignment=8 instantiation of pdt1(k1=1_4,l1=n)
!CHECK:        a1, ALLOCATABLE size=40 offset=0: ObjectEntity type: TYPE(pdt2(k2=1_4,l2=int(l1,kind=4)))
!CHECK:        k1: TypeParam type:INTEGER(4) Kind init:1_4
!CHECK:        l1: TypeParam type:INTEGER(4) Len init:n
!CHECK:        DerivedType scope: size=72 alignment=8 instantiation of pdt2(k2=1_4,l2=int(l1,kind=4))
!CHECK:          a2 size=64 offset=8: ObjectEntity type: TYPE(pdt1(k1=1_4,l1=int(l2,kind=4))) shape: 1_8:1_8
!CHECK:          j2 size=1 offset=0: ObjectEntity type: INTEGER(1)
!CHECK:          k2: TypeParam type:INTEGER(4) Kind init:1_4
!CHECK:          l2: TypeParam type:INTEGER(4) Len init:int(l1,kind=4)
!CHECK:          DerivedType scope: size=40 alignment=8 instantiation of pdt1(k1=1_4,l1=int(l2,kind=4))
!CHECK:            a1, ALLOCATABLE size=40 offset=0: ObjectEntity type: TYPE(pdt2(k2=1_4,l2=int(l1,kind=4)))
!CHECK:            k1: TypeParam type:INTEGER(4) Kind init:1_4
!CHECK:            l1: TypeParam type:INTEGER(4) Len init:int(l2,kind=4)
!CHECK:      DerivedType scope: size=72 alignment=8 instantiation of pdt2(k2=2_4,l2=n)
!CHECK:        a2 size=64 offset=8: ObjectEntity type: TYPE(pdt1(k1=2_4,l1=int(l2,kind=4))) shape: 1_8:2_8
!CHECK:        j2 size=2 offset=0: ObjectEntity type: INTEGER(2)
!CHECK:        k2: TypeParam type:INTEGER(4) Kind init:2_4
!CHECK:        l2: TypeParam type:INTEGER(4) Len init:n
!CHECK:        DerivedType scope: size=40 alignment=8 instantiation of pdt1(k1=2_4,l1=int(l2,kind=4))
!CHECK:          a1, ALLOCATABLE size=40 offset=0: ObjectEntity type: TYPE(pdt2(k2=2_4,l2=int(l1,kind=4)))
!CHECK:          k1: TypeParam type:INTEGER(4) Kind init:2_4
!CHECK:          l1: TypeParam type:INTEGER(4) Len init:int(l2,kind=4)
!CHECK:          DerivedType scope: size=72 alignment=8 instantiation of pdt2(k2=2_4,l2=int(l1,kind=4))
!CHECK:            a2 size=64 offset=8: ObjectEntity type: TYPE(pdt1(k1=2_4,l1=int(l2,kind=4))) shape: 1_8:2_8
!CHECK:            j2 size=2 offset=0: ObjectEntity type: INTEGER(2)
!CHECK:            k2: TypeParam type:INTEGER(4) Kind init:2_4
!CHECK:            l2: TypeParam type:INTEGER(4) Len init:int(l1,kind=4)

submodule(m) sm
 contains
  module procedure mf
    print *, len(str), x1%k1, x1%l1, res%k2, res%l2
    allocate(res%a2(1)%a1)
    res%a2(1)%a1%j2 = 2
  end procedure
  module procedure ms
!    type(pdt2(2.3)) x
!    x = f(3, "abc", pdt1(1,3)())
  end procedure
end submodule
!CHECK:    Module scope: sm size=0 alignment=1
!CHECK:      mf, MODULE, PUBLIC (Function): Subprogram result:TYPE(pdt2(k2=2_4,l2=n)) res (INTEGER(4) n,CHARACTER(n,1) str,TYPE(pdt1(k1=1_4,l1=n)) x1) moduleInterface: mf, MODULE, PUBLIC (Function): Subprogram isInterface result:TYPE(pdt2(k2=2_4,l2=n)) res (INTEGER(4) n,CHARACTER(n,1) str,TYPE(pdt1(k1=1_4,l1=n)) x1)
!CHECK:      Subprogram scope: mf size=112 alignment=8
!CHECK:        len, INTRINSIC, PURE (Function): ProcEntity
!CHECK:        n, INTENT(IN) size=4 offset=0: ObjectEntity dummy type: INTEGER(4)
!CHECK:        res size=40 offset=72: ObjectEntity funcResult type: TYPE(pdt2(k2=2_4,l2=n))
!CHECK:        str, INTENT(IN) size=24 offset=8: ObjectEntity dummy type: CHARACTER(n,1)
!CHECK:        x1, INTENT(IN) size=40 offset=32: ObjectEntity dummy type: TYPE(pdt1(k1=1_4,l1=n))
!CHECK:        DerivedType scope: size=72 alignment=8 instantiation of pdt2(k2=2_4,l2=n)
!CHECK:          a2 size=64 offset=8: ObjectEntity type: TYPE(pdt1(k1=2_4,l1=int(l2,kind=4))) shape: 1_8:2_8
!CHECK:          j2 size=2 offset=0: ObjectEntity type: INTEGER(2)
!CHECK:          k2: TypeParam type:INTEGER(4) Kind init:2_4
!CHECK:          l2: TypeParam type:INTEGER(4) Len init:n
!CHECK:          DerivedType scope: size=40 alignment=8 instantiation of pdt1(k1=2_4,l1=int(l2,kind=4))
!CHECK:            a1, ALLOCATABLE size=40 offset=0: ObjectEntity type: TYPE(pdt2(k2=2_4,l2=int(l1,kind=4)))
!CHECK:            k1: TypeParam type:INTEGER(4) Kind init:2_4
!CHECK:            l1: TypeParam type:INTEGER(4) Len init:int(l2,kind=4)
!CHECK:            DerivedType scope: size=72 alignment=8 instantiation of pdt2(k2=2_4,l2=int(l1,kind=4))
!CHECK:              a2 size=64 offset=8: ObjectEntity type: TYPE(pdt1(k1=2_4,l1=int(l2,kind=4))) shape: 1_8:2_8
!CHECK:              j2 size=2 offset=0: ObjectEntity type: INTEGER(2)
!CHECK:              k2: TypeParam type:INTEGER(4) Kind init:2_4
!CHECK:              l2: TypeParam type:INTEGER(4) Len init:int(l1,kind=4)
!CHECK:        DerivedType scope: size=40 alignment=8 instantiation of pdt1(k1=1_4,l1=n)
!CHECK:          a1, ALLOCATABLE size=40 offset=0: ObjectEntity type: TYPE(pdt2(k2=1_4,l2=int(l1,kind=4)))
!CHECK:          k1: TypeParam type:INTEGER(4) Kind init:1_4
!CHECK:          l1: TypeParam type:INTEGER(4) Len init:n
!CHECK:          DerivedType scope: size=72 alignment=8 instantiation of pdt2(k2=1_4,l2=int(l1,kind=4))
!CHECK:            a2 size=64 offset=8: ObjectEntity type: TYPE(pdt1(k1=1_4,l1=int(l2,kind=4))) shape: 1_8:1_8
!CHECK:            j2 size=1 offset=0: ObjectEntity type: INTEGER(1)
!CHECK:            k2: TypeParam type:INTEGER(4) Kind init:1_4
!CHECK:            l2: TypeParam type:INTEGER(4) Len init:int(l1,kind=4)
!CHECK:            DerivedType scope: size=40 alignment=8 instantiation of pdt1(k1=1_4,l1=int(l2,kind=4))
!CHECK:              a1, ALLOCATABLE size=40 offset=0: ObjectEntity type: TYPE(pdt2(k2=1_4,l2=int(l1,kind=4)))
!CHECK:              k1: TypeParam type:INTEGER(4) Kind init:1_4
!CHECK:              l1: TypeParam type:INTEGER(4) Len init:int(l2,kind=4)

program test
  use m
  type(pdt2(2,3)) x
  x = mf(3, "abc", pdt1(1,3)())
!  call ms(mf)
end program
!CHECK:  MainProgram scope: test size=88 alignment=8
!CHECK:    mf, MODULE (Function): Use from mf in m
!CHECK:    pdt1: Use from pdt1 in m
!CHECK:    pdt2: Use from pdt2 in m
!CHECK:    sm: Use from sm in m
!CHECK:    x size=88 offset=0: ObjectEntity type: TYPE(pdt2(k2=2_4,l2=3_4))
!CHECK:    DerivedType scope: size=88 alignment=8 instantiation of pdt2(k2=2_4,l2=3_4)
!CHECK:      a2 size=80 offset=8: ObjectEntity type: TYPE(pdt1(k1=2_4,l1=3_4)) shape: 1_8:2_8
!CHECK:      j2 size=2 offset=0: ObjectEntity type: INTEGER(2)
!CHECK:      k2: TypeParam type:INTEGER(4) Kind init:2_4
!CHECK:      l2: TypeParam type:INTEGER(4) Len init:3_4
!CHECK:      DerivedType scope: size=40 alignment=8 instantiation of pdt1(k1=2_4,l1=3_4)
!CHECK:        a1, ALLOCATABLE size=40 offset=0: ObjectEntity type: TYPE(pdt2(k2=2_4,l2=3_4))
!CHECK:        k1: TypeParam type:INTEGER(4) Kind init:2_4
!CHECK:        l1: TypeParam type:INTEGER(4) Len init:3_4
!CHECK:    DerivedType scope: size=40 alignment=8 instantiation of pdt1(k1=1_4,l1=3_4)
!CHECK:      a1, ALLOCATABLE size=40 offset=0: ObjectEntity type: TYPE(pdt2(k2=1_4,l2=3_4))
!CHECK:      k1: TypeParam type:INTEGER(4) Kind init:1_4
!CHECK:      l1: TypeParam type:INTEGER(4) Len init:3_4
!CHECK:      DerivedType scope: size=48 alignment=8 instantiation of pdt2(k2=1_4,l2=3_4) sourceRange=0 bytes
!CHECK:        a2 size=40 offset=8: ObjectEntity type: TYPE(pdt1(k1=1_4,l1=3_4)) shape: 1_8:1_8
!CHECK:        j2 size=1 offset=0: ObjectEntity type: INTEGER(1)
!CHECK:        k2: TypeParam type:INTEGER(4) Kind init:1_4
!CHECK:        l2: TypeParam type:INTEGER(4) Len init:3_4
!CHECK:        DerivedType scope: size=40 alignment=8 instantiation of pdt1(k1=1_4,l1=3_4) sourceRange=0 bytes
!CHECK:          a1, ALLOCATABLE size=40 offset=0: ObjectEntity type: TYPE(pdt2(k2=1_4,l2=3_4))
!CHECK:          k1: TypeParam type:INTEGER(4) Kind init:1_4
!CHECK:          l1: TypeParam type:INTEGER(4) Len init:3_4
