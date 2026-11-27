! RUN: %flang_fc1 -fopenmp -emit-llvm -debug-info-kind=standalone %s -o - | FileCheck %s

! Test that each construct that associates a name in more than one place gives
! each association its own scope.

module m
  type :: base
    integer :: a = 1
  end type
  type, extends(base) :: derived
    integer :: b = 2
  end type
contains
  subroutine ranks(x, y)
    integer, intent(in) :: x(..)
    integer, intent(out) :: y
    select rank(x)
    rank(0) ; y = x
    rank(1) ; y = x(1)
    end select
  end subroutine ranks

  subroutine types(p, y)
    class(base), intent(in) :: p
    integer, intent(out) :: y
    select type (p)
    type is (derived)
      y = p%b
    class default
      y = p%a
    end select
  end subroutine types

  subroutine assoc(n, c, y)
    integer, intent(in) :: n
    logical, intent(in) :: c
    integer, intent(out) :: y
    if (c) then
      associate (r => n * 3)
        y = r
      end associate
    else
      associate (r => n * 4)
        y = r
      end associate
    end if
  end subroutine assoc

  subroutine lin(a)
    integer, intent(out) :: a(8)
    integer :: i, k
    k = 100
    !$omp simd linear(k:1)
    do i = 1, 8
      a(i) = k
    end do
  end subroutine lin

  subroutine after(p, y)
    class(base), intent(in) :: p
    integer, intent(out) :: y
    select type (p)
    type is (derived)
      y = p%b
    class default
      y = p%a
      return
    end select
    y = y + 1
  end subroutine after
end module m

! CHECK-DAG: ![[RANKS:[0-9]+]] = distinct !DISubprogram(name: "ranks"
! CHECK-DAG: !DILocalVariable(name: "x", arg: 1, scope: ![[RANKS]],
! CHECK-DAG: ![[R0:[0-9]+]] = distinct !DILexicalBlock(scope: ![[RANKS]], {{.*}}line: 18,
! CHECK-DAG: !DILocalVariable(name: "x", scope: ![[R0]], {{.*}}type: ![[SCALAR:[0-9]+]])
! CHECK-DAG: ![[SCALAR]] = !DIBasicType(name: "integer(kind=4)"
! CHECK-DAG: ![[R1:[0-9]+]] = distinct !DILexicalBlock(scope: ![[RANKS]], {{.*}}line: 19,
! CHECK-DAG: !DILocalVariable(name: "x", scope: ![[R1]],

! CHECK-DAG: ![[TYPES:[0-9]+]] = distinct !DISubprogram(name: "types"
! CHECK-DAG: !DILocalVariable(name: "p", arg: 1, scope: ![[TYPES]],
! CHECK-DAG: ![[T1:[0-9]+]] = distinct !DILexicalBlock(scope: ![[TYPES]], {{.*}}line: 27,
! CHECK-DAG: !DILocalVariable(name: "p", scope: ![[T1]],
! CHECK-DAG: ![[T2:[0-9]+]] = distinct !DILexicalBlock(scope: ![[TYPES]], {{.*}}line: 29,
! CHECK-DAG: !DILocalVariable(name: "p", scope: ![[T2]],

! CHECK-DAG: ![[ASSOC:[0-9]+]] = distinct !DISubprogram(name: "assoc"
! CHECK-DAG: ![[A1:[0-9]+]] = distinct !DILexicalBlock(scope: ![[ASSOC]], {{.*}}line: 39,
! CHECK-DAG: !DILocalVariable(name: "r", scope: ![[A1]],
! CHECK-DAG: ![[A2:[0-9]+]] = distinct !DILexicalBlock(scope: ![[ASSOC]], {{.*}}line: 43,
! CHECK-DAG: !DILocalVariable(name: "r", scope: ![[A2]],

! CHECK-DAG: ![[LIN:[0-9]+]] = distinct !DISubprogram(name: "lin"
! CHECK-DAG: !DILocalVariable(name: "k", scope: ![[LIN]],
! CHECK-DAG: ![[SIMD:[0-9]+]] = distinct !DILexicalBlock(scope: ![[LIN]],
! CHECK-DAG: !DILocalVariable(name: "k", scope: ![[SIMD]],
! CHECK-DAG: !DILocation(line: 55, column: 7, scope: ![[SIMD]])

! The other branch returns, so the code after the construct joins the block of
! the first branch, but not its scope.
! CHECK-DAG: ![[AFTER:[0-9]+]] = distinct !DISubprogram(name: "after"
! CHECK-DAG: ![[AF1:[0-9]+]] = distinct !DILexicalBlock(scope: ![[AFTER]], {{.*}}line: 63,
! CHECK-DAG: !DILocation(line: 64, column: 7, scope: ![[AF1]])
! CHECK-DAG: !DILocation(line: 69, column: 5, scope: ![[AFTER]])
