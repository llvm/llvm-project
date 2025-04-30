!RUN: %flang -g -S -emit-llvm %s -o - | FileCheck %s

!CHECK: [[N1:![0-9]+]] = distinct !DISubprogram
!CHECK: !DILocalVariable(arg: 1, scope: [[N1]]
!CHECK: !DILocalVariable(arg: 2, scope: [[N1]]
!CHECK: !DILocalVariable(name: "a", arg: 3, scope: [[N1]]

module test
contains
  subroutine sub(a)
    implicit none
    integer(kind = 4) :: m
    real(kind = 8), intent(inout) :: a(:,:)
    m = size(a, 1)
    entry subsub(a)
    m = size(a, 1) + 1
    entry subsub1(a)
    m = size(a, 1) + 2
  end subroutine sub
end module
