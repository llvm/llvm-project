!RUN: %flang_fc1 -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s -check-prefix=UNPARSE
!RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s -check-prefix=TREE

subroutine test_prefetch_01(a, b)
    integer, intent(in) :: a
    integer, intent(inout) :: b(5)
    integer :: i = 2
    integer :: res

!TREE: | | DeclarationConstruct -> SpecificationConstruct -> CompilerDirective -> Prefetch -> Designator -> DataRef -> Name = 'a'

!UNPARSE:    !DIR$ PREFETCH a
    !dir$ prefetch a
    b(1) = a

!TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> CompilerDirective -> Prefetch -> Designator -> DataRef -> Name = 'b'

!UNPARSE:    !DIR$ PREFETCH b
    !dir$ prefetch b
    res = sum(b)

!TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> CompilerDirective -> Prefetch -> Designator -> DataRef -> Name = 'a'
!TREE: | | Designator -> DataRef -> ArrayElement
!TREE: | | | DataRef -> Name = 'b'
!TREE: | | | SectionSubscript -> SubscriptTriplet
!TREE: | | | | Scalar -> Integer -> Expr -> LiteralConstant -> IntLiteralConstant = '3'
!TREE: | | | | Scalar -> Integer -> Expr -> LiteralConstant -> IntLiteralConstant = '5'

!UNPARSE:    !DIR$ PREFETCH a, b(3:5)
    !dir$ prefetch a, b(3:5)
    res = a + b(4)

!TREE: | | ExecutionPartConstruct -> ExecutableConstruct -> CompilerDirective -> Prefetch -> Designator -> DataRef -> Name = 'res'
!TREE: | | Designator -> DataRef -> ArrayElement
!TREE: | | | DataRef -> Name = 'b'
!TREE: | | | SectionSubscript -> Integer -> Expr -> Add
!TREE: | | | | Expr -> Designator -> DataRef -> Name = 'i'
!TREE: | | | | Expr -> LiteralConstant -> IntLiteralConstant = '2'

!UNPARSE:    !DIR$ PREFETCH res, b(i+2)
    !dir$ prefetch res, b(i+2)
    res = res + b(i+2)
end subroutine

subroutine test_prefetch_02(n, a)
    integer, intent(in) :: n
    integer, intent(in) :: a(n)
    type :: t
        real, allocatable :: x(:, :)
    end type t
    type(t) :: p

    do i = 1, n
!TREE: | | | | ExecutionPartConstruct -> ExecutableConstruct -> CompilerDirective -> Prefetch -> Designator -> DataRef -> ArrayElement
!TREE: | | | | | DataRef -> StructureComponent
!TREE: | | | | | | DataRef -> Name = 'p'
!TREE: | | | | | | Name = 'x'
!TREE: | | | | | SectionSubscript -> Integer -> Expr -> Designator -> DataRef -> Name = 'i'
!TREE: | | | | | SectionSubscript -> SubscriptTriplet
!TREE: | | | | Designator -> DataRef -> Name = 'a'

!UNPARSE:  !DIR$ PREFETCH p%x(i,:), a
        !dir$ prefetch p%x(i, :), a
        do j = 1, n
!TREE: | | | | | | ExecutionPartConstruct -> ExecutableConstruct -> CompilerDirective -> Prefetch -> Designator -> DataRef -> ArrayElement
!TREE: | | | | | | | DataRef -> StructureComponent
!TREE: | | | | | | | | DataRef -> Name = 'p'
!TREE: | | | | | | | | Name = 'x'
!TREE: | | | | | | | SectionSubscript -> Integer -> Expr -> Designator -> DataRef -> Name = 'i'
!TREE: | | | | | | | SectionSubscript -> Integer -> Expr -> Designator -> DataRef -> Name = 'j'
!TREE: | | | | | | Designator -> DataRef -> ArrayElement
!TREE: | | | | | | | DataRef -> Name = 'a'
!TREE: | | | | | | | SectionSubscript -> Integer -> Expr -> Designator -> DataRef -> Name = 'i'

!UNPARSE:   !DIR$ PREFETCH p%x(i,j), a(i)
            !dir$ prefetch p%x(i, j), a(i)
            p%x(i, j) = p%x(i, j) ** a(j)
        end do
    end do
end subroutine
