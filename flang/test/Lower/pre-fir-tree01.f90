! RUN: %flang_fc1 -fdebug-pre-fir-tree %s | FileCheck %s

! Test structure of the Pre-FIR tree

! CHECK: Subroutine foo
subroutine foo()
  ! CHECK: <<DoConstruct>>
  ! CHECK: NonLabelDoStmt
  do i=1,5
    ! CHECK: PrintStmt
    print *, "hey"
    ! CHECK: <<DoConstruct>>
    ! CHECK: NonLabelDoStmt
    do j=1,5
      ! CHECK: PrintStmt
      print *, "hello", i, j
    ! CHECK: EndDoStmt
    end do
    ! CHECK: <<End DoConstruct>>
  ! CHECK: EndDoStmt
  end do
  ! CHECK: <<End DoConstruct>>
! CHECK: EndSubroutineStmt
end subroutine
! CHECK: End Subroutine foo

! CHECK: BlockData
block data
  integer, parameter :: n = 100
  integer, dimension(n) :: a, b, c
  common /arrays/ a, b, c
end
! CHECK: End BlockData

! CHECK: ModuleLike
module test_mod
interface
  ! check specification parts are not part of the PFT.
  ! CHECK-NOT: node
  module subroutine dump()
  end subroutine
end interface
 integer :: xdim
 real, allocatable :: pressure(:)
contains
  ! CHECK: Subroutine foo
  subroutine foo()
  ! CHECK: EndSubroutineStmt
    contains
    ! CHECK: Subroutine subfoo
    subroutine subfoo()
    ! CHECK: EndSubroutineStmt
  9 end subroutine
    ! CHECK: End Subroutine subfoo
    ! CHECK: Function subfoo2
    function subfoo2()
    ! CHECK: EndFunctionStmt
  9 end function
    ! CHECK: End Function subfoo2
  end subroutine
  ! CHECK: End Subroutine foo

  ! CHECK: Function foo2
  function foo2(i, j)
    integer i, j, foo2
    ! CHECK: AssignmentStmt
    foo2 = i + j
  ! CHECK: EndFunctionStmt
    contains
    ! CHECK: Subroutine subfoo
    subroutine subfoo()
    ! CHECK: EndSubroutineStmt
    end subroutine
    ! CHECK: End Subroutine subfoo
  end function
  ! CHECK: End Function foo2
end module
! CHECK: End ModuleLike

! CHECK: ModuleLike
submodule (test_mod) test_mod_impl
contains
  ! CHECK: Subroutine foo
  subroutine foo()
  ! CHECK: EndSubroutineStmt
    contains
    ! CHECK: Subroutine subfoo
    subroutine subfoo()
    ! CHECK: EndSubroutineStmt
    end subroutine
    ! CHECK: End Subroutine subfoo
    ! CHECK: Function subfoo2
    function subfoo2()
    ! CHECK: EndFunctionStmt
    end function
    ! CHECK: End Function subfoo2
  end subroutine
  ! CHECK: End Subroutine foo
  ! CHECK: MpSubprogram dump
  module procedure dump
    ! CHECK: FormatStmt
11  format (2E16.4, I6)
    ! CHECK: <<IfConstruct>>
    ! CHECK: IfThenStmt
    if (xdim > 100) then
      ! CHECK: PrintStmt
      print *, "test: ", xdim
    ! CHECK: ElseStmt
    else
      ! CHECK: WriteStmt
      write (*, 11) "test: ", xdim, pressure
    ! CHECK: EndIfStmt
    end if
    ! CHECK: <<End IfConstruct>>
  end procedure
end submodule
! CHECK: End ModuleLike

! CHECK: BlockData
block data named_block
 integer i, j, k
 common /indexes/ i, j, k
end
! CHECK: End BlockData

! CHECK: Function bar
function bar()
! CHECK: EndFunctionStmt
end function
! CHECK: End Function bar

! Test top level directives
!DIR$ INTEGER=64
! CHECK: CompilerDirective:
! CHECK: End CompilerDirective

! Test nested directive
! CHECK: Subroutine test_directive
subroutine test_directive()
  !DIR$ INTEGER=64
  ! CHECK: <<CompilerDirective>>
  ! CHECK: <<End CompilerDirective>>
end subroutine
! CHECK: EndSubroutine

! CHECK: Program <anonymous>
  ! check specification parts are not part of the PFT.
  ! CHECK-NOT: node
  use test_mod
  real, allocatable :: x(:)
  ! CHECK: AllocateStmt
  allocate(x(foo2(10, 30)))
end
! CHECK: End Program
