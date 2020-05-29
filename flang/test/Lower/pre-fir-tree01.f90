! RUN: %f18 -fdebug-pre-fir-tree -fparse-only %s | FileCheck %s

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
end subroutine
! CHECK: EndSubroutine foo

! CHECK: BlockData
block data
  integer, parameter :: n = 100
  integer, dimension(n) :: a, b, c
  common /arrays/ a, b, c
end
! CHECK: EndBlockData

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
    contains
    ! CHECK: Subroutine subfoo
    subroutine subfoo()
    end subroutine
    ! CHECK: EndSubroutine subfoo
    ! CHECK: Function subfoo2
    function subfoo2()
    end function
    ! CHECK: EndFunction subfoo2
  end subroutine
  ! CHECK: EndSubroutine foo

  ! CHECK: Function foo2
  function foo2(i, j)
    integer i, j, foo2
    ! CHECK: AssignmentStmt
    foo2 = i + j
    contains
    ! CHECK: Subroutine subfoo
    subroutine subfoo()
    end subroutine
    ! CHECK: EndSubroutine subfoo
  end function
  ! CHECK: EndFunction foo2
end module
! CHECK: EndModuleLike

! CHECK: ModuleLike
submodule (test_mod) test_mod_impl
contains
  ! CHECK: Subroutine foo
  subroutine foo()
    contains
    ! CHECK: Subroutine subfoo
    subroutine subfoo()
    end subroutine
    ! CHECK: EndSubroutine subfoo
    ! CHECK: Function subfoo2
    function subfoo2()
    end function
    ! CHECK: EndFunction subfoo2
  end subroutine
  ! CHECK: EndSubroutine foo
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
! CHECK: EndModuleLike

! CHECK: BlockData
block data named_block
 integer i, j, k
 common /indexes/ i, j, k
end
! CHECK: EndBlockData

! CHECK: Function bar
function bar()
end function
! CHECK: EndFunction bar

! CHECK: Program <anonymous>
  ! check specification parts are not part of the PFT.
  ! CHECK-NOT: node
  use test_mod
  real, allocatable :: x(:)
  ! CHECK: AllocateStmt
  allocate(x(foo2(10, 30)))
end
! CHECK: EndProgram
