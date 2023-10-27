! RUN: bbc -pft-test -o %t %s | FileCheck %s

! Test Pre-FIR Tree captures all the intended nodes from the parse-tree
! Coarray and OpenMP related nodes are tested in other files.

! CHECK: Program test_prog
program test_prog
  ! Check specification part is not part of the tree.
  interface
    subroutine incr(i)
      integer, intent(inout) :: i
    end subroutine
  end interface
  integer :: i, j, k
  real, allocatable, target :: x(:)
  real :: y(100)
  ! CHECK-NOT: node
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

  ! CHECK: <<AssociateConstruct>>
  ! CHECK: AssociateStmt
  associate (k => i + j)
    ! CHECK: AllocateStmt
    allocate(x(k))
  ! CHECK: EndAssociateStmt
  end associate
  ! CHECK: <<End AssociateConstruct>>

  ! CHECK: <<BlockConstruct!>>
  ! CHECK: BlockStmt
  block
    integer :: k, l
    real, pointer :: p(:)
    ! CHECK: PointerAssignmentStmt
    p => x
    ! CHECK: AssignmentStmt
    k = size(p)
    ! CHECK: AssignmentStmt
    l = 1
    ! CHECK: <<CaseConstruct!>>
    ! CHECK: SelectCaseStmt
    select case (k)
      ! CHECK: CaseStmt
      case (:0)
        ! CHECK: NullifyStmt
        nullify(p)
      ! CHECK: CaseStmt
      case (1)
        ! CHECK: <<IfConstruct>>
        ! CHECK: IfThenStmt
        if (p(1)>0.) then
          ! CHECK: PrintStmt
          print *, "+"
        ! CHECK: ElseIfStmt
        else if (p(1)==0.) then
          ! CHECK: PrintStmt
          print *, "0."
        ! CHECK: ElseStmt
        else
          ! CHECK: PrintStmt
          print *, "-"
        ! CHECK: EndIfStmt
        end if
        ! CHECK: <<End IfConstruct>>
        ! CHECK: CaseStmt
      case (2:10)
      ! CHECK: CaseStmt
      case default
        ! Note: label-do-loop are canonicalized into do constructs
        ! CHECK: <<DoConstruct!>>
        ! CHECK: NonLabelDoStmt
        do 22 while(l<=k)
          ! CHECK: IfStmt
          if (p(l)<0.) p(l)=cos(p(l))
          ! CHECK: CallStmt
22        call incr(l)
        ! CHECK: EndDoStmt
       ! CHECK: <<End DoConstruct!>>
      ! CHECK: CaseStmt
      case (100:)
    ! CHECK: EndSelectStmt
    end select
  ! CHECK: <<End CaseConstruct!>>
  ! CHECK: EndBlockStmt
  end block
  ! CHECK: <<End BlockConstruct!>>

  ! CHECK-NOT: WhereConstruct
  ! CHECK: WhereStmt
  where (x > 1.) x = x/2.

  ! CHECK: <<WhereConstruct>>
  ! CHECK: WhereConstructStmt
  where (x == 0.)
    ! CHECK: AssignmentStmt
    x = 0.01
  ! CHECK: MaskedElsewhereStmt
  elsewhere (x < 0.5)
    ! CHECK: AssignmentStmt
    x = x*2.
    ! CHECK: <<WhereConstruct>>
    where (y > 0.4)
      ! CHECK: AssignmentStmt
      y = y/2.
    end where
    ! CHECK: <<End WhereConstruct>>
  ! CHECK: ElsewhereStmt
  elsewhere
    ! CHECK: AssignmentStmt
    x = x + 1.
  ! CHECK: EndWhereStmt
  end where
  ! CHECK: <<End WhereConstruct>>

  ! CHECK-NOT: ForAllConstruct
  ! CHECK: ForallStmt
  forall (i = 1:5) x(i) = y(i)

  ! CHECK: <<ForallConstruct>>
  ! CHECK: ForallConstructStmt
  forall (i = 1:5)
    ! CHECK: AssignmentStmt
    x(i) = x(i) + y(10*i)
  ! CHECK: EndForallStmt
  end forall
  ! CHECK: <<End ForallConstruct>>

  ! CHECK: DeallocateStmt
  deallocate(x)
end

! CHECK: Module test
module test
  !! When derived type processing is implemented, remove all instances of:
  !!  - !![disable]
  !!  -  COM: 
  !![disable]type :: a_type
  !![disable]  integer :: x
  !![disable]end type
  !![disable]type, extends(a_type) :: b_type
  !![disable]  integer :: y
  !![disable]end type
  interface
     subroutine ss(aa)
       ! CHECK: CompilerDirective
       !DIR$ IGNORE_TKR aa
       integer :: aa
     end subroutine ss
  end interface
contains
  ! CHECK: Function foo
  function foo(x)
    real x(..)
    integer :: foo
    ! CHECK: <<SelectRankConstruct!>>
    ! CHECK: SelectRankStmt
    select rank(x)
      ! CHECK: SelectRankCaseStmt
      rank (0)
        ! CHECK: AssignmentStmt
        foo = 0
      ! CHECK: SelectRankCaseStmt
      rank (*)
        ! CHECK: AssignmentStmt
        foo = -1
      ! CHECK: SelectRankCaseStmt
      rank (1)
        ! CHECK: AssignmentStmt
        foo = 1
      ! CHECK: SelectRankCaseStmt
      rank default
        ! CHECK: AssignmentStmt
        foo = 2
    ! CHECK: EndSelectStmt
    end select
    ! CHECK: <<End SelectRankConstruct!>>
  end function

  ! CHECK: Function bar
  function bar(x)
    class(*) :: x
    ! CHECK: <<SelectTypeConstruct!>>
    ! CHECK: SelectTypeStmt
    select type(x)
      ! CHECK: TypeGuardStmt
      type is (integer)
        ! CHECK: AssignmentStmt
        bar = 0
      !![disable]! COM: CHECK: TypeGuardStmt
      !![disable]class is (a_type)
      !![disable]  ! COM: CHECK: AssignmentStmt
      !![disable]  bar = 1
      !![disable]  ! COM: CHECK: ReturnStmt
      !![disable]  return
      ! CHECK: TypeGuardStmt
      class default
        ! CHECK: AssignmentStmt
        bar = -1
    ! CHECK: EndSelectStmt
    end select
    ! CHECK: <<End SelectTypeConstruct!>>
  end function

  ! CHECK: Subroutine sub
  subroutine sub(a)
    real(4):: a
    ! CHECK: CompilerDirective
    !DIR$ IGNORE_TKR a
  end subroutine


end module

! CHECK: Subroutine altreturn
subroutine altreturn(i, j, *, *)
  ! CHECK: <<IfConstruct!>>
  if (i>j) then
    ! CHECK: ReturnStmt
    return 1
  else
    ! CHECK: ReturnStmt
    return 2
  end if
  ! CHECK: <<End IfConstruct!>>
end subroutine


! Remaining TODO

! CHECK: Subroutine iostmts
subroutine iostmts(filename, a, b, c)
  character(*) :: filename
  integer :: length
  logical :: file_is_opened
  real, a, b ,c
  ! CHECK: InquireStmt
  inquire(file=filename, opened=file_is_opened)
  ! CHECK: <<IfConstruct>>
  if (file_is_opened) then
    ! CHECK: OpenStmt
    open(10, FILE=filename)
  end if
  ! CHECK: <<End IfConstruct>>
  ! CHECK: ReadStmt
  read(10, *) length
  ! CHECK: RewindStmt
  rewind 10
  ! CHECK-NOT: NamelistStmt
  namelist /nlist/ a, b, c
  ! CHECK: WriteStmt
  write(10, NML=nlist)
  ! CHECK: BackspaceStmt
  backspace(10)
  ! CHECK: FormatStmt
1 format (1PE12.4)
  ! CHECK: WriteStmt
  write (10, 1) a
  ! CHECK: EndfileStmt
  endfile 10
  ! CHECK: FlushStmt
  flush 10
  ! CHECK: WaitStmt
  wait(10)
  ! CHECK: CloseStmt
  close(10)
end subroutine


! CHECK: Subroutine sub2
subroutine sub2()
  integer :: i, j, k, l
  i = 0
1 j = i
  ! CHECK: ContinueStmt
2 continue
  i = i+1
3 j = j+1
! CHECK: ArithmeticIfStmt
  if (j-i) 3, 4, 5
  ! CHECK: GotoStmt
4  goto 6

! FIXME: is name resolution on assigned goto broken/todo ?
! WILLCHECK: AssignStmt
!55 assign 6 to label
! WILLCHECK: AssignedGotoStmt
!66  go to label (5, 6)

! CHECK: ComputedGotoStmt
  go to (5, 6), 1 + mod(i, 2)
5 j = j + 1
6 i = i + j/2

  ! CHECK: <<DoConstruct!>>
  do1: do k=1,10
    ! CHECK: <<DoConstruct!>>
    do2: do l=5,20
      ! CHECK: CycleStmt
      cycle do1
      ! CHECK: ExitStmt
      exit do2
    end do do2
    ! CHECK: <<End DoConstruct!>>
  end do do1
  ! CHECK: <<End DoConstruct!>>

  ! CHECK: PauseStmt
  pause 7
  ! CHECK: StopStmt
  stop
end subroutine


! CHECK: Subroutine sub3
subroutine sub3()
 print *, "normal"
  ! CHECK: EntryStmt
 entry sub4entry()
 print *, "test"
end subroutine

! CHECK: Subroutine sub4
subroutine sub4()
  integer :: i
  print*, "test"
  data i /1/
end subroutine
