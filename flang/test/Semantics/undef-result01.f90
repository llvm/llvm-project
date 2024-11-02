! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror

!WARNING: Function result is never defined
function basic()
end

function defdByIntentOut()
  call intentout(defdByIntentOut)
 contains
  subroutine intentout(x)
    real, intent(out) :: x
  end
end

function defdByIntentInOut()
  call intentinout(defdByIntentInOut)
 contains
  subroutine intentInout(x)
    real, intent(out) :: x
  end
end

function defdByIntentInPtr()
  real, target :: defdByIntentInPtr
  call intentInPtr(defdByIntentInPtr)
 contains
  subroutine intentInPtr(p)
    real, intent(in), pointer :: p
  end
end

!WARNING: Function result is never defined
function notDefdByCall()
  call intentin(notDefdByCall)
 contains
  subroutine intentin(n)
    integer, intent(in) :: n
  end
end

!WARNING: Function result is never defined
function basicAlloc()
  real, allocatable :: basicAlloc
  allocate(basicAlloc)
end

function sourcedAlloc()
  real, allocatable :: sourcedAlloc
  allocate(sourcedAlloc, source=0.)
end

function defdByEntry()
  entry entry1
  entry1 = 0.
end

function defdByEntry2()
  entry entry2() result(entryResult)
  entryResult = 0.
end

function usedAsTarget()
  real, target :: usedAsTarget
  real, pointer :: p
  p => usedAsTarget
end

function entryUsedAsTarget()
  real, target :: entryResult
  real, pointer :: p
  entry entry5() result(entryResult)
  p => entryResult
end

function defdByCall()
  call implicitInterface(defdByCall)
end

function defdInInternal()
 contains
  subroutine internal
    defdInInternal = 0.
  end
end

function defdByEntryInInternal()
  entry entry3() result(entryResult)
 contains
  subroutine internal
    entryResult = 0.
  end
end

type(defaultInitialized) function defdByDefault()
  type defaultInitialized
    integer :: n = 123
  end type
end

integer function defdByDo()
  do defdByDo = 1, 10
  end do
end

function defdByRead()
  read(*,*) defdByRead
end function

function defdByNamelist()
  namelist /nml/ defdByNamelist
  read(*,nml=nml)
end

character(4) function defdByWrite()
  write(defdByWrite) 'abcd'
end

integer function defdBySize()
  real arr(10)
  read(*,size=defdBySize) arr
end

character(40) function defdByIomsg()
  !WARNING: IOMSG= is useless without either ERR= or IOSTAT=
  write(123,*,iomsg=defdByIomsg)
end

character(20) function defdByInquire()
  inquire(6,status=defdByInquire)
end

!WARNING: Function result is never defined
character(20) function notDefdByInquire()
  inquire(file=notDefdByInquire)
end

integer function defdByNewunit()
  open(newunit=defdByNewunit, file="foo.txt")
end

function defdByAssociate()
  associate(s => defdByAssociate)
    s = 1.
  end associate
end
