! RUN: %python %S/test_errors.py %s %flang_fc1
! Test error message priorities for DATA problems
module m
  integer useAlloc
  allocatable useAlloc
  integer, pointer :: usePtr(:)
 contains
  subroutine useProc
  end
end
function f(hostDummy, hostProc) result(hostResult)
  integer hostDummy, hostResult
  external hostProc
  integer hostAuto(hostDummy)
  integer, allocatable :: hostAlloc
  integer :: hostInit = 1
  integer, pointer :: hostPtr(:)
 contains
  subroutine test(innerDummy, innerProc)
    use m
    external innerProc
    integer innerAuto(innerDummy)
    integer, allocatable :: innerAlloc
    integer :: innerInit = 1
    integer, pointer :: innerPtr(:)
    !ERROR: Procedure 'useproc' must not be initialized in a DATA statement
    data useProc/0/
    !ERROR: Procedure 'hostproc' must not be initialized in a DATA statement
    data hostProc/0/
    !ERROR: Procedure 'innerproc' must not be initialized in a DATA statement
    data innerProc/0/
    !ERROR: Host-associated object 'hostdummy' must not be initialized in a DATA statement
    data hostDummy/1/
    !ERROR: Host-associated object 'hostresult' must not be initialized in a DATA statement
    data hostResult/1/
    !ERROR: Host-associated object 'hostauto' must not be initialized in a DATA statement
    data hostAuto/1/
    !ERROR: Host-associated object 'hostalloc' must not be initialized in a DATA statement
    data hostAlloc/1/
    !ERROR: Host-associated object 'hostinit' must not be initialized in a DATA statement
    data hostInit/1/
    !ERROR: Host-associated object 'hostptr' must not be initialized in a DATA statement
    data hostPtr(1)/1/
    !ERROR: USE-associated object 'usealloc' must not be initialized in a DATA statement
    data useAlloc/1/
    !ERROR: USE-associated object 'useptr' must not be initialized in a DATA statement
    data usePtr(1)/1/
    !ERROR: Dummy argument 'innerdummy' must not be initialized in a DATA statement
    data innerDummy/1/
    !ERROR: Automatic variable 'innerauto' must not be initialized in a DATA statement
    data innerAuto/1/
    !ERROR: Allocatable 'inneralloc' must not be initialized in a DATA statement
    data innerAlloc/1/
    !ERROR: Default-initialized 'innerinit' must not be initialized in a DATA statement
    data innerInit/1/
    !ERROR: Target of pointer 'innerptr' must not be initialized in a DATA statement
    data innerptr(1)/1/
  end
end
