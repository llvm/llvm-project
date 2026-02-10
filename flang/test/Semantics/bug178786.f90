!RUN: %flang_fc1 -fdebug-dump-symbols %s 2>&1 | FileCheck %s
!CHECK:    DerivedType scope: b size=4048 alignment=8 instantiation of b sourceRange=62 bytes
!CHECK:    DerivedType scope: e size=4080 alignment=8 instantiation of e sourceRange=48 bytes
!CHECK:      DerivedType scope: size=4048 alignment=8 instantiation of b(k=4_4) sourceRange=0 bytes
!CHECK:        DerivedType scope: size=4080 alignment=8 instantiation of e(k=4_4) sourceRange=0 bytes
!CHECK:          DerivedType scope: size=4048 alignment=8 instantiation of b(k=4_4) sourceRange=0 bytes
!CHECK:      DerivedType scope: size=4080 alignment=8 instantiation of e(k=4_4) sourceRange=0 bytes
!CHECK:        DerivedType scope: size=4048 alignment=8 instantiation of b(k=4_4) sourceRange=0 bytes
!CHECK:      DerivedType scope: size=4048 alignment=8 instantiation of b(k=4_4) sourceRange=0 bytes
!CHECK:        DerivedType scope: size=4080 alignment=8 instantiation of e(k=4_4) sourceRange=0 bytes
!CHECK:          DerivedType scope: size=4048 alignment=8 instantiation of b(k=4_4) sourceRange=0 bytes

module noPDTs
  type :: b
    integer::d1
    type (e),allocatable::n
    integer::dd1(1000)
  end type
  type,extends(b) :: e
    integer::d2
    character(:),allocatable::c
  end type
 contains
  subroutine alloc(d)
    class (b),allocatable :: d
    allocate(e::d)
  end subroutine alloc
  subroutine s1
    class (b),allocatable :: v
    call alloc(v)
    deallocate(v)
  end
end

module PDTs
  type :: b(k)
    integer, kind :: k
    integer(k)::d1
    type(e(k)),allocatable::n
    integer::dd1(1000)
  end type
  type,extends(b) :: e
    integer::d2
    character(:),allocatable::c
  end type
 contains
  subroutine alloc(d)
    class(b(kind(0))),allocatable :: d
    allocate(e(kind(0))::d)
  end subroutine alloc
  subroutine s2
    class(b(kind(0))),allocatable :: v
    call alloc(v)
    deallocate(v)
  end
end

use noPDTS, only: s1
use PDTS, only: s2
call s1
call s2
print *, 'ok'
end
