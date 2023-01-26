! RUN: bbc -polymorphic-type -emit-fir %s -o - | FileCheck %s

! Tests the generation of fir.dispatch_table operations.

module polymorphic_types
  type p1
    integer :: a
    integer :: b
  contains
    procedure :: proc1 => proc1_p1
    procedure :: aproc
    procedure :: zproc
  end type

  type, extends(p1) :: p2
    integer :: c
  contains
    procedure :: proc1 => proc1_p2
    procedure :: aproc2
  end type

  type, extends(p2) :: p3
    integer :: d
  contains
    procedure :: aproc3
  end type
contains


  subroutine proc1_p1(p)
    class(p1) :: p
  end subroutine

  subroutine aproc(p)
    class(p1) :: p
  end subroutine

  subroutine zproc(p)
    class(p1) :: p
  end subroutine

  subroutine proc1_p2(p)
    class(p2) :: p
  end subroutine

  subroutine aproc2(p)
    class(p2) :: p
  end subroutine

  subroutine aproc3(p)
    class(p3) :: p
  end subroutine

end module

! CHECK-LABEL: fir.dispatch_table @_QMpolymorphic_typesTp1 {
! CHECK:         fir.dt_entry "aproc", @_QMpolymorphic_typesPaproc
! CHECK:         fir.dt_entry "proc1", @_QMpolymorphic_typesPproc1_p1
! CHECK:         fir.dt_entry "zproc", @_QMpolymorphic_typesPzproc
! CHECK:       }

! CHECK-LABEL: fir.dispatch_table @_QMpolymorphic_typesTp2 extends("_QMpolymorphic_typesTp1") {
! CHECK:         fir.dt_entry "aproc", @_QMpolymorphic_typesPaproc
! CHECK:         fir.dt_entry "proc1", @_QMpolymorphic_typesPproc1_p2
! CHECK:         fir.dt_entry "zproc", @_QMpolymorphic_typesPzproc
! CHECK:         fir.dt_entry "aproc2", @_QMpolymorphic_typesPaproc2
! CHECK:       }

! CHECK-LABEL: fir.dispatch_table @_QMpolymorphic_typesTp3 extends("_QMpolymorphic_typesTp2") {
! CHECK:         fir.dt_entry "aproc", @_QMpolymorphic_typesPaproc
! CHECK:         fir.dt_entry "proc1", @_QMpolymorphic_typesPproc1_p2
! CHECK:         fir.dt_entry "zproc", @_QMpolymorphic_typesPzproc
! CHECK:         fir.dt_entry "aproc2", @_QMpolymorphic_typesPaproc2
! CHECK:         fir.dt_entry "aproc3", @_QMpolymorphic_typesPaproc3
! CHECK:       }
