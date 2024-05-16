! Check that InputDerivedType/OutputDeriverType APIs are used
! for io of derived types.
! RUN: bbc -emit-fir -o - %s | FileCheck %s

module p
  type :: person
     type(person), pointer :: next => null()
  end type person
  type :: club
     class(person), allocatable :: membership(:)
  end type club
contains
  subroutine pwf (dtv,unit,iotype,vlist,iostat,iomsg)
    class(person), intent(in) :: dtv
    integer, intent(in) :: unit
    character (len=*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character (len=*), intent(inout) :: iomsg
    print *, 'write'
  end subroutine pwf
  subroutine prf (dtv,unit,iotype,vlist,iostat,iomsg)
    class(person), intent(inout) :: dtv
    integer, intent(in) :: unit
    character (len=*), intent(in) :: iotype
    integer, intent(in) :: vlist(:)
    integer, intent(out) :: iostat
    character (len=*), intent(inout) :: iomsg
  end subroutine prf
  subroutine test1(dtv)
    interface read(formatted)
       module procedure prf
    end interface read(formatted)
    class(person), intent(inout) :: dtv
    read(7, fmt='(DT)') dtv%next
  end subroutine test1
! CHECK-LABEL:   func.func @_QMpPtest1(
! CHECK:           %{{.*}} = fir.call @_FortranAioInputDerivedType(%{{.*}}, %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>, !fir.ref<none>) -> i1

  subroutine test2(social_club)
    interface read(formatted)
       module procedure prf
    end interface read(formatted)
    class(club) :: social_club
    read(7, fmt='(DT)') social_club%membership(0)
  end subroutine test2
! CHECK-LABEL:   func.func @_QMpPtest2(
! CHECK:           %{{.*}} = fir.call @_FortranAioInputDerivedType(%{{.*}}, %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>, !fir.ref<none>) -> i1

  subroutine test3(dtv)
    interface write(formatted)
       module procedure pwf
    end interface write(formatted)
    class(person), intent(inout) :: dtv
    write(7, fmt='(DT)') dtv%next
  end subroutine test3
! CHECK-LABEL:   func.func @_QMpPtest3(
! CHECK:           %{{.*}} = fir.call @_FortranAioOutputDerivedType(%{{.*}}, %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>, !fir.ref<none>) -> i1

  subroutine test4(social_club)
    interface write(formatted)
       module procedure pwf
    end interface write(formatted)
    class(club) :: social_club
    write(7, fmt='(DT)') social_club%membership(0)
  end subroutine test4
! CHECK-LABEL:   func.func @_QMpPtest4(
! CHECK:           %{{.*}} = fir.call @_FortranAioOutputDerivedType(%{{.*}}, %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>, !fir.ref<none>) -> i1
end module p

