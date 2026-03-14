! RUN: %python %S/test_modfile.py %s %flang_fc1
module m
  type t1
  contains
    procedure, nopass :: s2
    procedure, nopass :: s3
    procedure :: r
    generic :: foo => s2
    generic :: read(formatted)=> r
  end type
  type, extends(t1) :: t2
  contains
    procedure, nopass :: s4
    generic :: foo => s3
    generic :: foo => s4
  end type
contains
  subroutine s2(i)
  end
  subroutine s3(r)
  end
  subroutine s4(z)
    complex :: z
  end
  subroutine r(dtv, unit, iotype, v_list, iostat, iomsg)
    class(t1), intent(inout) :: dtv
    integer, intent(in) :: unit
    character (len=*), intent(in) :: iotype
    integer, intent(in) :: v_list(:)
    integer, intent(out) :: iostat
    character (len=*), intent(inout) :: iomsg
  end
end

!Expect: m.mod
!module m
!  type::t1
!  contains
!    procedure,nopass::s2
!    procedure,nopass::s3
!    procedure::r
!    generic::foo=>s2
!    generic::read(formatted)=>r
!  end type
!  type,extends(t1)::t2
!  contains
!    procedure,nopass::s4
!    generic::foo=>s3
!    generic::foo=>s4
!  end type
!contains
!  subroutine s2(i)
!    integer(4)::i
!  end
!  subroutine s3(r)
!    real(4)::r
!  end
!  subroutine s4(z)
!    complex(4)::z
!  end
!  subroutine r(dtv,unit,iotype,v_list,iostat,iomsg)
!    class(t1),intent(inout)::dtv
!    integer(4),intent(in)::unit
!    character(*,1),intent(in)::iotype
!    integer(4),intent(in)::v_list(:)
!    integer(4),intent(out)::iostat
!    character(*,1),intent(inout)::iomsg
!  end
!end
