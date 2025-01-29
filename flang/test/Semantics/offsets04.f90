!RUN: %flang_fc1 -fdebug-dump-symbols %s | FileCheck %s

!REQUIRES: target={{.+}}-aix{{.*}}

! Size and alignment of bind(c) derived types
subroutine s1()
  use, intrinsic :: iso_c_binding
  type, bind(c) :: dt1
    character(c_char) :: x1    !CHECK: x1 size=1 offset=0:
    real(c_double) :: x2       !CHECK: x2 size=8 offset=4:
  end type
  type, bind(c) :: dt2
    character(c_char) :: x1(9) !CHECK: x1 size=9 offset=0:
    real(c_double) :: x2       !CHECK: x2 size=8 offset=12:
  end type
  type, bind(c) :: dt3
    integer(c_short) :: x1     !CHECK: x1 size=2 offset=0:
    real(c_double) :: x2       !CHECK: x2 size=8 offset=4:
  end type
  type, bind(c) :: dt4
    integer(c_int) :: x1       !CHECK: x1 size=4 offset=0:
    real(c_double) :: x2       !CHECK: x2 size=8 offset=4:
  end type
  type, bind(c) :: dt5
    real(c_double) :: x1       !CHECK: x1 size=8 offset=0:
    real(c_double) :: x2       !CHECK: x2 size=8 offset=8:
  end type
  type, bind(c) :: dt6
    integer(c_long) :: x1      !CHECK: x1 size=8 offset=0:
    character(c_char) :: x2    !CHECK: x2 size=1 offset=8:
    real(c_double) :: x3       !CHECK: x3 size=8 offset=12:
  end type
  type, bind(c) :: dt7
    integer(c_long) :: x1      !CHECK: x1 size=8 offset=0:
    integer(c_long) :: x2      !CHECK: x2 size=8 offset=8:
    character(c_char) :: x3    !CHECK: x3 size=1 offset=16:
    real(c_double) :: x4       !CHECK: x4 size=8 offset=20:
  end type
  type, bind(c) :: dt8
    character(c_char) :: x1         !CHECK: x1 size=1 offset=0:
    complex(c_double_complex) :: x2 !CHECK: x2 size=16 offset=4:
  end type
end subroutine

subroutine s2()
  use, intrinsic :: iso_c_binding
  type, bind(c) :: dt10
    character(c_char) :: x1
    real(c_double) :: x2
  end type
  type, bind(c) :: dt11
    type(dt10) :: y1           !CHECK: y1 size=12 offset=0:
    real(c_double) :: y2       !CHECK: y2 size=8 offset=12:
  end type
  type, bind(c) :: dt12
    character(c_char) :: y1    !CHECK: y1 size=1 offset=0:
    type(dt10) :: y2           !CHECK: y2 size=12 offset=4:
    character(c_char) :: y3    !CHECK: y3 size=1 offset=16:
  end type
  type, bind(c) :: dt13
    integer(c_short) :: y1     !CHECK: y1 size=2 offset=0:
    type(dt10) :: y2           !CHECK: y2 size=12 offset=4:
    character(c_char) :: y3    !CHECK: y3 size=1 offset=16:
  end type

  type, bind(c) :: dt20
    character(c_char) :: x1
    integer(c_short) :: x2
  end type
  type, bind(c) :: dt21
    real(c_double) :: y1       !CHECK: y1 size=8 offset=0:
    type(dt20) :: y2           !CHECK: y2 size=4 offset=8:
    real(c_double) :: y3       !CHECK: y3 size=8 offset=12:
  end type

  type, bind(c) :: dt30
    character(c_char) :: x1
    character(c_char) :: x2
  end type
  type, bind(c) :: dt31
     integer(c_long) :: y1     !CHECK: y1 size=8 offset=0:
     type(dt30) :: y2          !CHECK: y2 size=2 offset=8:
     real(c_double) :: y3      !CHECK: y3 size=8 offset=12:
  end type

  type, bind(c) :: dt40
    integer(c_short) :: x1
    real(c_double) :: x2
  end type
  type, bind(c) :: dt41
    real(c_double) :: y1       !CHECK: y1 size=8 offset=0:
    type(dt40) :: y2           !CHECK: y2 size=12 offset=8:
    real(c_double) :: y3       !CHECK: y3 size=8 offset=20:
  end type

  type, bind(c) :: dt50
    integer(c_short) :: x1
    complex(c_double_complex) :: x2
  end type
  type, bind(c) :: dt51
    real(c_double) :: y1            !CHECK: y1 size=8 offset=0:
    type(dt50) :: y2                !CHECK: y2 size=20 offset=8:
    complex(c_double_complex) :: y3 !CHECK: y3 size=16 offset=28:
  end type
end subroutine
