! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! C712 The value of scalar-int-constant-expr shall be nonnegative and 
! shall specify a representation method that exists on the processor.
! C714 The value of kind-param shall be nonnegative.
! C715 The value of kind-param shall specify a representation method that 
! exists on the processor.
! C719 The value of scalar-int-constant-expr shall be nonnegative and shall 
! specify a representation method that exists on the processor.
! C725 The optional comma in a length-selector is permitted only if no 
! double-colon separator appears in the typedeclaration- stmt.
! C727 The value of kind-param shall specify a representation method that 
! exists on the processor.
! C728 The value of kind-param shall specify a representation method that 
! exists on the processor.
!
!ERROR: INTEGER(KIND=0) is not a supported type
integer(kind=0) :: j0
!ERROR: INTEGER(KIND=-1) is not a supported type
integer(kind=-1) :: jm1
!ERROR: INTEGER(KIND=3) is not a supported type
integer(kind=3) :: j3
!ERROR: INTEGER(KIND=32) is not a supported type
integer(kind=32) :: j32
!ERROR: REAL(KIND=0) is not a supported type
real(kind=0) :: a0
!ERROR: REAL(KIND=-1) is not a supported type
real(kind=-1) :: am1
!ERROR: REAL(KIND=1) is not a supported type
real(kind=1) :: a1
!ERROR: REAL(KIND=7) is not a supported type
real(kind=7) :: a7
!ERROR: REAL(KIND=32) is not a supported type
real(kind=32) :: a32
!ERROR: COMPLEX(KIND=0) is not a supported type
complex(kind=0) :: z0
!ERROR: COMPLEX(KIND=-1) is not a supported type
complex(kind=-1) :: zm1
!ERROR: COMPLEX(KIND=1) is not a supported type
complex(kind=1) :: z1
!ERROR: COMPLEX(KIND=7) is not a supported type
complex(kind=7) :: z7
!ERROR: COMPLEX(KIND=32) is not a supported type
complex(kind=32) :: z32
!ERROR: COMPLEX*1 is not a supported type
complex*1 :: zs1
!ERROR: COMPLEX*2 is not a supported type
complex*2 :: zs2
!ERROR: COMPLEX*64 is not a supported type
complex*64 :: zs64
!ERROR: LOGICAL(KIND=0) is not a supported type
logical(kind=0) :: l0
!ERROR: LOGICAL(KIND=-1) is not a supported type
logical(kind=-1) :: lm1
!ERROR: LOGICAL(KIND=3) is not a supported type
logical(kind=3) :: l3
!ERROR: LOGICAL(KIND=16) is not a supported type
logical(kind=16) :: l16
integer, parameter :: negOne = -1
!ERROR: unsupported LOGICAL(KIND=0)
logical :: lvar0 = .true._0
logical :: lvar1 = .true._1
logical :: lvar2 = .true._2
!ERROR: unsupported LOGICAL(KIND=3)
logical :: lvar3 = .true._3
logical :: lvar4 = .true._4
!ERROR: unsupported LOGICAL(KIND=5)
logical :: lvar5 = .true._5
!ERROR: unsupported LOGICAL(KIND=-1)
logical :: lvar6 = .true._negOne
character (len=99, kind=1) :: cvar1
character (len=99, kind=2) :: cvar2
character *4, cvar3
character *(5), cvar4
!ERROR: KIND value (3) not valid for CHARACTER
character (len=99, kind=3) :: cvar5
!ERROR: KIND value (-1) not valid for CHARACTER
character (len=99, kind=-1) :: cvar6
character(len=*), parameter :: cvar7 = 1_"abcd"
character(len=*), parameter :: cvar8 = 2_"abcd"
!ERROR: CHARACTER(KIND=3) is not a supported type
character(len=*), parameter :: cvar9 = 3_"abcd"
character(len=*), parameter :: cvar10 = 4_"abcd"
!ERROR: CHARACTER(KIND=8) is not a supported type
character(len=*), parameter :: cvar11 = 8_"abcd"
end program

subroutine s(a, b)
  character(*,2) :: a
  !ERROR: KIND value (8) not valid for CHARACTER
  character(*,8) :: b
end
