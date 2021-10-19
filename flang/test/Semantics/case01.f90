! RUN: %python %S/test_errors.py %s %flang_fc1
! Test SELECT CASE Constraints: C1145, C1146, C1147, C1148, C1149
program selectCaseProg
   implicit none
   ! local variable declaration
   character :: grade1 = 'B'
   integer :: grade2 = 3
   logical :: grade3 = .false.
   real :: grade4 = 2.0
   character (len = 10) :: name = 'test'
   logical, parameter :: grade5 = .false.
   CHARACTER(KIND=1), parameter :: ASCII_parm1 = 'a', ASCII_parm2='b'
   CHARACTER(KIND=2), parameter :: UCS16_parm = 'c'
   CHARACTER(KIND=4), parameter :: UCS32_parm ='d'
   type scores
     integer :: val
   end type
   type (scores) :: score = scores(25)
   type (scores), parameter :: score_val = scores(50)

  ! Valid Cases
   select case (grade1)
      case ('A')
      case ('B')
      case ('C')
      case default
   end select

   select case (grade2)
      case (1)
      case (2)
      case (3)
      case default
   end select

   select case (grade3)
      case (.true.)
      case (.false.)
   end select

   select case (name)
      case default
      case ('now')
      case ('test')
   end select

  ! C1145
  !ERROR: SELECT CASE expression must be integer, logical, or character
  select case (grade4)
     case (1.0)
     case (2.0)
     case (3.0)
     case default
  end select

  !ERROR: SELECT CASE expression must be integer, logical, or character
  select case (score)
     case (score_val)
     case (scores(100))
  end select

  ! C1146
  select case (grade3)
     case default
     case (.true.)
     !ERROR: CASE DEFAULT conflicts with previous cases
     case default
  end select

  ! C1147
  select case (grade2)
     !ERROR: CASE value has type 'CHARACTER(1)' which is not compatible with the SELECT CASE expression's type 'INTEGER(4)'
     case (:'Z')
     case default
   end select

  select case (grade1)
     !ERROR: CASE value has type 'INTEGER(4)' which is not compatible with the SELECT CASE expression's type 'CHARACTER(KIND=1,LEN=1_8)'
     case (:1)
     case default
   end select

  select case (grade3)
     case default
     case (.true.)
     !ERROR: CASE value has type 'INTEGER(4)' which is not compatible with the SELECT CASE expression's type 'LOGICAL(4)'
     case (3)
  end select

  select case (grade2)
     case default
     case (2 :)
     !ERROR: CASE value has type 'LOGICAL(4)' which is not compatible with the SELECT CASE expression's type 'INTEGER(4)'
     case (.true. :)
     !ERROR: CASE value has type 'REAL(4)' which is not compatible with the SELECT CASE expression's type 'INTEGER(4)'
     case (1.0)
     !ERROR: CASE value has type 'CHARACTER(1)' which is not compatible with the SELECT CASE expression's type 'INTEGER(4)'
     case ('wow')
  end select

  select case (ASCII_parm1)
     case (ASCII_parm2)
     !ERROR: CASE value has type 'CHARACTER(4)' which is not compatible with the SELECT CASE expression's type 'CHARACTER(1)'
     case (UCS32_parm)
     !ERROR: CASE value has type 'CHARACTER(2)' which is not compatible with the SELECT CASE expression's type 'CHARACTER(1)'
     case (UCS16_parm)
     !ERROR: CASE value has type 'CHARACTER(4)' which is not compatible with the SELECT CASE expression's type 'CHARACTER(1)'
     case (4_"ucs-32")
     !ERROR: CASE value has type 'CHARACTER(2)' which is not compatible with the SELECT CASE expression's type 'CHARACTER(1)'
     case (2_"ucs-16")
     case default
   end select

  ! C1148
  select case (grade3)
     case default
     !ERROR: CASE range is not allowed for LOGICAL
     case (.true. :)
  end select

  ! C1149
  select case (grade3)
    case (.true.)
    case (.false.)
     !ERROR: CASE (.true._1) conflicts with previous cases
     case (.true.)
    !ERROR: CASE (.false._1) conflicts with previous cases
     case (grade5)
  end select

  select case (grade2)
     case (51:50) ! warning
     case (100:)
     case (:30)
     case (40)
     case (90)
     case (91:99)
     !ERROR: CASE (81_4:90_4) conflicts with previous cases
     case (81:90)
     !ERROR: CASE (:80_4) conflicts with previous cases
     case (:80)
     !ERROR: CASE (200_4) conflicts with previous cases
     case (200)
     case default
  end select

  select case (name)
     case ('hello')
     case ('hey')
     !ERROR: CASE (:"hh") conflicts with previous cases
     case (:'hh')
     !ERROR: CASE (:"hd") conflicts with previous cases
     case (:'hd')
     case ( 'hu':)
     case ('hi':'ho')
     !ERROR: CASE ("hj") conflicts with previous cases
     case ('hj')
     !ERROR: CASE ("ha") conflicts with previous cases
     case ('ha')
     !ERROR: CASE ("hz") conflicts with previous cases
     case ('hz')
     case default
   end select

end program

program test_overlap
  integer :: i
  !OK: these cases do not overlap
  select case(i)
    case(0:)
    case(:-1)
  end select
  select case(i)
    case(-1:)
    !ERROR: CASE (:0_4) conflicts with previous cases
    case(:0)
  end select
end
