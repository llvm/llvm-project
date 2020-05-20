! RUN: %S/test_errors.sh %s %t %f18
! C1135 A cycle-stmt shall not appear within a CHANGE TEAM, CRITICAL, or DO 
! CONCURRENT construct if it belongs to an outer construct.
!
! C1167 -- An exit-stmt shall not appear within a DO CONCURRENT construct if 
! it belongs to that construct or an outer construct.
!
! C1168 -- An exit-stmt shall not appear within a CHANGE TEAM or CRITICAL 
! construct if it belongs to an outer construct.

subroutine s1()
!ERROR: No matching DO construct for CYCLE statement
  cycle
end subroutine s1

subroutine s2()
!ERROR: No matching construct for EXIT statement
  exit
end subroutine s2

subroutine s3()
  level0: block
!ERROR: No matching DO construct for CYCLE statement
    cycle level0
  end block level0
end subroutine s3

subroutine s4()
  level0: do i = 1, 10
    level1: do concurrent (j = 1:20)
!ERROR: CYCLE must not leave a DO CONCURRENT statement
      cycle level0
    end do level1
  end do level0
end subroutine s4

subroutine s5()
  level0: do i = 1, 10
    level1: do concurrent (j = 1:20)
!ERROR: EXIT must not leave a DO CONCURRENT statement
      exit level0
    end do level1
  end do level0
end subroutine s5

subroutine s6()
  level0: do i = 1, 10
    level1: critical
!ERROR: CYCLE must not leave a CRITICAL statement
      cycle level0
    end critical level1
  end do level0
end subroutine s6

subroutine s7()
  level0: do i = 1, 10
    level1: critical
!ERROR: EXIT must not leave a CRITICAL statement
      exit level0
    end critical level1
  end do level0
end subroutine s7

subroutine s8()
  use :: iso_fortran_env
  type(team_type) team_var

  level0: do i = 1, 10
    level1: change team(team_var)
!ERROR: CYCLE must not leave a CHANGE TEAM statement
      cycle level0
    end team level1
  end do level0
end subroutine s8

subroutine s9()
  use :: iso_fortran_env
  type(team_type) team_var

  level0: do i = 1, 10
    level1: change team(team_var)
!ERROR: EXIT must not leave a CHANGE TEAM statement
      exit level0
    end team level1
  end do level0
end subroutine s9

subroutine s10(table)
! A complex, but all legal example

  integer :: table(..)

  type point
    real :: x, y
  end type point

  type, extends(point) :: color_point
    integer :: color
  end type color_point

  type(point), target :: target_var
  class(point), pointer :: p_or_c

  p_or_c => target_var
  level0: do i = 1, 10
    level1: associate (avar => ivar)
      level2: block
        level3: select case (l)
          case default
            print*, "default"
          case (1)
            level4: if (.true.) then
              level5: select rank(table)
                rank default
                  level6: select type ( a => p_or_c )
                  type is ( point )
                    cycle level0
                end select level6
              end select level5
            end if level4
        end select level3
      end block level2
    end associate level1
  end do level0
end subroutine s10

subroutine s11(table)
! A complex, but all legal example with a CYCLE statement

  integer :: table(..)

  type point
    real :: x, y
  end type point

  type, extends(point) :: color_point
    integer :: color
  end type color_point

  type(point), target :: target_var
  class(point), pointer :: p_or_c

  p_or_c => target_var
  level0: do i = 1, 10
    level1: associate (avar => ivar)
      level2: block
        level3: select case (l)
          case default
            print*, "default"
          case (1)
            level4: if (.true.) then
              level5: select rank(table)
                rank default
                  level6: select type ( a => p_or_c )
                  type is ( point )
                    cycle level0
                end select level6
              end select level5
            end if level4
        end select level3
      end block level2
    end associate level1
  end do level0
end subroutine s11

subroutine s12(table)
! A complex, but all legal example with an EXIT statement

  integer :: table(..)

  type point
    real :: x, y
  end type point

  type, extends(point) :: color_point
    integer :: color
  end type color_point

  type(point), target :: target_var
  class(point), pointer :: p_or_c

  p_or_c => target_var
  level0: do i = 1, 10
    level1: associate (avar => ivar)
      level2: block
        level3: select case (l)
          case default
            print*, "default"
          case (1)
            level4: if (.true.) then
              level5: select rank(table)
                rank default
                  level6: select type ( a => p_or_c )
                  type is ( point )
                    exit level0
                end select level6
              end select level5
            end if level4
        end select level3
      end block level2
    end associate level1
  end do level0
end subroutine s12

subroutine s13(table)
! Similar example without construct names

  integer :: table(..)

  type point
    real :: x, y
  end type point

  type, extends(point) :: color_point
    integer :: color
  end type color_point

  type(point), target :: target_var
  class(point), pointer :: p_or_c

  p_or_c => target_var
  do i = 1, 10
    associate (avar => ivar)
      block
        select case (l)
          case default
            print*, "default"
          case (1)
            if (.true.) then
              select rank(table)
                rank default
                  select type ( a => p_or_c )
                  type is ( point )
                    cycle
                end select
              end select
            end if
        end select
      end block
    end associate
  end do
end subroutine s13

subroutine s14(table)

  integer :: table(..)

  type point
    real :: x, y
  end type point

  type, extends(point) :: color_point
    integer :: color
  end type color_point

  type(point), target :: target_var
  class(point), pointer :: p_or_c

  p_or_c => target_var
  do i = 1, 10
    associate (avar => ivar)
      block
        critical
          select case (l)
            case default
              print*, "default"
            case (1)
              if (.true.) then
                select rank(table)
                  rank default
                    select type ( a => p_or_c )
                    type is ( point )
!ERROR: CYCLE must not leave a CRITICAL statement
                      cycle
!ERROR: EXIT must not leave a CRITICAL statement
                      exit
                  end select
                end select
              end if
          end select
        end critical
      end block
    end associate
  end do
end subroutine s14

subroutine s15(table)
! Illegal EXIT to an intermediated construct

  integer :: table(..)

  type point
    real :: x, y
  end type point

  type, extends(point) :: color_point
    integer :: color
  end type color_point

  type(point), target :: target_var
  class(point), pointer :: p_or_c

  p_or_c => target_var
  level0: do i = 1, 10
    level1: associate (avar => ivar)
      level2: block
        level3: select case (l)
          case default
            print*, "default"
          case (1)
            level4: if (.true.) then
              level5: critical
                level6: select rank(table)
                  rank default
                    level7: select type ( a => p_or_c )
                    type is ( point )
                      exit level6
!ERROR: EXIT must not leave a CRITICAL statement
                      exit level4
                  end select level7
                end select level6
              end critical level5
            end if level4
        end select level3
      end block level2
    end associate level1
  end do level0
end subroutine s15
