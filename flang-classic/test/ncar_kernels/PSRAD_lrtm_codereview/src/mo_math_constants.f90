
! KGEN-generated Fortran source file
!
! Filename    : mo_math_constants.f90
! Generated at: 2015-02-19 15:30:32
! KGEN version: 0.4.4



    MODULE mo_math_constants
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        PUBLIC
        ! Mathematical constants defined:
        !
        !--------------------------------------------------------------
        ! Fortran name | C name       | meaning                       |
        !--------------------------------------------------------------
        ! euler        | M_E          |  e                            |
        ! log2e        | M_LOG2E      |  log2(e)                      |
        ! log10e       | M_LOG10E     |  log10(e)                     |
        ! ln2          | M_LN2        |  ln(2)                        |
        ! ln10         | M_LN10       |  ln(10)                       |
        ! pi           | M_PI         |  pi                           |
        ! pi_2         | M_PI_2       |  pi/2                         |
        ! pi_4         | M_PI_4       |  pi/4                         |
        ! rpi          | M_1_PI       |  1/pi                         |
        ! rpi_2        | M_2_PI       |  2/pi                         |
        ! rsqrtpi_2    | M_2_SQRTPI   |  2/(sqrt(pi))                 |
        ! sqrt2        | M_SQRT2      |  sqrt(2)                      |
        ! sqrt1_2      | M_SQRT1_2    |  1/sqrt(2)                    |
        ! sqrt3        |              |  sqrt(3)                      |
        ! sqrt1_3      |              |  1/sqrt(3)                    |
        ! half angle of pentagon                                      |
        ! pi_5         |              |  pi/5                         |
        ! latitude of the lowest major  triangle corner               |
        ! and latitude of the major hexagonal faces centers           |
        ! phi0         |              |  pi/2 -2acos(1/(2*sin(pi/5))) |
        ! conversion factor from radians to degree                    |
        ! rad2deg      |              |  180/pi                       |
        ! conversion factor from degree to radians                    |
        ! deg2rad      |              |  pi/180                       |
        ! one_third    |              |  1/3                          |
        !-------------------------------------------------------------|
        REAL(KIND=wp), parameter :: pi        =  3.14159265358979323846264338327950288419717_wp

    ! read subroutines
    END MODULE mo_math_constants
