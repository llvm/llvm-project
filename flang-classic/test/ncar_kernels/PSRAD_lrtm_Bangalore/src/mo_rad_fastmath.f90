
! KGEN-generated Fortran source file
!
! Filename    : mo_rad_fastmath.f90
! Generated at: 2015-02-19 15:30:32
! KGEN version: 0.4.4



    MODULE mo_rad_fastmath
        USE mo_kind, ONLY: dp
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        PRIVATE
        PUBLIC tautrans, inv_expon, transmit
        !< Optical depth
        !< Exponential lookup table (EXP(-tau))
        !< Tau transition function
        ! i.e. the transition of the Planck function from that for the mean layer temperature
        ! to that for the layer boundary temperature as a function of optical depth.
        ! The "linear in tau" method is used to make the table.
        !< Value of tau below which expansion is used
        !< Smallest value for exponential table
        !< Pade approximation constant
        REAL(KIND=wp), parameter :: rec_6  = 1._wp/6._wp
        !
        ! The RRTMG tables are indexed with INT(tblint * x(i)/(bpade + x(i)) + 0.5_wp)
        !   But these yield unstable values in the SW solver for some parameter sets, so
        !   we'll remove this option (though the tables are initialized if people want them).
        ! RRTMG table lookups are approximated second-order Taylor series expansion
        !   (e.g. exp(-x) = 1._wp - x(1:n) + 0.5_wp * x(1:n)**2, tautrans = x/6._wp) for x < od_lo
        !
        CONTAINS

        ! read subroutines
        ! ------------------------------------------------------------

        ! ------------------------------------------------------------

        ! ------------------------------------------------------------

        FUNCTION inv_expon(x, n)
            !
            ! Compute EXP(-x) - but do it fast
            !
            INTEGER, intent(in) :: n
            REAL(KIND=dp), intent(in) :: x(n)
            REAL(KIND=dp) :: inv_expon(n)
            inv_expon(1:n) = exp(-x(1:n))
        END FUNCTION inv_expon
        ! ------------------------------------------------------------

        FUNCTION transmit(x, n)
            !
            ! Compute transmittance 1 - EXP(-x)
            !
            INTEGER, intent(in) :: n
            REAL(KIND=dp), intent(in) :: x(n)
            REAL(KIND=dp) :: transmit(n)
            !
            ! MASS and MKL libraries have exp(x) - 1 functions; we could
            !   use those here
            !
            transmit(1:n) = 1._wp - inv_expon(x,n)
        END FUNCTION transmit
        ! ------------------------------------------------------------

        FUNCTION tautrans(x, n)
            !
            ! Compute "tau transition" using linear-in-tau approximation
            !
            INTEGER, intent(in) :: n
            REAL(KIND=dp), intent(in) :: x(n)
            REAL(KIND=dp) :: tautrans(n)
            REAL(KIND=dp) :: y(n)
            !
            ! Default calculation is unstable (NaN) for the very lowest value of tau (3.6e-4)
            !
            y(:) = inv_expon(x,n)
            tautrans(:) = merge(1._wp - 2._wp*(1._wp/x(1:n) - y(:)/(1._wp-y(:))),                        x * rec_6,               &
                                                            x > 1.e-3_wp)
        END FUNCTION tautrans
        ! ------------------------------------------------------------
    END MODULE mo_rad_fastmath
