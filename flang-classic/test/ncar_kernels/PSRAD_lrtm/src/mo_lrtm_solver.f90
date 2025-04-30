
! KGEN-generated Fortran source file
!
! Filename    : mo_lrtm_solver.f90
! Generated at: 2015-02-19 15:30:36
! KGEN version: 0.4.4



    MODULE mo_lrtm_solver
        USE mo_kind, ONLY: wp
        USE mo_math_constants, ONLY: pi
        USE mo_rrtm_params, ONLY: nbndlw
        USE mo_rad_fastmath, ONLY: tautrans
        USE mo_rad_fastmath, ONLY: transmit
        IMPLICIT NONE
        REAL(KIND=wp), parameter :: fluxfac = 2.0e+04_wp * pi
        CONTAINS

        ! read subroutines
        ! -------------------------------------------------------------------------------

        SUBROUTINE lrtm_solver(kproma, kbdim, klev, tau, layplnk, levplnk, weights, secdiff, surfplanck, surfemis, fluxup, fluxdn)
            !
            ! Compute IR (no scattering) radiative transfer for a set of columns
            !   Based on AER code RRTMG_LW_RTNMC, including approximations used there
            ! Layers are ordered from botton to top (i.e. tau(1) is tau in lowest layer)
            ! Computes all-sky RT given a total optical thickness in each layer
            !
            INTEGER, intent(in) :: kbdim
            INTEGER, intent(in) :: klev
            INTEGER, intent(in) :: kproma
            !< Number of columns
            !< Maximum number of columns as declared in calling (sub)program
            !< number of layers (one fewer than levels)
            REAL(KIND=wp), intent(in) :: tau(kbdim,klev)
            REAL(KIND=wp), intent(in) :: layplnk(kbdim,klev)
            REAL(KIND=wp), intent(in) :: weights(kbdim,klev) !< dimension (kbdim, klev)
            !< Longwave optical thickness
            !< Planck function at layer centers
            !< Fraction of total Planck function for this g-point
            REAL(KIND=wp), intent(in) :: levplnk(kbdim, 0:klev)
            !< Planck function at layer edges, level i is the top of layer i
            REAL(KIND=wp), intent(in) :: secdiff(kbdim)
            REAL(KIND=wp), intent(in) :: surfemis(kbdim)
            REAL(KIND=wp), intent(in) :: surfplanck(kbdim) !< dimension (kbdim)
            !< Planck function at surface
            !< Surface emissivity
            !< secant of integration angle - depends on band, column water vapor
            REAL(KIND=wp), intent(out) :: fluxup(kbdim, 0:klev)
            REAL(KIND=wp), intent(out) :: fluxdn(kbdim, 0:klev) !< dimension (kbdim, 0:klev)
            !< Fluxes at the interfaces
            ! -----------
            INTEGER :: jk
            !< Loop index for layers
            REAL(KIND=wp) :: odepth(kbdim,klev)
            REAL(KIND=wp) :: tfn(kbdim)
            REAL(KIND=wp) :: dplnkup(kbdim,klev)
            REAL(KIND=wp) :: dplnkdn(kbdim,klev)
            REAL(KIND=wp) :: bbup(kbdim,klev)
            REAL(KIND=wp) :: bbdn(kbdim,klev)
            REAL(KIND=wp) :: trans(kbdim,klev)
            !< Layer transmissivity
            !< TFN_TBL
            !< Tau transition function; i.e. the transition of the Planck
            !< function from that for the mean layer temperature to that for
            !< the layer boundary temperature as a function of optical depth.
            !< The "linear in tau" method is used to make the table.
            !< Upward derivative of Planck function
            !< Downward derivative of Planck function
            !< Interpolated downward emission
            !< Interpolated upward emission
            !< Effective IR optical depth of layer
            REAL(KIND=wp) :: rad_dn(kbdim,0:klev)
            REAL(KIND=wp) :: rad_up(kbdim,0:klev)
            !< Radiance down at propagation angle
            !< Radiance down at propagation angle
            ! This secant and weight corresponds to the standard diffusivity
            ! angle.  The angle is redefined for some bands.
            REAL(KIND=wp), parameter :: wtdiff = 0.5_wp
            ! -----------
            !
            ! 1.0 Initial preparations
            ! Weight optical depth by 1/cos(diffusivity angle), which depends on band
            ! This will be used to compute layer transmittance
            ! -----
            !IBM* ASSERT(NODEPS)
            DO jk = 1, klev
                odepth(1:kproma,jk) = max(0._wp, secdiff(1:kproma) * tau(1:kproma,jk))
            END DO 
            !
            ! 2.0 Radiative transfer
            !
            ! -----
            !
            ! Plank function derivatives and total emission for linear-in-tau approximation
            !
            !IBM* ASSERT(NODEPS)
            DO jk = 1, klev
                tfn(1:kproma) = tautrans(odepth(:,jk), kproma)
                dplnkup(1:kproma,jk) = levplnk(1:kproma,jk) - layplnk(1:kproma,jk)
                dplnkdn(1:kproma,jk) = levplnk(1:kproma,jk-1) - layplnk(1:kproma,jk)
                bbup(1:kproma,jk) = weights(1:kproma,jk) * (layplnk(1:kproma,jk) + dplnkup(1:kproma,jk) * tfn(1:kproma))
                bbdn(1:kproma,jk) = weights(1:kproma,jk) * (layplnk(1:kproma,jk) + dplnkdn(1:kproma,jk) * tfn(1:kproma))
            END DO 
            ! -----
            ! 2.1 Downward radiative transfer
            !
            ! Level 0 is closest to the ground
            !
            rad_dn(:, klev) = 0. ! Upper boundary condition - no downwelling IR
            DO jk = klev, 1, -1
                trans(1:kproma,jk) = transmit(odepth(:,jk), kproma)
                ! RHS is a rearrangment of rad_dn(:,jk) * (1._wp - trans(:,jk)) + trans(:,jk) * bbdn(:)
                rad_dn(1:kproma,jk-1) = rad_dn(1:kproma,jk) + (bbdn(1:kproma,jk) - rad_dn(1:kproma,jk)) * trans(1:kproma,jk)
            END DO 
            !
            ! 2.2 Surface contribution, including reflection
            !
            rad_up(1:kproma, 0) = weights(1:kproma, 1) * surfemis(1:kproma) * surfplanck(1:kproma)                  + (1._wp - &
            surfemis(1:kproma)) * rad_dn(1:kproma, 0)
            !
            ! 2.3 Upward radiative transfer
            !
            DO jk = 1, klev
                rad_up(1:kproma,jk) = rad_up(1:kproma,jk-1) * (1._wp - trans(1:kproma,jk)) + trans(1:kproma,jk) * bbup(1:kproma,&
                jk)
            END DO 
            !
            ! 3.0 Covert intensities at diffusivity angles to fluxes
            !
            ! -----
            fluxup(1:kproma, 0:klev) = rad_up(1:kproma,:) * wtdiff * fluxfac
            fluxdn(1:kproma, 0:klev) = rad_dn(1:kproma,:) * wtdiff * fluxfac
        END SUBROUTINE lrtm_solver
        ! -------------------------------------------------------------------------------

        elemental FUNCTION find_secdiff(iband, pwvcm)
            INTEGER, intent(in) :: iband
            !< RRTMG LW band number
            REAL(KIND=wp), intent(in) :: pwvcm
            !< Precipitable water vapor (cm)
            REAL(KIND=wp) :: find_secdiff
            ! Compute diffusivity angle for Bands 2-3 and 5-9 to vary (between 1.50
            ! and 1.80) as a function of total column water vapor.  The function
            ! has been defined to minimize flux and cooling rate errors in these bands
            ! over a wide range of precipitable water values.
            REAL(KIND=wp), dimension(nbndlw), parameter :: a0 = (/ 1.66_wp,  1.55_wp,  1.58_wp,  1.66_wp, 1.54_wp, 1.454_wp,  &
            1.89_wp,  1.33_wp,                   1.668_wp, 1.66_wp,  1.66_wp,  1.66_wp, 1.66_wp,  1.66_wp,  1.66_wp,  1.66_wp /)
            REAL(KIND=wp), dimension(nbndlw), parameter :: a1 = (/ 0.00_wp,  0.25_wp,  0.22_wp,  0.00_wp,  0.13_wp, 0.446_wp, &
            -0.10_wp,  0.40_wp,                 -0.006_wp, 0.00_wp,  0.00_wp,  0.00_wp,  0.00_wp,  0.00_wp,  0.00_wp,  0.00_wp /)
            REAL(KIND=wp), dimension(nbndlw), parameter :: a2 = (/ 0.00_wp, -12.0_wp, -11.7_wp,  0.00_wp, -0.72_wp,-0.243_wp,  &
            0.19_wp,-0.062_wp,                  0.414_wp, 0.00_wp,  0.00_wp,  0.00_wp,  0.00_wp,  0.00_wp,  0.00_wp, 0.00_wp /)
            IF (iband == 1 .or. iband == 4 .or. iband >= 10) THEN
                find_secdiff = 1.66_wp
                ELSE
                find_secdiff = max(min(a0(iband) + a1(iband) * exp(a2(iband)*pwvcm), 1.8_wp), 1.5_wp)
            END IF 
        END FUNCTION find_secdiff
        ! -------------------------------------------------------------------------------
    END MODULE mo_lrtm_solver
