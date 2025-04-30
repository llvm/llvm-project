
! KGEN-generated Fortran source file
!
! Filename    : rrsw_cld.f90
! Generated at: 2015-07-27 00:38:36
! KGEN version: 0.4.13



    MODULE rrsw_cld
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        !      use parkind, only : jpim, jprb
        IMPLICIT NONE
        !------------------------------------------------------------------
        ! rrtmg_sw cloud property coefficients
        !
        ! Initial: J.-J. Morcrette, ECMWF, oct1999
        ! Revised: J. Delamere/MJIacono, AER, aug2005
        ! Revised: MJIacono, AER, nov2005
        ! Revised: MJIacono, AER, jul2006
        !------------------------------------------------------------------
        !
        !  name     type     purpose
        ! -----  :  ----   : ----------------------------------------------
        ! xxxliq1 : real   : optical properties (extinction coefficient, single
        !                    scattering albedo, assymetry factor) from
        !                    Hu & Stamnes, j. clim., 6, 728-742, 1993.
        ! xxxice2 : real   : optical properties (extinction coefficient, single
        !                    scattering albedo, assymetry factor) from streamer v3.0,
        !                    Key, streamer user's guide, cooperative institude
        !                    for meteorological studies, 95 pp., 2001.
        ! xxxice3 : real   : optical properties (extinction coefficient, single
        !                    scattering albedo, assymetry factor) from
        !                    Fu, j. clim., 9, 1996.
        ! xbari   : real   : optical property coefficients for five spectral
        !                    intervals (2857-4000, 4000-5263, 5263-7692, 7692-14285,
        !                    and 14285-40000 wavenumbers) following
        !                    Ebert and Curry, jgr, 97, 3831-3836, 1992.
        !------------------------------------------------------------------
        REAL(KIND=r8) :: extliq1(58,16:29)
        REAL(KIND=r8) :: ssaliq1(58,16:29)
        REAL(KIND=r8) :: asyliq1(58,16:29)
        REAL(KIND=r8) :: extice2(43,16:29)
        REAL(KIND=r8) :: ssaice2(43,16:29)
        REAL(KIND=r8) :: asyice2(43,16:29)
        REAL(KIND=r8) :: extice3(46,16:29)
        REAL(KIND=r8) :: ssaice3(46,16:29)
        REAL(KIND=r8) :: asyice3(46,16:29)
        REAL(KIND=r8) :: fdlice3(46,16:29)
        REAL(KIND=r8) :: abari(5)
        REAL(KIND=r8) :: bbari(5)
        REAL(KIND=r8) :: cbari(5)
        REAL(KIND=r8) :: dbari(5)
        REAL(KIND=r8) :: ebari(5)
        REAL(KIND=r8) :: fbari(5)
        PUBLIC kgen_read_externs_rrsw_cld
    CONTAINS

    ! write subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_rrsw_cld(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) extliq1
        READ(UNIT=kgen_unit) ssaliq1
        READ(UNIT=kgen_unit) asyliq1
        READ(UNIT=kgen_unit) extice2
        READ(UNIT=kgen_unit) ssaice2
        READ(UNIT=kgen_unit) asyice2
        READ(UNIT=kgen_unit) extice3
        READ(UNIT=kgen_unit) ssaice3
        READ(UNIT=kgen_unit) asyice3
        READ(UNIT=kgen_unit) fdlice3
        READ(UNIT=kgen_unit) abari
        READ(UNIT=kgen_unit) bbari
        READ(UNIT=kgen_unit) cbari
        READ(UNIT=kgen_unit) dbari
        READ(UNIT=kgen_unit) ebari
        READ(UNIT=kgen_unit) fbari
    END SUBROUTINE kgen_read_externs_rrsw_cld

    END MODULE rrsw_cld
