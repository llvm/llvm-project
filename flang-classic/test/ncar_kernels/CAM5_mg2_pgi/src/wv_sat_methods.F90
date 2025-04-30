
! KGEN-generated Fortran source file
!
! Filename    : wv_sat_methods.F90
! Generated at: 2015-03-31 09:44:41
! KGEN version: 0.4.5



    MODULE wv_sat_methods
        ! This portable module contains all 1 methods for estimating
        ! the saturation vapor pressure of water.
        !
        ! wv_saturation provides 1-specific interfaces and utilities
        ! based on these formulae.
        !
        ! Typical usage of this module:
        !
        ! Init:
        ! call wv_sat_methods_init(r8, <constants>, errstring)
        !
        ! Get scheme index from a name string:
        ! scheme_idx = wv_sat_get_scheme_idx(scheme_name)
        ! if (.not. wv_sat_valid_idx(scheme_idx)) <throw some error>
        !
        ! Get pressures:
        ! es = wv_sat_svp_water(t, scheme_idx)
        ! es = wv_sat_svp_ice(t, scheme_idx)
        !
        ! Use ice/water transition range:
        ! es = wv_sat_svp_trice(t, ttrice, scheme_idx)
        !
        ! Note that elemental functions cannot be pointed to, nor passed
        ! as arguments. If you need to do either, it is recommended to
        ! wrap the function so that it can be given an explicit (non-
        ! elemental) interface.
        IMPLICIT NONE
        PRIVATE
        INTEGER, parameter :: r8 = selected_real_kind(12) ! 8 byte real
        REAL(KIND=r8) :: tmelt ! Melting point of water at 1 atm (K)
        REAL(KIND=r8) :: h2otrip ! Triple point temperature of water (K)
        REAL(KIND=r8) :: tboil ! Boiling point of water at 1 atm (K)
        ! Ice-water transition range
        REAL(KIND=r8) :: epsilo ! Ice-water transition range
        REAL(KIND=r8) :: omeps ! 1._r8 - epsilo
        ! Indices representing individual schemes
        INTEGER, parameter :: oldgoffgratch_idx = 0
        INTEGER, parameter :: goffgratch_idx = 1
        INTEGER, parameter :: murphykoop_idx = 2
        INTEGER, parameter :: bolton_idx = 3
        ! Index representing the current default scheme.
        INTEGER, parameter :: initial_default_idx = goffgratch_idx
        INTEGER :: default_idx = initial_default_idx
        PUBLIC wv_sat_svp_water
        PUBLIC wv_sat_svp_ice
        ! pressure -> humidity conversion
        PUBLIC wv_sat_svp_to_qsat
        ! Combined qsat operations
        PUBLIC wv_sat_qsat_water
        PUBLIC wv_sat_qsat_ice
            PUBLIC kgen_read_externs_wv_sat_methods
        CONTAINS

        ! write subroutines
        ! No subroutines

        ! module extern variables

        SUBROUTINE kgen_read_externs_wv_sat_methods(kgen_unit)
            INTEGER, INTENT(IN) :: kgen_unit
            READ(UNIT=kgen_unit) tmelt
            READ(UNIT=kgen_unit) h2otrip
            READ(UNIT=kgen_unit) tboil
            READ(UNIT=kgen_unit) epsilo
            READ(UNIT=kgen_unit) omeps
            READ(UNIT=kgen_unit) default_idx
        END SUBROUTINE kgen_read_externs_wv_sat_methods

        !---------------------------------------------------------------------
        ! ADMINISTRATIVE FUNCTIONS
        !---------------------------------------------------------------------
        ! Get physical constants

        ! Look up index by name.

        ! Check validity of an index from the above routine.

        ! Set default scheme (otherwise, Goff & Gratch is default)
        ! Returns a logical representing success (.true.) or
        ! failure (.false.).

        ! Reset default scheme to initial value.
        ! The same thing can be accomplished with wv_sat_set_default;
        ! the real reason to provide this routine is to reset the
        ! module for testing purposes.

        !---------------------------------------------------------------------
        ! UTILITIES
        !---------------------------------------------------------------------
        ! Get saturation specific humidity given pressure and SVP.
        ! Specific humidity is limited to range 0-1.

        elemental FUNCTION wv_sat_svp_to_qsat(es, p) RESULT ( qs )
            REAL(KIND=r8), intent(in) :: es ! SVP
            REAL(KIND=r8), intent(in) :: p ! Current pressure.
            REAL(KIND=r8) :: qs
            ! If pressure is less than SVP, set qs to maximum of 1.
            IF ((p - es) <= 0._r8) THEN
                qs = 1.0_r8
                ELSE
                qs = epsilo*es / (p - omeps*es)
            END IF 
        END FUNCTION wv_sat_svp_to_qsat

        elemental SUBROUTINE wv_sat_qsat_water(t, p, es, qs, idx)
            !------------------------------------------------------------------!
            ! Purpose:                                                         !
            !   Calculate SVP over water at a given temperature, and then      !
            !   calculate and return saturation specific humidity.             !
            !------------------------------------------------------------------!
            ! Inputs
            REAL(KIND=r8), intent(in) :: t ! Temperature
            REAL(KIND=r8), intent(in) :: p ! Pressure
            ! Outputs
            REAL(KIND=r8), intent(out) :: es ! Saturation vapor pressure
            REAL(KIND=r8), intent(out) :: qs ! Saturation specific humidity
            INTEGER, intent(in), optional :: idx ! Scheme index
            es = wv_sat_svp_water(t, idx)
            qs = wv_sat_svp_to_qsat(es, p)
            ! Ensures returned es is consistent with limiters on qs.
            es = min(es, p)
        END SUBROUTINE wv_sat_qsat_water

        elemental SUBROUTINE wv_sat_qsat_ice(t, p, es, qs, idx)
            !------------------------------------------------------------------!
            ! Purpose:                                                         !
            !   Calculate SVP over ice at a given temperature, and then        !
            !   calculate and return saturation specific humidity.             !
            !------------------------------------------------------------------!
            ! Inputs
            REAL(KIND=r8), intent(in) :: t ! Temperature
            REAL(KIND=r8), intent(in) :: p ! Pressure
            ! Outputs
            REAL(KIND=r8), intent(out) :: es ! Saturation vapor pressure
            REAL(KIND=r8), intent(out) :: qs ! Saturation specific humidity
            INTEGER, intent(in), optional :: idx ! Scheme index
            es = wv_sat_svp_ice(t, idx)
            qs = wv_sat_svp_to_qsat(es, p)
            ! Ensures returned es is consistent with limiters on qs.
            es = min(es, p)
        END SUBROUTINE wv_sat_qsat_ice

        !---------------------------------------------------------------------
        ! SVP INTERFACE FUNCTIONS
        !---------------------------------------------------------------------

        elemental FUNCTION wv_sat_svp_water(t, idx) RESULT ( es )
            REAL(KIND=r8), intent(in) :: t
            INTEGER, intent(in), optional :: idx
            REAL(KIND=r8) :: es
            INTEGER :: use_idx
            IF (present(idx)) THEN
                use_idx = idx
                ELSE
                use_idx = default_idx
            END IF 
            SELECT CASE ( use_idx )
                CASE ( goffgratch_idx )
                es = goffgratch_svp_water(t)
                CASE ( murphykoop_idx )
                es = murphykoop_svp_water(t)
                CASE ( oldgoffgratch_idx )
                es = oldgoffgratch_svp_water(t)
                CASE ( bolton_idx )
                es = bolton_svp_water(t)
            END SELECT 
        END FUNCTION wv_sat_svp_water

        elemental FUNCTION wv_sat_svp_ice(t, idx) RESULT ( es )
            REAL(KIND=r8), intent(in) :: t
            INTEGER, intent(in), optional :: idx
            REAL(KIND=r8) :: es
            INTEGER :: use_idx
            IF (present(idx)) THEN
                use_idx = idx
                ELSE
                use_idx = default_idx
            END IF 
            SELECT CASE ( use_idx )
                CASE ( goffgratch_idx )
                es = goffgratch_svp_ice(t)
                CASE ( murphykoop_idx )
                es = murphykoop_svp_ice(t)
                CASE ( oldgoffgratch_idx )
                es = oldgoffgratch_svp_ice(t)
                CASE ( bolton_idx )
                es = bolton_svp_water(t)
            END SELECT 
        END FUNCTION wv_sat_svp_ice

        !---------------------------------------------------------------------
        ! SVP METHODS
        !---------------------------------------------------------------------
        ! Goff & Gratch (1946)

        elemental FUNCTION goffgratch_svp_water(t) RESULT ( es )
            REAL(KIND=r8), intent(in) :: t ! Temperature in Kelvin
            REAL(KIND=r8) :: es ! SVP in Pa
            ! uncertain below -70 C
            es = 10._r8**(-7.90298_r8*(tboil/t-1._r8)+        5.02808_r8*log10(tboil/t)-        1.3816e-7_r8*(10._r8**(11.344_r8*(&
            1._r8-t/tboil))-1._r8)+        8.1328e-3_r8*(10._r8**(-3.49149_r8*(tboil/t-1._r8))-1._r8)+        log10(1013.246_r8))*100._r8
        END FUNCTION goffgratch_svp_water

        elemental FUNCTION goffgratch_svp_ice(t) RESULT ( es )
            REAL(KIND=r8), intent(in) :: t ! Temperature in Kelvin
            REAL(KIND=r8) :: es ! SVP in Pa
            ! good down to -100 C
            es = 10._r8**(-9.09718_r8*(h2otrip/t-1._r8)-3.56654_r8*        log10(h2otrip/t)+0.876793_r8*(1._r8-t/h2otrip)+        &
            log10(6.1071_r8))*100._r8
        END FUNCTION goffgratch_svp_ice
        ! Murphy & Koop (2005)

        elemental FUNCTION murphykoop_svp_water(t) RESULT ( es )
            REAL(KIND=r8), intent(in) :: t ! Temperature in Kelvin
            REAL(KIND=r8) :: es ! SVP in Pa
            ! (good for 123 < T < 332 K)
            es = exp(54.842763_r8 - (6763.22_r8 / t) - (4.210_r8 * log(t)) +        (0.000367_r8 * t) + (tanh(0.0415_r8 * (t - &
            218.8_r8)) *        (53.878_r8 - (1331.22_r8 / t) - (9.44523_r8 * log(t)) +        0.014025_r8 * t)))
        END FUNCTION murphykoop_svp_water

        elemental FUNCTION murphykoop_svp_ice(t) RESULT ( es )
            REAL(KIND=r8), intent(in) :: t ! Temperature in Kelvin
            REAL(KIND=r8) :: es ! SVP in Pa
            ! (good down to 110 K)
            es = exp(9.550426_r8 - (5723.265_r8 / t) + (3.53068_r8 * log(t))        - (0.00728332_r8 * t))
        END FUNCTION murphykoop_svp_ice
        ! Old 1 implementation, also labelled Goff & Gratch (1946)
        ! The water formula differs only due to compiler-dependent order of
        ! operations, so differences are roundoff level, usually 0.
        ! The ice formula gives fairly close answers to the current
        ! implementation, but has been rearranged, and uses the
        ! 1 atm melting point of water as the triple point.
        ! Differences are thus small but above roundoff.
        ! A curious fact: although using the melting point of water was
        ! probably a mistake, it mildly improves accuracy for ice svp,
        ! since it compensates for a systematic error in Goff & Gratch.

        elemental FUNCTION oldgoffgratch_svp_water(t) RESULT ( es )
            REAL(KIND=r8), intent(in) :: t
            REAL(KIND=r8) :: es
            REAL(KIND=r8) :: ps
            REAL(KIND=r8) :: e1
            REAL(KIND=r8) :: e2
            REAL(KIND=r8) :: f1
            REAL(KIND=r8) :: f2
            REAL(KIND=r8) :: f3
            REAL(KIND=r8) :: f4
            REAL(KIND=r8) :: f5
            REAL(KIND=r8) :: f
            ps = 1013.246_r8
            e1 = 11.344_r8*(1.0_r8 - t/tboil)
            e2 = -3.49149_r8*(tboil/t - 1.0_r8)
            f1 = -7.90298_r8*(tboil/t - 1.0_r8)
            f2 = 5.02808_r8*log10(tboil/t)
            f3 = -1.3816_r8*(10.0_r8**e1 - 1.0_r8)/10000000.0_r8
            f4 = 8.1328_r8*(10.0_r8**e2 - 1.0_r8)/1000.0_r8
            f5 = log10(ps)
            f = f1 + f2 + f3 + f4 + f5
            es = (10.0_r8**f)*100.0_r8
        END FUNCTION oldgoffgratch_svp_water

        elemental FUNCTION oldgoffgratch_svp_ice(t) RESULT ( es )
            REAL(KIND=r8), intent(in) :: t
            REAL(KIND=r8) :: es
            REAL(KIND=r8) :: term1
            REAL(KIND=r8) :: term2
            REAL(KIND=r8) :: term3
            term1 = 2.01889049_r8/(tmelt/t)
            term2 = 3.56654_r8*log(tmelt/t)
            term3 = 20.947031_r8*(tmelt/t)
            es = 575.185606e10_r8*exp(-(term1 + term2 + term3))
        END FUNCTION oldgoffgratch_svp_ice
        ! Bolton (1980)
        ! zm_conv deep convection scheme contained this SVP calculation.
        ! It appears to be from D. Bolton, 1980, Monthly Weather Review.
        ! Unlike the other schemes, no distinct ice formula is associated
        ! with it. (However, a Bolton ice formula exists in CLUBB.)
        ! The original formula used degrees C, but this function
        ! takes Kelvin and internally converts.

        elemental FUNCTION bolton_svp_water(t) RESULT ( es )
            REAL(KIND=r8), parameter :: c1 = 611.2_r8
            REAL(KIND=r8), parameter :: c2 = 17.67_r8
            REAL(KIND=r8), parameter :: c3 = 243.5_r8
            REAL(KIND=r8), intent(in) :: t ! Temperature in Kelvin
            REAL(KIND=r8) :: es ! SVP in Pa
            es = c1*exp((c2*(t - tmelt))/((t - tmelt)+c3))
        END FUNCTION bolton_svp_water
    END MODULE wv_sat_methods
