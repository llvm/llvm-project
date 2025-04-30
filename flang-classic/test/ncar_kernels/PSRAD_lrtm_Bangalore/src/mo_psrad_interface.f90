
! KGEN-generated Fortran source file
!
! Filename    : mo_psrad_interface.f90
! Generated at: 2015-02-19 15:30:28
! KGEN version: 0.4.4



    MODULE mo_psrad_interface
    USE mo_spec_sampling, only : read_var_mod5 => kgen_read_var
        USE mo_kind, ONLY: wp
        USE mo_rrtm_params, ONLY: nbndlw
        USE mo_rrtm_params, ONLY: maxinpx
        USE mo_rrtm_params, ONLY: maxxsec
        USE mo_lrtm_driver, ONLY: lrtm
        USE mo_spec_sampling, ONLY: spec_sampling_strategy
        IMPLICIT NONE
        PUBLIC lw_strat
        PUBLIC read_externs_mo_psrad_interface
        INTEGER, PARAMETER :: kgen_dp = selected_real_kind(15, 307)
        PUBLIC psrad_interface
        type, public  ::  check_t
            logical :: Passed
            integer :: numFatal
            integer :: numTotal
            integer :: numIdentical
            integer :: numWarning
            integer :: VerboseLevel
            real(kind=kgen_dp) :: tolerance
        end type check_t
        TYPE(spec_sampling_strategy), save :: lw_strat
        !< Spectral sampling strategies for longwave, shortwave
        INTEGER, parameter :: rng_seed_size = 4
        CONTAINS

        ! module extern variables

        SUBROUTINE read_externs_mo_psrad_interface(kgen_unit)
        integer, intent(in) :: kgen_unit
        call read_var_mod5(lw_strat, kgen_unit)
        END SUBROUTINE read_externs_mo_psrad_interface

        subroutine kgen_init_check(check,tolerance)
          type(check_t), intent(inout) :: check
          real(kind=kgen_dp), intent(in), optional :: tolerance
           check%Passed   = .TRUE.
           check%numFatal = 0
           check%numWarning = 0
           check%numTotal = 0
           check%numIdentical = 0
           check%VerboseLevel = 1
           if(present(tolerance)) then
             check%tolerance = tolerance
           else
              check%tolerance = 1.E-14
           endif
        end subroutine kgen_init_check
        subroutine kgen_print_check(kname, check)
           character(len=*) :: kname
           type(check_t), intent(in) ::  check
           write (*,*)
           write (*,*) TRIM(kname),' KGENPrtCheck: Tolerance for normalized RMS: ',check%tolerance
           write (*,*) TRIM(kname),' KGENPrtCheck: Number of variables checked: ',check%numTotal
           write (*,*) TRIM(kname),' KGENPrtCheck: Number of Identical results: ',check%numIdentical
           write (*,*) TRIM(kname),' KGENPrtCheck: Number of warnings detected: ',check%numWarning
           write (*,*) TRIM(kname),' KGENPrtCheck: Number of fatal errors detected: ', check%numFatal
           if (check%numFatal> 0) then
                write(*,*) TRIM(kname),' KGENPrtCheck: verification FAILED'
           else
                write(*,*) TRIM(kname),' KGENPrtCheck: verification PASSED'
           endif
        end subroutine kgen_print_check
        !---------------------------------------------------------------------------
        !>
        !! @brief  Sets up (initializes) radation routines
        !!
        !! @remarks
        !!   Modify preset variables of module MO_RADIATION which control the
        !!   configuration of the radiation scheme.
        !

        !-----------------------------------------------------------------------------
        !>
        !! @brief arranges input and calls rrtm sw and lw routines
        !!
        !! @par Revision History
        !! Original Source Rewritten and renamed by B. Stevens (2009-08)
        !!
        !! @remarks
        !!   Because the RRTM indexes vertical levels differently than ECHAM a chief
        !!   function of thise routine is to reorder the input in the vertical.  In
        !!   addition some cloud physical properties are prescribed, which are
        !!   required to derive cloud optical properties
        !!
        !! @par The gases are passed into RRTM via two multi-constituent arrays:
        !!   zwkl and wx_r. zwkl has maxinpx species and  wx_r has maxxsec species
        !!   The species are identifed as follows.
        !!     ZWKL [#/cm2]          WX_R [#/cm2]
        !!    index = 1 => H20     index = 1 => n/a
        !!    index = 2 => CO2     index = 2 => CFC11
        !!    index = 3 =>  O3     index = 3 => CFC12
        !!    index = 4 => N2O     index = 4 => n/a
        !!    index = 5 => n/a
        !!    index = 6 => CH4
        !!    index = 7 => O2
        !

        SUBROUTINE psrad_interface(kbdim, klev, nb_sw, kproma, ktrac, tk_sfc, kgen_unit)
            integer, intent(in) :: kgen_unit

            ! read interface
            !interface kgen_read_var
            !    procedure read_var_real_wp_dim2
            !    procedure read_var_real_wp_dim1
            !    procedure read_var_real_wp_dim3
            !    procedure read_var_integer_4_dim2
            !end interface kgen_read_var



            ! verification interface
            !interface kgen_verify_var
            !    procedure verify_var_logical
            !    procedure verify_var_integer
            !    procedure verify_var_real
            !    procedure verify_var_character
            !    procedure verify_var_real_wp_dim2
            !    procedure verify_var_real_wp_dim1
            !    procedure verify_var_real_wp_dim3
            !    procedure verify_var_integer_4_dim2
            !end interface kgen_verify_var

            INTEGER*8 :: kgen_intvar, start_clock, stop_clock, rate_clock
            TYPE(check_t):: check_status
            REAL(KIND=kgen_dp) :: tolerance
            INTEGER, intent(in) :: kbdim
            INTEGER, intent(in) :: klev
            INTEGER, intent(in) :: nb_sw
            INTEGER, intent(in) :: kproma
            INTEGER, intent(in) :: ktrac
            !< aerosol control
            !< number of longitudes
            !< first dimension of 2-d arrays
            !< first dimension of 2-d arrays
            !< number of levels
            !< number of tracers
            !< type of convection
            !< number of shortwave bands
            !< land sea mask, land=.true.
            !< glacier mask, glacier=.true.
            REAL(KIND=wp), intent(in) :: tk_sfc(kbdim)
            !< surface emissivity
            !< mu0 for solar zenith angle
            !< geopotential above ground
            !< surface albedo for vis range and dir light
            !< surface albedo for NIR range and dir light
            !< surface albedo for vis range and dif light
            !< surface albedo for NIR range and dif light
            !< full level pressure in Pa
            !< half level pressure in Pa
            !< surface pressure in Pa
            !< full level temperature in K
            !< half level temperature in K
            !< surface temperature in K
            !< specific humidity in g/g
            !< specific liquid water content
            !< specific ice content in g/g
            !< cloud nuclei concentration
            !< fractional cloud cover
            !< total cloud cover in m2/m2
            !< o3  mass mixing ratio
            !< co2 mass mixing ratio
            !< ch4 mass mixing ratio
            !< n2o mass mixing ratio
            !< cfc volume mixing ratio
            !< o2  mass mixing ratio
            !< tracer mass mixing ratios
            !<   upward LW flux profile, all sky
            !<   upward LW flux profile, clear sky
            !< downward LW flux profile, all sky
            !< downward LW flux profile, clear sky
            !<   upward SW flux profile, all sky
            !<   upward SW flux profile, clear sky
            !< downward SW flux profile, all sky
            !< downward SW flux profile, clear sky
            !< Visible (250-680) fraction of net surface radiation
            !< Downward Photosynthetically Active Radiation (PAR) at surface
            !< Diffuse fraction of downward surface near-infrared radiation
            !< Diffuse fraction of downward surface visible radiation
            !< Diffuse fraction of downward surface PAR
            ! -------------------------------------------------------------------------------------
            !< loop indicies
            !< index for clear or cloudy
            REAL(KIND=wp) :: zsemiss     (kbdim,nbndlw)
            REAL(KIND=wp) :: pm_sfc      (kbdim)
            !< LW surface emissivity by band
            !< pressure thickness in Pa
            !< surface pressure in mb
            !< pressure thickness
            !< scratch array
            !
            ! --- vertically reversed _vr variables
            !
            REAL(KIND=wp) :: cld_frc_vr(kbdim,klev)
            REAL(KIND=wp) :: aer_tau_lw_vr(kbdim,klev,nbndlw)
            REAL(KIND=wp) :: pm_fl_vr  (kbdim,klev)
            REAL(KIND=wp) :: tk_fl_vr  (kbdim,klev)
            REAL(KIND=wp) :: tk_hl_vr  (kbdim,klev+1)
            REAL(KIND=wp) :: cld_tau_lw_vr(kbdim,klev,nbndlw)
            REAL(KIND=wp) :: wkl_vr       (kbdim,maxinpx,klev)
            REAL(KIND=wp) :: wx_vr        (kbdim,maxxsec,klev)
            REAL(KIND=wp) :: col_dry_vr(kbdim,klev)
            !< number of molecules/cm2 of
            !< full level pressure [mb]
            !< half level pressure [mb]
            !< full level temperature [K]
            !< half level temperature [K]
            !< cloud nuclei concentration
            !< secure cloud fraction
            !< specific ice water content
            !< ice water content per volume
            !< ice water path in g/m2
            !< specific liquid water content
            !< liquid water path in g/m2
            !< liquid water content per
            !< effective radius of liquid
            !< effective radius of ice
            !< number of molecules/cm2 of
            !< number of molecules/cm2 of
            !< LW optical thickness of clouds
            !< extincion
            !< asymmetry factor
            !< single scattering albedo
            !< LW optical thickness of aerosols
            !< aerosol optical thickness
            !< aerosol asymmetry factor
            !< aerosol single scattering albedo
            REAL(KIND=wp) :: flx_uplw_vr(kbdim,klev+1)
            REAL(KIND=wp), allocatable :: ref_flx_uplw_vr(:,:)
            REAL(KIND=wp) :: flx_uplw_clr_vr(kbdim,klev+1)
            REAL(KIND=wp), allocatable :: ref_flx_uplw_clr_vr(:,:)
            REAL(KIND=wp) :: flx_dnlw_clr_vr(kbdim,klev+1)
            REAL(KIND=wp), allocatable :: ref_flx_dnlw_clr_vr(:,:)
            REAL(KIND=wp) :: flx_dnlw_vr(kbdim,klev+1)
            REAL(KIND=wp), allocatable :: ref_flx_dnlw_vr(:,:)
            !< upward flux, total sky
            !< upward flux, clear sky
            !< downward flux, total sky
            !< downward flux, clear sky
            !
            ! Random seeds for sampling. Needs to get somewhere upstream
            !
            INTEGER :: rnseeds(kbdim,rng_seed_size)
            INTEGER, allocatable :: ref_rnseeds(:,:)
            !
            ! Number of g-points per time step. Determine here to allow automatic array allocation in
            !   lrtm, srtm subroutines.
            !
            INTEGER :: n_gpts_ts
            ! 1.0 Constituent properties
            !--------------------------------
            !IBM* ASSERT(NODEPS)
            !
            ! --- control for zero, infintesimal or negative cloud fractions
            !
            !
            ! --- main constituent reordering
            !
            !IBM* ASSERT(NODEPS)
            !IBM* ASSERT(NODEPS)
            !IBM* ASSERT(NODEPS)
            !
            ! --- CFCs are in volume mixing ratio
            !
            !IBM* ASSERT(NODEPS)
            !
            ! -- Convert to molecules/cm^2
            !
            !
            ! 2.0 Surface Properties
            ! --------------------------------
            !
            ! 3.0 Particulate Optical Properties
            ! --------------------------------
            !
            ! 3.5 Interface for submodels that provide aerosol and/or cloud radiative properties:
            ! -----------------------------------------------------------------------------------
            !
            ! 4.0 Radiative Transfer Routines
            ! --------------------------------
            !
            ! Seeds for random numbers come from least significant digits of pressure field
            !
            tolerance = 1.E-12
            CALL kgen_init_check(check_status, tolerance)
            READ(UNIT=kgen_unit) zsemiss
            READ(UNIT=kgen_unit) pm_sfc
            READ(UNIT=kgen_unit) cld_frc_vr
            READ(UNIT=kgen_unit) aer_tau_lw_vr
            READ(UNIT=kgen_unit) pm_fl_vr
            READ(UNIT=kgen_unit) tk_fl_vr
            READ(UNIT=kgen_unit) tk_hl_vr
            READ(UNIT=kgen_unit) cld_tau_lw_vr
            READ(UNIT=kgen_unit) wkl_vr
            READ(UNIT=kgen_unit) wx_vr
            READ(UNIT=kgen_unit) col_dry_vr
            READ(UNIT=kgen_unit) flx_uplw_vr
            READ(UNIT=kgen_unit) flx_uplw_clr_vr
            READ(UNIT=kgen_unit) flx_dnlw_clr_vr
            READ(UNIT=kgen_unit) flx_dnlw_vr
            READ(UNIT=kgen_unit) rnseeds
            READ(UNIT=kgen_unit) n_gpts_ts

            !call kgen_read_var(ref_flx_uplw_vr, kgen_unit)
            !call kgen_read_var(ref_flx_uplw_clr_vr, kgen_unit)
            !call kgen_read_var(ref_flx_dnlw_clr_vr, kgen_unit)
            !call kgen_read_var(ref_flx_dnlw_vr, kgen_unit)
            !call kgen_read_var(ref_rnseeds, kgen_unit)
            call read_var_real_wp_dim2(ref_flx_uplw_vr, kgen_unit)
            call read_var_real_wp_dim2(ref_flx_uplw_clr_vr, kgen_unit)
            call read_var_real_wp_dim2(ref_flx_dnlw_clr_vr, kgen_unit)
            call read_var_real_wp_dim2(ref_flx_dnlw_vr, kgen_unit)
            call read_var_integer_4_dim2(ref_rnseeds, kgen_unit)

            ! call to kernel
            CALL lrtm(kproma, kbdim, klev, pm_fl_vr, pm_sfc, tk_fl_vr, tk_hl_vr, tk_sfc, wkl_vr, wx_vr, col_dry_vr, zsemiss, cld_frc_vr, cld_tau_lw_vr, aer_tau_lw_vr, rnseeds, lw_strat, n_gpts_ts, flx_uplw_vr, flx_dnlw_vr, flx_uplw_clr_vr, flx_dnlw_clr_vr)
            ! kernel verification for output variables
            call verify_var_real_wp_dim2("flx_uplw_vr", check_status, flx_uplw_vr, ref_flx_uplw_vr)
            call verify_var_real_wp_dim2("flx_uplw_clr_vr", check_status, flx_uplw_clr_vr, ref_flx_uplw_clr_vr)
            call verify_var_real_wp_dim2("flx_dnlw_clr_vr", check_status, flx_dnlw_clr_vr, ref_flx_dnlw_clr_vr)
            call verify_var_real_wp_dim2("flx_dnlw_vr", check_status, flx_dnlw_vr, ref_flx_dnlw_vr)
            call verify_var_integer_4_dim2("rnseeds", check_status, rnseeds, ref_rnseeds)
            CALL kgen_print_check("lrtm", check_status)
            CALL system_clock(start_clock, rate_clock)
            DO kgen_intvar=1,100
                CALL lrtm(kproma, kbdim, klev, pm_fl_vr, pm_sfc, tk_fl_vr, tk_hl_vr, tk_sfc, wkl_vr, wx_vr, col_dry_vr, zsemiss, cld_frc_vr, cld_tau_lw_vr, aer_tau_lw_vr, rnseeds, lw_strat, n_gpts_ts, flx_uplw_vr, flx_dnlw_vr, flx_uplw_clr_vr, flx_dnlw_clr_vr)
            END DO
            CALL system_clock(stop_clock, rate_clock)
            WRITE(*,*)
            PRINT *, "Elapsed time (sec): ", (stop_clock - start_clock)/REAL(rate_clock*100)
            !
            ! Reset random seeds so SW doesn't depend on what's happened in LW but is also independent
            !
            !
            ! Potential pitfall - we're passing every argument but some may not be present
            !
            !
            ! 5.0 Post Processing
            ! --------------------------------
            !
            ! Lw fluxes are vertically revered but SW fluxes are not
            !
            !
            ! 6.0 Interface for submodel diagnosics after radiation calculation:
            ! ------------------------------------------------------------------
        CONTAINS

        ! read subroutines
        subroutine read_var_real_wp_dim2(var, kgen_unit)
            integer, intent(in) :: kgen_unit
            real(kind=wp), intent(out), dimension(:,:), allocatable :: var
            integer, dimension(2,2) :: kgen_bound
            logical is_save
            
            READ(UNIT = kgen_unit) is_save
            if ( is_save ) then
                READ(UNIT = kgen_unit) kgen_bound(1, 1)
                READ(UNIT = kgen_unit) kgen_bound(2, 1)
                READ(UNIT = kgen_unit) kgen_bound(1, 2)
                READ(UNIT = kgen_unit) kgen_bound(2, 2)
                ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
                READ(UNIT = kgen_unit) var
            end if
        end subroutine
        subroutine read_var_real_wp_dim1(var, kgen_unit)
            integer, intent(in) :: kgen_unit
            real(kind=wp), intent(out), dimension(:), allocatable :: var
            integer, dimension(2,1) :: kgen_bound
            logical is_save
            
            READ(UNIT = kgen_unit) is_save
            if ( is_save ) then
                READ(UNIT = kgen_unit) kgen_bound(1, 1)
                READ(UNIT = kgen_unit) kgen_bound(2, 1)
                ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
                READ(UNIT = kgen_unit) var
            end if
        end subroutine
        subroutine read_var_real_wp_dim3(var, kgen_unit)
            integer, intent(in) :: kgen_unit
            real(kind=wp), intent(out), dimension(:,:,:), allocatable :: var
            integer, dimension(2,3) :: kgen_bound
            logical is_save
            
            READ(UNIT = kgen_unit) is_save
            if ( is_save ) then
                READ(UNIT = kgen_unit) kgen_bound(1, 1)
                READ(UNIT = kgen_unit) kgen_bound(2, 1)
                READ(UNIT = kgen_unit) kgen_bound(1, 2)
                READ(UNIT = kgen_unit) kgen_bound(2, 2)
                READ(UNIT = kgen_unit) kgen_bound(1, 3)
                READ(UNIT = kgen_unit) kgen_bound(2, 3)
                ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1))
                READ(UNIT = kgen_unit) var
            end if
        end subroutine
        subroutine read_var_integer_4_dim2(var, kgen_unit)
            integer, intent(in) :: kgen_unit
            integer(kind=4), intent(out), dimension(:,:), allocatable :: var
            integer, dimension(2,2) :: kgen_bound
            logical is_save
            
            READ(UNIT = kgen_unit) is_save
            if ( is_save ) then
                READ(UNIT = kgen_unit) kgen_bound(1, 1)
                READ(UNIT = kgen_unit) kgen_bound(2, 1)
                READ(UNIT = kgen_unit) kgen_bound(1, 2)
                READ(UNIT = kgen_unit) kgen_bound(2, 2)
                ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
                READ(UNIT = kgen_unit) var
            end if
        end subroutine

        subroutine verify_var_logical(varname, check_status, var, ref_var)
            character(*), intent(in) :: varname
            type(check_t), intent(inout) :: check_status
            logical, intent(in) :: var, ref_var
        
            check_status%numTotal = check_status%numTotal + 1
            IF ( var .eqv. ref_var ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
                if(check_status%verboseLevel > 1) then
                    WRITE(*,*)
                    WRITE(*,*) trim(adjustl(varname)), " is IDENTICAL( ", var, " )."
                endif
            ELSE
                if(check_status%verboseLevel > 1) then
                    WRITE(*,*)
                    WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                    if(check_status%verboseLevel > 2) then
                        WRITE(*,*) "KERNEL: ", var
                        WRITE(*,*) "REF.  : ", ref_var
                    endif
                endif
                check_status%numFatal = check_status%numFatal + 1
            END IF
        end subroutine
        
        subroutine verify_var_integer(varname, check_status, var, ref_var)
            character(*), intent(in) :: varname
            type(check_t), intent(inout) :: check_status
            integer, intent(in) :: var, ref_var
        
            check_status%numTotal = check_status%numTotal + 1
            IF ( var == ref_var ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
                if(check_status%verboseLevel > 1) then
                    WRITE(*,*)
                    WRITE(*,*) trim(adjustl(varname)), " is IDENTICAL( ", var, " )."
                endif
            ELSE
                if(check_status%verboseLevel > 0) then
                    WRITE(*,*)
                    WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                    if(check_status%verboseLevel > 2) then
                        WRITE(*,*) "KERNEL: ", var
                        WRITE(*,*) "REF.  : ", ref_var
                    endif
                endif
                check_status%numFatal = check_status%numFatal + 1
            END IF
        end subroutine
        
        subroutine verify_var_real(varname, check_status, var, ref_var)
            character(*), intent(in) :: varname
            type(check_t), intent(inout) :: check_status
            real, intent(in) :: var, ref_var
        
            check_status%numTotal = check_status%numTotal + 1
            IF ( var == ref_var ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
                if(check_status%verboseLevel > 1) then
                    WRITE(*,*)
                    WRITE(*,*) trim(adjustl(varname)), " is IDENTICAL( ", var, " )."
                endif
            ELSE
                if(check_status%verboseLevel > 0) then
                    WRITE(*,*)
                    WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                    if(check_status%verboseLevel > 2) then
                        WRITE(*,*) "KERNEL: ", var
                        WRITE(*,*) "REF.  : ", ref_var
                    endif
                endif
                check_status%numFatal = check_status%numFatal + 1
            END IF
        end subroutine
        
        subroutine verify_var_character(varname, check_status, var, ref_var)
            character(*), intent(in) :: varname
            type(check_t), intent(inout) :: check_status
            character(*), intent(in) :: var, ref_var
        
            check_status%numTotal = check_status%numTotal + 1
            IF ( var == ref_var ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
                if(check_status%verboseLevel > 1) then
                    WRITE(*,*)
                    WRITE(*,*) trim(adjustl(varname)), " is IDENTICAL( ", var, " )."
                endif
            ELSE
                if(check_status%verboseLevel > 0) then
                    WRITE(*,*)
                    WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                    if(check_status%verboseLevel > 2) then
                        WRITE(*,*) "KERNEL: ", var
                        WRITE(*,*) "REF.  : ", ref_var
                    end if
                end if
                check_status%numFatal = check_status%numFatal + 1
            END IF
        end subroutine

        subroutine verify_var_real_wp_dim2(varname, check_status, var, ref_var)
            character(*), intent(in) :: varname
            type(check_t), intent(inout) :: check_status
            real(kind=wp), intent(in), dimension(:,:) :: var
            real(kind=wp), intent(in), allocatable, dimension(:,:) :: ref_var
            real(kind=wp) :: nrmsdiff, rmsdiff
            real(kind=wp), allocatable :: temp(:,:), temp2(:,:)
            integer :: n
        
        
            IF ( ALLOCATED(ref_var) ) THEN
                check_status%numTotal = check_status%numTotal + 1
                allocate(temp(SIZE(var,dim=1),SIZE(var,dim=2)))
                allocate(temp2(SIZE(var,dim=1),SIZE(var,dim=2)))
                IF ( ALL( var == ref_var ) ) THEN
                    check_status%numIdentical = check_status%numIdentical + 1            
                    if(check_status%verboseLevel > 1) then
                        WRITE(*,*)
                        WRITE(*,*) "All elements of ", trim(adjustl(varname)), " are IDENTICAL."
                        !WRITE(*,*) "KERNEL: ", var
                        !WRITE(*,*) "REF.  : ", ref_var
                        IF ( ALL( var == 0 ) ) THEN
                            if(check_status%verboseLevel > 2) then
                                WRITE(*,*) "All values are zero."
                            end if
                        END IF
                    end if
                ELSE
                    n = count(var/=ref_var)
                    where(ref_var .NE. 0)
                        temp  = ((var-ref_var)/ref_var)**2
                        temp2 = (var-ref_var)**2
                    elsewhere
                        temp  = (var-ref_var)**2
                        temp2 = temp
                    endwhere
                    nrmsdiff = sqrt(sum(temp)/real(n))
                    rmsdiff = sqrt(sum(temp2)/real(n))
        
                    if(check_status%verboseLevel > 0) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                        WRITE(*,*) count( var /= ref_var), " of ", size( var ), " elements are different."
                        if(check_status%verboseLevel > 1) then
                            WRITE(*,*) "Average - kernel ", sum(var)/real(size(var))
                            WRITE(*,*) "Average - reference ", sum(ref_var)/real(size(ref_var))
                        endif
                        WRITE(*,*) "RMS of difference is ",rmsdiff
                        WRITE(*,*) "Normalized RMS of difference is ",nrmsdiff
                    end if
        
                    if (nrmsdiff > check_status%tolerance) then
                        check_status%numFatal = check_status%numFatal+1
                    else
                        check_status%numWarning = check_status%numWarning+1
                    endif
                END IF
                deallocate(temp,temp2)
            END IF
        end subroutine

        subroutine verify_var_real_wp_dim1(varname, check_status, var, ref_var)
            character(*), intent(in) :: varname
            type(check_t), intent(inout) :: check_status
            real(kind=wp), intent(in), dimension(:) :: var
            real(kind=wp), intent(in), allocatable, dimension(:) :: ref_var
            real(kind=wp) :: nrmsdiff, rmsdiff
            real(kind=wp), allocatable :: temp(:), temp2(:)
            integer :: n
        
        
            IF ( ALLOCATED(ref_var) ) THEN
                check_status%numTotal = check_status%numTotal + 1
                allocate(temp(SIZE(var,dim=1)))
                allocate(temp2(SIZE(var,dim=1)))
                IF ( ALL( var == ref_var ) ) THEN
                    check_status%numIdentical = check_status%numIdentical + 1            
                    if(check_status%verboseLevel > 1) then
                        WRITE(*,*)
                        WRITE(*,*) "All elements of ", trim(adjustl(varname)), " are IDENTICAL."
                        !WRITE(*,*) "KERNEL: ", var
                        !WRITE(*,*) "REF.  : ", ref_var
                        IF ( ALL( var == 0 ) ) THEN
                            if(check_status%verboseLevel > 2) then
                                WRITE(*,*) "All values are zero."
                            end if
                        END IF
                    end if
                ELSE
                    n = count(var/=ref_var)
                    where(ref_var .NE. 0)
                        temp  = ((var-ref_var)/ref_var)**2
                        temp2 = (var-ref_var)**2
                    elsewhere
                        temp  = (var-ref_var)**2
                        temp2 = temp
                    endwhere
                    nrmsdiff = sqrt(sum(temp)/real(n))
                    rmsdiff = sqrt(sum(temp2)/real(n))
        
                    if(check_status%verboseLevel > 0) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                        WRITE(*,*) count( var /= ref_var), " of ", size( var ), " elements are different."
                        if(check_status%verboseLevel > 1) then
                            WRITE(*,*) "Average - kernel ", sum(var)/real(size(var))
                            WRITE(*,*) "Average - reference ", sum(ref_var)/real(size(ref_var))
                        endif
                        WRITE(*,*) "RMS of difference is ",rmsdiff
                        WRITE(*,*) "Normalized RMS of difference is ",nrmsdiff
                    end if
        
                    if (nrmsdiff > check_status%tolerance) then
                        check_status%numFatal = check_status%numFatal+1
                    else
                        check_status%numWarning = check_status%numWarning+1
                    endif
                END IF
                deallocate(temp,temp2)
            END IF
        end subroutine

        subroutine verify_var_real_wp_dim3(varname, check_status, var, ref_var)
            character(*), intent(in) :: varname
            type(check_t), intent(inout) :: check_status
            real(kind=wp), intent(in), dimension(:,:,:) :: var
            real(kind=wp), intent(in), allocatable, dimension(:,:,:) :: ref_var
            real(kind=wp) :: nrmsdiff, rmsdiff
            real(kind=wp), allocatable :: temp(:,:,:), temp2(:,:,:)
            integer :: n
        
        
            IF ( ALLOCATED(ref_var) ) THEN
                check_status%numTotal = check_status%numTotal + 1
                allocate(temp(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3)))
                allocate(temp2(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3)))
                IF ( ALL( var == ref_var ) ) THEN
                    check_status%numIdentical = check_status%numIdentical + 1            
                    if(check_status%verboseLevel > 1) then
                        WRITE(*,*)
                        WRITE(*,*) "All elements of ", trim(adjustl(varname)), " are IDENTICAL."
                        !WRITE(*,*) "KERNEL: ", var
                        !WRITE(*,*) "REF.  : ", ref_var
                        IF ( ALL( var == 0 ) ) THEN
                            if(check_status%verboseLevel > 2) then
                                WRITE(*,*) "All values are zero."
                            end if
                        END IF
                    end if
                ELSE
                    n = count(var/=ref_var)
                    where(ref_var .NE. 0)
                        temp  = ((var-ref_var)/ref_var)**2
                        temp2 = (var-ref_var)**2
                    elsewhere
                        temp  = (var-ref_var)**2
                        temp2 = temp
                    endwhere
                    nrmsdiff = sqrt(sum(temp)/real(n))
                    rmsdiff = sqrt(sum(temp2)/real(n))
        
                    if(check_status%verboseLevel > 0) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                        WRITE(*,*) count( var /= ref_var), " of ", size( var ), " elements are different."
                        if(check_status%verboseLevel > 1) then
                            WRITE(*,*) "Average - kernel ", sum(var)/real(size(var))
                            WRITE(*,*) "Average - reference ", sum(ref_var)/real(size(ref_var))
                        endif
                        WRITE(*,*) "RMS of difference is ",rmsdiff
                        WRITE(*,*) "Normalized RMS of difference is ",nrmsdiff
                    end if
        
                    if (nrmsdiff > check_status%tolerance) then
                        check_status%numFatal = check_status%numFatal+1
                    else
                        check_status%numWarning = check_status%numWarning+1
                    endif
                END IF
                deallocate(temp,temp2)
            END IF
        end subroutine

        subroutine verify_var_integer_4_dim2(varname, check_status, var, ref_var)
            character(*), intent(in) :: varname
            type(check_t), intent(inout) :: check_status
            integer(kind=4), intent(in), dimension(:,:) :: var
            integer(kind=4), intent(in), allocatable, dimension(:,:) :: ref_var
            integer(kind=4) :: nrmsdiff, rmsdiff
            integer(kind=4), allocatable :: temp(:,:), temp2(:,:)
            integer :: n
        
        
            IF ( ALLOCATED(ref_var) ) THEN
                check_status%numTotal = check_status%numTotal + 1
                allocate(temp(SIZE(var,dim=1),SIZE(var,dim=2)))
                allocate(temp2(SIZE(var,dim=1),SIZE(var,dim=2)))
                IF ( ALL( var == ref_var ) ) THEN
                    check_status%numIdentical = check_status%numIdentical + 1            
                    if(check_status%verboseLevel > 1) then
                        WRITE(*,*)
                        WRITE(*,*) "All elements of ", trim(adjustl(varname)), " are IDENTICAL."
                        !WRITE(*,*) "KERNEL: ", var
                        !WRITE(*,*) "REF.  : ", ref_var
                        IF ( ALL( var == 0 ) ) THEN
                            if(check_status%verboseLevel > 2) then
                                WRITE(*,*) "All values are zero."
                            end if
                        END IF
                    end if
                ELSE
                    n = count(var/=ref_var)
                    where(ref_var .NE. 0)
                        temp  = ((var-ref_var)/ref_var)**2
                        temp2 = (var-ref_var)**2
                    elsewhere
                        temp  = (var-ref_var)**2
                        temp2 = temp
                    endwhere
                    nrmsdiff = sqrt(sum(temp)/real(n))
                    rmsdiff = sqrt(sum(temp2)/real(n))
        
                    if(check_status%verboseLevel > 0) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                        WRITE(*,*) count( var /= ref_var), " of ", size( var ), " elements are different."
                        if(check_status%verboseLevel > 1) then
                            WRITE(*,*) "Average - kernel ", sum(var)/real(size(var))
                            WRITE(*,*) "Average - reference ", sum(ref_var)/real(size(ref_var))
                        endif
                        WRITE(*,*) "RMS of difference is ",rmsdiff
                        WRITE(*,*) "Normalized RMS of difference is ",nrmsdiff
                    end if
        
                    if (nrmsdiff > check_status%tolerance) then
                        check_status%numFatal = check_status%numFatal+1
                    else
                        check_status%numWarning = check_status%numWarning+1
                    endif
                END IF
                deallocate(temp,temp2)
            END IF
        end subroutine

        END SUBROUTINE psrad_interface
    END MODULE mo_psrad_interface
