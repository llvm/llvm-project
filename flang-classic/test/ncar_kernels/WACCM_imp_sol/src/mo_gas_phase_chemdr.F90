
! KGEN-generated Fortran source file
!
! Filename    : mo_gas_phase_chemdr.F90
! Generated at: 2015-05-13 11:02:21
! KGEN version: 0.4.10



    MODULE mo_gas_phase_chemdr
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        USE chem_mods, ONLY: gas_pcnst
        USE chem_mods, ONLY: rxntot
        USE chem_mods, ONLY: extcnt
        USE ppgrid, ONLY: pver
        USE ppgrid, ONLY: pcols
        IMPLICIT NONE
        PUBLIC gas_phase_chemdr
        PRIVATE
        ! index map to/from chemistry/constituents list
        !
        ! CCMI
        !
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables

        !-----------------------------------------------------------------------
        !-----------------------------------------------------------------------

        SUBROUTINE gas_phase_chemdr(lchnk, ncol, delt, kgen_unit)
                USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
            !-----------------------------------------------------------------------
            !     ... Chem_solver advances the volumetric mixing ratio
            !         forward one time step via a combination of explicit,
            !         ebi, hov, fully implicit, and/or rodas algorithms.
            !-----------------------------------------------------------------------
            USE chem_mods, ONLY: nfs
            USE chem_mods, ONLY: indexm
            USE mo_imp_sol, ONLY: imp_sol
            !
            ! LINOZ
            !
            !
            ! for aqueous chemistry and aerosol growth
            !
            IMPLICIT NONE
            !-----------------------------------------------------------------------
            !        ... Dummy arguments
            !-----------------------------------------------------------------------
            integer, intent(in) :: kgen_unit
            INTEGER*8 :: kgen_intvar, start_clock, stop_clock, rate_clock
            TYPE(check_t):: check_status
            REAL(KIND=kgen_dp) :: tolerance
            INTEGER, intent(in) :: lchnk ! chunk index
            INTEGER, intent(in) :: ncol ! number columns in chunk
            ! gas phase start index in q
            REAL(KIND=r8), intent(in) :: delt ! timestep (s)
            ! day of year
            ! surface pressure
            ! surface geopotential
            ! midpoint temperature (K)
            ! midpoint pressures (Pa)
            ! pressure delta about midpoints (Pa)
            ! zonal velocity (m/s)
            ! meridional velocity (m/s)
            ! cloud water (kg/kg)
            ! droplet number concentration (#/kg)
            ! midpoint geopotential height above the surface (m)
            ! interface geopotential height above the surface (m)
            ! interface pressures (Pa)
            ! species concentrations (kg/kg)
            ! longwave down at sfc
            ! sea-ice areal fraction
            ! ocean areal fraction
            ! albedo: shortwave, direct
            ! sfc temp (merged w/ocean if coupled)
            !
            !
            !
            ! species tendencies (kg/kg/s)
            ! constituent surface flux (kg/m^2/s)
            ! dry deposition flux (kg/m^2/s)
            !-----------------------------------------------------------------------
            !       ... Local variables
            !-----------------------------------------------------------------------
            ! chunk lat indicies
            ! chunk lon indicies
            REAL(KIND=r8) :: invariants(ncol,pver,nfs)
            ! column densities (molecules/cm^2)
            ! layer column densities (molecules/cm^2)
            REAL(KIND=r8) :: extfrc(ncol,pver,max(1,extcnt))
            REAL(KIND=r8) :: vmr(ncol,pver,gas_pcnst)
            REAL(KIND=r8) :: ref_vmr(ncol,pver,gas_pcnst) ! xported species (vmr)
            REAL(KIND=r8) :: reaction_rates(ncol,pver,max(1,rxntot)) ! reaction rates
            ! dry deposition velocity (cm/s)
            REAL(KIND=r8) :: het_rates(ncol,pver,max(1,gas_pcnst)) ! washout rate (1/s)
            ! water vapor volume mixing ratio
            ! mean wet atmospheric mass ( amu )
            ! midpoint geopotential in km
            ! midpoint geopotential in km realitive to surf
            ! trop sulfate aerosols
            ! pressure at midpoints ( hPa )
            ! cloud water mass mixing ratio (kg/kg)
            ! interface geopotential in km realitive to surf
            ! interface geopotential in km
            ! solar zenith angles
            ! surface height (m)
            ! chunk latitudes and longitudes (radians)
            ! solar zenith angles (degrees)
            ! radians to degrees conversion factor
            ! relative humidity
            ! wrk array for relative humidity
            ! wrk array for relative humidity
            INTEGER :: ltrop_sol(pcols) ! tropopause vertical index used in chem solvers
            ! stratospheric sad (1/cm)
            ! total trop. sad (cm^2/cm^3)
            ! surface wind speed (m/s)
            ! od diagnostics
            ! fraction of day
            ! o2 concentration (kg/kg)
            ! o concentration (kg/kg)
            ! chem working concentrations (kg/kg)
            ! chem working concentrations (kg/kg)
            ! hno3 gas phase concentration (mol/mol)
            ! hno3 condensed phase concentration (mol/mol)
            ! hcl gas phase concentration (mol/mol)
            ! hcl condensed phase concentration (mol/mol)
            ! h2o gas phase concentration (mol/mol)
            ! h2o condensed phase concentration (mol/mol)
            ! cloud water "ice" (kg/kg)
            ! radius of sulfate, nat, & ice ( cm )
            ! surf area density of sulfate, nat, & ice ( cm^2/cm^3 )
            ! chemistry species tendencies (kg/kg/s)
            ! specific humidity (kg/kg)
            ! for aerosol formation....
            !
            ! CCMI
            !
            REAL(KIND=r8), dimension(ncol,pver) :: o3s_loss
            REAL(KIND=r8) :: ref_o3s_loss(ncol,pver) ! tropospheric ozone loss for o3s
            !
            ! jfl
            !
            !
            ! aerosol reaction diagnostics
            ! initialize to NaN to hopefully catch user defined rxts that go unset
            !-----------------------------------------------------------------------
            !        ... Get chunck latitudes and longitudes
            !-----------------------------------------------------------------------
            ! convert to degrees
            !-----------------------------------------------------------------------
            !        ... Calculate cosine of zenith angle
            !            then cast back to angle (radians)
            !-----------------------------------------------------------------------
            !-----------------------------------------------------------------------
            !        ... Xform geopotential height from m to km
            !            and pressure from Pa to mb
            !-----------------------------------------------------------------------
            !-----------------------------------------------------------------------
            !        ... map incoming concentrations to working array
            !-----------------------------------------------------------------------
            !-----------------------------------------------------------------------
            !        ... Set atmosphere mean mass
            !-----------------------------------------------------------------------
            !-----------------------------------------------------------------------
            !        ... Xform from mmr to vmr
            !-----------------------------------------------------------------------
            !
            ! CCMI
            !
            ! reset STE tracer to specific vmr of 200 ppbv
            !
            !
            ! reset AOA_NH, NH_5, NH_50, NH_50W surface mixing ratios between 30N and 50N
            !
            !-----------------------------------------------------------------------
            !        ... force ion/electron balance
            !-----------------------------------------------------------------------
            !-----------------------------------------------------------------------
            !        ... Set the "invariants"
            !-----------------------------------------------------------------------
            !-----------------------------------------------------------------------
            !        ... stratosphere aerosol surface area
            !-----------------------------------------------------------------------
            !      NOTE: For gas-phase solver only.
            !            ratecon_sfstrat needs total hcl.
            !-----------------------------------------------------------------------
            !        ... Set the column densities at the upper boundary
            !-----------------------------------------------------------------------
            !-----------------------------------------------------------------------
            !       ...  Set rates for "tabular" and user specified reactions
            !-----------------------------------------------------------------------
            !-----------------------------------------------------------------
            ! ... zero out sulfate above tropopause
            !-----------------------------------------------------------------
            !-----------------------------------------------------------------
            !       ... compute the relative humidity
            !-----------------------------------------------------------------
            !-----------------------------------------------------------------------
            !        ... Compute the photolysis rates at time = t(n+1)
            !-----------------------------------------------------------------------
            !-----------------------------------------------------------------------
            !       ... Set the column densities
            !-----------------------------------------------------------------------
            !-----------------------------------------------------------------------
            !       ... Calculate the photodissociation rates
            !-----------------------------------------------------------------------
            !-----------------------------------------------------------------------
            !       ... Adjust the photodissociation rates
            !-----------------------------------------------------------------------
            !-----------------------------------------------------------------------
            !        ... Compute the extraneous frcing at time = t(n+1)
            !-----------------------------------------------------------------------
            !-----------------------------------------------------------------------
            !        ... Compute the extraneous frcing at time = t(n+1)
            !-----------------------------------------------------------------------
            !-----------------------------------------------------------------------
            !        ... Form the washout rates
            !-----------------------------------------------------------------------
            !
            ! CCMI
            !
            ! set loss to below the tropopause only
            !
            !
            ! save h2so4 before gas phase chem (for later new particle nucleation)
            ! mixing ratios before chemistry changes
            !=======================================================================
            !        ... Call the class solution algorithms
            !=======================================================================
            !-----------------------------------------------------------------------
            !       ... Solve for "Explicit" species
            !-----------------------------------------------------------------------
            !-----------------------------------------------------------------------
            !       ... Solve for "Implicit" species
            !-----------------------------------------------------------------------
            !
            tolerance = 1.E-14
            CALL kgen_init_check(check_status, tolerance)
            READ(UNIT=kgen_unit) invariants
            READ(UNIT=kgen_unit) extfrc
            READ(UNIT=kgen_unit) vmr
            READ(UNIT=kgen_unit) reaction_rates
            READ(UNIT=kgen_unit) het_rates
            READ(UNIT=kgen_unit) ltrop_sol
            READ(UNIT=kgen_unit) o3s_loss

            READ(UNIT=kgen_unit) ref_vmr
            READ(UNIT=kgen_unit) ref_o3s_loss

            !Uncomment following call(s) to generate perturbed input(s)
            !CALL kgen_perturb_real_r8_dim3( vmr )

            ! call to kernel
            CALL imp_sol(vmr, reaction_rates, het_rates, extfrc, delt, invariants(1,1,indexm), ncol, lchnk, ltrop_sol(:ncol), &
            o3s_loss=o3s_loss)
            ! kernel verification for output variables
            CALL kgen_verify_real_r8_dim3( "vmr", check_status, vmr, ref_vmr)
            CALL kgen_verify_real_r8_dim2( "o3s_loss", check_status, o3s_loss, ref_o3s_loss)
            CALL kgen_print_check("imp_sol", check_status)
            CALL system_clock(start_clock, rate_clock)
            DO kgen_intvar=1,10
                CALL imp_sol(vmr, reaction_rates, het_rates, extfrc, delt, invariants(1, 1, indexm), ncol, lchnk, ltrop_sol(: ncol), o3s_loss = o3s_loss)
            END DO
            CALL system_clock(stop_clock, rate_clock)
            WRITE(*,*)
            PRINT *, "Elapsed time (sec): ", (stop_clock - start_clock)/REAL(rate_clock*10)
            !
            ! jfl : CCMI : implement O3S here because mo_fstrat is not called
            !
            ! save h2so4 change by gas phase chem (for later new particle nucleation)
            !
            ! Aerosol processes ...
            !
            !
            ! LINOZ
            !
            !-----------------------------------------------------------------------
            !         ... Check for negative values and reset to zero
            !-----------------------------------------------------------------------
            !-----------------------------------------------------------------------
            !         ... Set upper boundary mmr values
            !-----------------------------------------------------------------------
            !-----------------------------------------------------------------------
            !         ... Set fixed lower boundary mmr values
            !-----------------------------------------------------------------------
            !-----------------------------------------------------------------------
            ! set NOy UBC
            !-----------------------------------------------------------------------
            !-----------------------------------------------------------------------
            !         ... Xform from vmr to mmr
            !-----------------------------------------------------------------------
            !-----------------------------------------------------------------------
            !         ... Form the tendencies
            !-----------------------------------------------------------------------
            !
            ! jfl
            !
            ! surface vmr
            !
            !
            !
            !
        CONTAINS

        ! write subroutines
            SUBROUTINE kgen_read_real_r8_dim3(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=r8), INTENT(OUT), ALLOCATABLE, DIMENSION(:,:,:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1,idx2,idx3
                INTEGER, DIMENSION(2,3) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    READ(UNIT = kgen_unit) kgen_bound(1, 2)
                    READ(UNIT = kgen_unit) kgen_bound(2, 2)
                    READ(UNIT = kgen_unit) kgen_bound(1, 3)
                    READ(UNIT = kgen_unit) kgen_bound(2, 3)
                    ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1))
                    READ(UNIT = kgen_unit) var
                    IF ( PRESENT(printvar) ) THEN
                        PRINT *, "** " // printvar // " **", var
                    END IF
                END IF
            END SUBROUTINE kgen_read_real_r8_dim3

            SUBROUTINE kgen_read_real_r8_dim2(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=r8), INTENT(OUT), ALLOCATABLE, DIMENSION(:,:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1,idx2
                INTEGER, DIMENSION(2,2) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    READ(UNIT = kgen_unit) kgen_bound(1, 2)
                    READ(UNIT = kgen_unit) kgen_bound(2, 2)
                    ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
                    READ(UNIT = kgen_unit) var
                    IF ( PRESENT(printvar) ) THEN
                        PRINT *, "** " // printvar // " **", var
                    END IF
                END IF
            END SUBROUTINE kgen_read_real_r8_dim2


        ! verify subroutines
            SUBROUTINE kgen_verify_real_r8_dim3( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=r8), intent(in), DIMENSION(:,:,:) :: var, ref_var
                real(KIND=r8) :: nrmsdiff, rmsdiff
                real(KIND=r8), allocatable, DIMENSION(:,:,:) :: temp, temp2
                integer :: n
                check_status%numTotal = check_status%numTotal + 1
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
                    allocate(temp(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3)))
                    allocate(temp2(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3)))
                
                    n = count(var/=ref_var)
                    where(abs(ref_var) > check_status%minvalue)
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
                
                    deallocate(temp,temp2)
                END IF
            END SUBROUTINE kgen_verify_real_r8_dim3

            SUBROUTINE kgen_verify_real_r8_dim2( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=r8), intent(in), DIMENSION(:,:) :: var, ref_var
                real(KIND=r8) :: nrmsdiff, rmsdiff
                real(KIND=r8), allocatable, DIMENSION(:,:) :: temp, temp2
                integer :: n
                check_status%numTotal = check_status%numTotal + 1
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
                    allocate(temp(SIZE(var,dim=1),SIZE(var,dim=2)))
                    allocate(temp2(SIZE(var,dim=1),SIZE(var,dim=2)))
                
                    n = count(var/=ref_var)
                    where(abs(ref_var) > check_status%minvalue)
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
                
                    deallocate(temp,temp2)
                END IF
            END SUBROUTINE kgen_verify_real_r8_dim2

        subroutine kgen_perturb_real_r8_dim3( var )
            real(kind=r8), intent(inout), dimension(:,:,:) :: var
            integer, allocatable :: rndm_seed(:)
            integer :: rndm_seed_sz
            real(kind=r8) :: pertval
            real(kind=r8) :: pertlim = 10e-15
            integer :: idx1,idx2,idx3
        
            call random_seed(size=rndm_seed_sz)
            allocate(rndm_seed(rndm_seed_sz))
            rndm_seed = 121869
            call random_seed(put=rndm_seed)
            do idx1=1,size(var, dim=1)
                do idx2=1,size(var, dim=2)
                    do idx3=1,size(var, dim=3)
                        call random_number(pertval)
                        pertval = 2.0_r8*pertlim*(0.5_r8 - pertval)
                        var(idx1,idx2,idx3) = var(idx1,idx2,idx3)*(1.0_r8 + pertval)
                    end do
                end do
            end do
            deallocate(rndm_seed)
        end subroutine
        END SUBROUTINE gas_phase_chemdr
    END MODULE mo_gas_phase_chemdr
