
! KGEN-generated Fortran source file
!
! Filename    : radiation.F90
! Generated at: 2015-07-07 00:48:23
! KGEN version: 0.4.13



    MODULE radiation
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
    USE physics_types, ONLY : kgen_read_mod42 => kgen_read
    USE physics_types, ONLY : kgen_verify_mod42 => kgen_verify
    USE camsrfexch, ONLY : kgen_read_mod43 => kgen_read
    USE camsrfexch, ONLY : kgen_verify_mod43 => kgen_verify
    USE rrtmg_state, ONLY : kgen_read_mod6 => kgen_read
    USE rrtmg_state, ONLY : kgen_verify_mod6 => kgen_verify
        !---------------------------------------------------------------------------------
        ! Purpose:
        !
        ! CAM interface to RRTMG
        !
        ! Revision history:
        ! May  2004, D. B. Coleman,  Initial version of interface module.
        ! July 2004, B. Eaton,       Use interfaces from new shortwave, longwave, and ozone modules.
        ! Feb  2005, B. Eaton,       Add namelist variables and control of when calcs are done.
        ! May  2008, Mike Iacono     Initial version for RRTMG
        ! Nov  2010, J. Kay          Add COSP simulator calls
        !---------------------------------------------------------------------------------
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        USE radconstants, ONLY: nswbands
        IMPLICIT NONE
        PRIVATE
        integer, parameter :: maxiter = 1
        character(len=80), parameter :: kname = "rad_rrtmg_sw"
        PUBLIC radiation_tend
        ! registers radiation physics buffer fields
        ! set default values of namelist variables in runtime_opts
        ! set namelist values from runtime_opts
        ! print namelist values to log
        ! provide read access to private module data
        ! calendar day of next radiation calculation
        ! query which radiation calcs are done this timestep
        ! calls radini
        ! moved from radctl.F90
        ! counter for cosp
        !initial value for cosp counter
        ! Private module data
        ! Default values for namelist variables
        ! freq. of shortwave radiation calc in time steps (positive)
        ! or hours (negative).
        ! frequency of longwave rad. calc. in time steps (positive)
        ! or hours (negative).
        ! Specifies length of time in timesteps (positive)
        ! or hours (negative) SW/LW radiation will be
        ! run continuously from the start of an
        ! initial or restart run
        ! calculate fluxes (up and down) per band.
        ! diagnostic  brightness temperatures at the top of the
        ! atmosphere for 7 TOVS/HIRS channels (2,4,6,8,10,11,12) and 4 TOVS/MSU
        ! channels (1,2,3,4).
        ! frequency (timesteps) of brightness temperature calcs
        !===============================================================================
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables
        !===============================================================================

        !================================================================================================

        !================================================================================================

        !===============================================================================

        !================================================================================================

        !================================================================================================

        !================================================================================================

        !================================================================================================

        !===============================================================================

        SUBROUTINE radiation_tend(fsns, fsnt, fsds, state, cam_out, cam_in, kgen_unit)
                USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
            !-----------------------------------------------------------------------
            !
            ! Purpose:
            ! Driver for radiation computation.
            !
            ! Method:
            ! Radiation uses cgs units, so conversions must be done from
            ! model fields to radiation fields.
            !
            ! Revision history:
            ! May 2004    D.B. Coleman     Merge of code from radctl.F90 and parts of tphysbc.F90.
            ! 2004-08-09  B. Eaton         Add pointer variables for constituents.
            ! 2004-08-24  B. Eaton         Access O3 and GHG constituents from chem_get_cnst.
            ! 2004-08-30  B. Eaton         Replace chem_get_cnst by rad_constituent_get.
            ! 2007-11-05  M. Iacono        Install rrtmg_lw and sw as radiation model.
            ! 2007-12-27  M. Iacono        Modify to use CAM cloud optical properties with rrtmg.
            !-----------------------------------------------------------------------
            USE physics_types, ONLY: physics_state
            USE camsrfexch, ONLY: cam_out_t
            USE camsrfexch, ONLY: cam_in_t
            USE parrrsw, ONLY: nbndsw
            USE ppgrid, only : pcols
            USE ppgrid, only : pver
            USE ppgrid, only : pverp
            USE radsw, ONLY: rad_rrtmg_sw
            USE rrtmg_state, ONLY: num_rrtmg_levs
            USE rrtmg_state, ONLY: rrtmg_state_t
            ! Arguments
            integer, intent(in) :: kgen_unit
            INTEGER*8 :: kgen_intvar, start_clock, stop_clock, rate_clock
            TYPE(check_t):: check_status
            REAL(KIND=kgen_dp) :: tolerance
            ! land fraction
            ! land fraction ramp
            ! land fraction
            ! Snow depth (liquid water equivalent)
            REAL(KIND=r8), intent(inout) :: fsns(pcols)
            REAL(KIND=r8) :: ref_fsns(pcols) ! Surface solar absorbed flux
            REAL(KIND=r8), intent(inout) :: fsnt(pcols)
            REAL(KIND=r8) :: ref_fsnt(pcols) ! Net column abs solar flux at model top
            ! Srf longwave cooling (up-down) flux
            ! Net outgoing lw flux at model top
            REAL(KIND=r8), intent(inout) :: fsds(pcols)
            REAL(KIND=r8) :: ref_fsds(pcols) ! Surface solar down flux
            TYPE(physics_state), intent(in), target :: state
            TYPE(cam_out_t), intent(inout) :: cam_out
            TYPE(cam_out_t) :: ref_cam_out
            TYPE(cam_in_t), intent(in) :: cam_in
            ! Local variables
            ! current timestep number
            ! Microwave brightness temperature
            ! Infrared brightness temperature
            ! surface temperature
            ! Model interface pressures (hPa)
            ! Land surface flag, sea=0, land=1
            ! Number of maximally overlapped regions
            ! Maximum values of pressure for each
            !    maximally overlapped region.
            !    0->pmxrgn(i,1) is range of pressure for
            !    1st region,pmxrgn(i,1)->pmxrgn(i,2) for
            !    2nd region, etc
            ! Cloud longwave emissivity
            ! Cloud longwave optical depth
            ! in-cloud cloud ice water path
            ! in-cloud cloud liquid water path
            ! Diagnostic total cloud cover
            !       "     low  cloud cover
            !       "     mid  cloud cover
            !       "     hgh  cloud cover
            ! Temporary workspace for outfld variables
            ! combined cloud radiative parameters are "in cloud" not "in cell"
            REAL(KIND=r8) :: c_cld_tau    (nbndsw,pcols,pver) ! cloud extinction optical depth
            REAL(KIND=r8) :: c_cld_tau_w  (nbndsw,pcols,pver) ! cloud single scattering albedo * tau
            REAL(KIND=r8) :: c_cld_tau_w_g(nbndsw,pcols,pver) ! cloud assymetry parameter * w * tau
            REAL(KIND=r8) :: c_cld_tau_w_f(nbndsw,pcols,pver) ! cloud forward scattered fraction * w * tau
            ! cloud absorption optics depth (LW)
            ! cloud radiative parameters are "in cloud" not "in cell"
            ! cloud extinction optical depth
            ! cloud single scattering albedo * tau
            ! cloud assymetry parameter * w * tau
            ! cloud forward scattered fraction * w * tau
            ! cloud absorption optics depth (LW)
            ! cloud radiative parameters are "in cloud" not "in cell"
            ! ice extinction optical depth
            ! ice single scattering albedo * tau
            ! ice assymetry parameter * tau * w
            ! ice forward scattered fraction * tau * w
            ! ice absorption optics depth (LW)
            ! cloud radiative parameters are "in cloud" not "in cell"
            ! snow extinction optical depth
            ! snow single scattering albedo * tau
            ! snow assymetry parameter * tau * w
            ! snow forward scattered fraction * tau * w
            ! snow absorption optics depth (LW)
            ! grid-box mean snow_tau for COSP only
            ! grid-box mean LW snow optical depth for COSP only
            ! cloud radiative parameters are "in cloud" not "in cell"
            ! liquid extinction optical depth
            ! liquid single scattering albedo * tau
            ! liquid assymetry parameter * tau * w
            ! liquid forward scattered fraction * tau * w
            ! liquid absorption optics depth (LW)
            ! tot gbx cloud visible sw optical depth for output on history files
            ! tot in-cloud visible sw optical depth for output on history files
            ! liq in-cloud visible sw optical depth for output on history files
            ! ice in-cloud visible sw optical depth for output on history files
            ! snow in-cloud visible sw optical depth for output on history files
            ! cloud fraction
            ! cloud fraction of just "snow clouds- whatever they are"
            REAL(KIND=r8) :: cldfprime(pcols,pver) ! combined cloud fraction (snow plus regular)
            REAL(KIND=r8), pointer, dimension(:,:) :: qrs
            REAL(KIND=r8), pointer :: ref_qrs(:,:) => NULL() ! shortwave radiative heating rate
            ! longwave  radiative heating rate
            REAL(KIND=r8) :: qrsc(pcols,pver)
            REAL(KIND=r8) :: ref_qrsc(pcols,pver) ! clearsky shortwave radiative heating rate
            ! clearsky longwave  radiative heating rate
            INTEGER :: ncol
            INTEGER :: lchnk
            ! current calendar day
            ! current latitudes(radians)
            ! current longitudes(radians)
            REAL(KIND=r8) :: coszrs(pcols) ! Cosine solar zenith angle
            ! flag to carry (QRS,QRL)*dp across time steps
            ! Local variables from radctl
            ! index
            REAL(KIND=r8) :: solin(pcols)
            REAL(KIND=r8) :: ref_solin(pcols) ! Solar incident flux
            REAL(KIND=r8) :: fsntoa(pcols)
            REAL(KIND=r8) :: ref_fsntoa(pcols) ! Net solar flux at TOA
            REAL(KIND=r8) :: fsutoa(pcols)
            REAL(KIND=r8) :: ref_fsutoa(pcols) ! Upwelling solar flux at TOA
            REAL(KIND=r8) :: fsntoac(pcols)
            REAL(KIND=r8) :: ref_fsntoac(pcols) ! Clear sky net solar flux at TOA
            REAL(KIND=r8) :: fsnirt(pcols)
            REAL(KIND=r8) :: ref_fsnirt(pcols) ! Near-IR flux absorbed at toa
            REAL(KIND=r8) :: fsnrtc(pcols)
            REAL(KIND=r8) :: ref_fsnrtc(pcols) ! Clear sky near-IR flux absorbed at toa
            REAL(KIND=r8) :: fsnirtsq(pcols)
            REAL(KIND=r8) :: ref_fsnirtsq(pcols) ! Near-IR flux absorbed at toa >= 0.7 microns
            REAL(KIND=r8) :: fsntc(pcols)
            REAL(KIND=r8) :: ref_fsntc(pcols) ! Clear sky total column abs solar flux
            REAL(KIND=r8) :: fsnsc(pcols)
            REAL(KIND=r8) :: ref_fsnsc(pcols) ! Clear sky surface abs solar flux
            REAL(KIND=r8) :: fsdsc(pcols)
            REAL(KIND=r8) :: ref_fsdsc(pcols) ! Clear sky surface downwelling solar flux
            ! Upward flux at top of model
            ! longwave cloud forcing
            ! shortwave cloud forcing
            ! Upward Clear Sky flux at top of model
            ! Clear sky lw flux at model top
            ! Clear sky lw flux at srf (up-down)
            ! Clear sky lw flux at srf (down)
            ! net longwave flux interpolated to 200 mb
            ! net clearsky longwave flux interpolated to 200 mb
            REAL(KIND=r8) :: fns(pcols,pverp)
            REAL(KIND=r8) :: ref_fns(pcols,pverp) ! net shortwave flux
            REAL(KIND=r8) :: fcns(pcols,pverp)
            REAL(KIND=r8) :: ref_fcns(pcols,pverp) ! net clear-sky shortwave flux
            ! fns interpolated to 200 mb
            ! fcns interpolated to 200 mb
            ! net longwave flux
            ! net clear-sky longwave flux
            ! Model mid-level pressures (dynes/cm2)
            ! Model interface pressures (dynes/cm2)
            REAL(KIND=r8) :: eccf ! Earth/sun distance factor
            ! Upward longwave flux in cgs units
            ! Temporary layer pressure thickness
            ! Model interface temperature
            REAL(KIND=r8) :: sfac(1:nswbands) ! time varying scaling factors due to Solar Spectral Irrad at 1 A.U. per band
            ! Ozone mass mixing ratio
            ! co2   mass mixing ratio
            ! co2 column mean mmr
            ! specific humidity
            REAL(KIND=r8), pointer, dimension(:,:,:) :: su => null()
            REAL(KIND=r8), pointer :: ref_su(:,:,:) => NULL() ! shortwave spectral flux up
            REAL(KIND=r8), pointer, dimension(:,:,:) :: sd => null()
            REAL(KIND=r8), pointer :: ref_sd(:,:,:) => NULL() ! shortwave spectral flux down
            ! longwave  spectral flux up
            ! longwave  spectral flux down
            ! Aerosol radiative properties
            REAL(KIND=r8) :: aer_tau    (pcols,0:pver,nbndsw) ! aerosol extinction optical depth
            REAL(KIND=r8) :: aer_tau_w  (pcols,0:pver,nbndsw) ! aerosol single scattering albedo * tau
            REAL(KIND=r8) :: aer_tau_w_g(pcols,0:pver,nbndsw) ! aerosol assymetry parameter * w * tau
            REAL(KIND=r8) :: aer_tau_w_f(pcols,0:pver,nbndsw) ! aerosol forward scattered fraction * w * tau
            ! aerosol absorption optics depth (LW)
            ! Gathered indicies of day and night columns
            !  chunk_column_index = IdxDay(daylight_column_index)
            INTEGER :: nday ! Number of daylight columns
            INTEGER :: nnite ! Number of night columns
            INTEGER, dimension(pcols) :: idxday ! Indicies of daylight coumns
            INTEGER, dimension(pcols) :: idxnite ! Indicies of night coumns
            ! index through climate/diagnostic radiation calls
            TYPE(rrtmg_state_t), pointer :: r_state ! contains the atm concentratiosn in layers needed for RRTMG
            !----------------------------------------------------------------------
            !  For CRM, make cloud equal to input observations:
            !
            ! Cosine solar zenith angle for current time step
            !
            ! Gather night/day column indices.
            ! do shortwave heating calc this timestep?
            ! do longwave heating calc this timestep?
                            tolerance = 8.E-13
                            CALL kgen_init_check(check_status, tolerance)
                            READ(UNIT=kgen_unit) c_cld_tau
                            READ(UNIT=kgen_unit) c_cld_tau_w
                            READ(UNIT=kgen_unit) c_cld_tau_w_g
                            READ(UNIT=kgen_unit) c_cld_tau_w_f
                            READ(UNIT=kgen_unit) cldfprime
                            CALL kgen_read_real_r8_dim2_ptr(qrs, kgen_unit)
                            READ(UNIT=kgen_unit) qrsc
                            READ(UNIT=kgen_unit) ncol
                            READ(UNIT=kgen_unit) lchnk
                            READ(UNIT=kgen_unit) coszrs
                            READ(UNIT=kgen_unit) solin
                            READ(UNIT=kgen_unit) fsntoa
                            READ(UNIT=kgen_unit) fsutoa
                            READ(UNIT=kgen_unit) fsntoac
                            READ(UNIT=kgen_unit) fsnirt
                            READ(UNIT=kgen_unit) fsnrtc
                            READ(UNIT=kgen_unit) fsnirtsq
                            READ(UNIT=kgen_unit) fsntc
                            READ(UNIT=kgen_unit) fsnsc
                            READ(UNIT=kgen_unit) fsdsc
                            READ(UNIT=kgen_unit) fns
                            READ(UNIT=kgen_unit) fcns
                            READ(UNIT=kgen_unit) eccf
                            READ(UNIT=kgen_unit) sfac
                            CALL kgen_read_real_r8_dim3_ptr(su, kgen_unit)
                            CALL kgen_read_real_r8_dim3_ptr(sd, kgen_unit)
                            READ(UNIT=kgen_unit) aer_tau
                            READ(UNIT=kgen_unit) aer_tau_w
                            READ(UNIT=kgen_unit) aer_tau_w_g
                            READ(UNIT=kgen_unit) aer_tau_w_f
                            READ(UNIT=kgen_unit) nday
                            READ(UNIT=kgen_unit) nnite
                            READ(UNIT=kgen_unit) idxday
                            READ(UNIT=kgen_unit) idxnite
                            CALL kgen_read_rrtmg_state_t_ptr(r_state, kgen_unit)

                            READ(UNIT=kgen_unit) ref_fsns
                            READ(UNIT=kgen_unit) ref_fsnt
                            READ(UNIT=kgen_unit) ref_fsds
                            CALL kgen_read_real_r8_dim2_ptr(ref_qrs, kgen_unit)
                            READ(UNIT=kgen_unit) ref_qrsc
                            READ(UNIT=kgen_unit) ref_solin
                            READ(UNIT=kgen_unit) ref_fsntoa
                            READ(UNIT=kgen_unit) ref_fsutoa
                            READ(UNIT=kgen_unit) ref_fsntoac
                            READ(UNIT=kgen_unit) ref_fsnirt
                            READ(UNIT=kgen_unit) ref_fsnrtc
                            READ(UNIT=kgen_unit) ref_fsnirtsq
                            READ(UNIT=kgen_unit) ref_fsntc
                            READ(UNIT=kgen_unit) ref_fsnsc
                            READ(UNIT=kgen_unit) ref_fsdsc
                            READ(UNIT=kgen_unit) ref_fns
                            READ(UNIT=kgen_unit) ref_fcns
                            CALL kgen_read_real_r8_dim3_ptr(ref_su, kgen_unit)
                            CALL kgen_read_real_r8_dim3_ptr(ref_sd, kgen_unit)
                            CALL kgen_read_mod43(ref_cam_out, kgen_unit)


                            ! call to kernel
                  call rad_rrtmg_sw( &
                       lchnk,        ncol,         num_rrtmg_levs, r_state,                    &
                       state%pmid,   cldfprime,                                                &
                       aer_tau,      aer_tau_w,    aer_tau_w_g,  aer_tau_w_f,                  &
                       eccf,         coszrs,       solin,        sfac,                         &
                       cam_in%asdir, cam_in%asdif, cam_in%aldir, cam_in%aldif,                 &
                       qrs,          qrsc,         fsnt,         fsntc,        fsntoa, fsutoa, &
                       fsntoac,      fsnirt,       fsnrtc,       fsnirtsq,     fsns,           &
                       fsnsc,        fsdsc,        fsds,         cam_out%sols, cam_out%soll,   &
                       cam_out%solsd,cam_out%solld,fns,          fcns,                         &
                       Nday,         Nnite,        IdxDay,       IdxNite,                      &
                       su,           sd,                                                       &
                       E_cld_tau=c_cld_tau, E_cld_tau_w=c_cld_tau_w, E_cld_tau_w_g=c_cld_tau_w_g, E_cld_tau_w_f=c_cld_tau_w_f, &
                       old_convert = .false.)
                            ! kernel verification for output variables
                            CALL kgen_verify_real_r8_dim1( "fsns", check_status, fsns, ref_fsns)
                            CALL kgen_verify_real_r8_dim1( "fsnt", check_status, fsnt, ref_fsnt)
                            CALL kgen_verify_real_r8_dim1( "fsds", check_status, fsds, ref_fsds)
                            CALL kgen_verify_mod43( "cam_out", check_status, cam_out, ref_cam_out)
                            CALL kgen_verify_real_r8_dim2_ptr( "qrs", check_status, qrs, ref_qrs)
                            CALL kgen_verify_real_r8_dim2( "qrsc", check_status, qrsc, ref_qrsc)
                            CALL kgen_verify_real_r8_dim1( "solin", check_status, solin, ref_solin)
                            CALL kgen_verify_real_r8_dim1( "fsntoa", check_status, fsntoa, ref_fsntoa)
                            CALL kgen_verify_real_r8_dim1( "fsutoa", check_status, fsutoa, ref_fsutoa)
                            CALL kgen_verify_real_r8_dim1( "fsntoac", check_status, fsntoac, ref_fsntoac)
                            CALL kgen_verify_real_r8_dim1( "fsnirt", check_status, fsnirt, ref_fsnirt)
                            CALL kgen_verify_real_r8_dim1( "fsnrtc", check_status, fsnrtc, ref_fsnrtc)
                            CALL kgen_verify_real_r8_dim1( "fsnirtsq", check_status, fsnirtsq, ref_fsnirtsq)
                            CALL kgen_verify_real_r8_dim1( "fsntc", check_status, fsntc, ref_fsntc)
                            CALL kgen_verify_real_r8_dim1( "fsnsc", check_status, fsnsc, ref_fsnsc)
                            CALL kgen_verify_real_r8_dim1( "fsdsc", check_status, fsdsc, ref_fsdsc)
                            CALL kgen_verify_real_r8_dim2( "fns", check_status, fns, ref_fns)
                            CALL kgen_verify_real_r8_dim2( "fcns", check_status, fcns, ref_fcns)
                            CALL kgen_verify_real_r8_dim3_ptr( "su", check_status, su, ref_su)
                            CALL kgen_verify_real_r8_dim3_ptr( "sd", check_status, sd, ref_sd)
                            CALL kgen_print_check("rad_rrtmg_sw", check_status)
                            CALL system_clock(start_clock, rate_clock)
                            print *,'ncol: ',ncol  
                            print *,'num_rrtmg_levs: ',num_rrtmg_levs
                            DO kgen_intvar=1,maxiter
                                CALL rad_rrtmg_sw(lchnk, ncol, num_rrtmg_levs, r_state, state % pmid, cldfprime, &
aer_tau, aer_tau_w, aer_tau_w_g, aer_tau_w_f, eccf, coszrs, solin, sfac, cam_in % asdir, cam_in % asdif, &
cam_in % aldir, cam_in % aldif, qrs, qrsc, fsnt, fsntc, fsntoa, fsutoa, fsntoac, fsnirt, fsnrtc, fsnirtsq, &
fsns, fsnsc, fsdsc, fsds, cam_out % sols, cam_out % soll, cam_out % solsd, cam_out % solld, fns, fcns, &
nday, nnite, idxday, idxnite, su, sd, e_cld_tau = c_cld_tau, e_cld_tau_w = c_cld_tau_w, &
e_cld_tau_w_g = c_cld_tau_w_g, e_cld_tau_w_f = c_cld_tau_w_f, old_convert = .FALSE.)
                            END DO
                            CALL system_clock(stop_clock, rate_clock)
                            WRITE(*,*)
                            PRINT *, TRIM(kname), ": Total time (sec): ", (stop_clock - start_clock)/REAL(rate_clock)
                            PRINT *, TRIM(kname), ": Elapsed time (usec): ", 1.0e6*(stop_clock - start_clock)/REAL(rate_clock*maxiter)
            !  if (dosw .or. dolw) then
            ! output rad inputs and resulting heating rates
            ! Compute net radiative heating tendency
            ! Compute heating rate for dtheta/dt
            ! convert radiative heating rates to Q*dp for energy conservation
        CONTAINS

        ! write subroutines
            SUBROUTINE kgen_read_real_r8_dim1(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=r8), INTENT(OUT), ALLOCATABLE, DIMENSION(:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1
                INTEGER, DIMENSION(2,1) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
                    READ(UNIT = kgen_unit) var
                    IF ( PRESENT(printvar) ) THEN
                        PRINT *, "** " // printvar // " **", var
                    END IF
                END IF
            END SUBROUTINE kgen_read_real_r8_dim1

            SUBROUTINE kgen_read_real_r8_dim2_ptr(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=r8), INTENT(OUT), POINTER, DIMENSION(:,:) :: var
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
            END SUBROUTINE kgen_read_real_r8_dim2_ptr

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

            SUBROUTINE kgen_read_real_r8_dim3_ptr(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                real(KIND=r8), INTENT(OUT), POINTER, DIMENSION(:,:,:) :: var
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
            END SUBROUTINE kgen_read_real_r8_dim3_ptr

            SUBROUTINE kgen_read_rrtmg_state_t_ptr(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                TYPE(rrtmg_state_t), INTENT(OUT), POINTER :: var
                LOGICAL :: is_true

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    ALLOCATE(var)
                    IF ( PRESENT(printvar) ) THEN
                        CALL kgen_read_mod6(var, kgen_unit, printvar=printvar//"%rrtmg_state")
                    ELSE
                        CALL kgen_read_mod6(var, kgen_unit)
                    END IF
                END IF
            END SUBROUTINE kgen_read_rrtmg_state_t_ptr


        ! verify subroutines
            SUBROUTINE kgen_verify_real_r8_dim1( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=r8), intent(in), DIMENSION(:) :: var, ref_var
                real(KIND=r8) :: nrmsdiff, rmsdiff
                real(KIND=r8), allocatable, DIMENSION(:) :: temp, temp2
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
                    allocate(temp(SIZE(var,dim=1)))
                    allocate(temp2(SIZE(var,dim=1)))
                
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
            END SUBROUTINE kgen_verify_real_r8_dim1

            SUBROUTINE kgen_verify_real_r8_dim2_ptr( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=r8), intent(in), DIMENSION(:,:), POINTER :: var, ref_var
                real(KIND=r8) :: nrmsdiff, rmsdiff
                real(KIND=r8), allocatable, DIMENSION(:,:) :: temp, temp2
                integer :: n
                IF ( ASSOCIATED(var) ) THEN
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
                END IF
            END SUBROUTINE kgen_verify_real_r8_dim2_ptr

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

            SUBROUTINE kgen_verify_real_r8_dim3_ptr( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=r8), intent(in), DIMENSION(:,:,:), POINTER :: var, ref_var
                real(KIND=r8) :: nrmsdiff, rmsdiff
                real(KIND=r8), allocatable, DIMENSION(:,:,:) :: temp, temp2
                integer :: n
                IF ( ASSOCIATED(var) ) THEN
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
                END IF
            END SUBROUTINE kgen_verify_real_r8_dim3_ptr

        END SUBROUTINE radiation_tend
        !===============================================================================

        !===============================================================================

        !===============================================================================
    END MODULE radiation
