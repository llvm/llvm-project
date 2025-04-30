
! KGEN-generated Fortran source file
!
! Filename    : micro_mg_cam.F90
! Generated at: 2015-03-31 09:44:40
! KGEN version: 0.4.5

#ifdef __aarch64__
#define  _TOL 1.E-12
#else
#define  _TOL 1.E-14
#endif

    MODULE micro_mg_cam
        !---------------------------------------------------------------------------------
        !
        !  1 Interfaces for MG microphysics
        !
        !---------------------------------------------------------------------------------
        !
        ! How to add new packed MG inputs to micro_mg_cam_tend:
        !
        ! If you have an input with first dimension [psetcols, pver], the procedure
        ! for adding inputs is as follows:
        !
        ! 1) In addition to any variables you need to declare for the "unpacked"
        !    (1 format) version, you must declare an allocatable or pointer array
        !    for the "packed" (MG format) version.
        !
        ! 2) After micro_mg_get_cols is called, allocate the "packed" array with
        !    size [mgncol, nlev].
        !
        ! 3) Add a call similar to the following line (look before the
        !    micro_mg_tend calls to see similar lines):
        !
        !      packed_array = packer%pack(original_array)
        !
        !    The packed array can then be passed into any of the MG schemes.
        !
        ! This same procedure will also work for 1D arrays of size psetcols, 3-D
        ! arrays with psetcols and pver as the first dimensions, and for arrays of
        ! dimension [psetcols, pverp]. You only have to modify the allocation of
        ! the packed array before the "pack" call.
        !
        !---------------------------------------------------------------------------------
        !
        ! How to add new packed MG outputs to micro_mg_cam_tend:
        !
        ! 1) As with inputs, in addition to the unpacked outputs you must declare
        !    an allocatable or pointer array for packed data. The unpacked and
        !    packed arrays must *also* be targets or pointers (but cannot be both).
        !
        ! 2) Again as for inputs, allocate the packed array using mgncol and nlev,
        !    which are set in micro_mg_get_cols.
        !
        ! 3) Add the field to post-processing as in the following line (again,
        !    there are many examples before the micro_mg_tend calls):
        !
        !      call post_proc%add_field(p(final_array),p(packed_array))
        !
        !    This registers the field for post-MG averaging, and to scatter to the
        !    final, unpacked version of the array.
        !
        !    By default, any columns/levels that are not operated on by MG will be
        !    set to 0 on output; this value can be adjusted using the "fillvalue"
        !    optional argument to post_proc%add_field.
        !
        !    Also by default, outputs from multiple substeps will be averaged after
        !    MG's substepping is complete. Passing the optional argument
        !    "accum_method=accum_null" will change this behavior so that the last
        !    substep is always output.
        !
        ! This procedure works on 1-D and 2-D outputs. Note that the final,
        ! unpacked arrays are not set until the call to
        ! "post_proc%process_and_unpack", which sets every single field that was
        ! added with post_proc%add_field.
        !
        !---------------------------------------------------------------------------------
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        IMPLICIT NONE
        PRIVATE
        PUBLIC kgen_read_externs_micro_mg_cam
        INTEGER, PARAMETER :: kgen_dp = selected_real_kind(15, 307)
        PUBLIC micro_mg_cam_tend
        type, public  ::  check_t
            logical :: Passed
            integer :: numFatal
            integer :: numTotal
            integer :: numIdentical
            integer :: numWarning
            integer :: VerboseLevel
            real(kind=kgen_dp) :: tolerance
            real(kind=kgen_dp) :: minvalue
        end type check_t
        ! Version number for MG.
        ! Second part of version number.
        ! type of precipitation fraction method
        ! berg efficiency factor
        ! Prognose cldliq flag
        ! Prognose cldice flag
        INTEGER :: num_steps ! Number of MG substeps
        ! Number of constituents
        ! Constituent names
        ! cloud liquid amount index
        ! cloud ice amount index
        ! cloud liquid number index
        ! cloud ice water index
        ! rain index
        ! snow index
        ! rain number index
        ! snow number index
        ! Physics buffer indices for fields registered by this module
        ! Fields for UNICON
        ! Evaporation area of stratiform precipitation
        ! Evaporation rate of stratiform rain [kg/kg/s]. >= 0.
        ! Evaporation rate of stratiform snow [kg/kg/s]. >= 0.
        ! Fields needed as inputs to COSP
        ! Fields needed by Park macrophysics
        ! Used to replace aspects of MG microphysics
        ! (e.g. by CARMA)
        ! Index fields for precipitation efficiency.
        ! Physics buffer indices for fields registered by other modules
        ! Pbuf fields needed for subcol_SILHS
        ! pbuf fields for heterogeneous freezing

        !===============================================================================
        CONTAINS

        ! write subroutines
        ! No subroutines

        ! module extern variables

        SUBROUTINE kgen_read_externs_micro_mg_cam(kgen_unit)
            INTEGER, INTENT(IN) :: kgen_unit
            READ(UNIT=kgen_unit) num_steps
        END SUBROUTINE kgen_read_externs_micro_mg_cam

        subroutine kgen_init_check(check, tolerance, minvalue)
          type(check_t), intent(inout) :: check
          real(kind=kgen_dp), intent(in), optional :: tolerance
          real(kind=kgen_dp), intent(in), optional :: minvalue
        
          check%Passed   = .TRUE.
          check%numFatal = 0
          check%numWarning = 0
          check%numTotal = 0
          check%numIdentical = 0
          check%VerboseLevel = 1
          if(present(tolerance)) then
             check%tolerance = tolerance
          else
              check%tolerance = _TOL
          endif
          if(present(minvalue)) then
             check%minvalue = minvalue
          else
              check%minvalue = 1.0D-15
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
        !===============================================================================


        !================================================================================================

        !===============================================================================

        !===============================================================================

        !===============================================================================

        !===============================================================================

        SUBROUTINE micro_mg_cam_tend(dtime, kgen_unit)
            USE micro_mg2_0, ONLY: micro_mg_tend2_0 => micro_mg_tend
            integer, intent(in) :: kgen_unit
            INTEGER*8 :: kgen_intvar, start_clock, stop_clock, rate_clock
            TYPE(check_t):: check_status
            REAL(KIND=kgen_dp) :: tolerance
            REAL(KIND=r8), intent(in) :: dtime
            ! Local variables
            ! ice nucleation number
            ! ice nucleation number (homogeneous)
            ! liquid activation number tendency
            ! Evaporation area of stratiform precipitation. 0<= am_evp_st <=1.
            ! Evaporation rate of stratiform rain [kg/kg/s]
            ! Evaporation rate of stratiform snow [kg/kg/s]
            ! [Total] Sfc flux of precip from stratiform [ m/s ]
            ! [Total] Sfc flux of snow from stratiform   [ m/s ]
            ! Surface flux of total cloud water from sedimentation
            ! Surface flux of cloud ice from sedimentation
            ! Sfc flux of precip from microphysics [ m/s ]
            ! Sfc flux of snow from microphysics [ m/s ]
            ! Relative humidity cloud fraction
            ! Old cloud fraction
            ! Evaporation of total precipitation (rain + snow)
            ! precipitation evaporation rate
            ! relative variance of cloud water
            ! optional accretion enhancement for experimentation
            ! Total precipitation (rain + snow)
            ! Ice effective diameter (meters) (AG: microns?)
            ! Size distribution shape parameter for radiation
            ! Size distribution slope parameter for radiation
            ! Snow effective diameter (m)
            ! array to hold rate1ord_cw2pr_st from microphysics
            ! Area over which precip evaporates
            ! Local evaporation of snow
            ! Local production of snow
            ! Rate of cond-evap of ice within the cloud
            ! Snow mixing ratio
            ! grid-box average rain flux (kg m^-2 s^-1)
            ! grid-box average snow flux (kg m^-2 s^-1)
            ! Rain mixing ratio
            ! Evaporation of falling cloud water
            ! Sublimation of falling cloud ice
            ! Residual condensation term to remove excess saturation
            ! Deposition/sublimation rate of cloud ice
            ! Mass-weighted cloud water fallspeed
            ! Mass-weighted cloud ice fallspeed
            ! Mass-weighted rain fallspeed
            ! Mass-weighted snow fallspeed
            ! Cloud water mixing ratio tendency from sedimentation
            ! Cloud ice mixing ratio tendency from sedimentation
            ! Rain mixing ratio tendency from sedimentation
            ! Snow mixing ratio tendency from sedimentation
            ! analytic radar reflectivity
            ! average reflectivity will zero points outside valid range
            ! average reflectivity in z.
            ! cloudsat reflectivity
            ! cloudsat average
            ! effective radius calculation for rain + cloud
            ! output number conc of ice nuclei available (1/m3)
            ! output number conc of CCN (1/m3)
            ! qc limiter ratio (1=no limit)
            ! Object that packs columns with clouds/precip.
            ! Packed versions of inputs.
            REAL(KIND=r8), allocatable :: packed_t(:,:)
            REAL(KIND=r8), allocatable :: packed_q(:,:)
            REAL(KIND=r8), allocatable :: packed_qc(:,:)
            REAL(KIND=r8), allocatable :: packed_nc(:,:)
            REAL(KIND=r8), allocatable :: packed_qi(:,:)
            REAL(KIND=r8), allocatable :: packed_ni(:,:)
            REAL(KIND=r8), allocatable :: packed_qr(:,:)
            REAL(KIND=r8), allocatable :: packed_nr(:,:)
            REAL(KIND=r8), allocatable :: packed_qs(:,:)
            REAL(KIND=r8), allocatable :: packed_ns(:,:)
            REAL(KIND=r8), allocatable :: packed_relvar(:,:)
            REAL(KIND=r8), allocatable :: packed_accre_enhan(:,:)
            REAL(KIND=r8), allocatable :: packed_p(:,:)
            REAL(KIND=r8), allocatable :: packed_pdel(:,:)
            ! This is only needed for MG1.5, and can be removed when support for
            ! that version is dropped.
            REAL(KIND=r8), allocatable :: packed_cldn(:,:)
            REAL(KIND=r8), allocatable :: packed_liqcldf(:,:)
            REAL(KIND=r8), allocatable :: packed_icecldf(:,:)
            REAL(KIND=r8), allocatable :: packed_naai(:,:)
            REAL(KIND=r8), allocatable :: packed_npccn(:,:)
            REAL(KIND=r8), allocatable :: packed_rndst(:,:,:)
            REAL(KIND=r8), allocatable :: packed_nacon(:,:,:)
            ! Optional outputs.
            REAL(KIND=r8), pointer :: packed_tnd_qsnow(:,:)
            REAL(KIND=r8), pointer :: packed_tnd_nsnow(:,:)
            REAL(KIND=r8), pointer :: packed_re_ice(:,:)
            REAL(KIND=r8), pointer :: packed_frzimm(:,:)
            REAL(KIND=r8), pointer :: packed_frzcnt(:,:)
            REAL(KIND=r8), pointer :: packed_frzdep(:,:)
            ! Output field post-processing.
            ! Packed versions of outputs.
            REAL(KIND=r8), allocatable, target :: packed_rate1ord_cw2pr_st(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_rate1ord_cw2pr_st(:,:)
            REAL(KIND=r8), allocatable, target :: packed_tlat(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_tlat(:,:)
            REAL(KIND=r8), allocatable, target :: packed_qvlat(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_qvlat(:,:)
            REAL(KIND=r8), allocatable, target :: packed_qctend(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_qctend(:,:)
            REAL(KIND=r8), allocatable, target :: packed_qitend(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_qitend(:,:)
            REAL(KIND=r8), allocatable, target :: packed_nctend(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_nctend(:,:)
            REAL(KIND=r8), allocatable, target :: packed_nitend(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_nitend(:,:)
            REAL(KIND=r8), allocatable, target :: packed_qrtend(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_qrtend(:,:)
            REAL(KIND=r8), allocatable, target :: packed_qstend(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_qstend(:,:)
            REAL(KIND=r8), allocatable, target :: packed_nrtend(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_nrtend(:,:)
            REAL(KIND=r8), allocatable, target :: packed_nstend(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_nstend(:,:)
            REAL(KIND=r8), allocatable, target :: packed_prect(:)
            REAL(KIND=r8), allocatable, target :: ref_packed_prect(:)
            REAL(KIND=r8), allocatable, target :: packed_preci(:)
            REAL(KIND=r8), allocatable, target :: ref_packed_preci(:)
            REAL(KIND=r8), allocatable, target :: packed_nevapr(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_nevapr(:,:)
            REAL(KIND=r8), allocatable, target :: packed_evapsnow(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_evapsnow(:,:)
            REAL(KIND=r8), allocatable, target :: packed_prain(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_prain(:,:)
            REAL(KIND=r8), allocatable, target :: packed_prodsnow(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_prodsnow(:,:)
            REAL(KIND=r8), allocatable, target :: packed_cmeout(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_cmeout(:,:)
            REAL(KIND=r8), allocatable, target :: packed_qsout(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_qsout(:,:)
            REAL(KIND=r8), allocatable, target :: packed_rflx(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_rflx(:,:)
            REAL(KIND=r8), allocatable, target :: packed_sflx(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_sflx(:,:)
            REAL(KIND=r8), allocatable, target :: packed_qrout(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_qrout(:,:)
            REAL(KIND=r8), allocatable, target :: packed_qcsevap(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_qcsevap(:,:)
            REAL(KIND=r8), allocatable, target :: packed_qisevap(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_qisevap(:,:)
            REAL(KIND=r8), allocatable, target :: packed_qvres(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_qvres(:,:)
            REAL(KIND=r8), allocatable, target :: packed_cmei(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_cmei(:,:)
            REAL(KIND=r8), allocatable, target :: packed_vtrmc(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_vtrmc(:,:)
            REAL(KIND=r8), allocatable, target :: packed_vtrmi(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_vtrmi(:,:)
            REAL(KIND=r8), allocatable, target :: packed_qcsedten(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_qcsedten(:,:)
            REAL(KIND=r8), allocatable, target :: packed_qisedten(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_qisedten(:,:)
            REAL(KIND=r8), allocatable, target :: packed_qrsedten(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_qrsedten(:,:)
            REAL(KIND=r8), allocatable, target :: packed_qssedten(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_qssedten(:,:)
            REAL(KIND=r8), allocatable, target :: packed_umr(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_umr(:,:)
            REAL(KIND=r8), allocatable, target :: packed_ums(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_ums(:,:)
            REAL(KIND=r8), allocatable, target :: packed_pra(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_pra(:,:)
            REAL(KIND=r8), allocatable, target :: packed_prc(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_prc(:,:)
            REAL(KIND=r8), allocatable, target :: packed_mnuccc(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_mnuccc(:,:)
            REAL(KIND=r8), allocatable, target :: packed_mnucct(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_mnucct(:,:)
            REAL(KIND=r8), allocatable, target :: packed_msacwi(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_msacwi(:,:)
            REAL(KIND=r8), allocatable, target :: packed_psacws(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_psacws(:,:)
            REAL(KIND=r8), allocatable, target :: packed_bergs(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_bergs(:,:)
            REAL(KIND=r8), allocatable, target :: packed_berg(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_berg(:,:)
            REAL(KIND=r8), allocatable, target :: packed_melt(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_melt(:,:)
            REAL(KIND=r8), allocatable, target :: packed_homo(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_homo(:,:)
            REAL(KIND=r8), allocatable, target :: packed_qcres(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_qcres(:,:)
            REAL(KIND=r8), allocatable, target :: packed_prci(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_prci(:,:)
            REAL(KIND=r8), allocatable, target :: packed_prai(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_prai(:,:)
            REAL(KIND=r8), allocatable, target :: packed_qires(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_qires(:,:)
            REAL(KIND=r8), allocatable, target :: packed_mnuccr(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_mnuccr(:,:)
            REAL(KIND=r8), allocatable, target :: packed_pracs(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_pracs(:,:)
            REAL(KIND=r8), allocatable, target :: packed_meltsdt(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_meltsdt(:,:)
            REAL(KIND=r8), allocatable, target :: packed_frzrdt(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_frzrdt(:,:)
            REAL(KIND=r8), allocatable, target :: packed_mnuccd(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_mnuccd(:,:)
            REAL(KIND=r8), allocatable, target :: packed_nrout(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_nrout(:,:)
            REAL(KIND=r8), allocatable, target :: packed_nsout(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_nsout(:,:)
            REAL(KIND=r8), allocatable, target :: packed_refl(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_refl(:,:)
            REAL(KIND=r8), allocatable, target :: packed_arefl(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_arefl(:,:)
            REAL(KIND=r8), allocatable, target :: packed_areflz(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_areflz(:,:)
            REAL(KIND=r8), allocatable, target :: packed_frefl(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_frefl(:,:)
            REAL(KIND=r8), allocatable, target :: packed_csrfl(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_csrfl(:,:)
            REAL(KIND=r8), allocatable, target :: packed_acsrfl(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_acsrfl(:,:)
            REAL(KIND=r8), allocatable, target :: packed_fcsrfl(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_fcsrfl(:,:)
            REAL(KIND=r8), allocatable, target :: packed_rercld(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_rercld(:,:)
            REAL(KIND=r8), allocatable, target :: packed_ncai(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_ncai(:,:)
            REAL(KIND=r8), allocatable, target :: packed_ncal(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_ncal(:,:)
            REAL(KIND=r8), allocatable, target :: packed_qrout2(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_qrout2(:,:)
            REAL(KIND=r8), allocatable, target :: packed_qsout2(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_qsout2(:,:)
            REAL(KIND=r8), allocatable, target :: packed_nrout2(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_nrout2(:,:)
            REAL(KIND=r8), allocatable, target :: packed_nsout2(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_nsout2(:,:)
            REAL(KIND=r8), allocatable, target :: packed_freqs(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_freqs(:,:)
            REAL(KIND=r8), allocatable, target :: packed_freqr(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_freqr(:,:)
            REAL(KIND=r8), allocatable, target :: packed_nfice(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_nfice(:,:)
            REAL(KIND=r8), allocatable, target :: packed_prer_evap(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_prer_evap(:,:)
            REAL(KIND=r8), allocatable, target :: packed_qcrat(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_qcrat(:,:)
            REAL(KIND=r8), allocatable, target :: packed_rel(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_rel(:,:)
            REAL(KIND=r8), allocatable, target :: packed_rei(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_rei(:,:)
            REAL(KIND=r8), allocatable, target :: packed_lambdac(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_lambdac(:,:)
            REAL(KIND=r8), allocatable, target :: packed_mu(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_mu(:,:)
            REAL(KIND=r8), allocatable, target :: packed_des(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_des(:,:)
            REAL(KIND=r8), allocatable, target :: packed_dei(:,:)
            REAL(KIND=r8), allocatable, target :: ref_packed_dei(:,:)
            ! Dummy arrays for cases where we throw away the MG version and
            ! recalculate sizes on the 1 grid to avoid time/subcolumn averaging
            ! issues.
            REAL(KIND=r8), allocatable :: rel_fn_dum(:,:)
            REAL(KIND=r8), allocatable :: ref_rel_fn_dum(:,:)
            REAL(KIND=r8), allocatable :: dsout2_dum(:,:)
            REAL(KIND=r8), allocatable :: ref_dsout2_dum(:,:)
            REAL(KIND=r8), allocatable :: drout_dum(:,:)
            REAL(KIND=r8), allocatable :: ref_drout_dum(:,:)
            REAL(KIND=r8), allocatable :: reff_rain_dum(:,:)
            REAL(KIND=r8), allocatable :: ref_reff_rain_dum(:,:)
            REAL(KIND=r8), allocatable :: reff_snow_dum(:,:)
            REAL(KIND=r8), allocatable :: ref_reff_snow_dum(:,:)
            ! Heterogeneous-only version of mnuccdo.
            ! physics buffer fields for COSP simulator
            ! MG grid-box mean flux_large_scale_cloud_rain+snow at interfaces (kg/m2/s)
            ! MG grid-box mean flux_large_scale_cloud_snow at interfaces (kg/m2/s)
            ! MG grid-box mean mixingratio_large_scale_cloud_rain+snow at interfaces (kg/kg)
            ! MG grid-box mean mixingratio_large_scale_cloud_snow at interfaces (kg/kg)
            ! MG diagnostic rain effective radius (um)
            ! MG diagnostic snow effective radius (um)
            ! convective cloud liquid effective radius (um)
            ! convective cloud ice effective radius (um)
            ! physics buffer fields used with CARMA
            ! external tendency on snow mass (kg/kg/s)
            ! external tendency on snow number(#/kg/s)
            ! ice effective radius (m)
            ! 1st order rate for direct conversion of
            ! strat. cloud water to precip (1/s)    ! rce 2010/05/01
            ! Sedimentation velocity of liquid stratus cloud droplet [ m/s ]
            ! Grid-mean microphysical tendency
            ! Grid-mean microphysical tendency
            ! Grid-mean microphysical tendency
            ! Grid-mean microphysical tendency
            ! Grid-mean microphysical tendency
            ! Grid-mean microphysical tendency
            ! In-liquid stratus microphysical tendency
            ! variables for heterogeneous freezing
            ! A local copy of state is used for diagnostic calculations
            ! Ice cloud fraction
            ! Liquid cloud fraction (combined into cloud)
            ! Liquid effective drop radius (microns)
            ! Ice effective drop size (microns)
            ! Total cloud fraction
            ! Convective cloud fraction
            ! Stratiform in-cloud ice water path for radiation
            ! Stratiform in-cloud liquid water path for radiation
            ! Cloud fraction for liquid+snow
            ! In-cloud snow water path
            ! In stratus ice mixing ratio
            ! In stratus water mixing ratio
            ! In cloud ice number conc
            ! In cloud water number conc
            ! Vertically-integrated in-cloud Liquid WP before microphysics
            ! Vertically-integrated in-cloud Ice WP before microphysics
            ! Averaging arrays for effective radius and number....
            ! Vertically-integrated droplet concentration
            ! In stratus ice mixing ratio
            ! In stratus water mixing ratio
            ! Cloud fraction used for precipitation.
            ! Average cloud top radius & number
            ! Variables for precip efficiency calculation
            ! LWP threshold
            ! accumulated precip across timesteps
            ! accumulated condensation across timesteps
            ! counter for # timesteps accumulated
            ! Variables for liquid water path and column condensation
            ! column liquid
            ! column condensation rate (units)
            ! precip efficiency for output
            ! fraction of time precip efficiency is written out
            ! average accumulated precipitation rate in pe calculation
            ! variables for autoconversion and accretion vertical averages
            ! vertical average autoconversion
            ! vertical average accretion
            ! ratio of vertical averages
            ! counters
            ! stratus ice mixing ratio - on grid
            ! stratus water mixing ratio - on grid
            ! Ice effective drop size at fixed number (indirect effect) (microns) - on grid
            INTEGER :: nlev ! number of levels where cloud physics is done
            INTEGER :: mgncol ! size of mgcols
            ! Columns with microphysics performed
            ! Flag to store whether accessing grid or sub-columns in pbuf_get_field
            CHARACTER(LEN=128) :: errstring
            CHARACTER(LEN=128) :: ref_errstring ! return status (non-blank for error return)
            ! For rrtmg optics. specified distribution.
            ! Convective size distribution effective radius (meters)
            ! Convective size distribution shape parameter
            ! Convective ice effective diameter (meters)
            !-------------------------------------------------------------------------------
            ! Find the number of levels used in the microphysics.
            ! Set the col_type flag to grid or subcolumn dependent on the value of use_subcol_microp
            !-----------------------
            ! These physics buffer fields are read only and not set in this parameterization
            ! If these fields do not have subcolumn data, copy the grid to the subcolumn if subcolumns is turned on
            ! If subcolumns is not turned on, then these fields will be grid data
            !-----------------------
            ! These physics buffer fields are calculated and set in this parameterization
            ! If subcolumns is turned on, then these fields will be calculated on a subcolumn grid, otherwise they will be a 
            ! normal grid
            !-----------------------
            ! If subcolumns is turned on, all calculated fields which are on subcolumns
            ! need to be retrieved on the grid as well for storing averaged values
            !-----------------------
            ! These are only on the grid regardless of whether subcolumns are turned on or not
            ! Only MG 1 defines this field so far.
            !-------------------------------------------------------------------------------------
            ! Microphysics assumes 'liquid stratus frac = ice stratus frac
            !                      = max( liquid stratus frac, ice stratus frac )'.
            ! Output initial in-cloud LWP (before microphysics)
            ! Initialize local state from input.
            ! Initialize ptend for output.
            ! the name 'cldwat' triggers special tests on cldliq
            ! and cldice in physics_update
            ! workaround an apparent pgi compiler bug on goldbach
            ! The following are all variables related to sizes, where it does not
            ! necessarily make sense to average over time steps. Instead, we keep
            ! the value from the last substep, which is what "accum_null" does.
            ! Allocate all the dummies with MG sizes.
            ! Pack input variables that are not updated during substeps.
            ! Allocate input variables that are updated during substeps.
                        tolerance = _TOL
                        CALL kgen_init_check(check_status, tolerance)
                        CALL kgen_read_real_r8_dim2_alloc(packed_t, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_q, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qc, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_nc, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qi, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_ni, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qr, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_nr, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qs, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_ns, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_relvar, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_accre_enhan, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_p, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_pdel, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_cldn, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_liqcldf, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_icecldf, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_naai, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_npccn, kgen_unit)
                        CALL kgen_read_real_r8_dim3_alloc(packed_rndst, kgen_unit)
                        CALL kgen_read_real_r8_dim3_alloc(packed_nacon, kgen_unit)
                        CALL kgen_read_real_r8_dim2_ptr(packed_tnd_qsnow, kgen_unit)
                        CALL kgen_read_real_r8_dim2_ptr(packed_tnd_nsnow, kgen_unit)
                        CALL kgen_read_real_r8_dim2_ptr(packed_re_ice, kgen_unit)
                        CALL kgen_read_real_r8_dim2_ptr(packed_frzimm, kgen_unit)
                        CALL kgen_read_real_r8_dim2_ptr(packed_frzcnt, kgen_unit)
                        CALL kgen_read_real_r8_dim2_ptr(packed_frzdep, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_rate1ord_cw2pr_st, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_tlat, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qvlat, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qctend, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qitend, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_nctend, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_nitend, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qrtend, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qstend, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_nrtend, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_nstend, kgen_unit)
                        CALL kgen_read_real_r8_dim1_alloc(packed_prect, kgen_unit)
                        CALL kgen_read_real_r8_dim1_alloc(packed_preci, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_nevapr, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_evapsnow, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_prain, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_prodsnow, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_cmeout, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qsout, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_rflx, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_sflx, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qrout, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qcsevap, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qisevap, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qvres, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_cmei, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_vtrmc, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_vtrmi, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qcsedten, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qisedten, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qrsedten, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qssedten, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_umr, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_ums, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_pra, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_prc, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_mnuccc, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_mnucct, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_msacwi, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_psacws, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_bergs, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_berg, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_melt, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_homo, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qcres, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_prci, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_prai, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qires, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_mnuccr, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_pracs, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_meltsdt, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_frzrdt, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_mnuccd, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_nrout, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_nsout, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_refl, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_arefl, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_areflz, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_frefl, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_csrfl, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_acsrfl, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_fcsrfl, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_rercld, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_ncai, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_ncal, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qrout2, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qsout2, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_nrout2, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_nsout2, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_freqs, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_freqr, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_nfice, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_prer_evap, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_qcrat, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_rel, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_rei, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_lambdac, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_mu, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_des, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(packed_dei, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(rel_fn_dum, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(dsout2_dum, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(drout_dum, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(reff_rain_dum, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(reff_snow_dum, kgen_unit)
                        READ(UNIT=kgen_unit) nlev
                        READ(UNIT=kgen_unit) mgncol
                        READ(UNIT=kgen_unit) errstring

                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_rate1ord_cw2pr_st, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_tlat, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_qvlat, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_qctend, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_qitend, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_nctend, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_nitend, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_qrtend, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_qstend, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_nrtend, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_nstend, kgen_unit)
                        CALL kgen_read_real_r8_dim1_alloc(ref_packed_prect, kgen_unit)
                        CALL kgen_read_real_r8_dim1_alloc(ref_packed_preci, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_nevapr, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_evapsnow, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_prain, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_prodsnow, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_cmeout, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_qsout, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_rflx, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_sflx, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_qrout, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_qcsevap, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_qisevap, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_qvres, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_cmei, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_vtrmc, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_vtrmi, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_qcsedten, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_qisedten, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_qrsedten, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_qssedten, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_umr, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_ums, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_pra, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_prc, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_mnuccc, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_mnucct, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_msacwi, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_psacws, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_bergs, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_berg, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_melt, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_homo, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_qcres, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_prci, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_prai, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_qires, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_mnuccr, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_pracs, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_meltsdt, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_frzrdt, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_mnuccd, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_nrout, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_nsout, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_refl, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_arefl, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_areflz, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_frefl, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_csrfl, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_acsrfl, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_fcsrfl, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_rercld, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_ncai, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_ncal, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_qrout2, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_qsout2, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_nrout2, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_nsout2, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_freqs, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_freqr, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_nfice, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_prer_evap, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_qcrat, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_rel, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_rei, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_lambdac, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_mu, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_des, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_packed_dei, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_rel_fn_dum, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_dsout2_dum, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_drout_dum, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_reff_rain_dum, kgen_unit)
                        CALL kgen_read_real_r8_dim2_alloc(ref_reff_snow_dum, kgen_unit)
                        READ(UNIT=kgen_unit) ref_errstring

                        ! call to kernel
                        CALL micro_mg_tend2_0(mgncol, nlev, dtime / num_steps, packed_t, packed_q, packed_qc, packed_qi, &
                            packed_nc, packed_ni, packed_qr, packed_qs, packed_nr, packed_ns, packed_relvar, &
                            packed_accre_enhan, packed_p, packed_pdel, packed_cldn, packed_liqcldf, packed_icecldf, &
                            packed_rate1ord_cw2pr_st, packed_naai, packed_npccn, packed_rndst, packed_nacon, packed_tlat, &
                            packed_qvlat, packed_qctend, packed_qitend, packed_nctend, packed_nitend, packed_qrtend, &
                            packed_qstend, packed_nrtend, packed_nstend, packed_rel, rel_fn_dum, packed_rei, packed_prect, &
                            packed_preci, packed_nevapr, packed_evapsnow, packed_prain, packed_prodsnow, packed_cmeout, &
                            packed_dei, packed_mu, packed_lambdac, packed_qsout, packed_des, packed_rflx, packed_sflx, &
                            packed_qrout, reff_rain_dum, reff_snow_dum, packed_qcsevap, packed_qisevap, packed_qvres, &
                            packed_cmei, packed_vtrmc, packed_vtrmi, packed_umr, packed_ums, packed_qcsedten, &
                            packed_qisedten, packed_qrsedten, packed_qssedten, packed_pra, packed_prc, packed_mnuccc, &
                            packed_mnucct, packed_msacwi, packed_psacws, packed_bergs, packed_berg, packed_melt, &
                            packed_homo, packed_qcres, packed_prci, packed_prai, packed_qires, packed_mnuccr, &
                            packed_pracs, packed_meltsdt, packed_frzrdt, packed_mnuccd, packed_nrout, packed_nsout, &
                            packed_refl, packed_arefl, packed_areflz, packed_frefl, packed_csrfl, packed_acsrfl, &
                            packed_fcsrfl, packed_rercld, packed_ncai, packed_ncal, packed_qrout2, packed_qsout2, &
                            packed_nrout2, packed_nsout2, drout_dum, dsout2_dum, packed_freqs, packed_freqr, &
                            packed_nfice, packed_qcrat, errstring, packed_tnd_qsnow, packed_tnd_nsnow, packed_re_ice, &
                            packed_prer_evap, packed_frzimm, packed_frzcnt, packed_frzdep)
                        ! kernel verification for output variables
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_rate1ord_cw2pr_st", check_status, packed_rate1ord_cw2pr_st, ref_packed_rate1ord_cw2pr_st)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_tlat", check_status, packed_tlat, ref_packed_tlat)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_qvlat", check_status, packed_qvlat, ref_packed_qvlat)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_qctend", check_status, packed_qctend, ref_packed_qctend)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_qitend", check_status, packed_qitend, ref_packed_qitend)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_nctend", check_status, packed_nctend, ref_packed_nctend)
                        ! Temporarily increase tolerance to 5.0e-13
                        check_status%tolerance = 5.E-13
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_nitend", check_status, packed_nitend, ref_packed_nitend)
                        check_status%tolerance = tolerance
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_qrtend", check_status, packed_qrtend, ref_packed_qrtend)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_qstend", check_status, packed_qstend, ref_packed_qstend)
                        ! Temporarily increase tolerance to 5.0e-14
                        check_status%tolerance = 5.E-14
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_nrtend", check_status, packed_nrtend, ref_packed_nrtend)
                        check_status%tolerance = tolerance
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_nstend", check_status, packed_nstend, ref_packed_nstend)
                        CALL kgen_verify_real_r8_dim1_alloc( "packed_prect", check_status, packed_prect, ref_packed_prect)
                        CALL kgen_verify_real_r8_dim1_alloc( "packed_preci", check_status, packed_preci, ref_packed_preci)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_nevapr", check_status, packed_nevapr, ref_packed_nevapr)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_evapsnow", check_status, packed_evapsnow, ref_packed_evapsnow)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_prain", check_status, packed_prain, ref_packed_prain)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_prodsnow", check_status, packed_prodsnow, ref_packed_prodsnow)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_cmeout", check_status, packed_cmeout, ref_packed_cmeout)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_qsout", check_status, packed_qsout, ref_packed_qsout)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_rflx", check_status, packed_rflx, ref_packed_rflx)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_sflx", check_status, packed_sflx, ref_packed_sflx)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_qrout", check_status, packed_qrout, ref_packed_qrout)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_qcsevap", check_status, packed_qcsevap, ref_packed_qcsevap)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_qisevap", check_status, packed_qisevap, ref_packed_qisevap)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_qvres", check_status, packed_qvres, ref_packed_qvres)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_cmei", check_status, packed_cmei, ref_packed_cmei)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_vtrmc", check_status, packed_vtrmc, ref_packed_vtrmc)
                        ! Temporarily increase tolerance to 5.0e-12
                        check_status%tolerance = 5.E-12
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_vtrmi", check_status, packed_vtrmi, ref_packed_vtrmi)
                        check_status%tolerance = tolerance
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_qcsedten", check_status, packed_qcsedten, ref_packed_qcsedten)
                        ! Temporarily increase tolerance to 1.0e-11
                        check_status%tolerance = 1.E-11 !djp djp
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_qisedten", check_status, packed_qisedten, ref_packed_qisedten)
                        check_status%tolerance = tolerance
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_qrsedten", check_status, packed_qrsedten, ref_packed_qrsedten)
                        ! Temporarily increase tolerance to 5.0e-12
                        check_status%tolerance = 1.E-11
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_qssedten", check_status, packed_qssedten, ref_packed_qssedten)
                        check_status%tolerance = tolerance
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_umr", check_status, packed_umr, ref_packed_umr)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_ums", check_status, packed_ums, ref_packed_ums)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_pra", check_status, packed_pra, ref_packed_pra)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_prc", check_status, packed_prc, ref_packed_prc)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_mnuccc", check_status, packed_mnuccc, ref_packed_mnuccc)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_mnucct", check_status, packed_mnucct, ref_packed_mnucct)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_msacwi", check_status, packed_msacwi, ref_packed_msacwi)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_psacws", check_status, packed_psacws, ref_packed_psacws)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_bergs", check_status, packed_bergs, ref_packed_bergs)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_berg", check_status, packed_berg, ref_packed_berg)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_melt", check_status, packed_melt, ref_packed_melt)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_homo", check_status, packed_homo, ref_packed_homo)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_qcres", check_status, packed_qcres, ref_packed_qcres)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_prci", check_status, packed_prci, ref_packed_prci)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_prai", check_status, packed_prai, ref_packed_prai)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_qires", check_status, packed_qires, ref_packed_qires)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_mnuccr", check_status, packed_mnuccr, ref_packed_mnuccr)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_pracs", check_status, packed_pracs, ref_packed_pracs)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_meltsdt", check_status, packed_meltsdt, ref_packed_meltsdt)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_frzrdt", check_status, packed_frzrdt, ref_packed_frzrdt)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_mnuccd", check_status, packed_mnuccd, ref_packed_mnuccd)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_nrout", check_status, packed_nrout, ref_packed_nrout)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_nsout", check_status, packed_nsout, ref_packed_nsout)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_refl", check_status, packed_refl, ref_packed_refl)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_arefl", check_status, packed_arefl, ref_packed_arefl)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_areflz", check_status, packed_areflz, ref_packed_areflz)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_frefl", check_status, packed_frefl, ref_packed_frefl)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_csrfl", check_status, packed_csrfl, ref_packed_csrfl)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_acsrfl", check_status, packed_acsrfl, ref_packed_acsrfl)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_fcsrfl", check_status, packed_fcsrfl, ref_packed_fcsrfl)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_rercld", check_status, packed_rercld, ref_packed_rercld)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_ncai", check_status, packed_ncai, ref_packed_ncai)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_ncal", check_status, packed_ncal, ref_packed_ncal)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_qrout2", check_status, packed_qrout2, ref_packed_qrout2)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_qsout2", check_status, packed_qsout2, ref_packed_qsout2)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_nrout2", check_status, packed_nrout2, ref_packed_nrout2)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_nsout2", check_status, packed_nsout2, ref_packed_nsout2)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_freqs", check_status, packed_freqs, ref_packed_freqs)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_freqr", check_status, packed_freqr, ref_packed_freqr)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_nfice", check_status, packed_nfice, ref_packed_nfice)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_prer_evap", check_status, packed_prer_evap, ref_packed_prer_evap)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_qcrat", check_status, packed_qcrat, ref_packed_qcrat)
                        ! Temporarily increase tolerance to 1.0e-11
                        check_status%tolerance = 1.E-11
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_rel", check_status, packed_rel, ref_packed_rel)
                        check_status%tolerance = tolerance
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_rei", check_status, packed_rei, ref_packed_rei)
                        ! Temporarily increase tolerance to 1.0e-11
                        check_status%tolerance = 1.E-11
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_lambdac", check_status, packed_lambdac, ref_packed_lambdac)
                        check_status%tolerance = tolerance
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_mu", check_status, packed_mu, ref_packed_mu)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_des", check_status, packed_des, ref_packed_des)
                        CALL kgen_verify_real_r8_dim2_alloc( "packed_dei", check_status, packed_dei, ref_packed_dei)
                        CALL kgen_verify_real_r8_dim2_alloc( "rel_fn_dum", check_status, rel_fn_dum, ref_rel_fn_dum)
                        CALL kgen_verify_real_r8_dim2_alloc( "dsout2_dum", check_status, dsout2_dum, ref_dsout2_dum)
                        CALL kgen_verify_real_r8_dim2_alloc( "drout_dum", check_status, drout_dum, ref_drout_dum)
                        CALL kgen_verify_real_r8_dim2_alloc( "reff_rain_dum", check_status, reff_rain_dum, ref_reff_rain_dum)
                        CALL kgen_verify_real_r8_dim2_alloc( "reff_snow_dum", check_status, reff_snow_dum, ref_reff_snow_dum)
                        CALL kgen_verify_character( "errstring", check_status, errstring, ref_errstring)
                        CALL kgen_print_check("micro_mg_tend", check_status)
                        CALL system_clock(start_clock, rate_clock)
                        DO kgen_intvar=1,10
                            CALL micro_mg_tend2_0(mgncol, nlev, dtime / num_steps, packed_t, packed_q, packed_qc, &
                                packed_qi, packed_nc, packed_ni, packed_qr, packed_qs, packed_nr, packed_ns, &
                                packed_relvar, packed_accre_enhan, packed_p, packed_pdel, packed_cldn, packed_liqcldf, &
                                packed_icecldf, packed_rate1ord_cw2pr_st, packed_naai, packed_npccn, packed_rndst, &
                                packed_nacon, packed_tlat, packed_qvlat, packed_qctend, packed_qitend, packed_nctend, &
                                packed_nitend, packed_qrtend, packed_qstend, packed_nrtend, packed_nstend, packed_rel, &
                                rel_fn_dum, packed_rei, packed_prect, packed_preci, packed_nevapr, packed_evapsnow, &
                                packed_prain, packed_prodsnow, packed_cmeout, packed_dei, packed_mu, packed_lambdac, &
                                packed_qsout, packed_des, packed_rflx, packed_sflx, packed_qrout, reff_rain_dum, &
                                reff_snow_dum, packed_qcsevap, packed_qisevap, packed_qvres, packed_cmei, packed_vtrmc, &
                                packed_vtrmi, packed_umr, packed_ums, packed_qcsedten, packed_qisedten, packed_qrsedten, &
                                packed_qssedten, packed_pra, packed_prc, packed_mnuccc, packed_mnucct, packed_msacwi, &
                                packed_psacws, packed_bergs, packed_berg, packed_melt, packed_homo, packed_qcres, &
                                packed_prci, packed_prai, packed_qires, packed_mnuccr, packed_pracs, packed_meltsdt, &
                                packed_frzrdt, packed_mnuccd, packed_nrout, packed_nsout, packed_refl, packed_arefl, &
                                packed_areflz, packed_frefl, packed_csrfl, packed_acsrfl, packed_fcsrfl, packed_rercld, &
                                packed_ncai, packed_ncal, packed_qrout2, packed_qsout2, packed_nrout2, packed_nsout2, &
                                drout_dum, dsout2_dum, packed_freqs, packed_freqr, packed_nfice, packed_qcrat, errstring, &
                                packed_tnd_qsnow, packed_tnd_nsnow, packed_re_ice, packed_prer_evap, packed_frzimm, &
                                packed_frzcnt, packed_frzdep)
                        END DO
                        CALL system_clock(stop_clock, rate_clock)
                        WRITE(*,*)
                        PRINT *, "Elapsed time (sec): ", (stop_clock - start_clock)/REAL(rate_clock*10)
            ! Divide ptend by substeps.
            ! Use summed outputs to produce averages
            ! Check to make sure that the microphysics code is respecting the flags that control
            ! whether MG should be prognosing cloud ice and cloud liquid or not.
            !! calculate effective radius of convective liquid and ice using dcon and deicon (not used by code, not useful for 
            ! COSP)
            !! hard-coded as average of hard-coded values used for deep/shallow convective detrainment (near line 1502/1505)
            ! Reassign rate1 if modal aerosols
            ! Sedimentation velocity for liquid stratus cloud droplet
            ! Microphysical tendencies for use in the macrophysics at the next time step
            ! Net micro_mg_cam condensation rate
            ! For precip, accumulate only total precip in prec_pcw and snow_pcw variables.
            ! Other precip output variables are set to 0
            ! Do not subscript by ncol here, because in physpkg we divide the whole
            ! array and need to avoid an FPE due to uninitialized data.
            ! ------------------------------------------------------------ !
            ! Compute in cloud ice and liquid mixing ratios                !
            ! Note that 'iclwp, iciwp' are used for radiation computation. !
            ! ------------------------------------------------------------ !
            ! Calculate cloud fraction for prognostic precip sizes.
            ! ------------------------------------------------------ !
            ! ------------------------------------------------------ !
            ! All code from here to the end is on grid columns only  !
            ! ------------------------------------------------------ !
            ! ------------------------------------------------------ !
            ! Average the fields which are needed later in this paramterization to be on the grid
            ! If on subcolumns, average the rest of the pbuf fields which were modified on subcolumns but are not used further in
            ! this parameterization  (no need to assign in the non-subcolumn case -- the else step)
            ! ------------------------------------- !
            ! Size distribution calculation         !
            ! ------------------------------------- !
            ! Calculate rho (on subcolumns if turned on) for size distribution
            ! parameter calculations and average it if needed
            !
            ! State instead of state_loc to preserve answers for MG1 (and in any
            ! case, it is unlikely to make much difference).
            ! Effective radius for cloud liquid, fixed number.
            ! Effective radius for cloud liquid, and size parameters
            ! mu_grid and lambdac_grid.
            ! Calculate ncic on the grid
            ! Rain/Snow effective diameter.
            ! Effective radius and diameter for cloud ice.
            ! Limiters for low cloud fraction.
            ! ------------------------------------- !
            ! Precipitation efficiency Calculation  !
            ! ------------------------------------- !
            !-----------------------------------------------------------------------
            ! Liquid water path
            ! Compute liquid water paths, and column condensation
            ! note: 1e-6 kgho2/kgair/s * 1000. pa / (9.81 m/s2) / 1000 kgh2o/m3 = 1e-7 m/s
            ! this is 1ppmv of h2o in 10hpa
            ! alternatively: 0.1 mm/day * 1.e-4 m/mm * 1/86400 day/s = 1.e-9
            !-----------------------------------------------------------------------
            ! precipitation efficiency calculation  (accumulate cme and precip)
            !minimum lwp threshold (kg/m3)
            ! zero out precip efficiency and total averaged precip
            ! accumulate precip and condensation
            !-----------------------------------------------------------------------
            ! vertical average of non-zero accretion, autoconversion and ratio.
            ! vars: vprco_grid(i),vprao_grid(i),racau_grid(i),cnt_grid
            ! --------------------- !
            ! History Output Fields !
            ! --------------------- !
            ! Column droplet concentration
            ! Averaging for new output fields
            ! Cloud top effective radius and number.
            ! Evaporation of stratiform precipitation fields for UNICON
            ! Assign the values to the pbuf pointers if they exist in pbuf
            ! --------------------------------------------- !
            ! General outfield calls for microphysics       !
            ! --------------------------------------------- !
            ! Output a handle of variables which are calculated on the fly
            ! Output fields which have not been averaged already, averaging if use_subcol_microp is true
            ! Example subcolumn outfld call
            ! Output fields which are already on the grid
            ! ptend_loc is deallocated in physics_update above
        CONTAINS

        ! write subroutines
            SUBROUTINE kgen_read_real_r8_dim2_alloc(var, kgen_unit, printvar)
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
            END SUBROUTINE kgen_read_real_r8_dim2_alloc

            SUBROUTINE kgen_read_real_r8_dim3_alloc(var, kgen_unit, printvar)
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
            END SUBROUTINE kgen_read_real_r8_dim3_alloc

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

            SUBROUTINE kgen_read_real_r8_dim1_alloc(var, kgen_unit, printvar)
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
            END SUBROUTINE kgen_read_real_r8_dim1_alloc


        ! verify subroutines
            SUBROUTINE kgen_verify_real_r8_dim2_alloc( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=r8), intent(in), DIMENSION(:,:), ALLOCATABLE :: var, ref_var
                real(KIND=r8) :: nrmsdiff, rmsdiff
                real(KIND=r8), allocatable, DIMENSION(:,:) :: temp, temp2
                integer :: n
                IF ( ALLOCATED(var) ) THEN
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
            END SUBROUTINE kgen_verify_real_r8_dim2_alloc

            SUBROUTINE kgen_verify_real_r8_dim1_alloc( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=r8), intent(in), DIMENSION(:), ALLOCATABLE :: var, ref_var
                real(KIND=r8) :: nrmsdiff, rmsdiff
                real(KIND=r8), allocatable, DIMENSION(:) :: temp, temp2
                integer :: n
                IF ( ALLOCATED(var) ) THEN
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
                END IF
            END SUBROUTINE kgen_verify_real_r8_dim1_alloc

            SUBROUTINE kgen_verify_character( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                character(LEN=128), intent(in) :: var, ref_var
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
            END SUBROUTINE kgen_verify_character

        END SUBROUTINE micro_mg_cam_tend


    END MODULE micro_mg_cam
