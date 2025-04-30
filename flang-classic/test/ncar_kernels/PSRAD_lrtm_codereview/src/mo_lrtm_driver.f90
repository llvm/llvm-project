
! KGEN-generated Fortran source file
!
! Filename    : mo_lrtm_driver.f90
! Generated at: 2015-02-19 15:30:29
! KGEN version: 0.4.4



    MODULE mo_lrtm_driver
        USE mo_kind, ONLY: wp
        USE mo_physical_constants, ONLY: amw
        USE mo_physical_constants, ONLY: amd
        USE mo_physical_constants, ONLY: grav
        USE mo_rrtm_params, ONLY: nbndlw
        USE mo_rrtm_params, ONLY: ngptlw
        USE mo_radiation_parameters, ONLY: do_gpoint
        USE mo_radiation_parameters, ONLY: i_overlap
        USE mo_radiation_parameters, ONLY: l_do_sep_clear_sky
        USE mo_radiation_parameters, ONLY: rad_undef
        USE mo_lrtm_setup, ONLY: ngb
        USE mo_lrtm_setup, ONLY: ngs
        USE mo_lrtm_setup, ONLY: delwave
        USE mo_lrtm_setup, ONLY: nspa
        USE mo_lrtm_setup, ONLY: nspb
        USE rrlw_planck, ONLY: totplanck
        USE mo_rrtm_coeffs, ONLY: lrtm_coeffs
        USE mo_lrtm_gas_optics, ONLY: gas_optics_lw
        USE mo_lrtm_solver, ONLY: find_secdiff
        USE mo_lrtm_solver, ONLY: lrtm_solver
        USE mo_cld_sampling, ONLY: sample_cld_state
        USE mo_spec_sampling, ONLY: spec_sampling_strategy
        USE mo_spec_sampling, ONLY: get_gpoint_set
        USE mo_taumol03, ONLY: taumol03_lwr,taumol03_upr
        USE mo_taumol04, ONLY: taumol04_lwr,taumol04_upr
        USE rrlw_planck, ONLY: chi_mls
        USE rrlw_kg03, ONLY: selfref
        USE rrlw_kg03, ONLY: forref
        USE rrlw_kg03, ONLY: ka_mn2o
        USE rrlw_kg03, ONLY: absa
        USE rrlw_kg03, ONLY: fracrefa
        USE rrlw_kg03, ONLY: kb_mn2o
        USE rrlw_kg03, ONLY: absb
        USE rrlw_kg03, ONLY: fracrefb
        IMPLICIT NONE
        PRIVATE
        PUBLIC lrtm
        CONTAINS

        ! read subroutines
        !-----------------------------------------------------------------------------
        !>
        !! @brief Prepares information for radiation call
        !!
        !! @remarks: This program is the driver subroutine for the longwave radiative
        !! transfer routine.  This routine is adapted from the AER LW RRTMG_LW model
        !! that itself has been adapted from RRTM_LW for improved efficiency.  Our
        !! routine does the spectral integration externally (the solver is explicitly
        !! called for each g-point, so as to facilitate sampling of g-points
        !! This routine:
        !!    1) calls INATM to read in the atmospheric profile from GCM;
        !!       all layering in RRTMG is ordered from surface to toa.
        !!    2) calls COEFFS to calculate various quantities needed for
        !!       the radiative transfer algorithm.  This routine is called only once for
        !!       any given thermodynamic state, i.e., it does not change if clouds chanege
        !!    3) calls TAUMOL to calculate gaseous optical depths for each
        !!       of the 16 spectral bands, this is updated band by band.
        !!    4) calls SOLVER (for both clear and cloudy profiles) to perform the
        !!       radiative transfer calculation with a maximum-random cloud
        !!       overlap method, or calls RTRN to use random cloud overlap.
        !!    5) passes the necessary fluxes and cooling rates back to GCM
        !!
        !

        SUBROUTINE lrtm(kproma, kbdim, klev, play, psfc, tlay, tlev, tsfc, wkl, wx, coldry, emis, cldfr, taucld, tauaer, rnseeds, &
        strategy, n_gpts_ts, uflx, dflx, uflxc, dflxc)
            INTEGER, intent(in) :: klev
            INTEGER, intent(in) :: kbdim
            INTEGER, intent(in) :: kproma
            !< Maximum block length
            !< Number of horizontal columns
            !< Number of model layers
            REAL(KIND=wp), intent(in) :: wx(:,:,:)
            REAL(KIND=wp), intent(in) :: cldfr(kbdim,klev)
            REAL(KIND=wp), intent(in) :: taucld(kbdim,klev,nbndlw)
            REAL(KIND=wp), intent(in) :: wkl(:,:,:)
            REAL(KIND=wp), intent(in) :: coldry(kbdim,klev)
            REAL(KIND=wp), intent(in) :: play(kbdim,klev)
            REAL(KIND=wp), intent(in) :: tlay(kbdim,klev)
            REAL(KIND=wp), intent(in) :: tauaer(kbdim,klev,nbndlw)
            REAL(KIND=wp), intent(in) :: tlev(kbdim,klev+1)
            REAL(KIND=wp), intent(in) :: tsfc(kbdim)
            REAL(KIND=wp), intent(in) :: psfc(kbdim)
            REAL(KIND=wp), intent(in) :: emis(kbdim,nbndlw)
            !< Layer pressures [hPa, mb] (kbdim,klev)
            !< Surface pressure [hPa, mb] (kbdim)
            !< Layer temperatures [K] (kbdim,klev)
            !< Interface temperatures [K] (kbdim,klev+1)
            !< Surface temperature [K] (kbdim)
            !< Gas volume mixing ratios
            !< CFC type gas volume mixing ratios
            !< Column dry amount
            !< Surface emissivity  (kbdim,nbndlw)
            !< Cloud fraction  (kbdim,klev)
            !< Coud optical depth (kbdim,klev,nbndlw)
            !< Aerosol optical depth (kbdim,klev,nbndlw)
            ! Variables for sampling cloud state and spectral points
            INTEGER, intent(inout) :: rnseeds(:, :) !< Seeds for random number generator (kbdim,:)
            TYPE(spec_sampling_strategy), intent(in) :: strategy
            INTEGER, intent(in   ) :: n_gpts_ts
            REAL(KIND=wp), intent(out) :: uflx (kbdim,0:klev)
            REAL(KIND=wp), intent(out) :: dflx (kbdim,0:klev)
            REAL(KIND=wp), intent(out) :: uflxc(kbdim,0:klev)
            REAL(KIND=wp), intent(out) :: dflxc(kbdim,0:klev)
            !< Tot sky longwave upward   flux [W/m2], (kbdim,0:klev)
            !< Tot sky longwave downward flux [W/m2], (kbdim,0:klev)
            !< Clr sky longwave upward   flux [W/m2], (kbdim,0:klev)
            !< Clr sky longwave downward flux [W/m2], (kbdim,0:klev)
            REAL(KIND=wp) :: taug(klev) !< Properties for one column at a time >! gas optical depth
            REAL(KIND=wp) :: rrpk_taug03(kbdim,klev)
            REAL(KIND=wp) :: rrpk_taug04(kbdim,klev)
            REAL(KIND=wp) :: fracs(kbdim,klev,n_gpts_ts)
            REAL(KIND=wp) :: taut  (kbdim,klev,n_gpts_ts)
            REAL(KIND=wp) :: tautot(kbdim,klev,n_gpts_ts)
            REAL(KIND=wp) :: pwvcm(kbdim)
            REAL(KIND=wp) :: secdiff(kbdim)
            !< Planck fraction per g-point
            !< precipitable water vapor [cm]
            !< diffusivity angle for RT calculation
            !< gaseous + aerosol optical depths for all columns
            !< cloud + gaseous + aerosol optical depths for all columns
            REAL(KIND=wp) :: planklay(kbdim,  klev,nbndlw)
            REAL(KIND=wp) :: planklev(kbdim,0:klev,nbndlw)
            REAL(KIND=wp) :: plankbnd(kbdim,       nbndlw) ! Properties for all bands
            ! Planck function at mid-layer
            ! Planck function at level interfaces
            ! Planck function at surface
            REAL(KIND=wp) :: layplnk(kbdim,  klev)
            REAL(KIND=wp) :: levplnk(kbdim,0:klev)
            REAL(KIND=wp) :: bndplnk(kbdim)
            REAL(KIND=wp) :: srfemis(kbdim) ! Properties for a single set of columns/g-points
            ! Planck function at mid-layer
            ! Planck function at level interfaces
            ! Planck function at surface
            ! Surface emission
            REAL(KIND=wp) :: zgpfd(kbdim,0:klev)
            REAL(KIND=wp) :: zgpfu(kbdim,0:klev)
            REAL(KIND=wp) :: zgpcu(kbdim,0:klev)
            REAL(KIND=wp) :: zgpcd(kbdim,0:klev)
            ! < gpoint clearsky downward flux
            ! < gpoint clearsky downward flux
            ! < gpoint fullsky downward flux
            ! < gpoint fullsky downward flux
            ! -----------------
            ! Variables for gas optics calculations
            INTEGER :: jt1     (kbdim,klev)
            INTEGER :: indfor  (kbdim,klev)
            INTEGER :: indself (kbdim,klev)
            INTEGER :: indminor(kbdim,klev)
            INTEGER :: laytrop (kbdim     )
            INTEGER :: jp      (kbdim,klev)
            INTEGER :: rrpk_jp (klev,kbdim)
            INTEGER :: jt      (kbdim,klev)
            INTEGER :: rrpk_jt (kbdim,0:1,klev)
            !< tropopause layer index
            !< lookup table index
            !< lookup table index
            !< lookup table index
            REAL(KIND=wp) :: wbrodl      (kbdim,klev)
            REAL(KIND=wp) :: selffac     (kbdim,klev)
            REAL(KIND=wp) :: colh2o      (kbdim,klev)
            REAL(KIND=wp) :: colo3       (kbdim,klev)
            REAL(KIND=wp) :: coln2o      (kbdim,klev)
            REAL(KIND=wp) :: colco       (kbdim,klev)
            REAL(KIND=wp) :: selffrac    (kbdim,klev)
            REAL(KIND=wp) :: colch4      (kbdim,klev)
            REAL(KIND=wp) :: colo2       (kbdim,klev)
            REAL(KIND=wp) :: colbrd      (kbdim,klev)
            REAL(KIND=wp) :: minorfrac   (kbdim,klev)
            REAL(KIND=wp) :: scaleminorn2(kbdim,klev)
            REAL(KIND=wp) :: scaleminor  (kbdim,klev)
            REAL(KIND=wp) :: forfac      (kbdim,klev)
            REAL(KIND=wp) :: colco2      (kbdim,klev)
            REAL(KIND=wp) :: forfrac     (kbdim,klev)
            !< column amount (h2o)
            !< column amount (co2)
            !< column amount (o3)
            !< column amount (n2o)
            !< column amount (co)
            !< column amount (ch4)
            !< column amount (o2)
            !< column amount (broadening gases)
            REAL(KIND=wp) :: wx_loc(size(wx, 2), size(wx, 3))
            !< Normalized CFC amounts (molecules/cm^2)
            REAL(KIND=wp) :: fac00(kbdim,klev)
            REAL(KIND=wp) :: fac01(kbdim,klev)
            REAL(KIND=wp) :: fac10(kbdim,klev)
            REAL(KIND=wp) :: fac11(kbdim,klev)
            REAL(KIND=wp) :: rrpk_fac0(kbdim,0:1,klev)
            REAL(KIND=wp) :: rrpk_fac1(kbdim,0:1,klev)
            REAL(KIND=wp) :: rat_n2oco2  (kbdim,klev)
            REAL(KIND=wp) :: rat_o3co2   (kbdim,klev)
            REAL(KIND=wp) :: rat_h2on2o  (kbdim,klev)
            REAL(KIND=wp) :: rat_n2oco2_1(kbdim,klev)
            REAL(KIND=wp) :: rat_h2on2o_1(kbdim,klev)
            REAL(KIND=wp) :: rat_h2oco2_1(kbdim,klev)
            REAL(KIND=wp) :: rat_h2oo3   (kbdim,klev)
            REAL(KIND=wp) :: rat_h2och4  (kbdim,klev)
            REAL(KIND=wp) :: rat_h2oco2  (kbdim,klev)
            REAL(KIND=wp) :: rrpk_rat_h2oco2  (kbdim,0:1,klev)
            REAL(KIND=wp) :: rrpk_rat_o3co2  (kbdim,0:1,klev)
            REAL(KIND=wp) :: rat_h2oo3_1 (kbdim,klev)
            REAL(KIND=wp) :: rat_o3co2_1 (kbdim,klev)
            REAL(KIND=wp) :: rat_h2och4_1(kbdim,klev)
            ! -----------------
            INTEGER :: jl,jlBegin,simdStep=96
            INTEGER :: ig
            INTEGER :: jk ! loop indicies
            INTEGER :: igs(kbdim, n_gpts_ts)
            INTEGER :: ibs(kbdim, n_gpts_ts)
            INTEGER :: ib
            INTEGER :: igpt
            INTEGER*8 :: start_clock,stop_clock,rate_clock
            REAL :: overall_time=0
            ! minimum val for clouds
            ! Variables for sampling strategy
            REAL(KIND=wp) :: gpt_scaling
            REAL(KIND=wp) :: clrsky_scaling(1:kbdim)
            REAL(KIND=wp) :: smp_tau(kbdim, klev, n_gpts_ts)
            LOGICAL :: cldmask(kbdim, klev, n_gpts_ts)
            LOGICAL :: colcldmask(kbdim,       n_gpts_ts) !< cloud mask in each cell
            !< cloud mask for each column
            !
            ! --------------------------------
            !
            ! 1.0 Choose a set of g-points to do consistent with the spectral sampling strategy
            !
            ! --------------------------------
            gpt_scaling = real(ngptlw,kind=wp)/real(n_gpts_ts,kind=wp)
            ! Standalone logic
            IF (do_gpoint == 0) THEN
                igs(1:kproma,1:n_gpts_ts) = get_gpoint_set(kproma, kbdim, strategy, rnseeds)
                ELSE IF (n_gpts_ts == 1) THEN ! Standalone logic
                IF (do_gpoint > ngptlw) RETURN
                igs(:, 1:n_gpts_ts) = do_gpoint
                ELSE
                PRINT *, "Asking for gpoint fluxes for too many gpoints!"
                STOP
            END IF 
            ! Save the band nunber associated with each gpoint
            DO jl = 1, kproma
                DO ig = 1, n_gpts_ts
                    ibs(jl, ig) = ngb(igs(jl, ig))
                END DO 
            END DO 
            !
            ! ---  2.0 Optical properties
            !
            ! ---  2.1 Cloud optical properties.
            ! --------------------------------
            ! Cloud optical depth is only saved for the band associated with this g-point
            !   We sample clouds first because we may want to adjust water vapor based
            !   on presence/absence of clouds
            !
            CALL sample_cld_state(kproma, kbdim, klev, n_gpts_ts, rnseeds(:,:), i_overlap, cldfr(:,:), cldmask(:,:,:))
            !IBM* ASSERT(NODEPS)
            DO ig = 1, n_gpts_ts
                DO jl = 1, kproma
                    smp_tau(jl,:,ig) = merge(taucld(jl,1:klev,ibs(jl,ig)), 0._wp, cldmask(jl,:,ig))
                END DO 
            END DO  ! Loop over samples - done with cloud optical depth calculations
            !
            ! Cloud masks for sorting out clear skies - by cell and by column
            !
            IF (.not. l_do_sep_clear_sky) THEN
                !
                ! Are any layers cloudy?
                !
                colcldmask(1:kproma,       1:n_gpts_ts) = any(cldmask(1:kproma,1:klev,1:n_gpts_ts), dim=2)
                !
                ! Clear-sky scaling is gpt_scaling/frac_clr or 0 if all samples are cloudy
                !
                clrsky_scaling(1:kproma) = gpt_scaling *                                                                          &
                         merge(real(n_gpts_ts,kind=wp) /                                        (real(n_gpts_ts - count(&
                colcldmask(1:kproma,:),dim=2),kind=wp)),                                                                          &
                  0._wp,                                                          any(.not. colcldmask(1:kproma,:),dim=2))
            END IF 
            !
            ! ---  2.2. Gas optical depth calculations
            !
            ! --------------------------------
            !
            ! 2.2.1  Calculate information needed by the radiative transfer routine
            ! that is specific to this atmosphere, especially some of the
            ! coefficients and indices needed to compute the optical depths
            ! by interpolating data from stored reference atmospheres.
            ! The coefficients are functions of temperature and pressure and remain the same
            ! for all g-point samples.
            ! If gas concentrations, temperatures, or pressures vary with sample (ig)
            !   the coefficients need to be calculated inside the loop over samples
            ! --------------------------------
            !
            ! Broadening gases -- the number of molecules per cm^2 of all gases not specified explicitly
            !   (water is excluded)
            wbrodl(1:kproma,1:klev) = coldry(1:kproma,1:klev) - sum(wkl(1:kproma,2:,1:klev), dim=2)
            CALL lrtm_coeffs(kproma, kbdim, klev, play, tlay, coldry, wkl, wbrodl, laytrop, jp, jt, jt1, colh2o, colco2, colo3, &
            coln2o, colco, colch4, colo2, colbrd, fac00, fac01, fac10, fac11, rat_h2oco2, rat_h2oco2_1, rat_h2oo3, rat_h2oo3_1, &
            rat_h2on2o, rat_h2on2o_1, rat_h2och4, rat_h2och4_1, rat_n2oco2, rat_n2oco2_1, rat_o3co2, rat_o3co2_1, selffac, &
            selffrac, indself, forfac, forfrac, indfor, minorfrac, scaleminor, scaleminorn2, indminor)
            !
            !  2.2.2 Loop over g-points calculating gas optical properties.
            !
            ! --------------------------------
            !IBM* ASSERT(NODEPS)
            !CALL system_clock(start_clock,rate_clock)
            rrpk_rat_h2oco2(:,0,:) = rat_h2oco2
            rrpk_rat_h2oco2(:,1,:) = (rat_h2oco2_1)
            rrpk_rat_o3co2(:,0,:) = rat_o3co2
            rrpk_rat_o3co2(:,1,:) = (rat_o3co2_1)
            rrpk_fac0(:,0,:) = fac00
            rrpk_fac0(:,1,:) = fac01
            rrpk_fac1(:,0,:) = fac10
            rrpk_fac1(:,1,:) = fac11
            rrpk_jt(:,0,:) = jt
            rrpk_jt(:,1,:) = jt1
            !CALL system_clock(stop_clock,rate_clock)
            !overall_time=overall_time + (stop_clock-start_clock)/REAL(rate_clock)
            !print *,n_gpts_ts
            !print *,"===",kproma
            DO ig = 1, n_gpts_ts
                igpt=igs(1,ig)
                IF(ngb(igpt) == 3) Then
                        CALL system_clock(start_clock, rate_clock)
                        jl=kproma
                        DO jlBegin = 1,kproma,simdStep
                                jl = jlBegin+simdStep-1
                                call taumol03_lwr(jl,jlBegin,laytrop(1), klev,                          &
                                    rrpk_rat_h2oco2(jlBegin:jl,:,:), colco2(jlBegin:jl,:), colh2o(jlBegin:jl,:), coln2o(jlBegin:jl,:), coldry(jlBegin:jl,:), &
                                    rrpk_fac0(jlBegin:jl,:,:), rrpk_fac1(jlBegin:jl,:,:), minorfrac(jlBegin:jl,:), &
                                    selffac(jlBegin:jl,:),selffrac(jlBegin:jl,:),forfac(jlBegin:jl,:),forfrac(jlBegin:jl,:), &
                                    jp(jlBegin:jl,:), rrpk_jt(jlBegin:jl,:,:), (igpt-ngs(ngb(igpt)-1)), indself(jlBegin:jl,:), &
                                    indfor(jlBegin:jl,:), indminor(jlBegin:jl,:), &
                                    rrpk_taug03(jlBegin:jl,:),fracs(jlBegin:jl,:,ig))
                                !print *,"Computing"
                                call taumol03_upr(jl,jlBegin,laytrop(1), klev,                          &
                                    rrpk_rat_h2oco2(jlBegin:jl,:,:), colco2(jlBegin:jl,:), colh2o(jlBegin:jl,:), coln2o(jlBegin:jl,:), coldry(jlBegin:jl,:), &
                                    rrpk_fac0(jlBegin:jl,:,:), rrpk_fac1(jlBegin:jl,:,:), minorfrac(jlBegin:jl,:), &
                                    forfac(jlBegin:jl,:),forfrac(jlBegin:jl,:),           &
                                    jp(jlBegin:jl,:), rrpk_jt(jlBegin:jl,:,:), (igpt-ngs(ngb(igpt)-1)), &
                                    indfor(jlBegin:jl,:), indminor(jlBegin:jl,:), &
                                    rrpk_taug03(jlBegin:jl,:),fracs(jlBegin:jl,:,ig))
                                !print *,"End Computing"
                        END DO
                        CALL system_clock(stop_clock, rate_clock)
                        overall_time=overall_time + (stop_clock-start_clock)/REAL(rate_clock)
                ENDIF
                IF(ngb(igpt) == 4) Then
                        !CALL system_clock(start_clock, rate_clock)
                        jl=kproma
                        call taumol04_lwr(jl,laytrop(1), klev,                          &
                            rrpk_rat_h2oco2(1:jl,:,:), colco2(1:jl,:), colh2o(1:jl,:),  coldry(1:jl,:), &
                            rrpk_fac0(1:jl,:,:), rrpk_fac1(1:jl,:,:), minorfrac(1:jl,:), &
                            selffac(1:jl,:),selffrac(1:jl,:),forfac(1:jl,:),forfrac(1:jl,:), &
                            jp(1:jl,:), rrpk_jt(1:jl,:,:), (igpt-ngs(ngb(igpt)-1)), indself(1:jl,:), &
                            indfor(1:jl,:), &
                            rrpk_taug04(1:jl,:),fracs(1:jl,:,ig))
                        call taumol04_upr(jl,laytrop(1), klev,                          &
                            rrpk_rat_o3co2(1:jl,:,:), colco2(1:jl,:), colo3(1:jl,:),  coldry(1:jl,:), &
                            rrpk_fac0(1:jl,:,:), rrpk_fac1(1:jl,:,:), minorfrac(1:jl,:), &
                            forfac(1:jl,:),forfrac(1:jl,:),           &
                            jp(1:jl,:), rrpk_jt(1:jl,:,:), (igpt-ngs(ngb(igpt)-1)), &
                            indfor(1:jl,:), &
                            rrpk_taug04(1:jl,:),fracs(1:jl,:,ig))
                        !CALL system_clock(stop_clock, rate_clock)
                        !overall_time=overall_time + (stop_clock-start_clock)/REAL(rate_clock)
                ENDIF
                DO jl = 1, kproma
                    ib = ibs(jl, ig)
                    igpt = igs(jl, ig)
                    !
                    ! Gas concentrations in colxx variables are normalized by 1.e-20_wp in lrtm_coeffs
                    !   CFC gas concentrations (wx) need the same normalization
                    !   Per Eli Mlawer the k values used in gas optics tables have been multiplied by 1e20
                    wx_loc(:,:) = 1.e-20_wp * wx(jl,:,:)
                    IF (ngb(igpt) == 3) THEN
                        taug = rrpk_taug03(jl,:)
                    ELSEIF (ngb(igpt) == 4) THEN
                        taug = rrpk_taug04(jl,:)
                    ELSE                    
                        CALL gas_optics_lw(klev, igpt, play        (jl,:), wx_loc    (:,:), coldry      (jl,:), laytrop     (jl), jp  &
                            (jl,:), jt        (jl,:), jt1         (jl,:), colh2o      (jl,:), colco2      (jl,:), colo3     (jl,:)&
                            , coln2o      (jl,:), colco       (jl,:), colch4      (jl,:), colo2     (jl,:), colbrd      (jl,:), fac00     &
                            (jl,:), fac01       (jl,:), fac10     (jl,:), fac11       (jl,:), rat_h2oco2  (jl,:), rat_h2oco2_1(jl,:), &
                            rat_h2oo3 (jl,:), rat_h2oo3_1 (jl,:), rat_h2on2o  (jl,:), rat_h2on2o_1(jl,:), rat_h2och4(jl,:), rat_h2och4_1(&
                            jl,:), rat_n2oco2  (jl,:), rat_n2oco2_1(jl,:), rat_o3co2 (jl,:), rat_o3co2_1 (jl,:), selffac     (jl,:), &
                            selffrac    (jl,:), indself   (jl,:), forfac      (jl,:), forfrac     (jl,:), indfor      (jl,:), minorfrac (&
                            jl,:), scaleminor  (jl,:), scaleminorn2(jl,:), indminor    (jl,:), fracs     (jl,:,ig), taug )
                    END IF
                    DO jk = 1, klev
                        taut(jl,jk,ig) = taug(jk) + tauaer(jl,jk,ib)
                    END DO 
                END DO  ! Loop over columns
            END DO  ! Loop over g point samples - done with gas optical depth calculations
            PRINT *, "Elapsed time (sec): ", overall_time
            overall_time=0
            tautot(1:kproma,:,:) = taut(1:kproma,:,:) + smp_tau(1:kproma,:,:) ! All-sky optical depth. Mask for 0 cloud optical depth?
            !
            ! ---  3.0 Compute radiative transfer.
            ! --------------------------------
            !
            ! Initialize fluxes to zero
            !
            uflx(1:kproma,0:klev) = 0.0_wp
            dflx(1:kproma,0:klev) = 0.0_wp
            uflxc(1:kproma,0:klev) = 0.0_wp
            dflxc(1:kproma,0:klev) = 0.0_wp
            !
            ! Planck function in each band at layers and boundaries
            !
            !IBM* ASSERT(NODEPS)
            DO ig = 1, nbndlw
                planklay(1:kproma,1:klev,ig) = planckfunction(tlay(1:kproma,1:klev  ),ig)
                planklev(1:kproma,0:klev,ig) = planckfunction(tlev(1:kproma,1:klev+1),ig)
                plankbnd(1:kproma,       ig) = planckfunction(tsfc(1:kproma         ),ig)
            END DO 
            !
            ! Precipitable water vapor in each column - this can affect the integration angle secdiff
            !
            pwvcm(1:kproma) = ((amw * sum(wkl(1:kproma,1,1:klev), dim=2)) /                        (amd * sum(coldry(1:kproma,&
            1:klev) + wkl(1:kproma,1,1:klev), dim=2))) *                       (1.e3_wp * psfc(1:kproma)) / (1.e2_wp * grav)
            !
            ! Compute radiative transfer for each set of samples
            !
            DO ig = 1, n_gpts_ts
                secdiff(1:kproma) = find_secdiff(ibs(1:kproma, ig), pwvcm(1:kproma))
                !IBM* ASSERT(NODEPS)
                DO jl = 1, kproma
                    ib = ibs(jl,ig)
                    layplnk(jl,1:klev) = planklay(jl,1:klev,ib)
                    levplnk(jl,0:klev) = planklev(jl,0:klev,ib)
                    bndplnk(jl) = plankbnd(jl,       ib)
                    srfemis(jl) = emis    (jl,       ib)
                END DO 
                !
                ! All sky fluxes
                !
                CALL lrtm_solver(kproma, kbdim, klev, tautot(:,:,ig), layplnk, levplnk, fracs(:,:,ig), secdiff, bndplnk, srfemis, &
                zgpfu, zgpfd)
                uflx(1:kproma,0:klev) = uflx (1:kproma,0:klev)                              + zgpfu(1:kproma,0:klev) * gpt_scaling
                dflx(1:kproma,0:klev) = dflx (1:kproma,0:klev)                              + zgpfd(1:kproma,0:klev) * gpt_scaling
                !
                ! Clear-sky fluxes
                !
                IF (l_do_sep_clear_sky) THEN
                    !
                    ! Remove clouds and do second RT calculation
                    !
                    CALL lrtm_solver(kproma, kbdim, klev, taut (:,:,ig), layplnk, levplnk, fracs(:,:,ig), secdiff, bndplnk, &
                    srfemis, zgpcu, zgpcd)
                    uflxc(1:kproma,0:klev) = uflxc(1:kproma,0:klev) + zgpcu(1:kproma,0:klev) * gpt_scaling
                    dflxc(1:kproma,0:klev) = dflxc(1:kproma,0:klev) + zgpcd(1:kproma,0:klev) * gpt_scaling
                    ELSE
                    !
                    ! Accumulate fluxes by excluding cloudy subcolumns, weighting to account for smaller sample size
                    !
                    !IBM* ASSERT(NODEPS)
                    DO jk = 0, klev
                        uflxc(1:kproma,jk) = uflxc(1:kproma,jk)                                                                   &
                                           + merge(0._wp,                                                                         &
                                                           zgpfu(1:kproma,jk) * clrsky_scaling(1:kproma),                         &
                                                                         colcldmask(1:kproma,ig))
                        dflxc(1:kproma,jk) = dflxc(1:kproma,jk)                                                                   &
                                           + merge(0._wp,                                                                         &
                                                           zgpfd(1:kproma,jk) * clrsky_scaling(1:kproma),                         &
                                                                         colcldmask(1:kproma,ig))
                    END DO 
                END IF 
            END DO  ! Loop over samples
            !
            ! ---  3.1 If computing clear-sky fluxes from samples, flag any columns where all samples were cloudy
            !
            ! --------------------------------
            IF (.not. l_do_sep_clear_sky) THEN
                !IBM* ASSERT(NODEPS)
                DO jl = 1, kproma
                    IF (all(colcldmask(jl,:))) THEN
                        uflxc(jl,0:klev) = rad_undef
                        dflxc(jl,0:klev) = rad_undef
                    END IF 
                END DO 
            END IF 
        END SUBROUTINE lrtm
        !----------------------------------------------------------------------------

        elemental FUNCTION planckfunction(temp, band)
            !
            ! Compute the blackbody emission in a given band as a function of temperature
            !
            REAL(KIND=wp), intent(in) :: temp
            INTEGER, intent(in) :: band
            REAL(KIND=wp) :: planckfunction
            INTEGER :: index
            REAL(KIND=wp) :: fraction
            index = min(max(1, int(temp - 159._wp)),180)
            fraction = temp - 159._wp - float(index)
            planckfunction = totplanck(index, band)                    + fraction * (totplanck(index+1, band) - totplanck(index, &
            band))
            planckfunction = planckfunction * delwave(band)
        END FUNCTION planckfunction
    END MODULE mo_lrtm_driver
