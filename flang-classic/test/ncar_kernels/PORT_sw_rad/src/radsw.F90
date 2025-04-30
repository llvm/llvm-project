
! KGEN-generated Fortran source file
!
! Filename    : radsw.F90
! Generated at: 2015-07-07 00:48:23
! KGEN version: 0.4.13



    MODULE radsw
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        !-----------------------------------------------------------------------
        !
        ! Purpose: Solar radiation calculations.
        !
        !-----------------------------------------------------------------------
        USE shr_kind_mod, ONLY: r8 => shr_kind_r8
        USE ppgrid, ONLY: pcols
        USE ppgrid, ONLY: pver
        USE ppgrid, ONLY: pverp
        USE scammod, ONLY: single_column
        USE scammod, ONLY: scm_crm_mode
        USE scammod, ONLY: have_asdir
        USE scammod, ONLY: asdirobs
        USE scammod, ONLY: have_asdif
        USE scammod, ONLY: asdifobs
        USE scammod, ONLY: have_aldir
        USE scammod, ONLY: aldirobs
        USE scammod, ONLY: have_aldif
        USE scammod, ONLY: aldifobs
        USE parrrsw, ONLY: nbndsw
        USE parrrsw, ONLY: ngptsw
        USE rrtmg_sw_rad, ONLY: rrtmg_sw
        IMPLICIT NONE
        PRIVATE
        ! fraction of solar irradiance in each band
        REAL(KIND=r8) :: solar_band_irrad(1:nbndsw) ! rrtmg-assumed solar irradiance in each sw band
        ! Public methods
        PUBLIC rad_rrtmg_sw
        ! initialize constants
        ! driver for solar radiation code
        !===============================================================================
            PUBLIC kgen_read_externs_radsw
        CONTAINS

        ! write subroutines

        ! module extern variables

        SUBROUTINE kgen_read_externs_radsw(kgen_unit)
            INTEGER, INTENT(IN) :: kgen_unit
            READ(UNIT=kgen_unit) solar_band_irrad
        END SUBROUTINE kgen_read_externs_radsw

        !===============================================================================

        SUBROUTINE rad_rrtmg_sw(lchnk, ncol, rrtmg_levs, r_state, e_pmid, e_cld, e_aer_tau, e_aer_tau_w, e_aer_tau_w_g, &
        e_aer_tau_w_f, eccf, e_coszrs, solin, sfac, e_asdir, e_asdif, e_aldir, e_aldif, qrs, qrsc, fsnt, fsntc, fsntoa, fsutoa, &
        fsntoac, fsnirtoa, fsnrtoac, fsnrtoaq, fsns, fsnsc, fsdsc, fsds, sols, soll, solsd, solld, fns, fcns, nday, nnite, idxday,&
         idxnite, su, sd, e_cld_tau, e_cld_tau_w, e_cld_tau_w_g, e_cld_tau_w_f, old_convert)
            !-----------------------------------------------------------------------
            !
            ! Purpose:
            ! Solar radiation code
            !
            ! Method:
            ! mji/rrtmg
            ! RRTMG, two-stream, with McICA
            !
            ! Divides solar spectrum into 14 intervals from 0.2-12.2 micro-meters.
            ! solar flux fractions specified for each interval. allows for
            ! seasonally and diurnally varying solar input.  Includes molecular,
            ! cloud, aerosol, and surface scattering, along with h2o,o3,co2,o2,cloud,
            ! and surface absorption. Computes delta-eddington reflections and
            ! transmissions assuming homogeneously mixed layers. Adds the layers
            ! assuming scattering between layers to be isotropic, and distinguishes
            ! direct solar beam from scattered radiation.
            !
            ! Longitude loops are broken into 1 or 2 sections, so that only daylight
            ! (i.e. coszrs > 0) computations are done.
            !
            ! Note that an extra layer above the model top layer is added.
            !
            ! mks units are used.
            !
            ! Special diagnostic calculation of the clear sky surface and total column
            ! absorbed flux is also done for cloud forcing diagnostics.
            !
            !-----------------------------------------------------------------------
            USE cmparray_mod, ONLY: cmpdaynite
            USE cmparray_mod, ONLY: expdaynite
            USE mcica_subcol_gen_sw, ONLY: mcica_subcol_sw
            USE physconst, ONLY: cpair
            USE rrtmg_state, ONLY: rrtmg_state_t
            ! Minimum cloud amount (as a fraction of the grid-box area) to
            ! distinguish from clear sky
            ! Decimal precision of cloud amount (0 -> preserve full resolution;
            ! 10^-n -> preserve n digits of cloud amount)
            ! Input arguments
            INTEGER, intent(in) :: lchnk ! chunk identifier
            INTEGER, intent(in) :: ncol ! number of atmospheric columns
            INTEGER, intent(in) :: rrtmg_levs ! number of levels rad is applied
            TYPE(rrtmg_state_t), intent(in) :: r_state
            INTEGER, intent(in) :: nday ! Number of daylight columns
            INTEGER, intent(in) :: nnite ! Number of night columns
            INTEGER, intent(in), dimension(pcols) :: idxday ! Indicies of daylight coumns
            INTEGER, intent(in), dimension(pcols) :: idxnite ! Indicies of night coumns
            REAL(KIND=r8), intent(in) :: e_pmid(pcols,pver) ! Level pressure (Pascals)
            REAL(KIND=r8), intent(in) :: e_cld(pcols,pver) ! Fractional cloud cover
            REAL(KIND=r8), intent(in) :: e_aer_tau    (pcols, 0:pver, nbndsw) ! aerosol optical depth
            REAL(KIND=r8), intent(in) :: e_aer_tau_w  (pcols, 0:pver, nbndsw) ! aerosol OD * ssa
            REAL(KIND=r8), intent(in) :: e_aer_tau_w_g(pcols, 0:pver, nbndsw) ! aerosol OD * ssa * asm
            REAL(KIND=r8), intent(in) :: e_aer_tau_w_f(pcols, 0:pver, nbndsw) ! aerosol OD * ssa * fwd
            REAL(KIND=r8), intent(in) :: eccf ! Eccentricity factor (1./earth-sun dist^2)
            REAL(KIND=r8), intent(in) :: e_coszrs(pcols) ! Cosine solar zenith angle
            REAL(KIND=r8), intent(in) :: e_asdir(pcols) ! 0.2-0.7 micro-meter srfc alb: direct rad
            REAL(KIND=r8), intent(in) :: e_aldir(pcols) ! 0.7-5.0 micro-meter srfc alb: direct rad
            REAL(KIND=r8), intent(in) :: e_asdif(pcols) ! 0.2-0.7 micro-meter srfc alb: diffuse rad
            REAL(KIND=r8), intent(in) :: e_aldif(pcols) ! 0.7-5.0 micro-meter srfc alb: diffuse rad
            REAL(KIND=r8), intent(in) :: sfac(nbndsw) ! factor to account for solar variability in each band
            REAL(KIND=r8), optional, intent(in) :: e_cld_tau    (nbndsw, pcols, pver) ! cloud optical depth
            REAL(KIND=r8), optional, intent(in) :: e_cld_tau_w  (nbndsw, pcols, pver) ! cloud optical
            REAL(KIND=r8), optional, intent(in) :: e_cld_tau_w_g(nbndsw, pcols, pver) ! cloud optical
            REAL(KIND=r8), optional, intent(in) :: e_cld_tau_w_f(nbndsw, pcols, pver) ! cloud optical
            LOGICAL, optional, intent(in) :: old_convert
            ! Output arguments
            REAL(KIND=r8), intent(out) :: solin(pcols) ! Incident solar flux
            REAL(KIND=r8), intent(out) :: qrs (pcols,pver) ! Solar heating rate
            REAL(KIND=r8), intent(out) :: qrsc(pcols,pver) ! Clearsky solar heating rate
            REAL(KIND=r8), intent(out) :: fsns(pcols) ! Surface absorbed solar flux
            REAL(KIND=r8), intent(out) :: fsnt(pcols) ! Total column absorbed solar flux
            REAL(KIND=r8), intent(out) :: fsntoa(pcols) ! Net solar flux at TOA
            REAL(KIND=r8), intent(out) :: fsutoa(pcols) ! Upward solar flux at TOA
            REAL(KIND=r8), intent(out) :: fsds(pcols) ! Flux shortwave downwelling surface
            REAL(KIND=r8), intent(out) :: fsnsc(pcols) ! Clear sky surface absorbed solar flux
            REAL(KIND=r8), intent(out) :: fsdsc(pcols) ! Clear sky surface downwelling solar flux
            REAL(KIND=r8), intent(out) :: fsntc(pcols) ! Clear sky total column absorbed solar flx
            REAL(KIND=r8), intent(out) :: fsntoac(pcols) ! Clear sky net solar flx at TOA
            REAL(KIND=r8), intent(out) :: sols(pcols) ! Direct solar rad on surface (< 0.7)
            REAL(KIND=r8), intent(out) :: soll(pcols) ! Direct solar rad on surface (>= 0.7)
            REAL(KIND=r8), intent(out) :: solsd(pcols) ! Diffuse solar rad on surface (< 0.7)
            REAL(KIND=r8), intent(out) :: solld(pcols) ! Diffuse solar rad on surface (>= 0.7)
            REAL(KIND=r8), intent(out) :: fsnirtoa(pcols) ! Near-IR flux absorbed at toa
            REAL(KIND=r8), intent(out) :: fsnrtoac(pcols) ! Clear sky near-IR flux absorbed at toa
            REAL(KIND=r8), intent(out) :: fsnrtoaq(pcols) ! Net near-IR flux at toa >= 0.7 microns
            REAL(KIND=r8), intent(out) :: fns(pcols,pverp) ! net flux at interfaces
            REAL(KIND=r8), intent(out) :: fcns(pcols,pverp) ! net clear-sky flux at interfaces
            REAL(KIND=r8), pointer, dimension(:,:,:) :: su ! shortwave spectral flux up
            REAL(KIND=r8), pointer, dimension(:,:,:) :: sd ! shortwave spectral flux down
            !---------------------------Local variables-----------------------------
            ! Local and reordered copies of the intent(in) variables
            REAL(KIND=r8) :: pmid(pcols,pver) ! Level pressure (Pascals)
            REAL(KIND=r8) :: cld(pcols,rrtmg_levs-1) ! Fractional cloud cover
            REAL(KIND=r8) :: cicewp(pcols,rrtmg_levs-1) ! in-cloud cloud ice water path
            REAL(KIND=r8) :: cliqwp(pcols,rrtmg_levs-1) ! in-cloud cloud liquid water path
            REAL(KIND=r8) :: rel(pcols,rrtmg_levs-1) ! Liquid effective drop size (microns)
            REAL(KIND=r8) :: rei(pcols,rrtmg_levs-1) ! Ice effective drop size (microns)
            REAL(KIND=r8) :: coszrs(pcols) ! Cosine solar zenith angle
            REAL(KIND=r8) :: asdir(pcols) ! 0.2-0.7 micro-meter srfc alb: direct rad
            REAL(KIND=r8) :: aldir(pcols) ! 0.7-5.0 micro-meter srfc alb: direct rad
            REAL(KIND=r8) :: asdif(pcols) ! 0.2-0.7 micro-meter srfc alb: diffuse rad
            REAL(KIND=r8) :: aldif(pcols) ! 0.7-5.0 micro-meter srfc alb: diffuse rad
            REAL(KIND=r8) :: h2ovmr(pcols,rrtmg_levs) ! h2o volume mixing ratio
            REAL(KIND=r8) :: o3vmr(pcols,rrtmg_levs) ! o3 volume mixing ratio
            REAL(KIND=r8) :: co2vmr(pcols,rrtmg_levs) ! co2 volume mixing ratio
            REAL(KIND=r8) :: ch4vmr(pcols,rrtmg_levs) ! ch4 volume mixing ratio
            REAL(KIND=r8) :: o2vmr(pcols,rrtmg_levs) ! o2  volume mixing ratio
            REAL(KIND=r8) :: n2ovmr(pcols,rrtmg_levs) ! n2o volume mixing ratio
            REAL(KIND=r8) :: tsfc(pcols) ! surface temperature
            INTEGER :: inflgsw ! flag for cloud parameterization method
            INTEGER :: iceflgsw ! flag for ice cloud parameterization method
            INTEGER :: liqflgsw ! flag for liquid cloud parameterization method
            INTEGER :: icld ! Flag for cloud overlap method
            ! 0=clear, 1=random, 2=maximum/random, 3=maximum
            INTEGER :: dyofyr ! Set to day of year for Earth/Sun distance calculation in
            ! rrtmg_sw, or pass in adjustment directly into adjes
            REAL(KIND=r8) :: solvar(nbndsw) ! solar irradiance variability in each band
            INTEGER, parameter :: nsubcsw = ngptsw ! rrtmg_sw g-point (quadrature point) dimension
            INTEGER :: permuteseed ! permute seed for sub-column generator
            ! cloud optical depth - diagnostic temp variable
            REAL(KIND=r8) :: tauc_sw(nbndsw, pcols, rrtmg_levs-1) ! cloud optical depth
            REAL(KIND=r8) :: ssac_sw(nbndsw, pcols, rrtmg_levs-1) ! cloud single scat. albedo
            REAL(KIND=r8) :: asmc_sw(nbndsw, pcols, rrtmg_levs-1) ! cloud asymmetry parameter
            REAL(KIND=r8) :: fsfc_sw(nbndsw, pcols, rrtmg_levs-1) ! cloud forward scattering fraction
            REAL(KIND=r8) :: tau_aer_sw(pcols, rrtmg_levs-1, nbndsw) ! aer optical depth
            REAL(KIND=r8) :: ssa_aer_sw(pcols, rrtmg_levs-1, nbndsw) ! aer single scat. albedo
            REAL(KIND=r8) :: asm_aer_sw(pcols, rrtmg_levs-1, nbndsw) ! aer asymmetry parameter
            REAL(KIND=r8) :: cld_stosw(nsubcsw, pcols, rrtmg_levs-1) ! stochastic cloud fraction
            REAL(KIND=r8) :: rei_stosw(pcols, rrtmg_levs-1) ! stochastic ice particle size
            REAL(KIND=r8) :: rel_stosw(pcols, rrtmg_levs-1) ! stochastic liquid particle size
            REAL(KIND=r8) :: cicewp_stosw(nsubcsw, pcols, rrtmg_levs-1) ! stochastic cloud ice water path
            REAL(KIND=r8) :: cliqwp_stosw(nsubcsw, pcols, rrtmg_levs-1) ! stochastic cloud liquid wter path
            REAL(KIND=r8) :: tauc_stosw(nsubcsw, pcols, rrtmg_levs-1) ! stochastic cloud optical depth (optional)
            REAL(KIND=r8) :: ssac_stosw(nsubcsw, pcols, rrtmg_levs-1) ! stochastic cloud single scat. albedo (optional)
            REAL(KIND=r8) :: asmc_stosw(nsubcsw, pcols, rrtmg_levs-1) ! stochastic cloud asymmetry parameter (optional)
            REAL(KIND=r8) :: fsfc_stosw(nsubcsw, pcols, rrtmg_levs-1) ! stochastic cloud forward scattering fraction (optional)
            REAL(KIND=r8), parameter :: dps = 1._r8/86400._r8 ! Inverse of seconds per day
            REAL(KIND=r8) :: swuflx(pcols,rrtmg_levs+1) ! Total sky shortwave upward flux (W/m2)
            REAL(KIND=r8) :: swdflx(pcols,rrtmg_levs+1) ! Total sky shortwave downward flux (W/m2)
            REAL(KIND=r8) :: swhr(pcols,rrtmg_levs) ! Total sky shortwave radiative heating rate (K/d)
            REAL(KIND=r8) :: swuflxc(pcols,rrtmg_levs+1) ! Clear sky shortwave upward flux (W/m2)
            REAL(KIND=r8) :: swdflxc(pcols,rrtmg_levs+1) ! Clear sky shortwave downward flux (W/m2)
            REAL(KIND=r8) :: swhrc(pcols,rrtmg_levs) ! Clear sky shortwave radiative heating rate (K/d)
            REAL(KIND=r8) :: swuflxs(nbndsw,pcols,rrtmg_levs+1) ! Shortwave spectral flux up
            REAL(KIND=r8) :: swdflxs(nbndsw,pcols,rrtmg_levs+1) ! Shortwave spectral flux down
            REAL(KIND=r8) :: dirdnuv(pcols,rrtmg_levs+1) ! Direct downward shortwave flux, UV/vis
            REAL(KIND=r8) :: difdnuv(pcols,rrtmg_levs+1) ! Diffuse downward shortwave flux, UV/vis
            REAL(KIND=r8) :: dirdnir(pcols,rrtmg_levs+1) ! Direct downward shortwave flux, near-IR
            REAL(KIND=r8) :: difdnir(pcols,rrtmg_levs+1) ! Diffuse downward shortwave flux, near-IR
            ! Added for net near-IR diagnostic
            REAL(KIND=r8) :: ninflx(pcols,rrtmg_levs+1) ! Net shortwave flux, near-IR
            REAL(KIND=r8) :: ninflxc(pcols,rrtmg_levs+1) ! Net clear sky shortwave flux, near-IR
            ! Other
            INTEGER :: ns
            INTEGER :: k
            INTEGER :: i ! indices
            ! Cloud radiative property arrays
            ! water cloud extinction optical depth
            ! ice cloud extinction optical depth
            ! liquid cloud single scattering albedo
            ! liquid cloud asymmetry parameter
            ! liquid cloud forward scattered fraction
            ! ice cloud single scattering albedo
            ! ice cloud asymmetry parameter
            ! ice cloud forward scattered fraction
            ! Aerosol radiative property arrays
            ! aerosol extinction optical depth
            ! aerosol single scattering albedo
            ! aerosol assymetry parameter
            ! aerosol forward scattered fraction
            ! CRM
            REAL(KIND=r8) :: fus(pcols,pverp) ! Upward flux (added for CRM)
            REAL(KIND=r8) :: fds(pcols,pverp) ! Downward flux (added for CRM)
            REAL(KIND=r8) :: fusc(pcols,pverp) ! Upward clear-sky flux (added for CRM)
            REAL(KIND=r8) :: fdsc(pcols,pverp) ! Downward clear-sky flux (added for CRM)
            INTEGER :: kk
            REAL(KIND=r8) :: pmidmb(pcols,rrtmg_levs) ! Level pressure (hPa)
            REAL(KIND=r8) :: pintmb(pcols,rrtmg_levs+1) ! Model interface pressure (hPa)
            REAL(KIND=r8) :: tlay(pcols,rrtmg_levs) ! mid point temperature
            REAL(KIND=r8) :: tlev(pcols,rrtmg_levs+1) ! interface temperature
            !-----------------------------------------------------------------------
            ! START OF CALCULATION
            !-----------------------------------------------------------------------
            ! Initialize output fields:
   fsds(1:ncol)     = 0.0_r8
   fsnirtoa(1:ncol) = 0.0_r8
   fsnrtoac(1:ncol) = 0.0_r8
   fsnrtoaq(1:ncol) = 0.0_r8
   fsns(1:ncol)     = 0.0_r8
   fsnsc(1:ncol)    = 0.0_r8
   fsdsc(1:ncol)    = 0.0_r8
   fsnt(1:ncol)     = 0.0_r8
   fsntc(1:ncol)    = 0.0_r8
   fsntoa(1:ncol)   = 0.0_r8
   fsutoa(1:ncol)   = 0.0_r8
   fsntoac(1:ncol)  = 0.0_r8
   solin(1:ncol)    = 0.0_r8
   sols(1:ncol)     = 0.0_r8
   soll(1:ncol)     = 0.0_r8
   solsd(1:ncol)    = 0.0_r8
   solld(1:ncol)    = 0.0_r8
   qrs (1:ncol,1:pver) = 0.0_r8
   qrsc(1:ncol,1:pver) = 0.0_r8
   fns(1:ncol,1:pverp) = 0.0_r8
   fcns(1:ncol,1:pverp) = 0.0_r8
   if (single_column.and.scm_crm_mode) then 
      fus(1:ncol,1:pverp) = 0.0_r8
      fds(1:ncol,1:pverp) = 0.0_r8
      fusc(:ncol,:pverp) = 0.0_r8
      fdsc(:ncol,:pverp) = 0.0_r8
   endif
   if (associated(su)) su(1:ncol,:,:) = 0.0_r8
   if (associated(sd)) sd(1:ncol,:,:) = 0.0_r8
            ! If night everywhere, return:
   if ( Nday == 0 ) then
     return
   endif
            ! Rearrange input arrays
   call CmpDayNite(E_pmid(:,pverp-rrtmg_levs+1:pver), pmid(:,1:rrtmg_levs-1), &
        Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, rrtmg_levs-1)
   call CmpDayNite(E_cld(:,pverp-rrtmg_levs+1:pver),  cld(:,1:rrtmg_levs-1), &
        Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, rrtmg_levs-1)
   call CmpDayNite(r_state%pintmb, pintmb, Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, rrtmg_levs+1)
   call CmpDayNite(r_state%pmidmb, pmidmb, Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, rrtmg_levs)
   call CmpDayNite(r_state%h2ovmr, h2ovmr, Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, rrtmg_levs)
   call CmpDayNite(r_state%o3vmr,  o3vmr,  Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, rrtmg_levs)
   call CmpDayNite(r_state%co2vmr, co2vmr, Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, rrtmg_levs)
   call CmpDayNite(E_coszrs, coszrs,    Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   call CmpDayNite(E_asdir,  asdir,     Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   call CmpDayNite(E_aldir,  aldir,     Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   call CmpDayNite(E_asdif,  asdif,     Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   call CmpDayNite(E_aldif,  aldif,     Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   call CmpDayNite(r_state%tlay,   tlay,   Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, rrtmg_levs)
   call CmpDayNite(r_state%tlev,   tlev,   Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, rrtmg_levs+1)
   call CmpDayNite(r_state%ch4vmr, ch4vmr, Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, rrtmg_levs)
   call CmpDayNite(r_state%o2vmr,  o2vmr,  Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, rrtmg_levs)
   call CmpDayNite(r_state%n2ovmr, n2ovmr, Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, rrtmg_levs)
            ! These fields are no longer input by CAM.
   cicewp = 0.0_r8
   cliqwp = 0.0_r8
   rel = 0.0_r8
   rei = 0.0_r8
            ! Aerosol daylight map
            ! Also convert to optical properties of rrtmg interface, even though
            !   these quantities are later multiplied back together inside rrtmg !
            ! Why does rrtmg use the factored quantities?
            ! There are several different ways this factoring could be done.
            ! Other ways might allow for better optimization
   do ns = 1, nbndsw
      do k  = 1, rrtmg_levs-1
         kk=(pverp-rrtmg_levs) + k
         do i  = 1, Nday
            if(E_aer_tau_w(IdxDay(i),kk,ns) > 1.e-80_r8) then
               asm_aer_sw(i,k,ns) = E_aer_tau_w_g(IdxDay(i),kk,ns)/E_aer_tau_w(IdxDay(i),kk,ns)
            else
               asm_aer_sw(i,k,ns) = 0._r8
            endif
            if(E_aer_tau(IdxDay(i),kk,ns) > 0._r8) then
               ssa_aer_sw(i,k,ns) = E_aer_tau_w(IdxDay(i),kk,ns)/E_aer_tau(IdxDay(i),kk,ns)
               tau_aer_sw(i,k,ns) = E_aer_tau(IdxDay(i),kk,ns)
            else
               ssa_aer_sw(i,k,ns) = 1._r8
               tau_aer_sw(i,k,ns) = 0._r8
            endif
         enddo
      enddo
   enddo
   if (scm_crm_mode) then
                ! overwrite albedos for CRM
      if(have_asdir) asdir = asdirobs(1)
      if(have_asdif) asdif = asdifobs(1)
      if(have_aldir) aldir = aldirobs(1)
      if(have_aldif) aldif = aldifobs(1)
   endif
            ! Define solar incident radiation
   do i = 1, Nday
      solin(i)  = sum(sfac(:)*solar_band_irrad(:)) * eccf * coszrs(i)
   end do
            ! Calculate cloud optical properties here if using CAM method, or if using one of the
            ! methods in RRTMG_SW, then pass in cloud physical properties and zero out cloud optical
            ! properties here
            ! Zero optional cloud optical property input arrays tauc_sw, ssac_sw, asmc_sw,
            ! if inputting cloud physical properties to RRTMG_SW
            !tauc_sw(:,:,:) = 0.0_r8
            !ssac_sw(:,:,:) = 1.0_r8
            !asmc_sw(:,:,:) = 0.0_r8
            !fsfc_sw(:,:,:) = 0.0_r8
            !
            ! Or, calculate and pass in CAM cloud shortwave optical properties to RRTMG_SW
            !if (present(old_convert)) print *, 'old_convert',old_convert
            !if (present(ancientmethod)) print *, 'ancientmethod',ancientmethod
   if (present(old_convert))then
      if (old_convert)then ! convert without limits ! convert without limits
         do i = 1, Nday
         do k = 1, rrtmg_levs-1
         kk=(pverp-rrtmg_levs) + k
         do ns = 1, nbndsw
           if (E_cld_tau_w(ns,IdxDay(i),kk) > 0._r8) then
              fsfc_sw(ns,i,k)=E_cld_tau_w_f(ns,IdxDay(i),kk)/E_cld_tau_w(ns,IdxDay(i),kk)
              asmc_sw(ns,i,k)=E_cld_tau_w_g(ns,IdxDay(i),kk)/E_cld_tau_w(ns,IdxDay(i),kk)
           else
              fsfc_sw(ns,i,k) = 0._r8
              asmc_sw(ns,i,k) = 0._r8
           endif
           tauc_sw(ns,i,k)=E_cld_tau(ns,IdxDay(i),kk)
           if (tauc_sw(ns,i,k) > 0._r8) then
              ssac_sw(ns,i,k)=E_cld_tau_w(ns,IdxDay(i),kk)/tauc_sw(ns,i,k)
           else
              tauc_sw(ns,i,k) = 0._r8
              fsfc_sw(ns,i,k) = 0._r8
              asmc_sw(ns,i,k) = 0._r8
              ssac_sw(ns,i,k) = 1._r8
           endif
         enddo
         enddo
         enddo
      else
                    ! eventually, when we are done with archaic versions, This set of code will become the default.
         do i = 1, Nday
         do k = 1, rrtmg_levs-1
         kk=(pverp-rrtmg_levs) + k
         do ns = 1, nbndsw
           if (E_cld_tau_w(ns,IdxDay(i),kk) > 0._r8) then
              fsfc_sw(ns,i,k)=E_cld_tau_w_f(ns,IdxDay(i),kk)/max(E_cld_tau_w(ns,IdxDay(i),kk), 1.e-80_r8)
              asmc_sw(ns,i,k)=E_cld_tau_w_g(ns,IdxDay(i),kk)/max(E_cld_tau_w(ns,IdxDay(i),kk), 1.e-80_r8)
           else
              fsfc_sw(ns,i,k) = 0._r8
              asmc_sw(ns,i,k) = 0._r8
           endif
           tauc_sw(ns,i,k)=E_cld_tau(ns,IdxDay(i),kk)
           if (tauc_sw(ns,i,k) > 0._r8) then
              ssac_sw(ns,i,k)=max(E_cld_tau_w(ns,IdxDay(i),kk),1.e-80_r8)/max(tauc_sw(ns,i,k),1.e-80_r8)
           else
              tauc_sw(ns,i,k) = 0._r8
              fsfc_sw(ns,i,k) = 0._r8
              asmc_sw(ns,i,k) = 0._r8
              ssac_sw(ns,i,k) = 1._r8
           endif
         enddo
         enddo
         enddo
      endif
   else
      do i = 1, Nday
      do k = 1, rrtmg_levs-1
      kk=(pverp-rrtmg_levs) + k
      do ns = 1, nbndsw
        if (E_cld_tau_w(ns,IdxDay(i),kk) > 0._r8) then
           fsfc_sw(ns,i,k)=E_cld_tau_w_f(ns,IdxDay(i),kk)/max(E_cld_tau_w(ns,IdxDay(i),kk), 1.e-80_r8)
           asmc_sw(ns,i,k)=E_cld_tau_w_g(ns,IdxDay(i),kk)/max(E_cld_tau_w(ns,IdxDay(i),kk), 1.e-80_r8)
        else
           fsfc_sw(ns,i,k) = 0._r8
           asmc_sw(ns,i,k) = 0._r8
        endif
        tauc_sw(ns,i,k)=E_cld_tau(ns,IdxDay(i),kk)
        if (tauc_sw(ns,i,k) > 0._r8) then
           ssac_sw(ns,i,k)=max(E_cld_tau_w(ns,IdxDay(i),kk),1.e-80_r8)/max(tauc_sw(ns,i,k),1.e-80_r8)
        else
           tauc_sw(ns,i,k) = 0._r8
           fsfc_sw(ns,i,k) = 0._r8
           asmc_sw(ns,i,k) = 0._r8
           ssac_sw(ns,i,k) = 1._r8
        endif
      enddo
      enddo
      enddo
   endif
            ! Call mcica sub-column generator for RRTMG_SW
            ! Call sub-column generator for McICA in radiation
            ! Select cloud overlap approach (1=random, 2=maximum-random, 3=maximum)
   icld = 2
            ! Set permute seed (must be offset between LW and SW by at least 140 to insure
            ! effective randomization)
   permuteseed = 1
   call mcica_subcol_sw(lchnk, Nday, rrtmg_levs-1, icld, permuteseed, pmid, &
      cld, cicewp, cliqwp, rei, rel, tauc_sw, ssac_sw, asmc_sw, fsfc_sw, &
      cld_stosw, cicewp_stosw, cliqwp_stosw, rei_stosw, rel_stosw, &
      tauc_stosw, ssac_stosw, asmc_stosw, fsfc_stosw)
            ! Call RRTMG_SW for all layers for daylight columns
            ! Select parameterization of cloud ice and liquid optical depths
            ! Use CAM shortwave cloud optical properties directly
   inflgsw = 0 
   iceflgsw = 0
   liqflgsw = 0
            ! Use E&C param for ice to mimic CAM3 for now
            !   inflgsw = 2
            !   iceflgsw = 1
            !   liqflgsw = 1
            ! Use merged Fu and E&C params for ice
            !   inflgsw = 2
            !   iceflgsw = 3
            !   liqflgsw = 1
            ! Set day of year for Earth/Sun distance calculation in rrtmg_sw, or
            ! set to zero and pass E/S adjustment (eccf) directly into array adjes
   dyofyr = 0
   tsfc(:ncol) = tlev(:ncol,rrtmg_levs+1)
   solvar(1:nbndsw) = sfac(1:nbndsw)
   call rrtmg_sw(lchnk, Nday, rrtmg_levs, icld,         &
                 pmidmb, pintmb, tlay, tlev, tsfc, &
                 h2ovmr, o3vmr, co2vmr, ch4vmr, o2vmr, n2ovmr, &
                 asdir, asdif, aldir, aldif, &
                 coszrs, eccf, dyofyr, solvar, &
                 inflgsw, iceflgsw, liqflgsw, &
                 cld_stosw, tauc_stosw, ssac_stosw, asmc_stosw, fsfc_stosw, &
                 cicewp_stosw, cliqwp_stosw, rei, rel, &
                 tau_aer_sw, ssa_aer_sw, asm_aer_sw, &
                 swuflx, swdflx, swhr, swuflxc, swdflxc, swhrc, &
                 dirdnuv, dirdnir, difdnuv, difdnir, ninflx, ninflxc, swuflxs, swdflxs)
            ! Flux units are in W/m2 on output from rrtmg_sw and contain output for
            ! extra layer above model top with vertical indexing from bottom to top.
            !
            ! Heating units are in J/kg/s on output from rrtmg_sw and contain output
            ! for extra layer above model top with vertical indexing from bottom to top.
            !
            ! Reverse vertical indexing to go from top to bottom for CAM output.
            ! Set the net absorted shortwave flux at TOA (top of extra layer)
   fsntoa(1:Nday) = swdflx(1:Nday,rrtmg_levs+1) - swuflx(1:Nday,rrtmg_levs+1)
   fsutoa(1:Nday) = swuflx(1:Nday,rrtmg_levs+1)
   fsntoac(1:Nday) = swdflxc(1:Nday,rrtmg_levs+1) - swuflxc(1:Nday,rrtmg_levs+1)
            ! Set net near-IR flux at top of the model
   fsnirtoa(1:Nday) = ninflx(1:Nday,rrtmg_levs)
   fsnrtoaq(1:Nday) = ninflx(1:Nday,rrtmg_levs)
   fsnrtoac(1:Nday) = ninflxc(1:Nday,rrtmg_levs)
            ! Set the net absorbed shortwave flux at the model top level
   fsnt(1:Nday) = swdflx(1:Nday,rrtmg_levs) - swuflx(1:Nday,rrtmg_levs)
   fsntc(1:Nday) = swdflxc(1:Nday,rrtmg_levs) - swuflxc(1:Nday,rrtmg_levs)
            ! Set the downwelling flux at the surface
   fsds(1:Nday) = swdflx(1:Nday,1)
   fsdsc(1:Nday) = swdflxc(1:Nday,1)
            ! Set the net shortwave flux at the surface
   fsns(1:Nday) = swdflx(1:Nday,1) - swuflx(1:Nday,1)
   fsnsc(1:Nday) = swdflxc(1:Nday,1) - swuflxc(1:Nday,1)
            ! Set the UV/vis and near-IR direct and dirruse downward shortwave flux at surface
   sols(1:Nday) = dirdnuv(1:Nday,1)
   soll(1:Nday) = dirdnir(1:Nday,1)
   solsd(1:Nday) = difdnuv(1:Nday,1)
   solld(1:Nday) = difdnir(1:Nday,1)
            ! Set the net, up and down fluxes at model interfaces
   fns (1:Nday,pverp-rrtmg_levs+1:pverp) =  swdflx(1:Nday,rrtmg_levs:1:-1) -  swuflx(1:Nday,rrtmg_levs:1:-1)
   fcns(1:Nday,pverp-rrtmg_levs+1:pverp) = swdflxc(1:Nday,rrtmg_levs:1:-1) - swuflxc(1:Nday,rrtmg_levs:1:-1)
   fus (1:Nday,pverp-rrtmg_levs+1:pverp) =  swuflx(1:Nday,rrtmg_levs:1:-1)
   fusc(1:Nday,pverp-rrtmg_levs+1:pverp) = swuflxc(1:Nday,rrtmg_levs:1:-1)
   fds (1:Nday,pverp-rrtmg_levs+1:pverp) =  swdflx(1:Nday,rrtmg_levs:1:-1)
   fdsc(1:Nday,pverp-rrtmg_levs+1:pverp) = swdflxc(1:Nday,rrtmg_levs:1:-1)
            ! Set solar heating, reverse layering
            ! Pass shortwave heating to CAM arrays and convert from K/d to J/kg/s
   qrs (1:Nday,pverp-rrtmg_levs+1:pver) = swhr (1:Nday,rrtmg_levs-1:1:-1)*cpair*dps
   qrsc(1:Nday,pverp-rrtmg_levs+1:pver) = swhrc(1:Nday,rrtmg_levs-1:1:-1)*cpair*dps
            ! Set spectral fluxes, reverse layering
            ! order=(/3,1,2/) maps the first index of swuflxs to the third index of su.
   if (associated(su)) then
      su(1:Nday,pverp-rrtmg_levs+1:pverp,:) = reshape(swuflxs(:,1:Nday,rrtmg_levs:1:-1), &
           (/Nday,rrtmg_levs,nbndsw/), order=(/3,1,2/))
   end if
   if (associated(sd)) then
      sd(1:Nday,pverp-rrtmg_levs+1:pverp,:) = reshape(swdflxs(:,1:Nday,rrtmg_levs:1:-1), &
           (/Nday,rrtmg_levs,nbndsw/), order=(/3,1,2/))
   end if
            ! Rearrange output arrays.
            !
            ! intent(out)
   call ExpDayNite(solin,	Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   call ExpDayNite(qrs,		Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, pver)
   call ExpDayNite(qrsc,	Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, pver)
   call ExpDayNite(fns,		Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, pverp)
   call ExpDayNite(fcns,	Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, pverp)
   call ExpDayNite(fsns,	Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   call ExpDayNite(fsnt,	Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   call ExpDayNite(fsntoa,	Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   call ExpDayNite(fsutoa,	Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   call ExpDayNite(fsds,	Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   call ExpDayNite(fsnsc,	Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   call ExpDayNite(fsdsc,	Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   call ExpDayNite(fsntc,	Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   call ExpDayNite(fsntoac,	Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   call ExpDayNite(sols,	Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   call ExpDayNite(soll,	Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   call ExpDayNite(solsd,	Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   call ExpDayNite(solld,	Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   call ExpDayNite(fsnirtoa,	Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   call ExpDayNite(fsnrtoac,	Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   call ExpDayNite(fsnrtoaq,	Nday, IdxDay, Nnite, IdxNite, 1, pcols)
   if (associated(su)) then
      call ExpDayNite(su,	Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, pverp, 1, nbndsw)
   end if
   if (associated(sd)) then
      call ExpDayNite(sd,	Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, pverp, 1, nbndsw)
   end if
            !  these outfld calls don't work for spmd only outfield in scm mode (nonspmd)
   if (single_column .and. scm_crm_mode) then 
                ! Following outputs added for CRM
      call ExpDayNite(fus,Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, pverp)
      call ExpDayNite(fds,Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, pverp)
      call ExpDayNite(fusc,Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, pverp)
      call ExpDayNite(fdsc,Nday, IdxDay, Nnite, IdxNite, 1, pcols, 1, pverp)
                !      call outfld('FUS     ',fus * 1.e-3_r8 ,pcols,lchnk)
                !      call outfld('FDS     ',fds * 1.e-3_r8 ,pcols,lchnk)
                !      call outfld('FUSC    ',fusc,pcols,lchnk)
                !      call outfld('FDSC    ',fdsc,pcols,lchnk)
   endif
        END SUBROUTINE rad_rrtmg_sw
        !-------------------------------------------------------------------------------

        !-------------------------------------------------------------------------------
    END MODULE radsw
