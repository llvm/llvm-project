!#define GENERATE_DRIVER




module wetdep

!----------------------------------------------------------------------- 
!
! Wet deposition routines for both aerosols and gas phase constituents.
! 
!-----------------------------------------------------------------------



use kinds_mod 
use params,    only: pcols, pver, gravit, rair, tmelt







implicit none
save
private

public :: wetdepa_v1  ! scavenging codes for very soluble aerosols -- CAM4 version
public :: wetdepa_v2  ! scavenging codes for very soluble aerosols -- CAM5 version
public :: wetdepg     ! scavenging of gas phase constituents by henry's law
public :: clddiag     ! calc of cloudy volume and rain mixing ratio

real(r8), parameter :: cmftau = 3600._r8
real(r8), parameter :: rhoh2o = 1000._r8            ! density of water
real(r8), parameter :: molwta = 28.97_r8            ! molecular weight dry air gm/mole

!==============================================================================
contains
!==============================================================================

subroutine clddiag(t, pmid, pdel, cmfdqr, evapc, &
                   cldt, cldcu, cldst, cme, evapr, &
                   prain, cldv, cldvcu, cldvst, rain, &
                   ncol)

   ! ------------------------------------------------------------------------------------ 
   ! Estimate the cloudy volume which is occupied by rain or cloud water as
   ! the max between the local cloud amount or the
   ! sum above of (cloud*positive precip production)      sum total precip from above
   !              ----------------------------------   x ------------------------
   ! sum above of     (positive precip           )        sum positive precip from above
   ! Author: P. Rasch
   !         Sungsu Park. Mar.2010 
   ! ------------------------------------------------------------------------------------

   ! Input arguments:
   real(r8), intent(in) :: t(pcols,pver)        ! temperature (K)
   real(r8), intent(in) :: pmid(pcols,pver)     ! pressure at layer midpoints
   real(r8), intent(in) :: pdel(pcols,pver)     ! pressure difference across layers
   real(r8), intent(in) :: cmfdqr(pcols,pver)   ! dq/dt due to convective rainout 
   real(r8), intent(in) :: evapc(pcols,pver)    ! Evaporation rate of convective precipitation ( >= 0 ) 
   real(r8), intent(in) :: cldt(pcols,pver)    ! total cloud fraction
   real(r8), intent(in) :: cldcu(pcols,pver)    ! Cumulus cloud fraction
   real(r8), intent(in) :: cldst(pcols,pver)    ! Stratus cloud fraction
   real(r8), intent(in) :: cme(pcols,pver)      ! rate of cond-evap within the cloud
   real(r8), intent(in) :: evapr(pcols,pver)    ! rate of evaporation of falling precipitation (kg/kg/s)
   real(r8), intent(in) :: prain(pcols,pver)    ! rate of conversion of condensate to precipitation (kg/kg/s)
   integer, intent(in) :: ncol

   ! Output arguments:
   real(r8), intent(out) :: cldv(pcols,pver)     ! fraction occupied by rain or cloud water 
   real(r8), intent(out) :: cldvcu(pcols,pver)   ! Convective precipitation volume
   real(r8), intent(out) :: cldvst(pcols,pver)   ! Stratiform precipitation volume
   real(r8), intent(out) :: rain(pcols,pver)     ! mixing ratio of rain (kg/kg)

   ! Local variables:
   integer  i, k
   real(r8) convfw         ! used in fallspeed calculation; taken from findmcnew
   real(r8) sumppr(pcols)        ! precipitation rate (kg/m2-s)
   real(r8) sumpppr(pcols)       ! sum of positive precips from above
   real(r8) cldv1(pcols)         ! precip weighted cloud fraction from above
   real(r8) lprec                ! local production rate of precip (kg/m2/s)
   real(r8) lprecp               ! local production rate of precip (kg/m2/s) if positive
   real(r8) rho                  ! air density
   real(r8) vfall
   real(r8) sumppr_cu(pcols)     ! Convective precipitation rate (kg/m2-s)
   real(r8) sumpppr_cu(pcols)    ! Sum of positive convective precips from above
   real(r8) cldv1_cu(pcols)      ! Convective precip weighted convective cloud fraction from above
   real(r8) lprec_cu             ! Local production rate of convective precip (kg/m2/s)
   real(r8) lprecp_cu            ! Local production rate of convective precip (kg/m2/s) if positive
   real(r8) sumppr_st(pcols)     ! Stratiform precipitation rate (kg/m2-s)
   real(r8) sumpppr_st(pcols)    ! Sum of positive stratiform precips from above
   real(r8) cldv1_st(pcols)      ! Stratiform precip weighted stratiform cloud fraction from above
   real(r8) lprec_st             ! Local production rate of stratiform precip (kg/m2/s)
   real(r8) lprecp_st            ! Local production rate of stratiform precip (kg/m2/s) if positive
   ! -----------------------------------------------------------------------

   convfw = 1.94_r8*2.13_r8*sqrt(rhoh2o*gravit*2.7e-4_r8)
   do i=1,ncol
      sumppr(i) = 0._r8
      cldv1(i) = 0._r8
      sumpppr(i) = 1.e-36_r8
      sumppr_cu(i)  = 0._r8
      cldv1_cu(i)   = 0._r8
      sumpppr_cu(i) = 1.e-36_r8
      sumppr_st(i)  = 0._r8
      cldv1_st(i)   = 0._r8
      sumpppr_st(i) = 1.e-36_r8
   end do

   do k = 1,pver
      do i = 1,ncol
         cldv(i,k) = &
            max(min(1._r8, &
            cldv1(i)/sumpppr(i) &
            )*sumppr(i)/sumpppr(i), &
            cldt(i,k) &
            )
         lprec = pdel(i,k)/gravit &
            *(prain(i,k)+cmfdqr(i,k)-evapr(i,k))
         lprecp = max(lprec,1.e-30_r8)
         cldv1(i) = cldv1(i)  + cldt(i,k)*lprecp
         sumppr(i) = sumppr(i) + lprec
         sumpppr(i) = sumpppr(i) + lprecp

         ! For convective precipitation volume at the top interface of each layer. Neglect the current layer.
         cldvcu(i,k)   = max(min(1._r8,cldv1_cu(i)/sumpppr_cu(i))*(sumppr_cu(i)/sumpppr_cu(i)),0._r8)
         lprec_cu      = (pdel(i,k)/gravit)*(cmfdqr(i,k)-evapc(i,k))
         lprecp_cu     = max(lprec_cu,1.e-30_r8)
         cldv1_cu(i)   = cldv1_cu(i) + cldcu(i,k)*lprecp_cu
         sumppr_cu(i)  = sumppr_cu(i) + lprec_cu
         sumpppr_cu(i) = sumpppr_cu(i) + lprecp_cu

         ! For stratiform precipitation volume at the top interface of each layer. Neglect the current layer.
         cldvst(i,k)   = max(min(1._r8,cldv1_st(i)/sumpppr_st(i))*(sumppr_st(i)/sumpppr_st(i)),0._r8)
         lprec_st      = (pdel(i,k)/gravit)*(prain(i,k)-evapr(i,k))
         lprecp_st     = max(lprec_st,1.e-30_r8)
         cldv1_st(i)   = cldv1_st(i) + cldst(i,k)*lprecp_st
         sumppr_st(i)  = sumppr_st(i) + lprec_st
         sumpppr_st(i) = sumpppr_st(i) + lprecp_st

         rain(i,k) = 0._r8
         if(t(i,k) .gt. tmelt) then
            rho = pmid(i,k)/(rair*t(i,k))
            vfall = convfw/sqrt(rho)
            rain(i,k) = sumppr(i)/(rho*vfall)
            if (rain(i,k).lt.1.e-14_r8) rain(i,k) = 0._r8
         endif
      end do
   end do

end subroutine clddiag

!==============================================================================

! This is the CAM5 version of wetdepa.

subroutine wetdepa_v2(t, p, q, pdel, &
                   cldt, cldc, cmfdqr, evapc, conicw, precs, conds, &
                       evaps, cwat, tracer, deltat, &
                       scavt, iscavt, cldv, cldvcu, cldvst, dlf, fracis, sol_fact, ncol, &
                       scavcoef, is_strat_cloudborne, rate1ord_cw2pr_st, qqcw, f_act_conv, &
                       icscavt, isscavt, bcscavt, bsscavt, &
                       sol_facti_in, sol_factbi_in, sol_factii_in, &
                       sol_factic_in, sol_factiic_in )

      !----------------------------------------------------------------------- 
      ! Purpose: 
      ! scavenging code for very soluble aerosols
      ! 
      ! Author: P. Rasch
      ! Modified by T. Bond 3/2003 to track different removals
      ! Sungsu Park. Mar.2010 : Impose consistencies with a few changes in physics.
      !-----------------------------------------------------------------------



      implicit none

      real(r8), intent(in) ::&
         t(pcols,pver),        &! temperature
         p(pcols,pver),        &! pressure
         q(pcols,pver),        &! moisture
         pdel(pcols,pver),     &! pressure thikness
         cldt(pcols,pver),    &! total cloud fraction
         cldc(pcols,pver),     &! convective cloud fraction
         cmfdqr(pcols,pver),   &! rate of production of convective precip
! Sungsu
         evapc(pcols,pver),    &! Evaporation rate of convective precipitation
! Sungsu
         conicw(pcols,pver),   &! convective cloud water
         cwat(pcols,pver),     &! cloud water amount 
         precs(pcols,pver),    &! rate of production of stratiform precip
         conds(pcols,pver),    &! rate of production of condensate
         evaps(pcols,pver),    &! rate of evaporation of precip
         cldv(pcols,pver),     &! total cloud fraction
! Sungsu
         cldvcu(pcols,pver),   &! Convective precipitation area at the top interface of each layer
         cldvst(pcols,pver),   &! Stratiform precipitation area at the top interface of each layer
         dlf(pcols,pver),      &! Detrainment of convective condensate [kg/kg/s]
! Sungsu
         deltat,               &! time step
         tracer(pcols,pver)     ! trace species

      ! If subroutine is called with just sol_fact:
            ! sol_fact is used for both in- and below-cloud scavenging
      ! If subroutine is called with optional argument sol_facti_in:
            ! sol_fact  is used for below cloud scavenging
            ! sol_facti is used for in cloud scavenging
         real(r8), intent(in) :: sol_fact ! solubility factor (fraction of aer scavenged below & in, or just below or sol_facti_in is provided)
         integer, intent(in) :: ncol
         real(r8), intent(in) :: scavcoef(pcols,pver) ! Dana and Hales coefficient (/mm) (0.1 if not MODAL_AERO)
         real(r8), intent(out) ::&
              scavt(pcols,pver),    &! scavenging tend 
              iscavt(pcols,pver),   &! incloud scavenging tends
              fracis(pcols,pver)     ! fraction of species not scavenged
         ! rce 2010/05/01
         ! is_strat_cloudborne = .true. if tracer is stratiform-cloudborne aerosol; else .false. 
         logical, intent(in), optional :: is_strat_cloudborne   
         ! rate1ord_cw2pr_st = 1st order rate for strat cw to precip (1/s) 
         real(r8), intent(in), optional  :: rate1ord_cw2pr_st(pcols,pver)
         ! qqcw = strat-cloudborne aerosol corresponding to tracer when is_strat_cloudborne==.false.; else 0.0 
         real(r8), intent(in), optional  :: qqcw(pcols,pver)
         ! f_act_conv = conv-cloud activation fraction when is_strat_cloudborne==.false.; else 0.0 
         real(r8), intent(in), optional  :: f_act_conv(pcols,pver)
         ! end rce 2010/05/01

         real(r8), intent(in), optional :: sol_facti_in   ! solubility factor (frac of aerosol scavenged in cloud)
         real(r8), intent(in), optional :: sol_factbi_in  ! solubility factor (frac of aerosol scavenged below cloud by ice)
         real(r8), intent(in), optional :: sol_factii_in  ! solubility factor (frac of aerosol scavenged in cloud by ice)
         real(r8), intent(in), optional :: sol_factic_in(pcols,pver)  ! sol_facti_in for convective clouds
         real(r8), intent(in), optional :: sol_factiic_in ! sol_factii_in for convective clouds
         

      real(r8), intent(out), optional ::    icscavt(pcols,pver)     ! incloud, convective
      real(r8), intent(out), optional ::    isscavt(pcols,pver)     ! incloud, stratiform
      real(r8), intent(out), optional ::    bcscavt(pcols,pver)     ! below cloud, convective
      real(r8), intent(out), optional ::    bsscavt(pcols,pver)     ! below cloud, stratiform



      ! local variables

      integer i                 ! x index
      integer k                 ! z index

      real(r8) adjfac               ! factor stolen from cmfmca
      real(r8) aqfrac               ! fraction of tracer in aqueous phase
      real(r8) cwatc                ! local convective total water amount 
      real(r8) cwats                ! local stratiform total water amount 
      real(r8) cwatp                ! local water amount falling from above precip
      real(r8) fracev(pcols)        ! fraction of precip from above that is evaporating
! Sungsu
      real(r8) fracev_cu(pcols)     ! Fraction of convective precip from above that is evaporating
! Sungsu
      real(r8) fracp(pcols)         ! fraction of cloud water converted to precip
      real(r8) gafrac               ! fraction of tracer in gas phasea
      real(r8) hconst               ! henry's law solubility constant when equation is expressed
                                ! in terms of mixing ratios
      real(r8) mpla                 ! moles / liter H2O entering the layer from above
      real(r8) mplb                 ! moles / liter H2O leaving the layer below
      real(r8) omsm                 ! 1 - (a small number)
      real(r8) part                 !  partial pressure of tracer in atmospheres
      real(r8) patm                 ! total pressure in atmospheres
      real(r8) pdog(pcols)          ! work variable (pdel/gravit)
      real(r8) rpdog(pcols)          ! work variable (gravit/pdel)
      real(r8) precabc(pcols)       ! conv precip from above (work array)
      real(r8) precabs(pcols)       ! strat precip from above (work array)
      real(r8) precbl               ! precip falling out of level (work array)
      real(r8) precmin              ! minimum convective precip causing scavenging
      real(r8) rat(pcols)           ! ratio of amount available to amount removed
      real(r8) scavab(pcols)        ! scavenged tracer flux from above (work array)
      real(r8) scavabc(pcols)       ! scavenged tracer flux from above (work array)
      real(r8) srcc(pcols)          ! tend for convective rain
      real(r8) srcs(pcols)          ! tend for stratiform rain
      real(r8) srct(pcols)          ! work variable
      real(r8) tracab(pcols)        ! column integrated tracer amount
!      real(r8) vfall                ! fall speed of precip
      real(r8) fins(pcols)          ! fraction of rem. rate by strat rain
      real(r8) finc(pcols)          ! fraction of rem. rate by conv. rain
      real(r8) srcs1(pcols)         ! work variable
      real(r8) srcs2(pcols)         ! work variable
      real(r8) tc(pcols)            ! temp in celcius
      real(r8) weight(pcols)        ! fraction of condensate which is ice
      real(r8) cldmabs(pcols)       ! maximum cloud at or above this level
      real(r8) cldmabc(pcols)       ! maximum cloud at or above this level
      real(r8) odds(pcols)          ! limit on removal rate (proportional to prec)
      real(r8) dblchek(pcols)
      logical :: found

    ! Jan.16.2009. Sungsu for wet scavenging below clouds.
    ! real(r8) cldovr_cu(pcols)     ! Convective precipitation area at the base of each layer
    ! real(r8) cldovr_st(pcols)     ! Stratiform precipitation area at the base of each layer

      real(r8) tracer_incu(pcols)
      real(r8) tracer_mean(pcols)

    ! End by Sungsu

      real(r8) sol_facti,  sol_factb  ! in cloud and below cloud fraction of aerosol scavenged
      real(r8) sol_factii, sol_factbi ! in cloud and below cloud fraction of aerosol scavenged by ice
      real(r8) sol_factic(pcols,pver)             ! sol_facti for convective clouds
      real(r8) sol_factiic            ! sol_factii for convective clouds
      ! sol_factic & solfact_iic added for MODAL_AERO.  
      ! For stratiform cloud, cloudborne aerosol is treated explicitly,
      !    and sol_facti is 1.0 for cloudborne, 0.0 for interstitial.
      ! For convective cloud, cloudborne aerosol is not treated explicitly,
      !    and sol_factic is 1.0 for both cloudborne and interstitial.
      real(r8) :: rdeltat



      ! ------------------------------------------------------------------------
!      omsm = 1.-1.e-10          ! used to prevent roundoff errors below zero
      omsm = 1._r8-2*epsilon(1._r8) ! used to prevent roundoff errors below zero
      precmin =  0.1_r8/8.64e4_r8      ! set critical value to 0.1 mm/day in kg/m2/s

      adjfac = deltat/(max(deltat,cmftau)) ! adjustment factor from hack scheme

      ! assume 4 m/s fall speed currently (should be improved)
!      vfall = 4.
	
      ! default (if other sol_facts aren't in call, set all to required sol_fact
      sol_facti = sol_fact
      sol_factb = sol_fact
      sol_factii = sol_fact
      sol_factbi = sol_fact

      if ( present(sol_facti_in) )  sol_facti = sol_facti_in
      if ( present(sol_factii_in) )  sol_factii = sol_factii_in
      if ( present(sol_factbi_in) )  sol_factbi = sol_factbi_in

      sol_factic  = sol_facti
      sol_factiic = sol_factii
      if ( present(sol_factic_in ) )  sol_factic  = sol_factic_in
      if ( present(sol_factiic_in) )  sol_factiic = sol_factiic_in

      ! this section of code is for highly soluble aerosols,
      ! the assumption is that within the cloud that
      ! all the tracer is in the cloud water
      !
      ! for both convective and stratiform clouds, 
      ! the fraction of cloud water converted to precip defines
      ! the amount of tracer which is pulled out.
      !

      do i = 1,pcols
         precabs(i) = 0
         precabc(i) = 0
         scavab(i) = 0
         scavabc(i) = 0
         tracab(i) = 0
         cldmabs(i) = 0
         cldmabc(i) = 0

       ! Jan.16. Sungsu 
       ! I added below to compute vertically projected cumulus and stratus fractions from the top to the
       ! current model layer by assuming a simple independent maximum overlapping assumption for 
       ! each cloud.
       ! cldovr_cu(i) = 0._r8
       ! cldovr_st(i) = 0._r8
       ! End by Sungsu

      end do

      do k = 1,pver
         do i = 1,ncol
            tc(i)     = t(i,k) - tmelt
            weight(i) = max(0._r8,min(-tc(i)*0.05_r8,1.0_r8)) ! fraction of condensate that is ice
            weight(i) = 0._r8                                 ! assume no ice

            pdog(i)  = pdel(i,k)/gravit
            rpdog(i) = gravit/pdel(i,k)
            rdeltat  = 1.0_r8/deltat

            ! ****************** Evaporation **************************
            ! calculate the fraction of strat precip from above 
            !                 which evaporates within this layer
            fracev(i) = evaps(i,k)*pdog(i) &
                     /max(1.e-12_r8,precabs(i))

            ! trap to ensure reasonable ratio bounds
            fracev(i) = max(0._r8,min(1._r8,fracev(i)))

! Sungsu : Same as above but convective precipitation part
            fracev_cu(i) = evapc(i,k)*pdog(i)/max(1.e-12_r8,precabc(i))
            fracev_cu(i) = max(0._r8,min(1._r8,fracev_cu(i)))
! Sungsu
            ! ****************** Convection ***************************
            ! now do the convective scavenging

            ! set odds proportional to fraction of the grid box that is swept by the 
            ! precipitation =precabc/rhoh20*(area of sphere projected on plane
            !                                /volume of sphere)*deltat
            ! assume the radius of a raindrop is 1 e-3 m from Rogers and Yau,
            ! unless the fraction of the area that is cloud is less than odds, in which
            ! case use the cloud fraction (assumes precabs is in kg/m2/s)
            ! is really: precabs*3/4/1000./1e-3*deltat
            ! here I use .1 from Balkanski
            !
            ! use a local rate of convective rain production for incloud scav
            !odds=max(min(1._r8, &
            !     cmfdqr(i,k)*pdel(i,k)/gravit*0.1_r8*deltat),0._r8)
            !++mcb -- change cldc to cldt; change cldt to cldv (9/17/96)
            !            srcs1 =  cldt(i,k)*odds*tracer(i,k)*(1.-weight) &
            ! srcs1 =  cldv(i,k)*odds*tracer(i,k)*(1.-weight) &
            !srcs1 =  cldc(i,k)*odds*tracer(i,k)*(1.-weight) &
            !         /deltat 

            ! fraction of convective cloud water converted to rain
            ! Dec.29.2009 : Sungsu multiplied cldc(i,k) to conicw(i,k) below
            ! fracp = cmfdqr(i,k)*deltat/max(1.e-8_r8,conicw(i,k))
            ! fracp = cmfdqr(i,k)*deltat/max(1.e-8_r8,cldc(i,k)*conicw(i,k))
            ! Sungsu: Below new formula of 'fracp' is necessary since 'conicw' is a LWC/IWC
            !         that has already precipitated out, that is, 'conicw' does not contain
            !         precipitation at all ! 
              fracp(i) = cmfdqr(i,k)*deltat/max(1.e-12_r8,cldc(i,k)*conicw(i,k)+(cmfdqr(i,k)+dlf(i,k))*deltat) ! Sungsu.Mar.19.2010.
            ! Dec.29.2009
            ! Note cmfdrq can be negative from evap of rain, so constrain it <-- This is wrong. cmfdqr does not
            ! contain evaporation of precipitation.
            fracp(i) = max(min(1._r8,fracp(i)),0._r8)

            !--mcb
            ! scavenge below cloud
            !            cldmabc(i) = max(cldc(i,k),cldmabc(i))
            !            cldmabc(i) = max(cldt(i,k),cldmabc(i))
            ! cldmabc(i) = max(cldv(i,k),cldmabc(i))
            ! cldmabc(i) = cldv(i,k)
            cldmabc(i) = cldvcu(i,k)

            ! Jan. 16. 2010. Sungsu
            ! cldmabc(i) = cldmabc(i) * cldovr_cu(i) / max( 0.01_r8, cldovr_cu(i) + cldovr_st(i) )
            ! End by Sungsu

        enddo
            ! remove that amount from within the convective area
!           srcs1 = cldc(i,k)*fracp*tracer(i,k)*(1._r8-weight)/deltat ! liquid only
!           srcs1 = cldc(i,k)*fracp*tracer(i,k)/deltat             ! any condensation
!           srcs1 = 0.
!           Jan.02.2010. Sungsu : cldt --> cldc below.
            ! rce 2010/05/01
            if (present(is_strat_cloudborne)) then  ! Tianyi, 2011/03/29
               if ( is_strat_cloudborne ) then
                  do i=1,ncol
                     ! only strat in-cloud removal affects strat-cloudborne aerosol
                     srcs1(i) = 0._r8
                     ! only strat in-cloud removal affects strat-cloudborne aerosol
                     srcs2(i) = 0._r8
                     !Note that using the temperature-determined weight doesn't make much sense here
                     srcc(i) = srcs1(i) + srcs2(i)  ! convective tend by both processes
                     finc(i) = srcs1(i)/(srcc(i) + 1.e-36_r8) ! fraction in-cloud
                     ! new code for stratiform incloud scav of cloudborne (modal) aerosol 
                     ! >> use the 1st order cw to precip rate calculated in microphysics routine
                     ! >> cloudborne aerosol resides in cloudy portion of grid cell, so do not apply "cldt" factor
                     ! fracp = rate1ord_cw2pr_st(i,k)*deltat
                     ! fracp = max(0._r8,min(1._r8,fracp))
                     fracp(i) = precs(i,k)*deltat/max(cwat(i,k)+precs(i,k)*deltat,1.e-12_r8) ! Sungsu. Mar.19.2010.
                     fracp(i) = max(0._r8,min(1._r8,fracp(i)))
                     srcs1(i) = sol_facti *fracp(i)*tracer(i,k)*rdeltat*(1._r8-weight(i)) &  ! Liquid
                              + sol_factii*fracp(i)*tracer(i,k)*rdeltat*(weight(i))          ! Ice
                     ! only strat in-cloud removal affects strat-cloudborne aerosol
                     srcs2(i) = 0._r8
                  enddo
               else
                  do i=1,ncol
                     tracer_incu(i) = f_act_conv(i,k)*(tracer(i,k)+& 
                           min(qqcw(i,k),tracer(i,k)*((cldt(i,k)-cldc(i,k))/max(0.01_r8,(1._r8-(cldt(i,k)-cldc(i,k)))))))              
                     srcs1(i) = sol_factic(i,k)*cldc(i,k)*fracp(i)*tracer_incu(i)*(1._r8-weight(i))*rdeltat &  ! Liquid
                              + sol_factiic    *cldc(i,k)*fracp(i)*tracer_incu(i)*(weight(i))*rdeltat          ! Ice

                     tracer_mean(i) = tracer(i,k)*(1._r8-cldc(i,k)*f_act_conv(i,k))-cldc(i,k)*f_act_conv(i,k)*&
                           min(qqcw(i,k),tracer(i,k)*((cldt(i,k)-cldc(i,k))/max(0.01_r8,(1._r8-(cldt(i,k)-cldc(i,k))))))
                     tracer_mean(i) = max(0._r8,tracer_mean(i)) 
                     odds(i)  = max(min(1._r8,precabc(i)/max(cldmabc(i),1.e-5_r8)*scavcoef(i,k)*deltat),0._r8) ! Dana and Hales coefficient (/mm)

                     srcs2(i) = sol_factb *cldmabc(i)*odds(i)*tracer_mean(i)*(1._r8-weight(i))*rdeltat & ! Liquid
                              + sol_factbi*cldmabc(i)*odds(i)*tracer_mean(i)*(weight(i))*rdeltat         ! Ice
                     !Note that using the temperature-determined weight doesn't make much sense here
                     srcc(i) = srcs1(i) + srcs2(i)  ! convective tend by both processes
                     finc(i) = srcs1(i)/(srcc(i) + 1.e-36_r8) ! fraction in-cloud
                     ! strat in-cloud removal only affects strat-cloudborne aerosol
                     srcs1(i) = 0._r8
                     odds(i) = precabs(i)/max(cldvst(i,k),1.e-5_r8)*scavcoef(i,k)*deltat
                     odds(i) = max(min(1._r8,odds(i)),0._r8)

                     srcs2(i) = sol_factb *cldvst(i,k)*odds(i)*tracer_mean(i)*(1._r8-weight(i))*rdeltat & ! Liquid
                              + sol_factbi*cldvst(i,k)*odds(i)*tracer_mean(i)*(weight(i))*rdeltat         ! Ice
                  enddo
               end if
            else
               do i=1,ncol
                  srcs1(i) = sol_factic(i,k)*cldc(i,k)*fracp(i)*tracer(i,k)*(1._r8-weight(i))*rdeltat &  ! liquid
                           +  sol_factiic*cldc(i,k)*fracp(i)*tracer(i,k)*(weight(i))*rdeltat      ! ice
                   odds(i) = max( &
                             min(1._r8,precabc(i)/max(cldmabc(i),1.e-5_r8) &
                           * scavcoef(i,k)*deltat),0._r8) ! Dana and Hales coefficient (/mm)
                   srcs2(i) = sol_factb*cldmabc(i)*odds(i)*tracer(i,k)*(1._r8-weight(i))*rdeltat & ! liquid
                            +  sol_factbi*cldmabc(i)*odds(i)*tracer(i,k)*(weight(i))*rdeltat    !ice
                   !Note that using the temperature-determined weight doesn't make much sense here
                   srcc(i) = srcs1(i) + srcs2(i)  ! convective tend by both processes
                   finc(i) = srcs1(i)/(srcc(i) + 1.e-36_r8) ! fraction in-cloud
                   ! fracp is the fraction of cloud water converted to precip
                   ! Sungsu modified fracp as the convectiv case.
                   !        Below new formula by Sungsu of 'fracp' is necessary since 'cwat' is a LWC/IWC
                   !        that has already precipitated out, that is, 'cwat' does not contain
                   !        precipitation at all ! 
                   !            fracp =  precs(i,k)*deltat/max(cwat(i,k),1.e-12_r8)
                   fracp(i) =  precs(i,k)*deltat/max(cwat(i,k)+precs(i,k)*deltat,1.e-12_r8) ! Sungsu. Mar.19.2010.
                   fracp(i) = max(0._r8,min(1._r8,fracp(i)))
                   !            fracp = 0.     ! for debug
                   ! assume the corresponding amnt of tracer is removed
                   !++mcb -- remove cldc; change cldt to cldv 
                   !            srcs1 = (cldt(i,k)-cldc(i,k))*fracp*tracer(i,k)/deltat
                   !            srcs1 = cldv(i,k)*fracp*tracer(i,k)/deltat &
                   !            srcs1 = cldt(i,k)*fracp*tracer(i,k)/deltat            ! all condensate
                   !            Jan.02.2010. Sungsu : cldt --> cldt - cldc below.
                   srcs1(i) = sol_facti*(cldt(i,k)-cldc(i,k))*fracp(i)*tracer(i,k)*rdeltat*(1._r8-weight(i)) &  ! liquid
                            + sol_factii*(cldt(i,k)-cldc(i,k))*fracp(i)*tracer(i,k)*rdeltat*(weight(i))       ! ice

                   odds(i) = precabs(i)/max(cldvst(i,k),1.e-5_r8)*scavcoef(i,k)*deltat
                   odds(i) = max(min(1._r8,odds(i)),0._r8)
                   srcs2  = sol_factb*(cldvst(i,k)*odds(i)) *tracer(i,k)*(1._r8-weight(i))*rdeltat & ! liquid
                          + sol_factbi*(cldvst(i,k)*odds(i)) *tracer(i,k)*(weight(i))*rdeltat       ! ice
               enddo
            end if

          do i=1,ncol

            !Note that using the temperature-determined weight doesn't make much sense here

            srcs(i) = srcs1(i) + srcs2(i)             ! total stratiform scavenging
            fins(i) = srcs1(i)/(srcs(i) + 1.e-36_r8)    ! fraction taken by incloud processes

            ! make sure we dont take out more than is there
            ! ratio of amount available to amount removed
            rat(i) = tracer(i,k)/max(deltat*(srcc(i)+srcs(i)),1.e-36_r8)
            if (rat(i).lt.1._r8) then
               srcs(i) = srcs(i)*rat(i)
               srcc(i) = srcc(i)*rat(i)
            endif
            srct(i) = (srcc(i)+srcs(i))*omsm

            
            ! fraction that is not removed within the cloud
            ! (assumed to be interstitial, and subject to convective transport)
            fracp(i) = deltat*srct(i)/max(cldvst(i,k)*tracer(i,k),1.e-36_r8)  ! amount removed
            fracp(i) = max(0._r8,min(1._r8,fracp(i)))
            fracis(i,k) = 1._r8 - fracp(i)

            ! tend is all tracer removed by scavenging, plus all re-appearing from evaporation above
            ! Sungsu added cumulus contribution in the below 3 blocks
         
            scavt(i,k) = -srct(i) + (fracev(i)*scavab(i)+fracev_cu(i)*scavabc(i))*rpdog(i)
            iscavt(i,k) = -(srcc(i)*finc(i) + srcs(i)*fins(i))*omsm

            if ( present(icscavt) ) icscavt(i,k) = -(srcc(i)*finc(i)) * omsm
            if ( present(isscavt) ) isscavt(i,k) = -(srcs(i)*fins(i)) * omsm
            if ( present(bcscavt) ) bcscavt(i,k) = -(srcc(i) * (1-finc(i))) * omsm +  &
                 fracev_cu(i)*scavabc(i)*rpdog(i)
            if ( present(bsscavt) ) bsscavt(i,k) = -(srcs(i) * (1-fins(i))) * omsm +  &
                 fracev(i)*scavab(i)*rpdog(i)

            dblchek(i) = tracer(i,k) + deltat*scavt(i,k)

            ! now keep track of scavenged mass and precip

            scavab(i) = scavab(i)*(1-fracev(i)) + srcs(i)*pdog(i)
            precabs(i) = precabs(i) + (precs(i,k) - evaps(i,k))*pdog(i)
            scavabc(i) = scavabc(i)*(1-fracev_cu(i)) + srcc(i)*pdog(i)
            precabc(i) = precabc(i) + (cmfdqr(i,k) - evapc(i,k))*pdog(i)
            tracab(i) = tracab(i) + tracer(i,k)*pdog(i)


       ! Jan.16.2010. Sungsu
       ! Compute convective and stratiform precipitation areas at the base interface
       ! of current layer. These are for computing 'below cloud scavenging' in the 
       ! next layer below.

       ! cldovr_cu(i) = max( cldovr_cu(i), cldc(i,k) )
       ! cldovr_st(i) = max( cldovr_st(i), max( 0._r8, cldt(i,k) - cldc(i,k) ) )

       ! cldovr_cu(i) = max( 0._r8, min ( 1._r8, cldovr_cu(i) ) )
       ! cldovr_st(i) = max( 0._r8, min ( 1._r8, cldovr_st(i) ) )

       ! End by Sungsu

         end do ! End of i = 1, ncol

         found = .false.
         do i = 1,ncol
            if ( dblchek(i) < 0._r8 ) then
               found = .true.
               exit
            end if
         end do

         if ( found ) then
            do i = 1,ncol
               if (dblchek(i) .lt. 0._r8) then
                  write(*,*) ' wetdapa: negative value ', i, k, tracer(i,k), &
                       dblchek(i), scavt(i,k), srct(i), rat(i), fracev(i)
               endif
            end do
         endif

      end do ! End of k = 1, pver


   end subroutine wetdepa_v2


!==============================================================================

! This is the frozen CAM4 version of wetdepa.


   subroutine wetdepa_v1( t, p, q, pdel, &
                       cldt, cldc, cmfdqr, conicw, precs, conds, &
                       evaps, cwat, tracer, deltat, &
                       scavt, iscavt, cldv, fracis, sol_fact, ncol, &
                       scavcoef,icscavt, isscavt, bcscavt, bsscavt, &
                       sol_facti_in, sol_factbi_in, sol_factii_in, &
                       sol_factic_in, sol_factiic_in )

      !----------------------------------------------------------------------- 
      ! Purpose: 
      ! scavenging code for very soluble aerosols
      ! 
      ! Author: P. Rasch
      ! Modified by T. Bond 3/2003 to track different removals
      !-----------------------------------------------------------------------

      implicit none

      real(r8), intent(in) ::&
         t(pcols,pver),        &! temperature
         p(pcols,pver),        &! pressure
         q(pcols,pver),        &! moisture
         pdel(pcols,pver),     &! pressure thikness
         cldt(pcols,pver),    &! total cloud fraction
         cldc(pcols,pver),     &! convective cloud fraction
         cmfdqr(pcols,pver),   &! rate of production of convective precip
         conicw(pcols,pver),   &! convective cloud water
         cwat(pcols,pver),     &! cloud water amount 
         precs(pcols,pver),    &! rate of production of stratiform precip
         conds(pcols,pver),    &! rate of production of condensate
         evaps(pcols,pver),    &! rate of evaporation of precip
         cldv(pcols,pver),     &! total cloud fraction
         deltat,               &! time step
         tracer(pcols,pver)     ! trace species
      ! If subroutine is called with just sol_fact:
            ! sol_fact is used for both in- and below-cloud scavenging
      ! If subroutine is called with optional argument sol_facti_in:
            ! sol_fact  is used for below cloud scavenging
            ! sol_facti is used for in cloud scavenging
         real(r8), intent(in) :: sol_fact ! solubility factor (fraction of aer scavenged below & in, or just below or sol_facti_in is provided)
         real(r8), intent(in), optional :: sol_facti_in   ! solubility factor (frac of aerosol scavenged in cloud)
         real(r8), intent(in), optional :: sol_factbi_in  ! solubility factor (frac of aerosol scavenged below cloud by ice)
         real(r8), intent(in), optional :: sol_factii_in  ! solubility factor (frac of aerosol scavenged in cloud by ice)
         real(r8), intent(in), optional :: sol_factic_in(pcols,pver)  ! sol_facti_in for convective clouds
         real(r8), intent(in), optional :: sol_factiic_in ! sol_factii_in for convective clouds
         real(r8), intent(in) :: scavcoef(pcols,pver) ! Dana and Hales coefficient (/mm) (0.1 if not MODAL_AERO)
         
      integer, intent(in) :: ncol

      real(r8), intent(out) ::&
         scavt(pcols,pver),    &! scavenging tend 
         iscavt(pcols,pver),   &! incloud scavenging tends
         fracis(pcols,pver)     ! fraction of species not scavenged

      real(r8), intent(out), optional ::    icscavt(pcols,pver)     ! incloud, convective
      real(r8), intent(out), optional ::    isscavt(pcols,pver)     ! incloud, stratiform
      real(r8), intent(out), optional ::    bcscavt(pcols,pver)     ! below cloud, convective
      real(r8), intent(out), optional ::    bsscavt(pcols,pver)     ! below cloud, stratiform

      ! local variables

      integer i                 ! x index
      integer k                 ! z index

      real(r8) adjfac               ! factor stolen from cmfmca
      real(r8) aqfrac               ! fraction of tracer in aqueous phase
      real(r8) cwatc                ! local convective total water amount 
      real(r8) cwats                ! local stratiform total water amount 
      real(r8) cwatp                ! local water amount falling from above precip
      real(r8) fracev(pcols)        ! fraction of precip from above that is evaporating
      real(r8) fracp                ! fraction of cloud water converted to precip
      real(r8) gafrac               ! fraction of tracer in gas phasea
      real(r8) hconst               ! henry's law solubility constant when equation is expressed
                                ! in terms of mixing ratios
      real(r8) mpla                 ! moles / liter H2O entering the layer from above
      real(r8) mplb                 ! moles / liter H2O leaving the layer below
      real(r8) omsm                 ! 1 - (a small number)
      real(r8) part                 !  partial pressure of tracer in atmospheres
      real(r8) patm                 ! total pressure in atmospheres
      real(r8) pdog                 ! work variable (pdel/gravit)
      real(r8) precabc(pcols)       ! conv precip from above (work array)
      real(r8) precabs(pcols)       ! strat precip from above (work array)
      real(r8) precbl               ! precip falling out of level (work array)
      real(r8) precmin              ! minimum convective precip causing scavenging
      real(r8) rat(pcols)           ! ratio of amount available to amount removed
      real(r8) scavab(pcols)        ! scavenged tracer flux from above (work array)
      real(r8) scavabc(pcols)       ! scavenged tracer flux from above (work array)
      real(r8) srcc                 ! tend for convective rain
      real(r8) srcs                 ! tend for stratiform rain
      real(r8) srct(pcols)          ! work variable
      real(r8) tracab(pcols)        ! column integrated tracer amount
!      real(r8) vfall                ! fall speed of precip
      real(r8) fins                 ! fraction of rem. rate by strat rain
      real(r8) finc                 ! fraction of rem. rate by conv. rain
      real(r8) srcs1                ! work variable
      real(r8) srcs2                ! work variable
      real(r8) tc                   ! temp in celcius
      real(r8) weight               ! fraction of condensate which is ice
      real(r8) cldmabs(pcols)       ! maximum cloud at or above this level
      real(r8) cldmabc(pcols)       ! maximum cloud at or above this level
      real(r8) odds                 ! limit on removal rate (proportional to prec)
      real(r8) dblchek(pcols)
      logical :: found

      real(r8) sol_facti,  sol_factb  ! in cloud and below cloud fraction of aerosol scavenged
      real(r8) sol_factii, sol_factbi ! in cloud and below cloud fraction of aerosol scavenged by ice
      real(r8) sol_factic(pcols,pver)             ! sol_facti for convective clouds
      real(r8) sol_factiic            ! sol_factii for convective clouds
      ! sol_factic & solfact_iic added for MODAL_AERO.  
      ! For stratiform cloud, cloudborne aerosol is treated explicitly,
      !    and sol_facti is 1.0 for cloudborne, 0.0 for interstitial.
      ! For convective cloud, cloudborne aerosol is not treated explicitly,
      !    and sol_factic is 1.0 for both cloudborne and interstitial.

      ! ------------------------------------------------------------------------
!      omsm = 1.-1.e-10          ! used to prevent roundoff errors below zero
      omsm = 1._r8-2*epsilon(1._r8) ! used to prevent roundoff errors below zero
      precmin =  0.1_r8/8.64e4_r8      ! set critical value to 0.1 mm/day in kg/m2/s

      adjfac = deltat/(max(deltat,cmftau)) ! adjustment factor from hack scheme

      ! assume 4 m/s fall speed currently (should be improved)
!      vfall = 4.
	
      ! default (if other sol_facts aren't in call, set all to required sol_fact
      sol_facti = sol_fact
      sol_factb = sol_fact
      sol_factii = sol_fact
      sol_factbi = sol_fact

      if ( present(sol_facti_in) )  sol_facti = sol_facti_in
      if ( present(sol_factii_in) )  sol_factii = sol_factii_in
      if ( present(sol_factbi_in) )  sol_factbi = sol_factbi_in

      sol_factic  = sol_facti
      sol_factiic = sol_factii
      if ( present(sol_factic_in ) )  sol_factic  = sol_factic_in
      if ( present(sol_factiic_in) )  sol_factiic = sol_factiic_in

      ! this section of code is for highly soluble aerosols,
      ! the assumption is that within the cloud that
      ! all the tracer is in the cloud water
      !
      ! for both convective and stratiform clouds, 
      ! the fraction of cloud water converted to precip defines
      ! the amount of tracer which is pulled out.
      !


   end subroutine wetdepa_v1

!==============================================================================

! wetdepg is currently being used for both CAM4 and CAM5 by making use of the
! cam_physpkg_is method.

   subroutine wetdepg( t, p, q, pdel, &
                       cldt, cldc, cmfdqr, evapc, precs, evaps, &
                       rain, cwat, tracer, deltat, molwt, &
                       solconst, scavt, iscavt, cldv, icwmr1, &
                       icwmr2, fracis, ncol )

      !----------------------------------------------------------------------- 
      ! Purpose: 
      ! scavenging of gas phase constituents by henry's law
      ! 
      ! Author: P. Rasch
      !-----------------------------------------------------------------------

      real(r8), intent(in) ::&
         t(pcols,pver),        &! temperature
         p(pcols,pver),        &! pressure
         q(pcols,pver),        &! moisture
         pdel(pcols,pver),     &! pressure thikness
         cldt(pcols,pver),     &! total cloud fraction
         cldc(pcols,pver),     &! convective cloud fraction
         cmfdqr(pcols,pver),   &! rate of production of convective precip
         rain (pcols,pver),    &! total rainwater mixing ratio
         cwat(pcols,pver),     &! cloud water amount 
         precs(pcols,pver),    &! rate of production of stratiform precip
         evaps(pcols,pver),    &! rate of evaporation of precip
! Sungsu
         evapc(pcols,pver),    &! Rate of evaporation of convective precipitation
! Sungsu 
         cldv(pcols,pver),     &! estimate of local volume occupied by clouds
         icwmr1 (pcols,pver),  &! in cloud water mixing ration for zhang scheme
         icwmr2 (pcols,pver),  &! in cloud water mixing ration for hack  scheme
         deltat,               &! time step
         tracer(pcols,pver),   &! trace species
         molwt                  ! molecular weights

      integer, intent(in) :: ncol

      real(r8) &
         solconst(pcols,pver)   ! Henry's law coefficient

      real(r8), intent(out) ::&
         scavt(pcols,pver),    &! scavenging tend 
         iscavt(pcols,pver),   &! incloud scavenging tends
         fracis(pcols, pver)    ! fraction of constituent that is insoluble

      ! local variables

      integer i                 ! x index
      integer k                 ! z index

      real(r8) adjfac               ! factor stolen from cmfmca
      real(r8) aqfrac               ! fraction of tracer in aqueous phase
      real(r8) cwatc                ! local convective total water amount 
      real(r8) cwats                ! local stratiform total water amount 
      real(r8) cwatl                ! local cloud liq water amount 
      real(r8) cwatp                ! local water amount falling from above precip
      real(r8) cwatpl               ! local water amount falling from above precip (liq)
      real(r8) cwatt                ! local sum of strat + conv total water amount 
      real(r8) cwatti               ! cwatt/cldv = cloudy grid volume mixing ratio
      real(r8) fracev               ! fraction of precip from above that is evaporating
      real(r8) fracp                ! fraction of cloud water converted to precip
      real(r8) gafrac               ! fraction of tracer in gas phasea
      real(r8) hconst               ! henry's law solubility constant when equation is expressed
                                ! in terms of mixing ratios
      real(r8) mpla                 ! moles / liter H2O entering the layer from above
      real(r8) mplb                 ! moles / liter H2O leaving the layer below
      real(r8) omsm                 ! 1 - (a small number)
      real(r8) part                 !  partial pressure of tracer in atmospheres
      real(r8) patm                 ! total pressure in atmospheres
      real(r8) pdog                 ! work variable (pdel/gravit)
      real(r8) precab(pcols)        ! precip from above (work array)
      real(r8) precbl               ! precip work variable
      real(r8) precxx               ! precip work variable
      real(r8) precxx2               !
      real(r8) precic               ! precip work variable
      real(r8) rat                  ! ratio of amount available to amount removed
      real(r8) scavab(pcols)        ! scavenged tracer flux from above (work array)
      real(r8) scavabc(pcols)       ! scavenged tracer flux from above (work array)
      !      real(r8) vfall                ! fall speed of precip
      real(r8) scavmax              ! an estimate of the max tracer avail for removal
      real(r8) scavbl               ! flux removed at bottom of layer
      real(r8) fins                 ! in cloud fraction removed by strat rain
      real(r8) finc                 ! in cloud fraction removed by conv rain
      real(r8) rate                 ! max removal rate estimate
      real(r8) scavlimt             ! limiting value 1
      real(r8) scavt1               ! limiting value 2
      real(r8) scavin               ! scavenging by incloud processes
      real(r8) scavbc               ! scavenging by below cloud processes
      real(r8) tc
      real(r8) weight               ! ice fraction
      real(r8) wtpl                 ! work variable
      real(r8) cldmabs(pcols)       ! maximum cloud at or above this level
      real(r8) cldmabc(pcols)       ! maximum cloud at or above this level
      !-----------------------------------------------------------

      omsm = 1._r8-2*epsilon(1._r8)   ! used to prevent roundoff errors below zero

      adjfac = deltat/(max(deltat,cmftau)) ! adjustment factor from hack scheme

      ! assume 4 m/s fall speed currently (should be improved)
      !      vfall = 4.

      ! zero accumulators
      do i = 1,pcols
         precab(i) = 1.e-36_r8
         scavab(i) = 0._r8
         cldmabs(i) = 0._r8
      end do

      do k = 1,pver
         do i = 1,ncol

            tc     = t(i,k) - tmelt
            weight = max(0._r8,min(-tc*0.05_r8,1.0_r8)) ! fraction of condensate that is ice

            cldmabs(i) = max(cldmabs(i),cldt(i,k))

            ! partitioning coefs for gas and aqueous phase
            !              take as a cloud water amount, the sum of the stratiform amount
            !              plus the convective rain water amount 

            ! convective amnt is just the local precip rate from the hack scheme
            !              since there is no storage of water, this ignores that falling from above
            !            cwatc = cmfdqr(i,k)*deltat/adjfac
            !++mcb -- test cwatc
            cwatc = (icwmr1(i,k) + icwmr2(i,k)) * (1._r8-weight)
            !--mcb 

            ! strat cloud water amount and also ignore the part falling from above
            cwats = cwat(i,k) 

            ! cloud water as liq
            !++mcb -- add cwatc later (in cwatti)
            !            cwatl = (1.-weight)*(cwatc+cwats)
            cwatl = (1._r8-weight)*cwats
            ! cloud water as ice
            !*not used        cwati = weight*(cwatc+cwats)

            ! total suspended condensate as liquid
            cwatt = cwatl + rain(i,k)

            ! incloud version 
            !++mcb -- add cwatc here
            cwatti = cwatt/max(cldv(i,k), 0.00001_r8) + cwatc

            ! partitioning terms
            patm = p(i,k)/1.013e5_r8 ! pressure in atmospheres
            hconst = molwta*patm*solconst(i,k)*cwatti/rhoh2o
            aqfrac = hconst/(1._r8+hconst)
            gafrac = 1/(1._r8+hconst)
            fracis(i,k) = gafrac


            ! partial pressure of the tracer in the gridbox in atmospheres
            part = patm*gafrac*tracer(i,k)*molwta/molwt

            ! use henrys law to give moles tracer /liter of water
            ! in this volume 
            ! then convert to kg tracer /liter of water (kg tracer / kg water)
            mplb = solconst(i,k)*part*molwt/1000._r8


            pdog = pdel(i,k)/gravit

            ! this part of precip will be carried downward but at a new molarity of mpl 
            precic = pdog*(precs(i,k) + cmfdqr(i,k))

            ! we cant take out more than entered, plus that available in the cloud
            !                  scavmax = scavab(i)+tracer(i,k)*cldt(i,k)/deltat*pdog
            scavmax = scavab(i)+tracer(i,k)*cldv(i,k)/deltat*pdog

            ! flux of tracer by incloud processes
            scavin = precic*(1._r8-weight)*mplb

            ! fraction of precip which entered above that leaves below
            if (.TRUE.) then
               ! Sungsu added evaporation of convective precipitation below.
               precxx = precab(i)-pdog*(evaps(i,k)+evapc(i,k))
            else
               precxx = precab(i)-pdog*evaps(i,k)
            end if
            precxx = max (precxx,0.0_r8)

            ! flux of tracer by below cloud processes
            !++mcb -- removed wtpl because it is now not assigned and previously
            !          when it was assigned it was unnecessary:  if(tc.gt.0)wtpl=1
            if (tc.gt.0) then
               !               scavbc = precxx*wtpl*mplb ! if liquid
               scavbc = precxx*mplb ! if liquid
            else
               precxx2=max(precxx,1.e-36_r8)
               scavbc = scavab(i)*precxx2/(precab(i)) ! if ice
            endif

            scavbl = min(scavbc + scavin, scavmax)

            ! first guess assuming that henries law works
            scavt1 = (scavab(i)-scavbl)/pdog*omsm

            ! pjr this should not be required, but we put it in to make sure we cant remove too much
            ! remember, scavt1 is generally negative (indicating removal)
            scavt1 = max(scavt1,-tracer(i,k)*cldv(i,k)/deltat)

            !++mcb -- remove this limitation for gas species
            !c use the dana and hales or balkanski limit on scavenging
            !c            rate = precab(i)*0.1
            !            rate = (precic + precxx)*0.1
            !            scavlimt = -tracer(i,k)*cldv(i,k)
            !     $           *rate/(1.+rate*deltat)

            !            scavt(i,k) = max(scavt1, scavlimt)

            ! instead just set scavt to scavt1
            scavt(i,k) = scavt1
            !--mcb

            ! now update the amount leaving the layer
            scavbl = scavab(i) - scavt(i,k)*pdog 

            ! in cloud amount is that formed locally over the total flux out bottom
            fins = scavin/(scavin + scavbc + 1.e-36_r8)
            iscavt(i,k) = scavt(i,k)*fins

            scavab(i) = scavbl
            precab(i) = max(precxx + precic,1.e-36_r8)

        
            
         end do
      end do
      
   end subroutine wetdepg

!##############################################################################

end module wetdep
