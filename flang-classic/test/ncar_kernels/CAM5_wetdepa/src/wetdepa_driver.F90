 ! Generating file: wetdepa_v2.spo
 program wetdepa_v2_driver

 use wetdep

 implicit none
 integer :: i,j,k,n1,n2,n3
 integer :: it
 integer, parameter :: i4 = selected_int_kind ( 6)  ! 4 byte integer
 integer, parameter :: r4 = selected_real_kind ( 6) ! 4 byte real
 integer, parameter :: r8 = selected_real_kind (12) ! 8 byte real
 integer(i4) :: val1_i4,val2_i4
 real(r4) :: val1_r4,val2_r4
 real(r8) :: val1_r8,val2_r8, rel_r8
 real(r8), parameter :: eps = 1.E-14
  real(r8), parameter :: Infinity_t = 290.00_r8
  real(r8), parameter :: Infinity_p = 53174.1653037401_r8
  real(r8), parameter :: Infinity_q = 1.092586539789276E-002
  real(r8), parameter :: Infinity_pdel = 2318.55362653732_r8
  real(r8), parameter :: Underflow = 0.0
  logical :: errorDetected
  real(r8)  start_time, stop_time
  integer :: start_clock,stop_clock,rate_clock

  real(r8), dimension(          16 ,          30 ) :: t
!DIR$ ATTRIBUTES ALIGN: 64 :: t
  real(r8), dimension(          16 ,          30 ) :: p
  real(r8), dimension(          16 ,          30 ) :: q
  real(r8), dimension(          16 ,          30 ) :: pdel
  real(r8), dimension(          16 ,          30 ) :: cldt
  real(r8), dimension(          16 ,          30 ) :: cldc
  real(r8), dimension(          16 ,          30 ) :: cmfdqr
  real(r8), dimension(          16 ,          30 ) :: evapc
  real(r8), dimension(          16 ,          30 ) :: conicw
  real(r8), dimension(          16 ,          30 ) :: cwat
  real(r8), dimension(          16 ,          30 ) :: precs
  real(r8), dimension(          16 ,          30 ) :: conds
  real(r8), dimension(          16 ,          30 ) :: evaps
  real(r8), dimension(          16 ,          30 ) :: cldv
  real(r8), dimension(          16 ,          30 ) :: cldvcu
  real(r8), dimension(          16 ,          30 ) :: cldvst
  real(r8), dimension(          16 ,          30 ) :: dlf
  real(r8) :: deltat
  real(r8), dimension(          16 ,          30 ) :: tracer
  real(r8) :: sol_fact
  real(r8), dimension(          16 ,          30 ) :: scavcoef
  real(r8), dimension(          16 ,          30 ) :: rate1ord_cw2pr_st
  real(r8), dimension(          16 ,          30 ) :: qqcw
  real(r8), dimension(          16 ,          30 ) :: f_act_conv
  real(r8) :: sol_facti_in
  real(r8) :: sol_factbi_in
  real(r8) :: sol_factii_in
  real(r8), dimension(          16 ,          30 ) :: sol_factic_in
  real(r8) :: sol_factiic_in
 logical :: is_strat_cloudborne
  
   integer, parameter :: ntrials = 10000

  real(r8), dimension(          16 ,          30 ) :: scavt, scavt_out
  real(r8), dimension(          16 ,          30 ) :: iscavt, iscavt_out
  real(r8), dimension(          16 ,          30 ) :: fracis, fracis_out
  real(r8), dimension(          16 ,          30 ) :: icscavt, icscavt_out
  real(r8), dimension(          16 ,          30 ) :: isscavt, isscavt_out
  real(r8), dimension(          16 ,          30 ) :: bcscavt, bcscavt_out
  real(r8), dimension(          16 ,          30 ) :: bsscavt, bsscavt_out
 integer(i4) :: ncol


 t(           : ,           : )=   249.034386263986_r8     
 p(           : ,           : )=   364.346569404006_r8     
 q(           : ,           : )=  2.461868225941993E-006
 pdel(        : ,           : )=   277.645234018564_r8     
 cldt(        : ,           : )=  0.626255763599366_r8     
 cldc(        : ,           : )=  5.880468503166033E-004
 cmfdqr(      : ,          : )=  1.241832531064138E-009
 evapc(       : ,          : )=  1.060404526009187E-009
 conicw(      :  ,         : )=  5.185935053792856E-004
 cwat(        : ,          : )=  5.877465715111163E-012
 precs(       :  ,         : )=  1.085056588888535E-008
 conds(       : ,          : )= -1.292209588098710E-009
 evaps(       : ,          : )=  1.317921505262640E-008
 cldv(        : ,          : )=  0.989423625165677_r8     
 cldvcu(      : ,          : )=  0.226541172855994_r8     
 cldvst(      :,           : )=  0.961717478206716_r8     
 dlf(         : ,          : )=  1.344445793338103E-007

   errorDetected = .false.
 ! real(r8) :: deltat
 deltat =    1800.00000000000     
 ! real(r8), dimension(          16 ,          30 ) :: tracer
 tracer(           : ,           : )=   6067770.36711884_r8     

 sol_fact =   0.100000000000000     
 ! integer(i4) :: ncol
 ncol =           14
 scavcoef(           : ,          : )=  1.024901244576826E-003

 is_strat_cloudborne = .FALSE.
 ! real(r8), dimension(          16 ,          30 ) :: rate1ord_cw2pr_st

 rate1ord_cw2pr_st(           : ,           : )=  0.000000000000000E+000

 ! real(r8), dimension(          16 ,          30 ) :: qqcw
 qqcw(           : ,          : )=   32847851.8054793_r8     

 ! real(r8), dimension(          16 ,          30 ) :: f_act_conv

 f_act_conv(           : ,           : )=  0.800000000000000_r8     

 ! real(r8) :: sol_facti_in
 sol_facti_in =   0.000000000000000E+000
 ! real(r8) :: sol_factbi_in
 sol_factbi_in =   0.100000000000000_r8     
 ! real(r8) :: sol_factii_in
 sol_factii_in =   0.000000000000000E+000
 ! real(r8), dimension(          16 ,          30 ) :: sol_factic_in

 sol_factic_in(           : ,           : )=  0.400000000000000_r8     

 ! real(r8) :: sol_factiic_in
 sol_factiic_in =   0.400000000000000_r8     
 !   
 ! Insert your call to subroutine here
 ! call wetdepa_v2()
 !   
   call system_clock(start_clock,rate_clock)
   call cpu_time(start_time)
   do it=1,ntrials
   call wetdepa_v2(t, p, q, pdel, &
                   cldt, cldc, cmfdqr, evapc, conicw, precs, conds, &
                       evaps, cwat, tracer, deltat, &
                       scavt_out, iscavt_out, cldv, cldvcu, cldvst, dlf, fracis_out, sol_fact, ncol, &
                       scavcoef, is_strat_cloudborne, rate1ord_cw2pr_st, qqcw, f_act_conv, &
                       icscavt_out, isscavt_out, bcscavt_out, bsscavt_out, &
                       sol_facti_in, sol_factbi_in, sol_factii_in, &
                       sol_factic_in, sol_factiic_in )

 ! real(r8), dimension(          16 ,          30 ) :: scavt_out
 scavt(           : ,           : )= -0.015489807056568383_r8
 iscavt(          : ,           : )= -0.015489807056568383_r8
 isscavt(         : ,           : )= 0.000000000000000E+000
 icscavt(:,:) = -0.015489807056568383_r8
 bcscavt(         : ,           :)=  0.000000000000000E+000
 fracis(           : ,           : )= 0.999995222047063_r8
 enddo
    call cpu_time(stop_time)
call system_clock(stop_clock,rate_clock)

 n1=SIZE(scavt,dim=1)
 n2=SIZE(scavt,dim=2)
 do i=1,1
 do j=1,1
       val1_r8 = scavt(i,j)
       val2_r8 = scavt_out(i,j)
       rel_r8 = (val1_r8-val2_r8)/val1_r8 
       if(abs(rel_r8) > eps) then
        errorDetected=.TRUE.
       write(*,80) 'scavt:', val1_r8,val2_r8
          print *, 'relerror: scavt(',i,',',j,'): ',rel_r8
    endif
 enddo
 enddo
 80 format(A, f25.18, f25.18)
 ! real(r8), dimension(          16 ,          30 ) :: iscavt_out
 n1=SIZE(iscavt,dim=1)
 n2=SIZE(iscavt,dim=2)
 do i=1,1
 do j=1,1
       val1_r8 = iscavt(i,j)
       val2_r8 = iscavt_out(i,j)
       rel_r8 = (val1_r8-val2_r8)/val1_r8 
       if(abs(rel_r8) > eps) then
        errorDetected=.TRUE.
!       print *, 'error: iscavt(',i,',',j,'): ',val1_r8,' != ',val2_r8
          print *, 'relerror: iscavt(',i,',',j,'): ',rel_r8
    endif
 enddo
 enddo
 ! real(r8), dimension(          16 ,          30 ) :: fracis_out


 n1=SIZE(fracis,dim=1)
 n2=SIZE(fracis,dim=2)
 do i=1,1
 do j=1,1
       val1_r8 = fracis(i,j)
       val2_r8 = fracis_out(i,j)
       rel_r8 = (val1_r8-val2_r8)/val1_r8 
       if(abs(rel_r8) > eps) then
        errorDetected=.TRUE.
       print *, 'error: fracis(',i,',',j,'): ',val1_r8,' != ',val2_r8
          print *, 'relerror: fracis(',i,',',j,'): ',rel_r8
        endif
 enddo
 enddo
 ! real(r8), dimension(          16 ,          30 ) :: icscavt_out
 n1=SIZE(icscavt,dim=1)
 n2=SIZE(icscavt,dim=2)
 do i=1,1
 do j=1,1
       val1_r8 = icscavt(i,j)
       val2_r8 = icscavt_out(i,j)
       rel_r8 = (val1_r8-val2_r8)/val1_r8 
       if(abs(rel_r8) > eps) then
        errorDetected=.TRUE.
!       print *, 'error: icscavt(',i,',',j,'): ',val1_r8,' != ',val2_r8
          print *, 'relerror: icscavt(',i,',',j,'): ',rel_r8
    endif
 enddo
 enddo
 ! real(r8), dimension(          16 ,          30 ) :: isscavt_out
 n1=SIZE(isscavt,dim=1)
 n2=SIZE(isscavt,dim=2)
 do i=1,1
 do j=1,1
    if(isscavt(i,j) .ne. isscavt_out(i,j)) then
       val1_r8 = isscavt(i,j)
       val2_r8 = isscavt_out(i,j)
        errorDetected=.TRUE.
       print *, 'error: isscavt(',i,',',j,'): ',val1_r8,' != ',val2_r8
    endif
 enddo
 enddo
 ! real(r8), dimension(          16 ,          30 ) :: bcscavt_out
 n1=SIZE(bcscavt,dim=1)
 n2=SIZE(bcscavt,dim=2)
 do i=1,1
 do j=1,1
    if(bcscavt(i,j) .ne. bcscavt_out(i,j)) then
       val1_r8 = bcscavt(i,j)
       val2_r8 = bcscavt_out(i,j)
        errorDetected=.TRUE.
       print *, 'error: bcscavt(',i,',',j,'): ',val1_r8,' != ',val2_r8
    endif
 enddo
 enddo
 ! real(r8), dimension(          16 ,          30 ) :: bsscavt_out
 n1=SIZE(bsscavt,dim=1)
 n2=SIZE(bsscavt,dim=2)
 do i=1,1
 do j=1,1
    if(bsscavt(i,j) .ne. bsscavt_out(i,j)) then
       val1_r8 = bsscavt(i,j)
       val2_r8 = bsscavt_out(i,j)
       rel_r8  = (bsscavt(i,j) - bsscavt_out(i,j))/bsscavt(i,j)
       print *, 'error: bsscavt(',i,',',j,') =',val1_r8, val2_r8
        errorDetected=.TRUE.
!       print *, 'relerror: bsscavt(',i,',',j,'): ',rel_r8
    endif
 enddo
 enddo
 if(errorDetected) then 
    print *,'Detected error'
    print *, 'FAILED'
 else
    print *,'Correct exection'
    print *,'PASSED'
!    write(*,'(a,f10.3,a)')  ' completed in ', 1.0E6*(real(stop_clock-start_clock,kind=r8)/real(rate_clock,kind=r8)), ' usec'
    write(*,'(a,f10.7)')  'total time(sec): ', (stop_time-start_time)
    write(*,'(a,f10.3)') 'time per call (usec): ',1e6*(stop_time-start_time)/dble(ntrials)
 endif
 end program wetdepa_v2_driver
