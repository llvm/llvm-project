    module rrtmg_lw_rad
        use shr_kind_mod, only: r8 => shr_kind_r8
        use ppgrid,       only: pcols, begchunk, endchunk
        use rrlw_vsn
        use mcica_subcol_gen_lw, only: mcica_subcol_lw
        use rrtmg_lw_cldprmc, only: cldprmc
        use rrtmg_lw_rtrnmc, only: rtrnmc
        use rrtmg_lw_setcoef, only: setcoef
        use rrtmg_lw_taumol, only: taumol
        implicit none
        public :: rrtmg_lw, inatm
        contains
        ! START OF STATE GENERATION BLOCK
        subroutine rrtmg_lw             (lchnk   ,ncol    ,nlay    ,icld    ,                                play    ,plev    ,tlay    ,tlev    ,tsfc    ,h2ovmr  ,              o3vmr   ,co2vmr  ,ch4vmr  ,o2vmr   ,n2ovmr  ,             cfc11vmr,cfc12vmr,              cfc22vmr,ccl4vmr ,emis    ,inflglw ,iceflglw,liqflglw,              cldfmcl ,taucmcl ,ciwpmcl ,clwpmcl ,reicmcl ,relqmcl ,              tauaer  ,              uflx    ,dflx    ,hr      ,uflxc   ,dflxc,  hrc, uflxs, dflxs )
            USE mpi
            use parrrtm, only : nbndlw, ngptlw, maxxsec, mxmol
            use rrlw_con, only: fluxfac, heatfac, oneminus, pi
            use rrlw_wvn, only: ng, ngb, nspa, nspb, wavenum1, wavenum2, delwave
            ! START OF SPECIFICATION PART OF STATE GENERATION BLOCK
            INTEGER :: kgen_mpi_rank, kgen_mpi_size, kgen_cur_rank
            CHARACTER(LEN=16) ::kgen_mpi_rank_conv
            INTEGER, DIMENSION(1), PARAMETER :: kgen_mpi_rank_at = (/ 0 /)
            INTEGER :: kgen_ierr, kgen_unit
            INTEGER, SAVE :: kgen_counter = 1
            CHARACTER(LEN=16) :: kgen_counter_conv
            INTEGER, DIMENSION(1), PARAMETER :: kgen_counter_at = (/ 1 /)
            CHARACTER(LEN=1024) :: kgen_filepath
            integer, intent(in) :: lchnk
            integer, intent(in) :: ncol
            integer, intent(in) :: nlay
            integer, intent(inout) :: icld
            real(kind=r8), intent(in) :: play(:,:)
            real(kind=r8), intent(in) :: plev(:,:)
            real(kind=r8), intent(in) :: tlay(:,:)
            real(kind=r8), intent(in) :: tlev(:,:)
            real(kind=r8), intent(in) :: tsfc(:)
            real(kind=r8), intent(in) :: h2ovmr(:,:)
            real(kind=r8), intent(in) :: o3vmr(:,:)
            real(kind=r8), intent(in) :: co2vmr(:,:)
            real(kind=r8), intent(in) :: ch4vmr(:,:)
            real(kind=r8), intent(in) :: o2vmr(:,:)
            real(kind=r8), intent(in) :: n2ovmr(:,:)
            real(kind=r8), intent(in) :: cfc11vmr(:,:)
            real(kind=r8), intent(in) :: cfc12vmr(:,:)
            real(kind=r8), intent(in) :: cfc22vmr(:,:)
            real(kind=r8), intent(in) :: ccl4vmr(:,:)
            real(kind=r8), intent(in) :: emis(:,:)
            integer, intent(in) :: inflglw
            integer, intent(in) :: iceflglw
            integer, intent(in) :: liqflglw
            real(kind=r8), intent(in) :: cldfmcl(:,:,:)
            real(kind=r8), intent(in) :: ciwpmcl(:,:,:)
            real(kind=r8), intent(in) :: clwpmcl(:,:,:)
            real(kind=r8), intent(in) :: reicmcl(:,:)
            real(kind=r8), intent(in) :: relqmcl(:,:)
            real(kind=r8), intent(in) :: taucmcl(:,:,:)
            real(kind=r8), intent(in) :: tauaer(:,:,:)
            real(kind=r8), intent(out) :: uflx(:,:)
            real(kind=r8), intent(out) :: dflx(:,:)
            real(kind=r8), intent(out) :: hr(:,:)
            real(kind=r8), intent(out) :: uflxc(:,:)
            real(kind=r8), intent(out) :: dflxc(:,:)
            real(kind=r8), intent(out) :: hrc(:,:)
            real(kind=r8), intent(out) :: uflxs(:,:,:)
            real(kind=r8), intent(out) :: dflxs(:,:,:)
            integer :: istart
            integer :: iend
            integer :: iout
            integer :: iaer
            integer :: iplon
            integer :: imca
            integer :: ims
            integer :: k
            integer :: ig
            real(kind=r8) :: pavel(nlay)
            real(kind=r8) :: tavel(nlay)
            real(kind=r8) :: pz(0:nlay)
            real(kind=r8) :: tz(0:nlay)
            real(kind=r8) :: tbound
            real(kind=r8) :: coldry(nlay)
            real(kind=r8) :: wbrodl(nlay)
            real(kind=r8) :: wkl(mxmol,nlay)
            real(kind=r8) :: wx(maxxsec,nlay)
            real(kind=r8) :: pwvcm
            real(kind=r8) :: semiss(nbndlw)
            real(kind=r8) :: fracs(nlay,ngptlw)
            real(kind=r8) :: taug(nlay,ngptlw)
            real(kind=r8) :: taut(nlay,ngptlw)
            real(kind=r8) :: taua(nlay,nbndlw)
            integer :: laytrop
            integer :: jp(nlay)
            integer :: jt(nlay)
            integer :: jt1(nlay)
            real(kind=r8) :: planklay(nlay,nbndlw)
            real(kind=r8) :: planklev(0:nlay,nbndlw)
            real(kind=r8) :: plankbnd(nbndlw)
            real(kind=r8) :: colh2o(nlay)
            real(kind=r8) :: colco2(nlay)
            real(kind=r8) :: colo3(nlay)
            real(kind=r8) :: coln2o(nlay)
            real(kind=r8) :: colco(nlay)
            real(kind=r8) :: colch4(nlay)
            real(kind=r8) :: colo2(nlay)
            real(kind=r8) :: colbrd(nlay)
            integer :: indself(nlay)
            integer :: indfor(nlay)
            real(kind=r8) :: selffac(nlay)
            real(kind=r8) :: selffrac(nlay)
            real(kind=r8) :: forfac(nlay)
            real(kind=r8) :: forfrac(nlay)
            integer :: indminor(nlay)
            real(kind=r8) :: minorfrac(nlay)
            real(kind=r8) :: scaleminor(nlay)
            real(kind=r8) :: scaleminorn2(nlay)
            real(kind=r8) ::                          fac00(nlay), fac01(nlay),                          fac10(nlay), fac11(nlay)
            real(kind=r8) ::                          rat_h2oco2(nlay),rat_h2oco2_1(nlay),                          rat_h2oo3(nlay),rat_h2oo3_1(nlay),                          rat_h2on2o(nlay),rat_h2on2o_1(nlay),                          rat_h2och4(nlay),rat_h2och4_1(nlay),                          rat_n2oco2(nlay),rat_n2oco2_1(nlay),                          rat_o3co2(nlay),rat_o3co2_1(nlay)
            integer :: ncbands
            integer :: inflag
            integer :: iceflag
            integer :: liqflag
            real(kind=r8) :: cldfmc(ngptlw,nlay)
            real(kind=r8) :: ciwpmc(ngptlw,nlay)
            real(kind=r8) :: clwpmc(ngptlw,nlay)
            real(kind=r8) :: relqmc(nlay)
            real(kind=r8) :: reicmc(nlay)
            real(kind=r8) :: dgesmc(nlay)
            real(kind=r8) :: taucmc(ngptlw,nlay)
            real(kind=r8) :: totuflux(0:nlay)
            real(kind=r8) :: totdflux(0:nlay)
            real(kind=r8) :: totufluxs(nbndlw,0:nlay)
            real(kind=r8) :: totdfluxs(nbndlw,0:nlay)
            real(kind=r8) :: fnet(0:nlay)
            real(kind=r8) :: htr(0:nlay)
            real(kind=r8) :: totuclfl(0:nlay)
            real(kind=r8) :: totdclfl(0:nlay)
            real(kind=r8) :: fnetc(0:nlay)
            real(kind=r8) :: htrc(0:nlay)
            ! START OF EXECUTION PART OF STATE GENERATION BLOCK
            oneminus = 1._r8 - 1.e-6_r8
            pi = 2._r8 * asin(1._r8)
            fluxfac = pi * 2.e4_r8
            istart = 1
            iend = 16
            iout = 0
            ims = 1
            if (icld.lt.0.or.icld.gt.3) icld = 2
            iaer = 10
            do iplon = 1, ncol
                ! START OF STATE GENERATION
                !$OMP MASTER
                CALL mpi_comm_rank ( MPI_COMM_WORLD, kgen_mpi_rank, kgen_ierr )
                IF ( kgen_ierr /= mpi_success ) THEN
                    CALL kgen_error_stop( "MPI ERROR" )
                END IF
                CALL mpi_comm_size ( MPI_COMM_WORLD, kgen_mpi_size, kgen_ierr )
                IF ( kgen_ierr /= mpi_success ) THEN
                    CALL kgen_error_stop( "MPI ERROR" )
                END IF
                kgen_cur_rank = 0
                kgen_unit = -1
                DO WHILE(kgen_cur_rank < kgen_mpi_size)
                    IF ( ANY(kgen_mpi_rank == kgen_mpi_rank_at) .AND. kgen_cur_rank == kgen_mpi_rank ) THEN
                        WRITE( kgen_mpi_rank_conv, * ) kgen_mpi_rank
                        IF ( ANY(kgen_counter == kgen_counter_at) ) THEN
                            WRITE( kgen_counter_conv, * ) kgen_counter
                            kgen_filepath = "../data/inatm." // TRIM(ADJUSTL(kgen_counter_conv)) // "." // TRIM(ADJUSTL(kgen_mpi_rank_conv))
                            kgen_unit = kgen_get_newunit(kgen_mpi_rank+kgen_counter)
                            OPEN (UNIT=kgen_unit, FILE=kgen_filepath, STATUS="REPLACE", ACCESS="STREAM", FORM="UNFORMATTED", ACTION="WRITE", IOSTAT=kgen_ierr, CONVERT="BIG_ENDIAN")
                            IF ( kgen_ierr /= 0 ) THEN
                                CALL kgen_error_stop( "FILE OPEN ERROR: " // TRIM(ADJUSTL(kgen_filepath)) )
                            END IF
                            PRINT *, "KGEN writes input state variables at count = ", kgen_counter, " on mpirank = ", kgen_mpi_rank
                            WRITE(UNIT = kgen_unit) lbound(taucmcl, 1)
                            WRITE(UNIT = kgen_unit) ubound(taucmcl, 1)
                            WRITE(UNIT = kgen_unit) lbound(taucmcl, 2)
                            WRITE(UNIT = kgen_unit) ubound(taucmcl, 2)
                            WRITE(UNIT = kgen_unit) lbound(taucmcl, 3)
                            WRITE(UNIT = kgen_unit) ubound(taucmcl, 3)
                            WRITE(UNIT = kgen_unit) taucmcl
                            WRITE(UNIT = kgen_unit) lbound(ch4vmr, 1)
                            WRITE(UNIT = kgen_unit) ubound(ch4vmr, 1)
                            WRITE(UNIT = kgen_unit) lbound(ch4vmr, 2)
                            WRITE(UNIT = kgen_unit) ubound(ch4vmr, 2)
                            WRITE(UNIT = kgen_unit) ch4vmr
                            WRITE(UNIT = kgen_unit) icld
                            WRITE(UNIT = kgen_unit) lbound(emis, 1)
                            WRITE(UNIT = kgen_unit) ubound(emis, 1)
                            WRITE(UNIT = kgen_unit) lbound(emis, 2)
                            WRITE(UNIT = kgen_unit) ubound(emis, 2)
                            WRITE(UNIT = kgen_unit) emis
                            WRITE(UNIT = kgen_unit) lbound(tlay, 1)
                            WRITE(UNIT = kgen_unit) ubound(tlay, 1)
                            WRITE(UNIT = kgen_unit) lbound(tlay, 2)
                            WRITE(UNIT = kgen_unit) ubound(tlay, 2)
                            WRITE(UNIT = kgen_unit) tlay
                            WRITE(UNIT = kgen_unit) lbound(reicmcl, 1)
                            WRITE(UNIT = kgen_unit) ubound(reicmcl, 1)
                            WRITE(UNIT = kgen_unit) lbound(reicmcl, 2)
                            WRITE(UNIT = kgen_unit) ubound(reicmcl, 2)
                            WRITE(UNIT = kgen_unit) reicmcl
                            WRITE(UNIT = kgen_unit) nlay
                            WRITE(UNIT = kgen_unit) lbound(cfc11vmr, 1)
                            WRITE(UNIT = kgen_unit) ubound(cfc11vmr, 1)
                            WRITE(UNIT = kgen_unit) lbound(cfc11vmr, 2)
                            WRITE(UNIT = kgen_unit) ubound(cfc11vmr, 2)
                            WRITE(UNIT = kgen_unit) cfc11vmr
                            WRITE(UNIT = kgen_unit) lbound(tsfc, 1)
                            WRITE(UNIT = kgen_unit) ubound(tsfc, 1)
                            WRITE(UNIT = kgen_unit) tsfc
                            WRITE(UNIT = kgen_unit) lbound(relqmcl, 1)
                            WRITE(UNIT = kgen_unit) ubound(relqmcl, 1)
                            WRITE(UNIT = kgen_unit) lbound(relqmcl, 2)
                            WRITE(UNIT = kgen_unit) ubound(relqmcl, 2)
                            WRITE(UNIT = kgen_unit) relqmcl
                            WRITE(UNIT = kgen_unit) lbound(o3vmr, 1)
                            WRITE(UNIT = kgen_unit) ubound(o3vmr, 1)
                            WRITE(UNIT = kgen_unit) lbound(o3vmr, 2)
                            WRITE(UNIT = kgen_unit) ubound(o3vmr, 2)
                            WRITE(UNIT = kgen_unit) o3vmr
                            WRITE(UNIT = kgen_unit) lbound(n2ovmr, 1)
                            WRITE(UNIT = kgen_unit) ubound(n2ovmr, 1)
                            WRITE(UNIT = kgen_unit) lbound(n2ovmr, 2)
                            WRITE(UNIT = kgen_unit) ubound(n2ovmr, 2)
                            WRITE(UNIT = kgen_unit) n2ovmr
                            WRITE(UNIT = kgen_unit) lbound(plev, 1)
                            WRITE(UNIT = kgen_unit) ubound(plev, 1)
                            WRITE(UNIT = kgen_unit) lbound(plev, 2)
                            WRITE(UNIT = kgen_unit) ubound(plev, 2)
                            WRITE(UNIT = kgen_unit) plev
                            WRITE(UNIT = kgen_unit) lbound(play, 1)
                            WRITE(UNIT = kgen_unit) ubound(play, 1)
                            WRITE(UNIT = kgen_unit) lbound(play, 2)
                            WRITE(UNIT = kgen_unit) ubound(play, 2)
                            WRITE(UNIT = kgen_unit) play
                            WRITE(UNIT = kgen_unit) lbound(tauaer, 1)
                            WRITE(UNIT = kgen_unit) ubound(tauaer, 1)
                            WRITE(UNIT = kgen_unit) lbound(tauaer, 2)
                            WRITE(UNIT = kgen_unit) ubound(tauaer, 2)
                            WRITE(UNIT = kgen_unit) lbound(tauaer, 3)
                            WRITE(UNIT = kgen_unit) ubound(tauaer, 3)
                            WRITE(UNIT = kgen_unit) tauaer
                            WRITE(UNIT = kgen_unit) lbound(clwpmcl, 1)
                            WRITE(UNIT = kgen_unit) ubound(clwpmcl, 1)
                            WRITE(UNIT = kgen_unit) lbound(clwpmcl, 2)
                            WRITE(UNIT = kgen_unit) ubound(clwpmcl, 2)
                            WRITE(UNIT = kgen_unit) lbound(clwpmcl, 3)
                            WRITE(UNIT = kgen_unit) ubound(clwpmcl, 3)
                            WRITE(UNIT = kgen_unit) clwpmcl
                            WRITE(UNIT = kgen_unit) lbound(o2vmr, 1)
                            WRITE(UNIT = kgen_unit) ubound(o2vmr, 1)
                            WRITE(UNIT = kgen_unit) lbound(o2vmr, 2)
                            WRITE(UNIT = kgen_unit) ubound(o2vmr, 2)
                            WRITE(UNIT = kgen_unit) o2vmr
                            WRITE(UNIT = kgen_unit) lbound(co2vmr, 1)
                            WRITE(UNIT = kgen_unit) ubound(co2vmr, 1)
                            WRITE(UNIT = kgen_unit) lbound(co2vmr, 2)
                            WRITE(UNIT = kgen_unit) ubound(co2vmr, 2)
                            WRITE(UNIT = kgen_unit) co2vmr
                            WRITE(UNIT = kgen_unit) lbound(ccl4vmr, 1)
                            WRITE(UNIT = kgen_unit) ubound(ccl4vmr, 1)
                            WRITE(UNIT = kgen_unit) lbound(ccl4vmr, 2)
                            WRITE(UNIT = kgen_unit) ubound(ccl4vmr, 2)
                            WRITE(UNIT = kgen_unit) ccl4vmr
                            WRITE(UNIT = kgen_unit) iceflglw
                            WRITE(UNIT = kgen_unit) lbound(cfc12vmr, 1)
                            WRITE(UNIT = kgen_unit) ubound(cfc12vmr, 1)
                            WRITE(UNIT = kgen_unit) lbound(cfc12vmr, 2)
                            WRITE(UNIT = kgen_unit) ubound(cfc12vmr, 2)
                            WRITE(UNIT = kgen_unit) cfc12vmr
                            WRITE(UNIT = kgen_unit) lbound(tlev, 1)
                            WRITE(UNIT = kgen_unit) ubound(tlev, 1)
                            WRITE(UNIT = kgen_unit) lbound(tlev, 2)
                            WRITE(UNIT = kgen_unit) ubound(tlev, 2)
                            WRITE(UNIT = kgen_unit) tlev
                            WRITE(UNIT = kgen_unit) lbound(h2ovmr, 1)
                            WRITE(UNIT = kgen_unit) ubound(h2ovmr, 1)
                            WRITE(UNIT = kgen_unit) lbound(h2ovmr, 2)
                            WRITE(UNIT = kgen_unit) ubound(h2ovmr, 2)
                            WRITE(UNIT = kgen_unit) h2ovmr
                            WRITE(UNIT = kgen_unit) inflglw
                            WRITE(UNIT = kgen_unit) lbound(ciwpmcl, 1)
                            WRITE(UNIT = kgen_unit) ubound(ciwpmcl, 1)
                            WRITE(UNIT = kgen_unit) lbound(ciwpmcl, 2)
                            WRITE(UNIT = kgen_unit) ubound(ciwpmcl, 2)
                            WRITE(UNIT = kgen_unit) lbound(ciwpmcl, 3)
                            WRITE(UNIT = kgen_unit) ubound(ciwpmcl, 3)
                            WRITE(UNIT = kgen_unit) ciwpmcl
                            WRITE(UNIT = kgen_unit) lbound(cldfmcl, 1)
                            WRITE(UNIT = kgen_unit) ubound(cldfmcl, 1)
                            WRITE(UNIT = kgen_unit) lbound(cldfmcl, 2)
                            WRITE(UNIT = kgen_unit) ubound(cldfmcl, 2)
                            WRITE(UNIT = kgen_unit) lbound(cldfmcl, 3)
                            WRITE(UNIT = kgen_unit) ubound(cldfmcl, 3)
                            WRITE(UNIT = kgen_unit) cldfmcl
                            WRITE(UNIT = kgen_unit) liqflglw
                            WRITE(UNIT = kgen_unit) lbound(cfc22vmr, 1)
                            WRITE(UNIT = kgen_unit) ubound(cfc22vmr, 1)
                            WRITE(UNIT = kgen_unit) lbound(cfc22vmr, 2)
                            WRITE(UNIT = kgen_unit) ubound(cfc22vmr, 2)
                            WRITE(UNIT = kgen_unit) cfc22vmr
                            WRITE(UNIT = kgen_unit) iaer
                            WRITE(UNIT = kgen_unit) iplon
                            CALL sleep(1)
                        END IF
                    END IF
                    kgen_cur_rank = kgen_cur_rank + 1
                    call mpi_barrier( MPI_COMM_WORLD, kgen_ierr )
                END DO
                !$OMP END MASTER
                !$OMP BARRIER

                IF ( kgen_unit > 0 ) THEN
                CALL inatm(iplon, nlay, icld, iaer, play, plev, tlay, tlev, tsfc, h2ovmr, o3vmr, co2vmr, ch4vmr, o2vmr, n2ovmr, cfc11vmr, cfc12vmr, cfc22vmr, ccl4vmr, emis, inflglw, iceflglw, liqflglw, cldfmcl, taucmcl, ciwpmcl, clwpmcl, reicmcl, relqmcl, tauaer, pavel, pz, tavel, tz, tbound, semiss, coldry, wkl, wbrodl, wx, pwvcm, inflag, iceflag, liqflag, cldfmc, taucmc, ciwpmc, clwpmc, reicmc, dgesmc, relqmc, taua, kgen_unit)
                ELSE
                call inatm (iplon, nlay, icld, iaer,               play, plev, tlay, tlev, tsfc, h2ovmr,               o3vmr, co2vmr, ch4vmr, o2vmr, n2ovmr, cfc11vmr, cfc12vmr,               cfc22vmr, ccl4vmr, emis, inflglw, iceflglw, liqflglw,               cldfmcl, taucmcl, ciwpmcl, clwpmcl, reicmcl, relqmcl, tauaer,               pavel, pz, tavel, tz, tbound, semiss, coldry,               wkl, wbrodl, wx, pwvcm, inflag, iceflag, liqflag,               cldfmc, taucmc, ciwpmc, clwpmc, reicmc, dgesmc, relqmc, taua)
                END IF

                !$OMP BARRIER
                !$OMP MASTER
                kgen_cur_rank = 0
                DO WHILE(kgen_cur_rank < kgen_mpi_size)
                    IF ( ANY(kgen_mpi_rank == kgen_mpi_rank_at) .AND. kgen_cur_rank == kgen_mpi_rank ) THEN
                        IF ( ANY(kgen_counter == kgen_counter_at) ) THEN
                            PRINT *, "KGEN writes output state variables at count = ", kgen_counter, " on mpirank = ", kgen_mpi_rank
                            WRITE(UNIT = kgen_unit) iceflag
                            WRITE(UNIT = kgen_unit) wkl
                            WRITE(UNIT = kgen_unit) coldry
                            WRITE(UNIT = kgen_unit) clwpmc
                            WRITE(UNIT = kgen_unit) cldfmc
                            WRITE(UNIT = kgen_unit) relqmc
                            WRITE(UNIT = kgen_unit) ciwpmc
                            WRITE(UNIT = kgen_unit) wbrodl
                            WRITE(UNIT = kgen_unit) tavel
                            WRITE(UNIT = kgen_unit) liqflag
                            WRITE(UNIT = kgen_unit) tz
                            WRITE(UNIT = kgen_unit) pz
                            WRITE(UNIT = kgen_unit) tbound
                            WRITE(UNIT = kgen_unit) reicmc
                            WRITE(UNIT = kgen_unit) semiss
                            WRITE(UNIT = kgen_unit) pavel
                            WRITE(UNIT = kgen_unit) dgesmc
                            WRITE(UNIT = kgen_unit) pwvcm
                            WRITE(UNIT = kgen_unit) inflag
                            WRITE(UNIT = kgen_unit) wx
                            WRITE(UNIT = kgen_unit) taua
                            WRITE(UNIT = kgen_unit) taucmc
                            ENDFILE kgen_unit
                            CALL sleep(1)
                            CLOSE (UNIT=kgen_unit)
                        END IF
                    END IF
                    kgen_cur_rank = kgen_cur_rank + 1
                    CALL mpi_barrier( MPI_COMM_WORLD, kgen_ierr )
                END DO
                PRINT *, "kgen_counter = ", kgen_counter, " at rank ", kgen_mpi_rank
                IF ( kgen_counter > maxval(kgen_counter_at) ) THEN
                    CALL sleep(2)
                    PRINT *, "kgen_counter is larger than maximum counter. Exit program..."
                    CALL mpi_abort( MPI_COMM_WORLD, 1, kgen_ierr)
                END IF
                kgen_counter = kgen_counter + 1
                !$OMP END MASTER
                ! END OF STATE GENERATION
                call cldprmc(nlay, inflag, iceflag, liqflag, cldfmc, ciwpmc,                       clwpmc, reicmc, dgesmc, relqmc, ncbands, taucmc)
                call setcoef(nlay, istart, pavel, tavel, tz, tbound, semiss,                       coldry, wkl, wbrodl,                       laytrop, jp, jt, jt1, planklay, planklev, plankbnd,                       colh2o, colco2, colo3, coln2o, colco, colch4, colo2,                       colbrd, fac00, fac01, fac10, fac11,                       rat_h2oco2, rat_h2oco2_1, rat_h2oo3, rat_h2oo3_1,                       rat_h2on2o, rat_h2on2o_1, rat_h2och4, rat_h2och4_1,                       rat_n2oco2, rat_n2oco2_1, rat_o3co2, rat_o3co2_1,                       selffac, selffrac, indself, forfac, forfrac, indfor,                       minorfrac, scaleminor, scaleminorn2, indminor)
                call taumol(nlay, pavel, wx, coldry,                      laytrop, jp, jt, jt1, planklay, planklev, plankbnd,                      colh2o, colco2, colo3, coln2o, colco, colch4, colo2,                      colbrd, fac00, fac01, fac10, fac11,                      rat_h2oco2, rat_h2oco2_1, rat_h2oo3, rat_h2oo3_1,                      rat_h2on2o, rat_h2on2o_1, rat_h2och4, rat_h2och4_1,                      rat_n2oco2, rat_n2oco2_1, rat_o3co2, rat_o3co2_1,                      selffac, selffrac, indself, forfac, forfrac, indfor,                      minorfrac, scaleminor, scaleminorn2, indminor,                      fracs, taug)
                if (iaer .eq. 0) then
                    do k = 1, nlay
                        do ig = 1, ngptlw
                            taut(k,ig) = taug(k,ig)
                        enddo
                    enddo
                    elseif (iaer .eq. 10) then
                    do k = 1, nlay
                        do ig = 1, ngptlw
                            taut(k,ig) = taug(k,ig) + taua(k,ngb(ig))
                        enddo
                    enddo
                endif
                call rtrnmc(nlay, istart, iend, iout, pz, semiss, ncbands,                      cldfmc, taucmc, planklay, planklev, plankbnd,                      pwvcm, fracs, taut,                      totuflux, totdflux, fnet, htr,                      totuclfl, totdclfl, fnetc, htrc, totufluxs, totdfluxs )
                do k = 0, nlay
                    uflx(iplon,k+1) = totuflux(k)
                    dflx(iplon,k+1) = totdflux(k)
                    uflxc(iplon,k+1) = totuclfl(k)
                    dflxc(iplon,k+1) = totdclfl(k)
                    uflxs(:,iplon,k+1) = totufluxs(:,k)
                    dflxs(:,iplon,k+1) = totdfluxs(:,k)
                enddo
                do k = 0, nlay-1
                    hr(iplon,k+1) = htr(k)
                    hrc(iplon,k+1) = htrc(k)
                enddo
            enddo
            CONTAINS
            ! END OF STATE GENERATION BLOCK

            
            FUNCTION kgen_get_newunit(seed) RESULT(new_unit)
               INTEGER, PARAMETER :: UNIT_MIN=100, UNIT_MAX=1000000
               LOGICAL :: is_opened
               INTEGER :: nunit, new_unit, counter
               INTEGER, INTENT(IN) :: seed
            
               new_unit = -1
               
               DO counter=UNIT_MIN, UNIT_MAX
                   inquire(UNIT=counter, OPENED=is_opened)
                   IF (.NOT. is_opened) THEN
                       new_unit = counter
                       EXIT
                   END IF
               END DO
            END FUNCTION

            
            SUBROUTINE kgen_error_stop( msg )
                IMPLICIT NONE
                CHARACTER(LEN=*), INTENT(IN) :: msg
            
                WRITE (*,*) msg
                STOP 1
            END SUBROUTINE
        end subroutine rrtmg_lw
        SUBROUTINE inatm(iplon, nlay, icld, iaer, play, plev, tlay, tlev, tsfc, h2ovmr, o3vmr, co2vmr, ch4vmr, o2vmr, n2ovmr, cfc11vmr, cfc12vmr, cfc22vmr, ccl4vmr, emis, inflglw, iceflglw, liqflglw, cldfmcl, taucmcl, ciwpmcl, clwpmcl, reicmcl, relqmcl, tauaer, pavel, pz, tavel, tz, tbound, semiss, coldry, wkl, wbrodl, wx, pwvcm, inflag, iceflag, liqflag, cldfmc, taucmc, ciwpmc, clwpmc, reicmc, dgesmc, relqmc, taua, kgen_unit)
        use parrrtm, only : nbndlw, ngptlw, nmol, maxxsec, mxmol
        use rrlw_con, only: fluxfac, heatfac, oneminus, pi, grav, avogad
        use rrlw_wvn, only: ng, nspa, nspb, wavenum1, wavenum2, delwave, ixindx
        INTEGER, OPTIONAL, INTENT(IN) :: kgen_unit
        integer, intent(in) :: iplon
        integer, intent(in) :: nlay
        integer, intent(in) :: icld
        integer, intent(in) :: iaer
        real(kind=r8), intent(in) :: play(:,:)
        real(kind=r8), intent(in) :: plev(:,:)
        real(kind=r8), intent(in) :: tlay(:,:)
        real(kind=r8), intent(in) :: tlev(:,:)
        real(kind=r8), intent(in) :: tsfc(:)
        real(kind=r8), intent(in) :: h2ovmr(:,:)
        real(kind=r8), intent(in) :: o3vmr(:,:)
        real(kind=r8), intent(in) :: co2vmr(:,:)
        real(kind=r8), intent(in) :: ch4vmr(:,:)
        real(kind=r8), intent(in) :: o2vmr(:,:)
        real(kind=r8), intent(in) :: n2ovmr(:,:)
        real(kind=r8), intent(in) :: cfc11vmr(:,:)
        real(kind=r8), intent(in) :: cfc12vmr(:,:)
        real(kind=r8), intent(in) :: cfc22vmr(:,:)
        real(kind=r8), intent(in) :: ccl4vmr(:,:)
        real(kind=r8), intent(in) :: emis(:,:)
        integer, intent(in) :: inflglw
        integer, intent(in) :: iceflglw
        integer, intent(in) :: liqflglw
        real(kind=r8), intent(in) :: cldfmcl(:,:,:)
        real(kind=r8), intent(in) :: ciwpmcl(:,:,:)
        real(kind=r8), intent(in) :: clwpmcl(:,:,:)
        real(kind=r8), intent(in) :: reicmcl(:,:)
        real(kind=r8), intent(in) :: relqmcl(:,:)
        real(kind=r8), intent(in) :: taucmcl(:,:,:)
        real(kind=r8), intent(in) :: tauaer(:,:,:)
        real(kind=r8), intent(out) :: pavel(:)
        real(kind=r8), intent(out) :: tavel(:)
        real(kind=r8), intent(out) :: pz(0:)
        real(kind=r8), intent(out) :: tz(0:)
        real(kind=r8), intent(out) :: tbound
        real(kind=r8), intent(out) :: coldry(:)
        real(kind=r8), intent(out) :: wbrodl(:)
        real(kind=r8), intent(out) :: wkl(:,:)
        real(kind=r8), intent(out) :: wx(:,:)
        real(kind=r8), intent(out) :: pwvcm
        real(kind=r8), intent(out) :: semiss(:)
        integer, intent(out) :: inflag
        integer, intent(out) :: iceflag
        integer, intent(out) :: liqflag
        real(kind=r8), intent(out) :: cldfmc(:,:)
        real(kind=r8), intent(out) :: ciwpmc(:,:)
        real(kind=r8), intent(out) :: clwpmc(:,:)
        real(kind=r8), intent(out) :: relqmc(:)
        real(kind=r8), intent(out) :: reicmc(:)
        real(kind=r8), intent(out) :: dgesmc(:)
        real(kind=r8), intent(out) :: taucmc(:,:)
        real(kind=r8), intent(out) :: taua(:,:)
        real(kind=r8), parameter :: amd = 28.9660_r8
        real(kind=r8), parameter :: amw = 18.0160_r8
        real(kind=r8), parameter :: amdw = 1.607793_r8
        real(kind=r8), parameter :: amdc = 0.658114_r8
        real(kind=r8), parameter :: amdo = 0.603428_r8
        real(kind=r8), parameter :: amdm = 1.805423_r8
        real(kind=r8), parameter :: amdn = 0.658090_r8
        real(kind=r8), parameter :: amdc1 = 0.210852_r8
        real(kind=r8), parameter :: amdc2 = 0.239546_r8
        real(kind=r8), parameter :: sbc = 5.67e-08_r8
        integer :: isp, l, ix, n, imol, ib, ig
        real(kind=r8) :: amm, amttl, wvttl, wvsh, summol
        IF ( present(kgen_unit) ) THEN
            WRITE(UNIT = kgen_unit) avogad
            WRITE(UNIT = kgen_unit) grav
            WRITE(UNIT = kgen_unit) ixindx
        END IF
        wkl(:,:) = 0.0_r8
        wx(:,:) = 0.0_r8
        cldfmc(:,:) = 0.0_r8
        taucmc(:,:) = 0.0_r8
        ciwpmc(:,:) = 0.0_r8
        clwpmc(:,:) = 0.0_r8
        reicmc(:) = 0.0_r8
        dgesmc(:) = 0.0_r8
        relqmc(:) = 0.0_r8
        taua(:,:) = 0.0_r8
        amttl = 0.0_r8
        wvttl = 0.0_r8
        tbound = tsfc(iplon)
        pz(0) = plev(iplon,nlay+1)
        tz(0) = tlev(iplon,nlay+1)
        do l = 1, nlay
            pavel(l) = play(iplon,nlay-l+1)
            tavel(l) = tlay(iplon,nlay-l+1)
            pz(l) = plev(iplon,nlay-l+1)
            tz(l) = tlev(iplon,nlay-l+1)
            wkl(1,l) = h2ovmr(iplon,nlay-l+1)
            wkl(2,l) = co2vmr(iplon,nlay-l+1)
            wkl(3,l) = o3vmr(iplon,nlay-l+1)
            wkl(4,l) = n2ovmr(iplon,nlay-l+1)
            wkl(6,l) = ch4vmr(iplon,nlay-l+1)
            wkl(7,l) = o2vmr(iplon,nlay-l+1)
            amm = (1._r8 - wkl(1,l)) * amd + wkl(1,l) * amw
            coldry(l) = (pz(l-1)-pz(l)) * 1.e3_r8 * avogad /                      (1.e2_r8 * grav * amm * (1._r8 + wkl(1,l)))
            wx(1,l) = ccl4vmr(iplon,nlay-l+1)
            wx(2,l) = cfc11vmr(iplon,nlay-l+1)
            wx(3,l) = cfc12vmr(iplon,nlay-l+1)
            wx(4,l) = cfc22vmr(iplon,nlay-l+1)
        enddo
        coldry(nlay) = (pz(nlay-1)) * 1.e3_r8 * avogad /                         (1.e2_r8 * grav * amm * (1._r8 + wkl(1,nlay-1)))
        do l = 1, nlay
            summol = 0.0_r8
            do imol = 2, nmol
                summol = summol + wkl(imol,l)
            enddo
            wbrodl(l) = coldry(l) * (1._r8 - summol)
            do imol = 1, nmol
                wkl(imol,l) = coldry(l) * wkl(imol,l)
            enddo
            amttl = amttl + coldry(l)+wkl(1,l)
            wvttl = wvttl + wkl(1,l)
            do ix = 1,maxxsec
                if (ixindx(ix) .ne. 0) then
                    wx(ixindx(ix),l) = coldry(l) * wx(ix,l) * 1.e-20_r8
                endif
            enddo
        enddo
        wvsh = (amw * wvttl) / (amd * amttl)
        pwvcm = wvsh * (1.e3_r8 * pz(0)) / (1.e2_r8 * grav)
        do n=1,nbndlw
            semiss(n) = emis(iplon,n)
        enddo
        if (iaer .ge. 1) then
            do l = 1, nlay-1
                do ib = 1, nbndlw
                    taua(l,ib) = tauaer(iplon,nlay-l,ib)
                enddo
            enddo
        endif
        if (icld .ge. 1) then
            inflag = inflglw
            iceflag = iceflglw
            liqflag = liqflglw
            do l = 1, nlay-1
                do ig = 1, ngptlw
                    cldfmc(ig,l) = cldfmcl(ig,iplon,nlay-l)
                    taucmc(ig,l) = taucmcl(ig,iplon,nlay-l)
                    ciwpmc(ig,l) = ciwpmcl(ig,iplon,nlay-l)
                    clwpmc(ig,l) = clwpmcl(ig,iplon,nlay-l)
                enddo
                reicmc(l) = reicmcl(iplon,nlay-l)
                if (iceflag .eq. 3) then
                    dgesmc(l) = 1.5396_r8 * reicmcl(iplon,nlay-l)
                endif
                relqmc(l) = relqmcl(iplon,nlay-l)
            enddo
            cldfmc(:,nlay) = 0.0_r8
            taucmc(:,nlay) = 0.0_r8
            ciwpmc(:,nlay) = 0.0_r8
            clwpmc(:,nlay) = 0.0_r8
            reicmc(nlay) = 0.0_r8
            dgesmc(nlay) = 0.0_r8
            relqmc(nlay) = 0.0_r8
            taua(nlay,:) = 0.0_r8
        endif
        IF ( present(kgen_unit) ) THEN
        END IF
    end subroutine inatm
end module rrtmg_lw_rad
