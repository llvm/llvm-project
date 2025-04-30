        program kgen_kernel_gradient_sphere

        INTEGER(KIND=4)  , PARAMETER :: real_kind = 8

        REAL(KIND=real_kind)  , PARAMETER :: rearth = 6.376d6

        REAL(KIND=real_kind)  , PARAMETER :: rrearth = 1.0_real_kind/rearth

        INTEGER , PARAMETER  :: np = 4

        INTEGER , Parameter  :: nelem = 30*64

        INTEGER , PARAMETER  :: nc = 4

        INTEGER , PARAMETER  :: nip = 3

        INTEGER , PARAMETER  :: nipm = nip-1

        INTEGER , PARAMETER  :: nep = nipm*nc+1

            TYPE  :: derivative_t
              REAL(KIND=real_kind) dvv(np,np)
              REAL(KIND=real_kind) dvv_diag(np,np)
              REAL(KIND=real_kind) dvv_twt(np,np)
              REAL(KIND=real_kind) mvv_twt(np,np)
              ! diagonal matrix of GLL weights
              REAL(KIND=real_kind) mfvm(np,nc+1)
              REAL(KIND=real_kind) cfvm(np,nc)
              REAL(KIND=real_kind) sfvm(np,nep)
              REAL(KIND=real_kind) legdg(np,np)
            END TYPE derivative_t

        TYPE :: element_t
           REAL(KIND=real_kind) dinv(2,2,np,np)
        END TYPE element_t

        TYPE :: element_t2
           REAL(KIND=real_kind) dinv2(np,np,2,2)
           REAL(KIND=real_kind) ds(np,np,2)
        END TYPE element_t2

        type (element_t), allocatable :: elem(:)
        type (element_t2), allocatable :: elem2(:)


        REAL(KIND=real_kind) s(np, np,nelem)
        TYPE(derivative_t) deriv
        REAL(KIND=real_kind), DIMENSION(2, 2, np, np,nelem) :: dinv
        REAL(KIND=real_kind), DIMENSION(np,np,2,2) :: dinv2b
        REAL(KIND=real_kind), dimension(np,np,2,2,nelem) :: dinv2 
        REAL(KIND=real_kind) KGEN_RESULT_ds(np, np, 2,nelem)
        REAL(KIND=real_kind), dimension(np,np,2) ::  KGEN_RESULT_ds2b
        REAL(KIND=real_kind) KGEN_ds(np, np, 2)

        !JMD manual timer additions
        integer*8 c1,c2,cr,cm
        real*8 dt
        real*8 flops 
        integer :: itmax
        character(len=80), parameter :: kname1='[kernel_gradient_sphere_v1]'
        character(len=80), parameter :: kname2a='[kernel_gradient_sphere_v2a]'
        character(len=80), parameter :: kname2b='[kernel_gradient_sphere_v2b]'
        character(len=80), parameter :: kname2c='[kernel_gradient_sphere_v2c]'
        character(len=80), parameter :: kname2d='[kernel_gradient_sphere_v2d]'
        character(len=80), parameter :: kname2e='[kernel_gradient_sphere_v2e]'
        character(len=80), parameter :: kname2f='[kernel_gradient_sphere_v2f]'
        integer :: it
        !JMD
!DIR$ ATTRIBUTES ALIGN:64 :: element_t2
!DIR$ ATTRIBUTES align:64 :: elem, elem2
!DIR$ ATTRIBUTES ALIGN:64 :: KGEN_RESULT_ds

        allocate(elem(nelem))
        allocate(elem2(nelem))
        itmax = ceiling(real(10000000,kind=real_kind)/real(nelem,kind=real_kind))


            ! populate dummy initial values
            do j=1,np
                do i=1,np
                    s(i,j,:) = 0.6_real_kind * i*j
                    deriv%Dvv(i,j) = 0.8_real_kind * j

                    Dinv(1,1,i,j,:) = 0.2_real_kind * j
                    Dinv(2,1,i,j,:) = 0.3_real_kind * i*j
                    Dinv(2,1,i,j,:) = 0.4_real_kind * i
                    Dinv(2,2,i,j,:) = 0.5_real_kind * j
                    Dinv2(i,j,1,1,:) = Dinv(1,1,i,j,:)
                    Dinv2(i,j,1,2,:) = Dinv(1,2,i,j,:)
                    Dinv2(i,j,2,1,:) = Dinv(2,1,i,j,:)
                    Dinv2(i,j,2,2,:) = Dinv(2,2,i,j,:)
                end do
            end do
            do ie=1,nelem
               elem(ie)%dinv   = Dinv(:,:,:,:,ie)
               elem2(ie)%dinv2 = Dinv2(:,:,:,:,ie)
            enddo
            dinv2b = Dinv2(:,:,:,:,1)

            ! reference result
            ! KGEN_ds = gradient_sphere_ref(s,deriv,dinv(:,:,:,:,1))
            KGEN_ds = gradient_sphere_ref(s,deriv,elem(1)%dinv)

            call system_clock(c1,cr,cm)
            ! modified result
            do it=1,itmax
               do ie=1,nelem
!                  KGEN_RESULT_ds(:,:,:,ie) = gradient_sphere_v1(s(:,:,ie),deriv,dinv(:,:,:,:,ie))
                  KGEN_RESULT_ds(:,:,:,ie)= gradient_sphere_v1(s(:,:,ie),deriv,elem(ie)%dinv)
               enddo
            enddo
            call system_clock(c2,cr,cm)
            dt = dble(c2-c1)/dble(cr)
!            flops = real(nelem,kind=real_kind)*real(4*np*np*np + 5*np*np,kind=real_kind)*real(itmax,kind=real_kind)
            print *, TRIM(kname1), ' total time (sec): ',dt
            print *, TRIM(kname1), ' time per call (usec): ',1.e6*dt/dble(itmax)

#if 0
            call system_clock(c1,cr,cm)
            ! modified result
            do it=1,itmax
               do ie=1,nelem
                  KGEN_RESULT_ds(:,:,:,ie)  = gradient_sphere_v2(s(:,:,ie),deriv,dinv2(:,:,:,:,ie))
               enddo
            enddo
            call system_clock(c2,cr,cm)
            dt = dble(c2-c1)/dble(cr)
            print *, TRIM(kname2a), ' total time (sec): ',dt
            print *, TRIM(kname2a), ' time per call (usec): ',1.e6*dt/dble(itmax)
#endif

            if(nelem==1) then 
               call system_clock(c1,cr,cm)
               ! modified result
               do it=1,itmax
                  do ie=1,nelem
                     KGEN_RESULT_ds2b = gradient_sphere_v2(s(:,:,ie),deriv,elem2(ie)%dinv2)
                  enddo
               enddo
               call system_clock(c2,cr,cm)
               dt = dble(c2-c1)/dble(cr)
               print *, TRIM(kname2b), ' total time (sec): ',dt
               print *, TRIM(kname2b), ' time per call (usec): ',1.e6*dt/dble(itmax)
            endif

#if 0
            call system_clock(c1,cr,cm)
            ! modified result
            do it=1,itmax
               do ie=1,nelem
                   elem2(ie)%ds  = gradient_sphere_v2(s(:,:,ie),deriv,elem2(ie)%dinv2)
               enddo
            enddo
            call system_clock(c2,cr,cm)
            dt = dble(c2-c1)/dble(cr)
            print *, TRIM(kname2c), ' total time (sec): ',dt
            print *, TRIM(kname2c), ' time per call (usec): ',1.e6*dt/dble(itmax)
#endif

            call system_clock(c1,cr,cm)
            ! modified result
            do it=1,itmax
               do ie=1,nelem
                  elem2(ie)%ds  = gradient_sphere_v2(s(:,:,ie),deriv,elem2(ie)%dinv2)
               enddo
            enddo
            call system_clock(c2,cr,cm)
            dt = dble(c2-c1)/dble(cr)
            print *, TRIM(kname2d), ' total time (sec): ',dt
            print *, TRIM(kname2d), ' time per call (usec): ',1.e6*dt/dble(itmax)

            if (nelem == 1) then 
               call system_clock(c1,cr,cm)
               ! modified result
               do it=1,itmax
                  do ie=1,nelem
                     KGEN_RESULT_ds2b  = gradient_sphere_v2(s(:,:,ie),deriv,dinv2(:,:,:,:,ie))
                  enddo
               enddo
               call system_clock(c2,cr,cm)
               dt = dble(c2-c1)/dble(cr)
               print *, TRIM(kname2e), ' total time (sec): ',dt
               print *, TRIM(kname2e), ' time per call (usec): ',1.e6*dt/dble(itmax)
            endif

#if 0
            call system_clock(c1,cr,cm)
            ! modified result
            do it=1,itmax
               do ie=1,nelem
                  KGEN_RESULT_ds(:,:,:,ie) = gradient_sphere_v2(s(:,:,ie),deriv,dinv2(:,:,:,:,ie))
               enddo
            enddo
            call system_clock(c2,cr,cm)
            dt = dble(c2-c1)/dble(cr)
            print *, TRIM(kname2f), ' total time (sec): ',dt
            print *, TRIM(kname2f), ' time per call (usec): ',1.e6*dt/dble(itmax)
#endif






            IF ( ALL( KGEN_ds == KGEN_RESULT_ds(:,:,:,1) ) ) THEN
                WRITE(*,*) "ds is identical."
                WRITE(*,*) "PASSED"
!                WRITE(*,*) "Modified: ", KGEN_ds
!                WRITE(*,*) "Reference:  ", KGEN_RESULT_ds(:,:,:,1)
            ELSE
                WRITE(*,*) "ds is NOT identical."
                WRITE(*,*) "FAILED"
                WRITE(*,*) COUNT( KGEN_ds /= KGEN_RESULT_ds(:,:,:,1)), " of ", SIZE( KGEN_RESULT_ds(:,:,:,1) ), " elements are different."
                WRITE(*,*) "RMS of difference is ", SQRT(SUM((KGEN_ds - KGEN_RESULT_ds(:,:,:,1))**2)/SIZE(KGEN_ds))
                WRITE(*,*) "Minimum difference is ", MINVAL(ABS(KGEN_ds - KGEN_RESULT_ds(:,:,:,1)))
                WRITE(*,*) "Maximum difference is ", MAXVAL(ABS(KGEN_ds - KGEN_RESULT_ds(:,:,:,1)))
                WRITE(*,*) "Mean value of kernel-generated ds is ", SUM(KGEN_RESULT_ds(:,:,:,1))/SIZE(KGEN_RESULT_ds(:,:,:,1))
                WRITE(*,*) "Mean value of original ds is ", SUM(KGEN_ds)/SIZE(KGEN_ds)
                WRITE(*,*) ""
                STOP
            END IF

        contains

        function gradient_sphere_ref(s,deriv,Dinv) result(ds)
        !
        !   input s:  scalar
        !   output  ds: spherical gradient of s, lat-lon coordinates
        !

              type (derivative_t), intent(in) :: deriv
              real(kind=real_kind), intent(in), dimension(2,2,np,np) :: Dinv
              real(kind=real_kind), intent(in) :: s(np,np)

              real(kind=real_kind) :: ds(np,np,2)

              integer i
              integer j
              integer l

              real(kind=real_kind) ::  dsdx00
              real(kind=real_kind) ::  dsdy00
              real(kind=real_kind) ::  v1(np,np),v2(np,np)

              do j=1,np
                    do l=1,np
                          dsdx00=0.0d0
                          dsdy00=0.0d0
                          do i=1,np
                                dsdx00 = dsdx00 + deriv%Dvv(i,l  )*s(i,j  )
                                dsdy00 = dsdy00 + deriv%Dvv(i,l  )*s(j  ,i)
                          end do
                          v1(l  ,j  ) = dsdx00*rrearth
                          v2(j  ,l  ) = dsdy00*rrearth
                    end do
              end do
        ! convert covarient to latlon
              do j=1,np
                    do i=1,np
                          ds(i,j,1)=Dinv(1,1,i,j)*v1(i,j) + Dinv(2,1,i,j)*v2(i,j)
                          ds(i,j,2)=Dinv(1,2,i,j)*v1(i,j) + Dinv(2,2,i,j)*v2(i,j)
                    enddo
              enddo

        end function gradient_sphere_ref

!DIR$ ATTRIBUTES FORCEINLINE :: gradient_sphere_v1
        function gradient_sphere_v1(s,deriv,Dinv) result(ds)
        !
        !   input s:  scalar
        !   output  ds: spherical gradient of s, lat-lon coordinates
        !

              type (derivative_t), intent(in) :: deriv
              real(kind=real_kind), intent(in), dimension(2,2,np,np) :: Dinv
              real(kind=real_kind), intent(in) :: s(np,np)

              real(kind=real_kind) :: ds(np,np,2)

              integer i
              integer j
              integer l

              real(kind=real_kind) ::  dsdx00
              real(kind=real_kind) ::  dsdy00
              real(kind=real_kind) ::  v1(np,np),v2(np,np)

              do j=1,np
                    do l=1,np
                          dsdx00=0.0d0
                          dsdy00=0.0d0
                          do i=1,np
                                dsdx00 = dsdx00 + deriv%Dvv(i,l  )*s(i,j  )
                                dsdy00 = dsdy00 + deriv%Dvv(i,l  )*s(j  ,i)
                          end do
                          v1(l  ,j  ) = dsdx00*rrearth
                          v2(j  ,l  ) = dsdy00*rrearth
                    end do
              end do
        ! convert covarient to latlon
              do j=1,np
                    do i=1,np
                          ds(i,j,1)=Dinv(1,1,i,j)*v1(i,j) + Dinv(2,1,i,j)*v2(i,j)
                          ds(i,j,2)=Dinv(1,2,i,j)*v1(i,j) + Dinv(2,2,i,j)*v2(i,j)
                    enddo
              enddo

        end function gradient_sphere_v1

!DIR$ ATTRIBUTES FORCEINLINE :: gradient_sphere_v2
        function gradient_sphere_v2(s,deriv,Dinv) result(ds)
        !
        !   input s:  scalar
        !   output  ds: spherical gradient of s, lat-lon coordinates
        !

              type (derivative_t), intent(in) :: deriv
              real(kind=real_kind), intent(in), dimension(np,np,2,2) :: Dinv
              real(kind=real_kind), intent(in) :: s(np,np)

              real(kind=real_kind) :: ds(np,np,2)
!DIR$ ATTRIBUTES ALIGN:64 :: ds

              integer i
              integer j
              integer l

              real(kind=real_kind) ::  dsdx00
              real(kind=real_kind) ::  dsdy00
              real(kind=real_kind) ::  v1(np,np),v2(np,np)

              do j=1,np
                    do l=1,np
                          dsdx00=0.0d0
                          dsdy00=0.0d0
!DIR$ UNROLL(4)
                          do i=1,np
                                dsdx00 = dsdx00 + deriv%Dvv(i,l  )*s(i,j  )
                                dsdy00 = dsdy00 + deriv%Dvv(i,l  )*s(j  ,i)
                          end do
                          v1(l  ,j  ) = dsdx00*rrearth
                          v2(j  ,l  ) = dsdy00*rrearth
                    end do
              end do
        ! convert covarient to latlon
              do j=1,np
                    do i=1,np
                          ds(i,j,1)=Dinv(i,j,1,1)*v1(i,j) + Dinv(i,j,2,1)*v2(i,j)
                          ds(i,j,2)=Dinv(i,j,1,2)*v1(i,j) + Dinv(i,j,2,2)*v2(i,j)
                    enddo
              enddo

        end function gradient_sphere_v2


        end program kgen_kernel_gradient_sphere
