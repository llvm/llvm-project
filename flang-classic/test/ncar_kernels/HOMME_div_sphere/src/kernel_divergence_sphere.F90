        program kgen_kernel_divergence_sphere

        INTEGER , PARAMETER  :: np = 4

        INTEGER(KIND=4)  , PARAMETER :: real_kind = 8

        REAL(KIND=real_kind)  , PARAMETER :: rearth = 6.376d6

        REAL(KIND=real_kind)  , PARAMETER :: rrearth = 1.0_real_kind/rearth

        INTEGER , PARAMETER  :: nc = 4

        INTEGER , PARAMETER :: nelem = 64*30

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

        INTEGER(KIND=4)  , PARAMETER :: int_kind = 4

        INTEGER , PARAMETER  :: npsq = np*np

            TYPE  :: index_t
              INTEGER(KIND=int_kind) ia(npsq), ja(npsq)
              INTEGER(KIND=int_kind) is, ie
              INTEGER(KIND=int_kind) numuniquepts
              INTEGER(KIND=int_kind) uniqueptoffset
            END TYPE index_t

        INTEGER(KIND=4)  , PARAMETER :: long_kind = 8

        INTEGER , PARAMETER  :: nlev = 20

            TYPE  :: elem_accum_t
              REAL(KIND=real_kind) u(np,np,nlev)
              REAL(KIND=real_kind) t(np,np,nlev)
              REAL(KIND=real_kind) ke(np,np,nlev)
            END TYPE elem_accum_t

            TYPE  :: derived_state_t
              REAL(KIND=real_kind) dummmy
              REAL(KIND=real_kind) vstar(np,np,2,nlev)
            END TYPE derived_state_t

        INTEGER  , PARAMETER :: timelevels = 3

            TYPE  :: elem_state_t
              REAL(KIND=real_kind) p(np,np,nlev,timelevels)
              REAL(KIND=real_kind) phis(np,np)
              REAL(KIND=real_kind) gradps(np,np,2)
              REAL(KIND=real_kind) v(np,np,2,nlev,timelevels)
              REAL(KIND=real_kind) couv(np,np,2,nlev)
              REAL(KIND=real_kind) uv(np,np,2,nlev)
              REAL(KIND=real_kind) uv0(np,np,2,nlev)
              REAL(KIND=real_kind) pgrads(np,np,2,nlev)
              REAL(KIND=real_kind) psi(np,np,nlev)
              REAL(KIND=real_kind) phi(np,np,nlev)
              REAL(KIND=real_kind) ht(np,np,nlev)
              REAL(KIND=real_kind) t(np,np,nlev,timelevels)
              REAL(KIND=real_kind) q(np,np,nlev,timelevels)
              REAL(KIND=real_kind) pt3d(np,np,nlev)
              REAL(KIND=real_kind) qt3d(np,np,nlev)
              REAL(KIND=real_kind) peta(np,np,nlev)
              REAL(KIND=real_kind) dp3d(np,np,nlev)
              REAL(KIND=real_kind) zeta(np,np,nlev)
              REAL(KIND=real_kind) pr3d(np,np,nlev+1)
              REAL(KIND=real_kind) pr3d_ref(np,np,nlev+1)
              REAL(KIND=real_kind) gp3d(np,np,nlev+1)
              REAL(KIND=real_kind) ptop(np,np)
              REAL(KIND=real_kind) sgp(np,np)
              REAL(KIND=real_kind) tbar(nlev)
            END TYPE elem_state_t

            TYPE  :: rotation_t
              INTEGER nbr
              INTEGER reverse
              REAL(KIND=real_kind), dimension(:,:,:), pointer :: r => null()
            END TYPE rotation_t

        INTEGER(KIND=4)  , PARAMETER :: log_kind = 4

            TYPE  :: cartesian3d_t
              REAL(KIND=real_kind) x
              REAL(KIND=real_kind) y
              REAL(KIND=real_kind) z
            END TYPE cartesian3d_t

            TYPE  :: edgedescriptor_t
              INTEGER(KIND=int_kind) use_rotation
              INTEGER(KIND=int_kind) padding
              INTEGER(KIND=int_kind), pointer :: putmapp(:) => null()
              INTEGER(KIND=int_kind), pointer :: getmapp(:) => null()
              INTEGER(KIND=int_kind), pointer :: putmapp_ghost(:) => null()
              INTEGER(KIND=int_kind), pointer :: getmapp_ghost(:) => null()
              INTEGER(KIND=int_kind), pointer :: globalid(:) => null()
              INTEGER(KIND=int_kind), pointer :: loc2buf(:) => null()
              TYPE(cartesian3d_t), pointer :: neigh_corners(:,:) => null()
              INTEGER actual_neigh_edges
              LOGICAL(KIND=log_kind), pointer :: reverse(:) => null()
              TYPE(rotation_t), dimension(:), pointer :: rot => null()
            END TYPE edgedescriptor_t

        INTEGER  , PARAMETER :: num_neighbors = 8

            TYPE  :: gridvertex_t
              INTEGER, pointer :: nbrs(:) => null()
              INTEGER, pointer :: nbrs_face(:) => null()
              INTEGER, pointer :: nbrs_wgt(:) => null()
              INTEGER, pointer :: nbrs_wgt_ghost(:) => null()
              INTEGER nbrs_ptr(num_neighbors + 1)
              INTEGER face_number
              INTEGER number
              INTEGER processor_number
              INTEGER spacecurve
            END TYPE gridvertex_t

            TYPE  :: cartesian2d_t
              REAL(KIND=real_kind) x
              REAL(KIND=real_kind) y
            END TYPE cartesian2d_t

            TYPE  :: spherical_polar_t
              REAL(KIND=real_kind) r
              REAL(KIND=real_kind) lon
              REAL(KIND=real_kind) lat
            END TYPE spherical_polar_t

            TYPE  :: element_t
              INTEGER(KIND=int_kind) localid
              INTEGER(KIND=int_kind) globalid
              TYPE(spherical_polar_t) spherep(np,np)
              TYPE(cartesian2d_t) cartp(np,np)
              TYPE(cartesian2d_t) corners(4)
              REAL(KIND=real_kind) u2qmap(4,2)
              TYPE(cartesian3d_t) corners3d(4)
              REAL(KIND=real_kind) area
              REAL(KIND=real_kind) max_eig
              REAL(KIND=real_kind) min_eig
              REAL(KIND=real_kind) max_eig_ratio
              REAL(KIND=real_kind) dx_short
              REAL(KIND=real_kind) dx_long
              REAL(KIND=real_kind) variable_hyperviscosity(np,np)
              REAL(KIND=real_kind) hv_courant
              REAL(KIND=real_kind) tensorvisc(2,2,np,np)
              INTEGER(KIND=int_kind) node_numbers(4)
              INTEGER(KIND=int_kind) node_multiplicity(4)
              TYPE(gridvertex_t) vertex
              TYPE(edgedescriptor_t) desc
              TYPE(elem_state_t) state
              TYPE(derived_state_t) derived
              TYPE(elem_accum_t) accum
              REAL(KIND=real_kind) met(2,2,np,np)
              REAL(KIND=real_kind) metinv(2,2,np,np)
              REAL(KIND=real_kind) metdet(np,np)
              REAL(KIND=real_kind) rmetdet(np,np)
              REAL(KIND=real_kind) d(2,2,np,np)
              REAL(KIND=real_kind) dinv(2,2,np,np)
              REAL(KIND=real_kind) vec_sphere2cart(np,np,3,2)
              REAL(KIND=real_kind) dinv2(np,np,2,2)
              REAL(KIND=real_kind) mp(np,np)
              REAL(KIND=real_kind) rmp(np,np)
              REAL(KIND=real_kind) spheremp(np,np)
              REAL(KIND=real_kind) rspheremp(np,np)
              INTEGER(KIND=long_kind) gdofp(np,np)
              REAL(KIND=real_kind) fcor(np,np)
              TYPE(index_t) idxp
              TYPE(index_t), pointer :: idxv
              INTEGER facenum
              INTEGER dummy
            END TYPE element_t


                REAL(KIND=real_kind) v(np, np, 2)
!JMD !dir$ attributes align : 64 :: v



                TYPE(derivative_t) deriv


                TYPE(element_t) elem
        !JMD manual timer additions
        integer*8 c1,c2,cr,cm
        integer*8 c12,c22,cr2
        real*8 dt, dt2
        integer :: itmax=10000
        character(len=80), parameter :: kname='[kernel_divergence_sphere]'
        character(len=80), parameter :: kname2='[kernel_divergence_sphere_v2]'
        integer :: it
        !JMD

        REAL(KIND=real_kind) :: DinvTemp(np,np,2,2)
        REAL(KIND=real_kind) :: DvvTemp(np,np)


                REAL(KIND=real_kind) KGEN_RESULT_div(np, np,nelem)
                REAL(KIND=real_kind) KGEN_RESULT_div_v2(np, np,nelem)
                REAL(KIND=real_kind) KGEN_div(np, np)


            ! populate dummy initial values
            do j=1,np
                do i=1,np
                    elem%metdet(i,j) = 0.1_real_kind * i
                    elem%Dinv(1,1,i,j) = 0.2_real_kind * j
                    elem%Dinv(1,2,i,j) = 0.3_real_kind * i*j
                    elem%Dinv(2,1,i,j) = 0.4_real_kind * i
                    elem%Dinv(2,2,i,j) = 0.5_real_kind * j
                    v(i,j,1) = 0.6_real_kind * i*j
                    v(i,j,2) = 0.7_real_kind * i
                    deriv%Dvv(i,j) = 0.8_real_kind * j
                    elem%rmetdet(i,j) = 1.0_real_kind / elem%metdet(i,j)
                    elem%Dinv2(i,j,1,1) = elem%Dinv(1,1,i,j)
                    elem%Dinv2(i,j,1,2) = elem%Dinv(1,2,i,j)
                    elem%Dinv2(i,j,2,1) = elem%Dinv(2,1,i,j)
                    elem%Dinv2(i,j,2,2) = elem%Dinv(2,2,i,j)
                end do
            end do
            DinvTemp(:,:,1,1) = elem%Dinv(1,1,:,:)
            DinvTemp(:,:,1,2) = elem%Dinv(1,2,:,:)
            DinvTemp(:,:,2,1) = elem%Dinv(2,1,:,:)
            DinvTemp(:,:,2,2) = elem%Dinv(2,2,:,:)

            ! reference result
            KGEN_div = divergence_sphere_ref(v,deriv,elem)

            dvvTemp(:,:) = deriv%dvv(:,:)
            call system_clock(c12,cr2,cm)
            ! modified result
            do it=1,itmax
               do ie=1,nelem
!JMD               KGEN_RESULT_div = divergence_sphere_v2(v,deriv,elem,DinvTemp)
               KGEN_RESULT_div(:,:,ie) = divergence_sphere_v2(v,dvvTemp,elem,DinvTemp)
               enddo
            enddo
            call system_clock(c22,cr2,cm)
            dt2 = dble(c22-c12)/dble(cr2)
            print *, TRIM(kname2), ' total time (sec): ',dt2
            print *, TRIM(kname2), ' time per call (usec): ',1.e6*dt2/dble(itmax)

            ! populate dummy initial values
            do j=1,np
                do i=1,np
                    elem%metdet(i,j) = 0.1_real_kind * i
                    elem%Dinv(1,1,i,j) = 0.2_real_kind * j
                    elem%Dinv(1,2,i,j) = 0.3_real_kind * i*j
                    elem%Dinv(2,1,i,j) = 0.4_real_kind * i
                    elem%Dinv(2,2,i,j) = 0.5_real_kind * j
                    v(i,j,1) = 0.6_real_kind * i*j
                    v(i,j,2) = 0.7_real_kind * i
                    deriv%Dvv(i,j) = 0.8_real_kind * j
                    elem%rmetdet(i,j) = 1.0_real_kind / elem%metdet(i,j)
                    elem%Dinv2(i,j,1,1) = elem%Dinv(1,1,i,j)
                    elem%Dinv2(i,j,1,2) = elem%Dinv(1,2,i,j)
                    elem%Dinv2(i,j,2,1) = elem%Dinv(2,1,i,j)
                    elem%Dinv2(i,j,2,2) = elem%Dinv(2,2,i,j)
                end do
            end do
            DinvTemp(:,:,1,1) = elem%Dinv(1,1,:,:)
            DinvTemp(:,:,1,2) = elem%Dinv(1,2,:,:)
            DinvTemp(:,:,2,1) = elem%Dinv(2,1,:,:)
            DinvTemp(:,:,2,2) = elem%Dinv(2,2,:,:)


            call system_clock(c1,cr,cm)
            ! modified result
            do it=1,itmax
               do ie=1,nelem
               KGEN_RESULT_div(:,:,ie) = divergence_sphere(v,deriv,elem)
               enddo
            enddo
            call system_clock(c2,cr,cm)
            dt = dble(c2-c1)/dble(cr)
            print *, TRIM(kname), ' total time (sec): ',dt
            print *, TRIM(kname), ' time per call (usec): ',1.e6*dt/dble(itmax)


            IF ( ALL( KGEN_div == KGEN_RESULT_div(:,:,1) ) ) THEN
                WRITE(*,*) "div is identical.  Test PASSED"
                WRITE(*,*) "Modified: ", KGEN_div
                WRITE(*,*) "Reference:  ", KGEN_RESULT_div(:,:,1)
            ELSE
                WRITE(*,*) "div is NOT identical.  Test FAILED"
                WRITE(*,*) COUNT( KGEN_div /= KGEN_RESULT_div(:,:,1)), " of ", SIZE( KGEN_RESULT_div ), " elements are different."
                WRITE(*,*) "RMS of difference is ", SQRT(SUM((KGEN_div - KGEN_RESULT_div(:,:,1))**2)/SIZE(KGEN_div))
                WRITE(*,*) "Minimum difference is ", MINVAL(ABS(KGEN_div - KGEN_RESULT_div(:,:,1)))
                WRITE(*,*) "Maximum difference is ", MAXVAL(ABS(KGEN_div - KGEN_RESULT_div(:,:,1)))
                WRITE(*,*) "Mean value of kernel-generated div is ", SUM(KGEN_RESULT_div(:,:,1))/SIZE(KGEN_RESULT_div(:,:,1))
                WRITE(*,*) "Mean value of original div is ", SUM(KGEN_div)/SIZE(KGEN_div)
                WRITE(*,*) ""
                STOP
            END IF

        contains

        function divergence_sphere_ref(v,deriv,elem) result(div)
        !
        !   input:  v = velocity in lat-lon coordinates
        !   ouput:  div(v)  spherical divergence of v
        !
              real(kind=real_kind), intent(in) :: v(np,np,2)
        ! in lat-lon coordinates
              type (derivative_t), intent(in) :: deriv
              type (element_t), intent(in) :: elem
              real(kind=real_kind) :: div(np,np)

        ! Local

              integer i
              integer j
              integer l

              real(kind=real_kind) ::  dudx00
              real(kind=real_kind) ::  dvdy00
              real(kind=real_kind) ::  gv(np,np,2),vvtemp(np,np)

        ! convert to contra variant form and multiply by g
              do j=1,np
                    do i=1,np
                          gv(i,j,1)=elem%metdet(i,j)*(elem%Dinv(1,1,i,j)*v(i,j,1) + elem%Dinv(1,2,i,j)*v(i,j,2))
                          gv(i,j,2)=elem%metdet(i,j)*(elem%Dinv(2,1,i,j)*v(i,j,1) + elem%Dinv(2,2,i,j)*v(i,j,2))
                    enddo
              enddo

        ! compute d/dx and d/dy
              do j=1,np
                    do l=1,np
                          dudx00=0.0d0
                          dvdy00=0.0d0
                          do i=1,np
                                dudx00 = dudx00 + deriv%Dvv(i,l  )*gv(i,j  ,1)
                                dvdy00 = dvdy00 + deriv%Dvv(i,l  )*gv(j  ,i,2)
                          end do
                          div(l  ,j  ) = dudx00
                          vvtemp(j  ,l  ) = dvdy00
                    end do
              end do


              do j=1,np
                    do i=1,np
                          div(i,j)=(div(i,j)+vvtemp(i,j))*(elem%rmetdet(i,j)*rrearth)
                    end do
              end do

        end function divergence_sphere_ref

        function divergence_sphere(v,deriv,elem) result(div)
        !
        !   input:  v = velocity in lat-lon coordinates
        !   ouput:  div(v)  spherical divergence of v
        !
              real(kind=real_kind), intent(in) :: v(np,np,2)
        ! in lat-lon coordinates
              type (derivative_t), intent(in) :: deriv
              type (element_t), intent(in) :: elem
              real(kind=real_kind) :: div(np,np)

        ! Local

              integer i
              integer j
              integer l

              real(kind=real_kind) :: dudx00
              real(kind=real_kind) :: dvdy00
              real(kind=real_kind) :: gv(np,np,2)
              real(kind=real_kind) :: vvtemp(np,np)

        ! convert to contra variant form and multiply by g
              do j=1,np
                    do i=1,np
                          gv(i,j,1)=elem%metdet(i,j)*(elem%Dinv(1,1,i,j)*v(i,j,1) + elem%Dinv(1,2,i,j)*v(i,j,2))
                          gv(i,j,2)=elem%metdet(i,j)*(elem%Dinv(2,1,i,j)*v(i,j,1) + elem%Dinv(2,2,i,j)*v(i,j,2))
                    enddo
              enddo

        ! compute d/dx and d/dy
              do j=1,np
                    do l=1,np
                          dudx00=0.0d0
                          dvdy00=0.0d0
                          do i=1,np
                                dudx00 = dudx00 + deriv%Dvv(i,l  )*gv(i,j  ,1)
                                dvdy00 = dvdy00 + deriv%Dvv(i,l  )*gv(j  ,i,2)
                          end do
                          div(l  ,j  ) = dudx00
                          vvtemp(j  ,l  ) = dvdy00
                    end do
              end do


              do j=1,np
                    do i=1,np
                          div(i,j)=(div(i,j)+vvtemp(i,j))*(elem%rmetdet(i,j)*rrearth)
                    end do
              end do

        end function divergence_sphere

!DIR$ ATTRIBUTES FORCEINLINE :: divergence_sphere_v2
        function divergence_sphere_v2(v,dvv,elem,Dinv2) result(div)
        !
        !   input:  v = velocity in lat-lon coordinates
        !   ouput:  div(v)  spherical divergence of v
        !
              real(kind=real_kind), intent(in) :: v(np,np,2)
        ! in lat-lon coordinates
              !JMD type (derivative_t), intent(in) :: deriv
              type (element_t), intent(in) :: elem
              real(kind=real_kind), intent(in) :: Dinv2(np,np,2,2)
              real(kind=real_kind), intent(in) :: dvv(np,np)
              real(kind=real_kind) :: div(np,np)

        ! Local

              integer i
              integer j
              integer l

              real(kind=real_kind) :: dudx00
              real(kind=real_kind) :: dvdy00
              real(kind=real_kind) :: gv1(np,np),gv2(np,np)
              real(kind=real_kind) :: vvtemp(np,np)

        ! convert to contra variant form and multiply by g
              do j=1,np
                    do i=1,np
!JMD                          gv1(i,j)=metdet(i,j)*(Dinv(1,1,i,j)*v(i,j,1) + Dinv(1,2,i,j)*v(i,j,2))
!JMD                          gv2(i,j)=metdet(i,j)*(Dinv(2,1,i,j)*v(i,j,1) + Dinv(2,2,i,j)*v(i,j,2))
                          gv1(i,j)=elem%metdet(i,j)*(elem%Dinv2(i,j,1,1)*v(i,j,1) + elem%Dinv2(i,j,1,2)*v(i,j,2))
                          gv2(i,j)=elem%metdet(i,j)*(elem%Dinv2(i,j,2,1)*v(i,j,1) + elem%Dinv2(i,j,2,2)*v(i,j,2))
                    enddo
              enddo

        ! compute d/dx and d/dy
              do j=1,np
                    do l=1,np
                         dudx00=0.0d0
                         dvdy00=0.0d0
!DIR$ UNROLL(4)
                         do i=1,np
                                dudx00 = Dvv(i,l  )*gv1(i,j  )
                                dvdy00 = Dvv(i,l  )*gv2(j  ,i)

                          end do
                          div(l  ,j  ) = dudx00
                          vvtemp(j  ,l  ) = dvdy00
                    end do
              end do


              do j=1,np
                    do i=1,np
                          div(i,j)=(div(i,j)+vvtemp(i,j))*(elem%rmetdet(i,j)*rrearth)
                    end do
              end do

        end function divergence_sphere_v2



        end program kgen_kernel_divergence_sphere
