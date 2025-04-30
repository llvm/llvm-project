
! KGEN-generated Fortran source file
!
! Filename    : element_mod.F90
! Generated at: 2015-04-12 19:17:34
! KGEN version: 0.4.9



    MODULE element_mod
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
    USE coordinate_systems_mod, ONLY : kgen_read_mod6 => kgen_read
    USE coordinate_systems_mod, ONLY : kgen_verify_mod6 => kgen_verify
    USE gridgraph_mod, ONLY : kgen_read_mod8 => kgen_read
    USE gridgraph_mod, ONLY : kgen_verify_mod8 => kgen_verify
    USE edge_mod, ONLY : kgen_read_mod9 => kgen_read
    USE edge_mod, ONLY : kgen_verify_mod9 => kgen_verify
        USE kinds, ONLY: int_kind
        USE kinds, ONLY: real_kind
        USE kinds, ONLY: long_kind
        USE coordinate_systems_mod, ONLY: spherical_polar_t
        USE coordinate_systems_mod, ONLY: cartesian2d_t
        USE coordinate_systems_mod, ONLY: cartesian3d_t
        USE dimensions_mod, ONLY: np
        USE dimensions_mod, ONLY: nlev
        USE dimensions_mod, ONLY: qsize_d
        USE dimensions_mod, ONLY: nlevp
        USE dimensions_mod, ONLY: npsq
        USE edge_mod, ONLY: edgedescriptor_t
        USE gridgraph_mod, ONLY: gridvertex_t
        IMPLICIT NONE
        PRIVATE
        INTEGER, public, parameter :: timelevels = 3
        ! =========== PRIMITIVE-EQUATION DATA-STRUCTURES =====================
        TYPE, public :: elem_state_t
            ! prognostic variables for preqx solver
            ! prognostics must match those in prim_restart_mod.F90
            ! vertically-lagrangian code advects dp3d instead of ps_v
            ! tracers Q, Qdp always use 2 level time scheme
            REAL(KIND=real_kind) :: v   (np,np,2,nlev,timelevels) ! velocity                           1
            REAL(KIND=real_kind) :: t   (np,np,nlev,timelevels) ! temperature                        2
            REAL(KIND=real_kind) :: dp3d(np,np,nlev,timelevels) ! delta p on levels                  8
            REAL(KIND=real_kind) :: lnps(np,np,timelevels) ! log surface pressure               3
            REAL(KIND=real_kind) :: ps_v(np,np,timelevels) ! surface pressure                   4
            REAL(KIND=real_kind) :: phis(np,np) ! surface geopotential (prescribed)  5
            REAL(KIND=real_kind) :: q   (np,np,nlev,qsize_d) ! Tracer concentration               6
            REAL(KIND=real_kind) :: qdp (np,np,nlev,qsize_d,2) ! Tracer mass                        7
        END TYPE elem_state_t
        ! num prognistics variables (for prim_restart_mod.F90)
        !___________________________________________________________________
        TYPE, public :: derived_state_t
            ! diagnostic variables for preqx solver
            ! storage for subcycling tracers/dynamics
            ! if (compute_mean_flux==1) vn0=time_avg(U*dp) else vn0=U at tracer-time t
            REAL(KIND=real_kind) :: vn0  (np,np,2,nlev) ! velocity for SE tracer advection
            REAL(KIND=real_kind) :: vstar(np,np,2,nlev) ! velocity on Lagrangian surfaces
            REAL(KIND=real_kind) :: dpdiss_biharmonic(np,np,nlev) ! mean dp dissipation tendency, if nu_p>0
            REAL(KIND=real_kind) :: dpdiss_ave(np,np,nlev) ! mean dp used to compute psdiss_tens
            ! diagnostics for explicit timestep
            REAL(KIND=real_kind) :: phi(np,np,nlev) ! geopotential
            REAL(KIND=real_kind) :: omega_p(np,np,nlev) ! vertical tendency (derived)
            REAL(KIND=real_kind) :: eta_dot_dpdn(np,np,nlevp) ! mean vertical flux from dynamics
            ! semi-implicit diagnostics: computed in explict-component, reused in Helmholtz-component.
            REAL(KIND=real_kind) :: grad_lnps(np,np,2) ! gradient of log surface pressure
            REAL(KIND=real_kind) :: zeta(np,np,nlev) ! relative vorticity
            REAL(KIND=real_kind) :: div(np,np,nlev,timelevels) ! divergence
            ! tracer advection fields used for consistency and limiters
            REAL(KIND=real_kind) :: dp(np,np,nlev) ! for dp_tracers at physics timestep
            REAL(KIND=real_kind) :: divdp(np,np,nlev) ! divergence of dp
            REAL(KIND=real_kind) :: divdp_proj(np,np,nlev) ! DSSed divdp
            ! forcing terms for 1
            REAL(KIND=real_kind) :: fq(np,np,nlev,qsize_d, 1) ! tracer forcing
            REAL(KIND=real_kind) :: fm(np,np,2,nlev, 1) ! momentum forcing
            REAL(KIND=real_kind) :: ft(np,np,nlev, 1) ! temperature forcing
            REAL(KIND=real_kind) :: omega_prescribed(np,np,nlev) ! prescribed vertical tendency
            ! forcing terms for both 1 and HOMME
            ! FQps for conserving dry mass in the presence of precipitation
            REAL(KIND=real_kind) :: pecnd(np,np,nlev) ! pressure perturbation from condensate
            REAL(KIND=real_kind) :: fqps(np,np,timelevels) ! forcing of FQ on ps_v
        END TYPE derived_state_t
        !___________________________________________________________________
        TYPE, public :: elem_accum_t
            ! the "4" timelevels represents data computed at:
            !  1  t-.5
            !  2  t+.5   after dynamics
            !  3  t+.5   after forcing
            !  4  t+.5   after Robert
            ! after calling TimeLevelUpdate, all times above decrease by 1.0
            REAL(KIND=real_kind) :: kener(np,np,4)
            REAL(KIND=real_kind) :: pener(np,np,4)
            REAL(KIND=real_kind) :: iener(np,np,4)
            REAL(KIND=real_kind) :: iener_wet(np,np,4)
            REAL(KIND=real_kind) :: qvar(np,np,qsize_d,4) ! Q variance at half time levels
            REAL(KIND=real_kind) :: qmass(np,np,qsize_d,4) ! Q mass at half time levels
            REAL(KIND=real_kind) :: q1mass(np,np,qsize_d) ! Q mass at full time levels
        END TYPE elem_accum_t
        ! ============= DATA-STRUCTURES COMMON TO ALL SOLVERS ================
        TYPE, public :: index_t
            INTEGER(KIND=int_kind) :: ia(npsq), ja(npsq)
            INTEGER(KIND=int_kind) :: is, ie
            INTEGER(KIND=int_kind) :: numuniquepts
            INTEGER(KIND=int_kind) :: uniqueptoffset
        END TYPE index_t
        !___________________________________________________________________
        TYPE, public :: element_t
            INTEGER(KIND=int_kind) :: localid
            INTEGER(KIND=int_kind) :: globalid
            ! Coordinate values of element points
            TYPE(spherical_polar_t) :: spherep(np,np) ! Spherical coords of GLL points
            ! Equ-angular gnomonic projection coordinates
            TYPE(cartesian2d_t) :: cartp(np,np) ! gnomonic coords of GLL points
            TYPE(cartesian2d_t) :: corners(4) ! gnomonic coords of element corners
            REAL(KIND=real_kind) :: u2qmap(4,2) ! bilinear map from ref element to quad in cubedsphere coordinates
            ! SHOULD BE REMOVED
            ! 3D cartesian coordinates
            TYPE(cartesian3d_t) :: corners3d(4)
            ! Element diagnostics
            REAL(KIND=real_kind) :: area ! Area of element
            REAL(KIND=real_kind) :: normdinv ! some type of norm of Dinv used for CFL
            REAL(KIND=real_kind) :: dx_short ! short length scale in km
            REAL(KIND=real_kind) :: dx_long ! long length scale in km
            REAL(KIND=real_kind) :: variable_hyperviscosity(np,np) ! hyperviscosity based on above
            REAL(KIND=real_kind) :: hv_courant ! hyperviscosity courant number
            REAL(KIND=real_kind) :: tensorvisc(2,2,np,np) !og, matrix V for tensor viscosity
            ! Edge connectivity information
            !     integer(kind=int_kind)   :: node_numbers(4)
            !     integer(kind=int_kind)   :: node_multiplicity(4)                 ! number of elements sharing corner node
            TYPE(gridvertex_t) :: vertex ! element grid vertex information
            TYPE(edgedescriptor_t) :: desc
            TYPE(elem_state_t) :: state
            TYPE(derived_state_t) :: derived
            TYPE(elem_accum_t) :: accum
            ! Metric terms
            REAL(KIND=real_kind) :: met(2,2,np,np) ! metric tensor on velocity and pressure grid
            REAL(KIND=real_kind) :: metinv(2,2,np,np) ! metric tensor on velocity and pressure grid
            REAL(KIND=real_kind) :: metdet(np,np) ! g = SQRT(det(g_ij)) on velocity and pressure grid
            REAL(KIND=real_kind) :: rmetdet(np,np) ! 1/metdet on velocity pressure grid
            REAL(KIND=real_kind) :: d(2,2,np,np) ! Map covariant field on cube to vector field on the sphere
            REAL(KIND=real_kind) :: dinv(2,2,np,np) ! Map vector field on the sphere to covariant v on cube
            ! Convert vector fields from spherical to rectangular components
            ! The transpose of this operation is its pseudoinverse.
            REAL(KIND=real_kind) :: vec_sphere2cart(np,np,3,2)
            ! Mass matrix terms for an element on a cube face
            REAL(KIND=real_kind) :: mp(np,np) ! mass matrix on v and p grid
            REAL(KIND=real_kind) :: rmp(np,np) ! inverse mass matrix on v and p grid
            ! Mass matrix terms for an element on the sphere
            ! This mass matrix is used when solving the equations in weak form
            ! with the natural (surface area of the sphere) inner product
            REAL(KIND=real_kind) :: spheremp(np,np) ! mass matrix on v and p grid
            REAL(KIND=real_kind) :: rspheremp(np,np) ! inverse mass matrix on v and p grid
            INTEGER(KIND=long_kind) :: gdofp(np,np) ! global degree of freedom (P-grid)
            REAL(KIND=real_kind) :: fcor(np,np) ! Coreolis term
            TYPE(index_t) :: idxp
            TYPE(index_t), pointer :: idxv
            INTEGER :: facenum
            ! force element_t to be a multiple of 8 bytes.
            ! on BGP, code will crash (signal 7, or signal 15) if 8 byte alignment is off
            ! check core file for:
            ! core.63:Generated by interrupt..(Alignment Exception DEAR=0xa1ef671c ESR=0x01800000 CCR0=0x4800a002)
            INTEGER :: dummy
        END TYPE element_t
        !___________________________________________________________________

        ! read interface
        PUBLIC kgen_read
        INTERFACE kgen_read
            MODULE PROCEDURE kgen_read_elem_state_t
            MODULE PROCEDURE kgen_read_derived_state_t
            MODULE PROCEDURE kgen_read_elem_accum_t
            MODULE PROCEDURE kgen_read_index_t
            MODULE PROCEDURE kgen_read_element_t
        END INTERFACE kgen_read

        PUBLIC kgen_verify
        INTERFACE kgen_verify
            MODULE PROCEDURE kgen_verify_elem_state_t
            MODULE PROCEDURE kgen_verify_derived_state_t
            MODULE PROCEDURE kgen_verify_elem_accum_t
            MODULE PROCEDURE kgen_verify_index_t
            MODULE PROCEDURE kgen_verify_element_t
        END INTERFACE kgen_verify

        CONTAINS

        ! write subroutines
            SUBROUTINE kgen_read_index_t_ptr(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                TYPE(index_t), INTENT(OUT), POINTER :: var
                LOGICAL :: is_true

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    ALLOCATE(var)
                    IF ( PRESENT(printvar) ) THEN
                        CALL kgen_read_index_t(var, kgen_unit, printvar=printvar//"%index_t")
                    ELSE
                        CALL kgen_read_index_t(var, kgen_unit)
                    END IF
                END IF
            END SUBROUTINE kgen_read_index_t_ptr

            SUBROUTINE kgen_read_cartesian2d_t_dim2(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                TYPE(cartesian2d_t), INTENT(OUT), DIMENSION(:,:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1,idx2
                INTEGER, DIMENSION(2,2) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    READ(UNIT = kgen_unit) kgen_bound(1, 2)
                    READ(UNIT = kgen_unit) kgen_bound(2, 2)
                    DO idx1=kgen_bound(1,1), kgen_bound(2, 1)
                        DO idx2=kgen_bound(1,2), kgen_bound(2, 2)
                    IF ( PRESENT(printvar) ) THEN
                                CALL kgen_read_mod6(var(idx1,idx2), kgen_unit, printvar=printvar)
                    ELSE
                                CALL kgen_read_mod6(var(idx1,idx2), kgen_unit)
                    END IF
                        END DO
                    END DO
                END IF
            END SUBROUTINE kgen_read_cartesian2d_t_dim2

            SUBROUTINE kgen_read_cartesian3d_t_dim1(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                TYPE(cartesian3d_t), INTENT(OUT), DIMENSION(:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1
                INTEGER, DIMENSION(2,1) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    DO idx1=kgen_bound(1,1), kgen_bound(2, 1)
                    IF ( PRESENT(printvar) ) THEN
                            CALL kgen_read_mod6(var(idx1), kgen_unit, printvar=printvar)
                    ELSE
                            CALL kgen_read_mod6(var(idx1), kgen_unit)
                    END IF
                    END DO
                END IF
            END SUBROUTINE kgen_read_cartesian3d_t_dim1

            SUBROUTINE kgen_read_cartesian2d_t_dim1(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                TYPE(cartesian2d_t), INTENT(OUT), DIMENSION(:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1
                INTEGER, DIMENSION(2,1) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    DO idx1=kgen_bound(1,1), kgen_bound(2, 1)
                    IF ( PRESENT(printvar) ) THEN
                            CALL kgen_read_mod6(var(idx1), kgen_unit, printvar=printvar)
                    ELSE
                            CALL kgen_read_mod6(var(idx1), kgen_unit)
                    END IF
                    END DO
                END IF
            END SUBROUTINE kgen_read_cartesian2d_t_dim1

            SUBROUTINE kgen_read_spherical_polar_t_dim2(var, kgen_unit, printvar)
                INTEGER, INTENT(IN) :: kgen_unit
                CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
                TYPE(spherical_polar_t), INTENT(OUT), DIMENSION(:,:) :: var
                LOGICAL :: is_true
                INTEGER :: idx1,idx2
                INTEGER, DIMENSION(2,2) :: kgen_bound

                READ(UNIT = kgen_unit) is_true

                IF ( is_true ) THEN
                    READ(UNIT = kgen_unit) kgen_bound(1, 1)
                    READ(UNIT = kgen_unit) kgen_bound(2, 1)
                    READ(UNIT = kgen_unit) kgen_bound(1, 2)
                    READ(UNIT = kgen_unit) kgen_bound(2, 2)
                    DO idx1=kgen_bound(1,1), kgen_bound(2, 1)
                        DO idx2=kgen_bound(1,2), kgen_bound(2, 2)
                    IF ( PRESENT(printvar) ) THEN
                                CALL kgen_read_mod6(var(idx1,idx2), kgen_unit, printvar=printvar)
                    ELSE
                                CALL kgen_read_mod6(var(idx1,idx2), kgen_unit)
                    END IF
                        END DO
                    END DO
                END IF
            END SUBROUTINE kgen_read_spherical_polar_t_dim2

        ! No module extern variables
        SUBROUTINE kgen_read_elem_state_t(var, kgen_unit, printvar)
            INTEGER, INTENT(IN) :: kgen_unit
            CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
            TYPE(elem_state_t), INTENT(out) :: var
            READ(UNIT=kgen_unit) var%v
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%v **", var%v
            END IF
            READ(UNIT=kgen_unit) var%t
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%t **", var%t
            END IF
            READ(UNIT=kgen_unit) var%dp3d
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%dp3d **", var%dp3d
            END IF
            READ(UNIT=kgen_unit) var%lnps
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%lnps **", var%lnps
            END IF
            READ(UNIT=kgen_unit) var%ps_v
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%ps_v **", var%ps_v
            END IF
            READ(UNIT=kgen_unit) var%phis
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%phis **", var%phis
            END IF
            READ(UNIT=kgen_unit) var%q
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%q **", var%q
            END IF
            READ(UNIT=kgen_unit) var%qdp
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%qdp **", var%qdp
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_read_derived_state_t(var, kgen_unit, printvar)
            INTEGER, INTENT(IN) :: kgen_unit
            CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
            TYPE(derived_state_t), INTENT(out) :: var
            READ(UNIT=kgen_unit) var%vn0
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%vn0 **", var%vn0
            END IF
            READ(UNIT=kgen_unit) var%vstar
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%vstar **", var%vstar
            END IF
            READ(UNIT=kgen_unit) var%dpdiss_biharmonic
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%dpdiss_biharmonic **", var%dpdiss_biharmonic
            END IF
            READ(UNIT=kgen_unit) var%dpdiss_ave
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%dpdiss_ave **", var%dpdiss_ave
            END IF
            READ(UNIT=kgen_unit) var%phi
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%phi **", var%phi
            END IF
            READ(UNIT=kgen_unit) var%omega_p
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%omega_p **", var%omega_p
            END IF
            READ(UNIT=kgen_unit) var%eta_dot_dpdn
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%eta_dot_dpdn **", var%eta_dot_dpdn
            END IF
            READ(UNIT=kgen_unit) var%grad_lnps
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%grad_lnps **", var%grad_lnps
            END IF
            READ(UNIT=kgen_unit) var%zeta
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%zeta **", var%zeta
            END IF
            READ(UNIT=kgen_unit) var%div
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%div **", var%div
            END IF
            READ(UNIT=kgen_unit) var%dp
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%dp **", var%dp
            END IF
            READ(UNIT=kgen_unit) var%divdp
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%divdp **", var%divdp
            END IF
            READ(UNIT=kgen_unit) var%divdp_proj
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%divdp_proj **", var%divdp_proj
            END IF
            READ(UNIT=kgen_unit) var%fq
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%fq **", var%fq
            END IF
            READ(UNIT=kgen_unit) var%fm
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%fm **", var%fm
            END IF
            READ(UNIT=kgen_unit) var%ft
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%ft **", var%ft
            END IF
            READ(UNIT=kgen_unit) var%omega_prescribed
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%omega_prescribed **", var%omega_prescribed
            END IF
            READ(UNIT=kgen_unit) var%pecnd
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%pecnd **", var%pecnd
            END IF
            READ(UNIT=kgen_unit) var%fqps
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%fqps **", var%fqps
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_read_elem_accum_t(var, kgen_unit, printvar)
            INTEGER, INTENT(IN) :: kgen_unit
            CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
            TYPE(elem_accum_t), INTENT(out) :: var
            READ(UNIT=kgen_unit) var%kener
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%kener **", var%kener
            END IF
            READ(UNIT=kgen_unit) var%pener
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%pener **", var%pener
            END IF
            READ(UNIT=kgen_unit) var%iener
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%iener **", var%iener
            END IF
            READ(UNIT=kgen_unit) var%iener_wet
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%iener_wet **", var%iener_wet
            END IF
            READ(UNIT=kgen_unit) var%qvar
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%qvar **", var%qvar
            END IF
            READ(UNIT=kgen_unit) var%qmass
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%qmass **", var%qmass
            END IF
            READ(UNIT=kgen_unit) var%q1mass
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%q1mass **", var%q1mass
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_read_index_t(var, kgen_unit, printvar)
            INTEGER, INTENT(IN) :: kgen_unit
            CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
            TYPE(index_t), INTENT(out) :: var
            READ(UNIT=kgen_unit) var%ia
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%ia **", var%ia
            END IF
            READ(UNIT=kgen_unit) var%ja
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%ja **", var%ja
            END IF
            READ(UNIT=kgen_unit) var%is
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%is **", var%is
            END IF
            READ(UNIT=kgen_unit) var%ie
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%ie **", var%ie
            END IF
            READ(UNIT=kgen_unit) var%numuniquepts
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%numuniquepts **", var%numuniquepts
            END IF
            READ(UNIT=kgen_unit) var%uniqueptoffset
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%uniqueptoffset **", var%uniqueptoffset
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_read_element_t(var, kgen_unit, printvar)
            INTEGER, INTENT(IN) :: kgen_unit
            CHARACTER(*), INTENT(IN), OPTIONAL :: printvar
            TYPE(element_t), INTENT(out) :: var
            READ(UNIT=kgen_unit) var%localid
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%localid **", var%localid
            END IF
            READ(UNIT=kgen_unit) var%globalid
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%globalid **", var%globalid
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_spherical_polar_t_dim2(var%spherep, kgen_unit, printvar=printvar//"%spherep")
            ELSE
                CALL kgen_read_spherical_polar_t_dim2(var%spherep, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_cartesian2d_t_dim2(var%cartp, kgen_unit, printvar=printvar//"%cartp")
            ELSE
                CALL kgen_read_cartesian2d_t_dim2(var%cartp, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_cartesian2d_t_dim1(var%corners, kgen_unit, printvar=printvar//"%corners")
            ELSE
                CALL kgen_read_cartesian2d_t_dim1(var%corners, kgen_unit)
            END IF
            READ(UNIT=kgen_unit) var%u2qmap
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%u2qmap **", var%u2qmap
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_cartesian3d_t_dim1(var%corners3d, kgen_unit, printvar=printvar//"%corners3d")
            ELSE
                CALL kgen_read_cartesian3d_t_dim1(var%corners3d, kgen_unit)
            END IF
            READ(UNIT=kgen_unit) var%area
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%area **", var%area
            END IF
            READ(UNIT=kgen_unit) var%normdinv
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%normdinv **", var%normdinv
            END IF
            READ(UNIT=kgen_unit) var%dx_short
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%dx_short **", var%dx_short
            END IF
            READ(UNIT=kgen_unit) var%dx_long
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%dx_long **", var%dx_long
            END IF
            READ(UNIT=kgen_unit) var%variable_hyperviscosity
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%variable_hyperviscosity **", var%variable_hyperviscosity
            END IF
            READ(UNIT=kgen_unit) var%hv_courant
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%hv_courant **", var%hv_courant
            END IF
            READ(UNIT=kgen_unit) var%tensorvisc
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%tensorvisc **", var%tensorvisc
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_mod8(var%vertex, kgen_unit, printvar=printvar//"%vertex")
            ELSE
                CALL kgen_read_mod8(var%vertex, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_mod9(var%desc, kgen_unit, printvar=printvar//"%desc")
            ELSE
                CALL kgen_read_mod9(var%desc, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_elem_state_t(var%state, kgen_unit, printvar=printvar//"%state")
            ELSE
                CALL kgen_read_elem_state_t(var%state, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_derived_state_t(var%derived, kgen_unit, printvar=printvar//"%derived")
            ELSE
                CALL kgen_read_derived_state_t(var%derived, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_elem_accum_t(var%accum, kgen_unit, printvar=printvar//"%accum")
            ELSE
                CALL kgen_read_elem_accum_t(var%accum, kgen_unit)
            END IF
            READ(UNIT=kgen_unit) var%met
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%met **", var%met
            END IF
            READ(UNIT=kgen_unit) var%metinv
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%metinv **", var%metinv
            END IF
            READ(UNIT=kgen_unit) var%metdet
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%metdet **", var%metdet
            END IF
            READ(UNIT=kgen_unit) var%rmetdet
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%rmetdet **", var%rmetdet
            END IF
            READ(UNIT=kgen_unit) var%d
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%d **", var%d
            END IF
            READ(UNIT=kgen_unit) var%dinv
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%dinv **", var%dinv
            END IF
            READ(UNIT=kgen_unit) var%vec_sphere2cart
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%vec_sphere2cart **", var%vec_sphere2cart
            END IF
            READ(UNIT=kgen_unit) var%mp
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%mp **", var%mp
            END IF
            READ(UNIT=kgen_unit) var%rmp
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%rmp **", var%rmp
            END IF
            READ(UNIT=kgen_unit) var%spheremp
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%spheremp **", var%spheremp
            END IF
            READ(UNIT=kgen_unit) var%rspheremp
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%rspheremp **", var%rspheremp
            END IF
            READ(UNIT=kgen_unit) var%gdofp
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%gdofp **", var%gdofp
            END IF
            READ(UNIT=kgen_unit) var%fcor
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%fcor **", var%fcor
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_index_t(var%idxp, kgen_unit, printvar=printvar//"%idxp")
            ELSE
                CALL kgen_read_index_t(var%idxp, kgen_unit)
            END IF
            IF ( PRESENT(printvar) ) THEN
                CALL kgen_read_index_t_ptr(var%idxv, kgen_unit, printvar=printvar//"%idxv")
            ELSE
                CALL kgen_read_index_t_ptr(var%idxv, kgen_unit)
            END IF
            READ(UNIT=kgen_unit) var%facenum
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%facenum **", var%facenum
            END IF
            READ(UNIT=kgen_unit) var%dummy
            IF ( PRESENT(printvar) ) THEN
                print *, "** " // printvar // "%dummy **", var%dummy
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_verify_elem_state_t(varname, check_status, var, ref_var)
            CHARACTER(*), INTENT(IN) :: varname
            TYPE(check_t), INTENT(INOUT) :: check_status
            TYPE(check_t) :: dtype_check_status
            TYPE(elem_state_t), INTENT(IN) :: var, ref_var

            check_status%numTotal = check_status%numTotal + 1
            CALL kgen_init_check(dtype_check_status)
            CALL kgen_verify_real_real_kind_dim5("v", dtype_check_status, var%v, ref_var%v)
            CALL kgen_verify_real_real_kind_dim4("t", dtype_check_status, var%t, ref_var%t)
            CALL kgen_verify_real_real_kind_dim4("dp3d", dtype_check_status, var%dp3d, ref_var%dp3d)
            CALL kgen_verify_real_real_kind_dim3("lnps", dtype_check_status, var%lnps, ref_var%lnps)
            CALL kgen_verify_real_real_kind_dim3("ps_v", dtype_check_status, var%ps_v, ref_var%ps_v)
            CALL kgen_verify_real_real_kind_dim2("phis", dtype_check_status, var%phis, ref_var%phis)
            CALL kgen_verify_real_real_kind_dim4("q", dtype_check_status, var%q, ref_var%q)
            CALL kgen_verify_real_real_kind_dim5("qdp", dtype_check_status, var%qdp, ref_var%qdp)
            IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
            ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                check_status%numFatal = check_status%numFatal + 1
            ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                check_status%numWarning = check_status%numWarning + 1
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_verify_derived_state_t(varname, check_status, var, ref_var)
            CHARACTER(*), INTENT(IN) :: varname
            TYPE(check_t), INTENT(INOUT) :: check_status
            TYPE(check_t) :: dtype_check_status
            TYPE(derived_state_t), INTENT(IN) :: var, ref_var

            check_status%numTotal = check_status%numTotal + 1
            CALL kgen_init_check(dtype_check_status)
            CALL kgen_verify_real_real_kind_dim4("vn0", dtype_check_status, var%vn0, ref_var%vn0)
            CALL kgen_verify_real_real_kind_dim4("vstar", dtype_check_status, var%vstar, ref_var%vstar)
            CALL kgen_verify_real_real_kind_dim3("dpdiss_biharmonic", dtype_check_status, var%dpdiss_biharmonic, ref_var%dpdiss_biharmonic)
            CALL kgen_verify_real_real_kind_dim3("dpdiss_ave", dtype_check_status, var%dpdiss_ave, ref_var%dpdiss_ave)
            CALL kgen_verify_real_real_kind_dim3("phi", dtype_check_status, var%phi, ref_var%phi)
            CALL kgen_verify_real_real_kind_dim3("omega_p", dtype_check_status, var%omega_p, ref_var%omega_p)
            CALL kgen_verify_real_real_kind_dim3("eta_dot_dpdn", dtype_check_status, var%eta_dot_dpdn, ref_var%eta_dot_dpdn)
            CALL kgen_verify_real_real_kind_dim3("grad_lnps", dtype_check_status, var%grad_lnps, ref_var%grad_lnps)
            CALL kgen_verify_real_real_kind_dim3("zeta", dtype_check_status, var%zeta, ref_var%zeta)
            CALL kgen_verify_real_real_kind_dim4("div", dtype_check_status, var%div, ref_var%div)
            CALL kgen_verify_real_real_kind_dim3("dp", dtype_check_status, var%dp, ref_var%dp)
            CALL kgen_verify_real_real_kind_dim3("divdp", dtype_check_status, var%divdp, ref_var%divdp)
            CALL kgen_verify_real_real_kind_dim3("divdp_proj", dtype_check_status, var%divdp_proj, ref_var%divdp_proj)
            CALL kgen_verify_real_real_kind_dim5("fq", dtype_check_status, var%fq, ref_var%fq)
            CALL kgen_verify_real_real_kind_dim5("fm", dtype_check_status, var%fm, ref_var%fm)
            CALL kgen_verify_real_real_kind_dim4("ft", dtype_check_status, var%ft, ref_var%ft)
            CALL kgen_verify_real_real_kind_dim3("omega_prescribed", dtype_check_status, var%omega_prescribed, ref_var%omega_prescribed)
            CALL kgen_verify_real_real_kind_dim3("pecnd", dtype_check_status, var%pecnd, ref_var%pecnd)
            CALL kgen_verify_real_real_kind_dim3("fqps", dtype_check_status, var%fqps, ref_var%fqps)
            IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
            ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                check_status%numFatal = check_status%numFatal + 1
            ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                check_status%numWarning = check_status%numWarning + 1
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_verify_elem_accum_t(varname, check_status, var, ref_var)
            CHARACTER(*), INTENT(IN) :: varname
            TYPE(check_t), INTENT(INOUT) :: check_status
            TYPE(check_t) :: dtype_check_status
            TYPE(elem_accum_t), INTENT(IN) :: var, ref_var

            check_status%numTotal = check_status%numTotal + 1
            CALL kgen_init_check(dtype_check_status)
            CALL kgen_verify_real_real_kind_dim3("kener", dtype_check_status, var%kener, ref_var%kener)
            CALL kgen_verify_real_real_kind_dim3("pener", dtype_check_status, var%pener, ref_var%pener)
            CALL kgen_verify_real_real_kind_dim3("iener", dtype_check_status, var%iener, ref_var%iener)
            CALL kgen_verify_real_real_kind_dim3("iener_wet", dtype_check_status, var%iener_wet, ref_var%iener_wet)
            CALL kgen_verify_real_real_kind_dim4("qvar", dtype_check_status, var%qvar, ref_var%qvar)
            CALL kgen_verify_real_real_kind_dim4("qmass", dtype_check_status, var%qmass, ref_var%qmass)
            CALL kgen_verify_real_real_kind_dim3("q1mass", dtype_check_status, var%q1mass, ref_var%q1mass)
            IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
            ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                check_status%numFatal = check_status%numFatal + 1
            ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                check_status%numWarning = check_status%numWarning + 1
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_verify_index_t(varname, check_status, var, ref_var)
            CHARACTER(*), INTENT(IN) :: varname
            TYPE(check_t), INTENT(INOUT) :: check_status
            TYPE(check_t) :: dtype_check_status
            TYPE(index_t), INTENT(IN) :: var, ref_var

            check_status%numTotal = check_status%numTotal + 1
            CALL kgen_init_check(dtype_check_status)
            CALL kgen_verify_integer_int_kind_dim1("ia", dtype_check_status, var%ia, ref_var%ia)
            CALL kgen_verify_integer_int_kind_dim1("ja", dtype_check_status, var%ja, ref_var%ja)
            CALL kgen_verify_integer_int_kind("is", dtype_check_status, var%is, ref_var%is)
            CALL kgen_verify_integer_int_kind("ie", dtype_check_status, var%ie, ref_var%ie)
            CALL kgen_verify_integer_int_kind("numuniquepts", dtype_check_status, var%numuniquepts, ref_var%numuniquepts)
            CALL kgen_verify_integer_int_kind("uniqueptoffset", dtype_check_status, var%uniqueptoffset, ref_var%uniqueptoffset)
            IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
            ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                check_status%numFatal = check_status%numFatal + 1
            ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                check_status%numWarning = check_status%numWarning + 1
            END IF
        END SUBROUTINE
        SUBROUTINE kgen_verify_element_t(varname, check_status, var, ref_var)
            CHARACTER(*), INTENT(IN) :: varname
            TYPE(check_t), INTENT(INOUT) :: check_status
            TYPE(check_t) :: dtype_check_status
            TYPE(element_t), INTENT(IN) :: var, ref_var

            check_status%numTotal = check_status%numTotal + 1
            CALL kgen_init_check(dtype_check_status)
            CALL kgen_verify_integer_int_kind("localid", dtype_check_status, var%localid, ref_var%localid)
            CALL kgen_verify_integer_int_kind("globalid", dtype_check_status, var%globalid, ref_var%globalid)
            CALL kgen_verify_spherical_polar_t_dim2("spherep", dtype_check_status, var%spherep, ref_var%spherep)
            CALL kgen_verify_cartesian2d_t_dim2("cartp", dtype_check_status, var%cartp, ref_var%cartp)
            CALL kgen_verify_cartesian2d_t_dim1("corners", dtype_check_status, var%corners, ref_var%corners)
            CALL kgen_verify_real_real_kind_dim2("u2qmap", dtype_check_status, var%u2qmap, ref_var%u2qmap)
            CALL kgen_verify_cartesian3d_t_dim1("corners3d", dtype_check_status, var%corners3d, ref_var%corners3d)
            CALL kgen_verify_real_real_kind("area", dtype_check_status, var%area, ref_var%area)
            CALL kgen_verify_real_real_kind("normdinv", dtype_check_status, var%normdinv, ref_var%normdinv)
            CALL kgen_verify_real_real_kind("dx_short", dtype_check_status, var%dx_short, ref_var%dx_short)
            CALL kgen_verify_real_real_kind("dx_long", dtype_check_status, var%dx_long, ref_var%dx_long)
            CALL kgen_verify_real_real_kind_dim2("variable_hyperviscosity", dtype_check_status, var%variable_hyperviscosity, ref_var%variable_hyperviscosity)
            CALL kgen_verify_real_real_kind("hv_courant", dtype_check_status, var%hv_courant, ref_var%hv_courant)
            CALL kgen_verify_real_real_kind_dim4("tensorvisc", dtype_check_status, var%tensorvisc, ref_var%tensorvisc)
            CALL kgen_verify_mod8("vertex", dtype_check_status, var%vertex, ref_var%vertex)
            CALL kgen_verify_mod9("desc", dtype_check_status, var%desc, ref_var%desc)
            CALL kgen_verify_elem_state_t("state", dtype_check_status, var%state, ref_var%state)
            CALL kgen_verify_derived_state_t("derived", dtype_check_status, var%derived, ref_var%derived)
            CALL kgen_verify_elem_accum_t("accum", dtype_check_status, var%accum, ref_var%accum)
            CALL kgen_verify_real_real_kind_dim4("met", dtype_check_status, var%met, ref_var%met)
            CALL kgen_verify_real_real_kind_dim4("metinv", dtype_check_status, var%metinv, ref_var%metinv)
            CALL kgen_verify_real_real_kind_dim2("metdet", dtype_check_status, var%metdet, ref_var%metdet)
            CALL kgen_verify_real_real_kind_dim2("rmetdet", dtype_check_status, var%rmetdet, ref_var%rmetdet)
            CALL kgen_verify_real_real_kind_dim4("d", dtype_check_status, var%d, ref_var%d)
            CALL kgen_verify_real_real_kind_dim4("dinv", dtype_check_status, var%dinv, ref_var%dinv)
            CALL kgen_verify_real_real_kind_dim4("vec_sphere2cart", dtype_check_status, var%vec_sphere2cart, ref_var%vec_sphere2cart)
            CALL kgen_verify_real_real_kind_dim2("mp", dtype_check_status, var%mp, ref_var%mp)
            CALL kgen_verify_real_real_kind_dim2("rmp", dtype_check_status, var%rmp, ref_var%rmp)
            CALL kgen_verify_real_real_kind_dim2("spheremp", dtype_check_status, var%spheremp, ref_var%spheremp)
            CALL kgen_verify_real_real_kind_dim2("rspheremp", dtype_check_status, var%rspheremp, ref_var%rspheremp)
            CALL kgen_verify_integer_long_kind_dim2("gdofp", dtype_check_status, var%gdofp, ref_var%gdofp)
            CALL kgen_verify_real_real_kind_dim2("fcor", dtype_check_status, var%fcor, ref_var%fcor)
            CALL kgen_verify_index_t("idxp", dtype_check_status, var%idxp, ref_var%idxp)
            CALL kgen_verify_index_t_ptr("idxv", dtype_check_status, var%idxv, ref_var%idxv)
            CALL kgen_verify_integer("facenum", dtype_check_status, var%facenum, ref_var%facenum)
            CALL kgen_verify_integer("dummy", dtype_check_status, var%dummy, ref_var%dummy)
            IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                check_status%numIdentical = check_status%numIdentical + 1
            ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                check_status%numFatal = check_status%numFatal + 1
            ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                check_status%numWarning = check_status%numWarning + 1
            END IF
        END SUBROUTINE
            SUBROUTINE kgen_verify_real_real_kind_dim5( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=real_kind), intent(in), DIMENSION(:,:,:,:,:) :: var, ref_var
                real(KIND=real_kind) :: nrmsdiff, rmsdiff
                real(KIND=real_kind), allocatable, DIMENSION(:,:,:,:,:) :: temp, temp2
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
                    allocate(temp(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3),SIZE(var,dim=4),SIZE(var,dim=5)))
                    allocate(temp2(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3),SIZE(var,dim=4),SIZE(var,dim=5)))
                
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
            END SUBROUTINE kgen_verify_real_real_kind_dim5

            SUBROUTINE kgen_verify_real_real_kind_dim4( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=real_kind), intent(in), DIMENSION(:,:,:,:) :: var, ref_var
                real(KIND=real_kind) :: nrmsdiff, rmsdiff
                real(KIND=real_kind), allocatable, DIMENSION(:,:,:,:) :: temp, temp2
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
                    allocate(temp(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3),SIZE(var,dim=4)))
                    allocate(temp2(SIZE(var,dim=1),SIZE(var,dim=2),SIZE(var,dim=3),SIZE(var,dim=4)))
                
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
            END SUBROUTINE kgen_verify_real_real_kind_dim4

            SUBROUTINE kgen_verify_real_real_kind_dim3( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=real_kind), intent(in), DIMENSION(:,:,:) :: var, ref_var
                real(KIND=real_kind) :: nrmsdiff, rmsdiff
                real(KIND=real_kind), allocatable, DIMENSION(:,:,:) :: temp, temp2
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
            END SUBROUTINE kgen_verify_real_real_kind_dim3

            SUBROUTINE kgen_verify_real_real_kind_dim2( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=real_kind), intent(in), DIMENSION(:,:) :: var, ref_var
                real(KIND=real_kind) :: nrmsdiff, rmsdiff
                real(KIND=real_kind), allocatable, DIMENSION(:,:) :: temp, temp2
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
            END SUBROUTINE kgen_verify_real_real_kind_dim2

            SUBROUTINE kgen_verify_integer_int_kind_dim1( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                integer(KIND=int_kind), intent(in), DIMENSION(:) :: var, ref_var
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
                    if(check_status%verboseLevel > 0) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                        WRITE(*,*) count( var /= ref_var), " of ", size( var ), " elements are different."
                    end if
                
                    check_status%numFatal = check_status%numFatal+1
                END IF
            END SUBROUTINE kgen_verify_integer_int_kind_dim1

            SUBROUTINE kgen_verify_integer_int_kind( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                integer(KIND=int_kind), intent(in) :: var, ref_var
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
            END SUBROUTINE kgen_verify_integer_int_kind

            SUBROUTINE kgen_verify_spherical_polar_t_dim2( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                type(check_t) :: dtype_check_status
                TYPE(spherical_polar_t), intent(in), DIMENSION(:,:) :: var, ref_var
                integer :: idx1,idx2
                check_status%numTotal = check_status%numTotal + 1
                CALL kgen_init_check(dtype_check_status)
                DO idx1=LBOUND(var,1), UBOUND(var,1)
                    DO idx2=LBOUND(var,2), UBOUND(var,2)
                        CALL kgen_verify_mod6(varname, dtype_check_status, var(idx1,idx2), ref_var(idx1,idx2))
                    END DO
                END DO
                IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                    check_status%numIdentical = check_status%numIdentical + 1
                ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                    check_status%numFatal = check_status%numFatal + 1
                ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                    check_status%numWarning = check_status%numWarning + 1
                END IF
            END SUBROUTINE kgen_verify_spherical_polar_t_dim2

            SUBROUTINE kgen_verify_cartesian2d_t_dim2( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                type(check_t) :: dtype_check_status
                TYPE(cartesian2d_t), intent(in), DIMENSION(:,:) :: var, ref_var
                integer :: idx1,idx2
                check_status%numTotal = check_status%numTotal + 1
                CALL kgen_init_check(dtype_check_status)
                DO idx1=LBOUND(var,1), UBOUND(var,1)
                    DO idx2=LBOUND(var,2), UBOUND(var,2)
                        CALL kgen_verify_mod6(varname, dtype_check_status, var(idx1,idx2), ref_var(idx1,idx2))
                    END DO
                END DO
                IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                    check_status%numIdentical = check_status%numIdentical + 1
                ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                    check_status%numFatal = check_status%numFatal + 1
                ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                    check_status%numWarning = check_status%numWarning + 1
                END IF
            END SUBROUTINE kgen_verify_cartesian2d_t_dim2

            SUBROUTINE kgen_verify_cartesian2d_t_dim1( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                type(check_t) :: dtype_check_status
                TYPE(cartesian2d_t), intent(in), DIMENSION(:) :: var, ref_var
                integer :: idx1
                check_status%numTotal = check_status%numTotal + 1
                CALL kgen_init_check(dtype_check_status)
                DO idx1=LBOUND(var,1), UBOUND(var,1)
                    CALL kgen_verify_mod6(varname, dtype_check_status, var(idx1), ref_var(idx1))
                END DO
                IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                    check_status%numIdentical = check_status%numIdentical + 1
                ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                    check_status%numFatal = check_status%numFatal + 1
                ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                    check_status%numWarning = check_status%numWarning + 1
                END IF
            END SUBROUTINE kgen_verify_cartesian2d_t_dim1

            SUBROUTINE kgen_verify_cartesian3d_t_dim1( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                type(check_t) :: dtype_check_status
                TYPE(cartesian3d_t), intent(in), DIMENSION(:) :: var, ref_var
                integer :: idx1
                check_status%numTotal = check_status%numTotal + 1
                CALL kgen_init_check(dtype_check_status)
                DO idx1=LBOUND(var,1), UBOUND(var,1)
                    CALL kgen_verify_mod6(varname, dtype_check_status, var(idx1), ref_var(idx1))
                END DO
                IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                    check_status%numIdentical = check_status%numIdentical + 1
                ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                    check_status%numFatal = check_status%numFatal + 1
                ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                    check_status%numWarning = check_status%numWarning + 1
                END IF
            END SUBROUTINE kgen_verify_cartesian3d_t_dim1

            SUBROUTINE kgen_verify_real_real_kind( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                real(KIND=real_kind), intent(in) :: var, ref_var
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
            END SUBROUTINE kgen_verify_real_real_kind

            SUBROUTINE kgen_verify_integer_long_kind_dim2( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                integer(KIND=long_kind), intent(in), DIMENSION(:,:) :: var, ref_var
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
                    if(check_status%verboseLevel > 0) then
                        WRITE(*,*)
                        WRITE(*,*) trim(adjustl(varname)), " is NOT IDENTICAL."
                        WRITE(*,*) count( var /= ref_var), " of ", size( var ), " elements are different."
                    end if
                
                    check_status%numFatal = check_status%numFatal+1
                END IF
            END SUBROUTINE kgen_verify_integer_long_kind_dim2

            SUBROUTINE kgen_verify_index_t_ptr( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                type(check_t) :: dtype_check_status
                TYPE(index_t), intent(in), POINTER :: var, ref_var
                IF ( ASSOCIATED(var) ) THEN

                check_status%numTotal = check_status%numTotal + 1
                CALL kgen_init_check(dtype_check_status)
                CALL kgen_verify_integer_int_kind_dim1("ia", dtype_check_status, var%ia, ref_var%ia)
                CALL kgen_verify_integer_int_kind_dim1("ja", dtype_check_status, var%ja, ref_var%ja)
                CALL kgen_verify_integer_int_kind("is", dtype_check_status, var%is, ref_var%is)
                CALL kgen_verify_integer_int_kind("ie", dtype_check_status, var%ie, ref_var%ie)
                CALL kgen_verify_integer_int_kind("numuniquepts", dtype_check_status, var%numuniquepts, ref_var%numuniquepts)
                CALL kgen_verify_integer_int_kind("uniqueptoffset", dtype_check_status, var%uniqueptoffset, ref_var%uniqueptoffset)
                IF ( dtype_check_status%numTotal == dtype_check_status%numIdentical ) THEN
                    check_status%numIdentical = check_status%numIdentical + 1
                ELSE IF ( dtype_check_status%numFatal > 0 ) THEN
                    check_status%numFatal = check_status%numFatal + 1
                ELSE IF ( dtype_check_status%numWarning > 0 ) THEN
                    check_status%numWarning = check_status%numWarning + 1
                END IF
                END IF
            END SUBROUTINE kgen_verify_index_t_ptr

            SUBROUTINE kgen_verify_integer( varname, check_status, var, ref_var)
                character(*), intent(in) :: varname
                type(check_t), intent(inout) :: check_status
                integer, intent(in) :: var, ref_var
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
            END SUBROUTINE kgen_verify_integer

        ! ===================== ELEMENT_MOD METHODS ==========================

        !___________________________________________________________________

        !___________________________________________________________________

        !___________________________________________________________________

        !___________________________________________________________________

        !___________________________________________________________________

    END MODULE element_mod
