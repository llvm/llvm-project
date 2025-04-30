
! KGEN-generated Fortran source file
!
! Filename    : control_mod.F90
! Generated at: 2015-04-12 19:17:34
! KGEN version: 0.4.9



    MODULE control_mod
        USE kgen_utils_mod, ONLY : kgen_dp, check_t, kgen_init_check, kgen_print_check
        USE kinds, ONLY: real_kind
        ! time integration (explicit, semi_imp, or full imp)
        ! none of this is used anymore:
        !  u grad(Q) formulation
        ! div(u dp/dn Q ) formulation
        ! Tracer transport type
        ! We potentially have five types of tracer advection. However, not all of them
        ! may be chosen at runtime due to compile-type restrictions on arrays
        !shallow water advection tests:
        !kmass points to a level with density.  other levels contain test tracers
        ! m s^-2
        ! 0 = leapfrog
        ! 1 = RK (foward-in-time)
        ! number of RK stages to use
        ! Forcing Type
        ! ftype = 0  HOMME ApplyColumn() type forcing process split
        ! ftype = -1   ignore forcing  (used for testing energy balance)
        ! use cp or cp* in T equation
        !  -1: No fixer, use non-staggered formula
        !   0: No Fixer, use staggered in time formula
        !       (only for leapfrog)
        !   1 or 4:  Enable fixer, non-staggered formula
        ! ratio of dynamics tsteps to tracer tsteps
        ! for vertically lagrangian dynamics, apply remap
        ! every rsplit tracer timesteps
        ! Defines if the program is to use its own physics (HOMME standalone), valid values 1,2,3
        ! physics = 0, no physics
        ! physics = 1, Use physics
        ! leapfrog-trapazoidal frequency
        ! interspace a lf-trapazoidal step every LFTfreq leapfrogs
        ! 0 = disabled
        ! compute_mean_flux:  obsolete, not used
        ! vert_remap_q_alg:    0  default value, Zerroukat monotonic splines
        !                      1  PPM vertical remap with mirroring at the boundaries
        !                         (solid wall bc's, high-order throughout)
        !                      2  PPM vertical remap without mirroring at the boundaries
        !                         (no bc's enforced, first-order at two cells bordering top and bottom boundaries)
        ! -1 = chosen at run time
        !  0 = equi-angle Gnomonic (default)
        !  1 = equi-spaced Gnomonic (not yet coded)
        !  2 = element-local projection  (for var-res)
        !  3 = parametric (not yet coded)
        !tolerance to define smth small, was introduced for lim 8 in 2d and 3d
        ! if semi_implicit, type of preconditioner:
        ! choices block_jacobi or identity
        ! partition methods
        ! options: "cube" is supported
        ! options: if cube: "swtc1","swtc2",or "swtc6"
        ! generic test case param
        ! remap frequency of synopsis of system state (steps)
        ! selected remapping option
        ! output frequency of synopsis of system state (steps)
        ! frequency in steps of field accumulation
        ! model day to start accumulation
        ! model day to stop  accumulation
        ! max iterations of solver
        ! solver tolerance (convergence criteria)
        ! debug level of CG solver
        ! Boyd Vandeven filter Transfer fn parameters
        ! Fischer-Mullen filter Transfer fn parameters
        ! vertical formulation (ecmwf,ccm1)
        ! vertical grid spacing (equal,unequal)
        ! vertical coordinate system (sigma,hybrid)
        ! set for refined exodus meshes (variable viscosity)
        ! upper bound for Courant number
        ! (only used for variable viscosity, recommend 1.9 in namelist)
        ! viscosity (momentum equ)
        ! viscsoity (momentum equ, div component)
        ! default = nu   T equ. viscosity
        ! default = nu   tracer viscosity
        ! default = 0    ps equ. viscosity
        ! top-of-the-model viscosity
        ! number of subcycles for hyper viscsosity timestep
        ! number of subcycles for hyper viscsosity timestep on TRACERS
        ! laplace**hypervis_order.  0=not used  1=regular viscosity, 2=grad**4
        ! 0 = use laplace on eta surfaces
        ! 1 = use (approx.) laplace on p surfaces
        REAL(KIND=real_kind), public :: hypervis_power=0 ! if not 0, use variable hyperviscosity based on element area
        REAL(KIND=real_kind), public :: hypervis_scaling=0 ! use tensor hyperviscosity
        !
        !three types of hyper viscosity are supported right now:
        ! (1) const hv:    nu * del^2 del^2
        ! (2) scalar hv:   nu(lat,lon) * del^2 del^2
        ! (3) tensor hv,   nu * ( \div * tensor * \grad ) * del^2
        !
        ! (1) default:  hypervis_power=0, hypervis_scaling=0
        ! (2) Original version for var-res grids. (M. Levy)
        !            scalar coefficient within each element
        !            hypervisc_scaling=0
        !            set hypervis_power>0 and set fine_ne, max_hypervis_courant
        ! (3) tensor HV var-res grids
        !            tensor within each element:
        !            set hypervis_scaling > 0 (typical values would be 3.2 or 4.0)
        !            hypervis_power=0
        !            (\div * tensor * \grad) operator uses cartesian laplace
        !
        ! hyperviscosity parameters used for smoothing topography
        ! 0 = disable
        ! 0 = disabled
        ! fix the velocities?
        ! initial perturbation in JW test case
        ! initial perturbation in JW test case
        PUBLIC kgen_read_externs_control_mod
    CONTAINS

    ! write subroutines
    ! No subroutines

    ! module extern variables

    SUBROUTINE kgen_read_externs_control_mod(kgen_unit)
        INTEGER, INTENT(IN) :: kgen_unit
        READ(UNIT=kgen_unit) hypervis_power
        READ(UNIT=kgen_unit) hypervis_scaling
    END SUBROUTINE kgen_read_externs_control_mod

    END MODULE control_mod
