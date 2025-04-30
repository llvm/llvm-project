
! KGEN-generated Fortran source file
!
! Filename    : dimensions_mod.F90
! Generated at: 2015-03-16 09:25:32
! KGEN version: 0.4.5



    MODULE dimensions_mod
        IMPLICIT NONE
        PRIVATE
        ! set MAX number of tracers.  actual number of tracers is a run time argument
        ! SE tracers: default is 4
        ! fvm tracers
        ! FI # dependent variables
        INTEGER, parameter, public :: np = 4
        ! fvm dimensions:
        !number of Gausspoints for the fvm integral approximation
        !Max. Courant number
        !halo width needed for reconstruction - phl
        !total halo width where reconstruction is needed (nht<=nc) - phl
        !(different from halo needed for elements on edges and corners
        !  integer, parameter, public :: ns=3         !quadratic halo interpolation - recommended setting for nc=3
        !  integer, parameter, public :: ns=4         !cubic halo interpolation     - recommended setting for nc=4
        !nhc determines width of halo exchanged with neighboring elements
        !
        ! constants for SPELT
        !
        !number of interpolation values, works only for this
        ! number of points in an element
        ! dg degree for hybrid cg/dg element  0=disabled
        INTEGER, parameter, public :: nlev=26
        INTEGER, parameter, public :: nlevp=nlev+1
        !  params for a mesh
        !  integer, public, parameter :: max_elements_attached_to_node = 7
        !  integer, public, parameter :: s_nv = 2*max_elements_attached_to_node
        !default for non-refined mesh (note that these are *not* parameters now)
        !max_elements_attached_to_node-3
        !4 + 4*max_corner_elem
        ! total number of elements
        ! number of elements per MPI task
        ! max number of elements on any MPI task
        ! This is the number of physics processors/ per dynamics processor
        CONTAINS

        ! write subroutines
        ! No subroutines
        ! No module extern variables

    END MODULE dimensions_mod
