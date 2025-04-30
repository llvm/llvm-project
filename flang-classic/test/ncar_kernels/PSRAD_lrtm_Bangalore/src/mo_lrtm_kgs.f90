
! KGEN-generated Fortran source file
!
! Filename    : mo_lrtm_kgs.f90
! Generated at: 2015-02-19 15:30:31
! KGEN version: 0.4.4



    MODULE rrlw_planck
        USE mo_kind, ONLY: wp
        USE mo_rrtm_params, ONLY: nbndlw
        REAL(KIND=wp) :: chi_mls(7,59)
        REAL(KIND=wp) :: totplanck(181,nbndlw) !< planck function for each band
        !< for band 16
        PUBLIC read_externs_rrlw_planck

    ! read interface
    PUBLIC kgen_read_var
    interface kgen_read_var
        module procedure read_var_real_wp_dim2
    end interface kgen_read_var

    CONTAINS

    ! module extern variables

    SUBROUTINE read_externs_rrlw_planck(kgen_unit)
    integer, intent(in) :: kgen_unit
    READ(UNIT=kgen_unit) chi_mls
    READ(UNIT=kgen_unit) totplanck
    END SUBROUTINE read_externs_rrlw_planck


    ! read subroutines
    subroutine read_var_real_wp_dim2(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:), allocatable :: var
        integer, dimension(2,2) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    END MODULE rrlw_planck

    MODULE rrlw_kg01
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        !< original abs coefficients
        INTEGER, parameter :: ng1  = 10 !< combined abs. coefficients
        REAL(KIND=wp) :: fracrefa(ng1)
        REAL(KIND=wp) :: fracrefb(ng1)
        REAL(KIND=wp) :: absa(65,ng1)
        REAL(KIND=wp) :: absb(235,ng1)
        REAL(KIND=wp) :: ka_mn2(19,ng1)
        REAL(KIND=wp) :: kb_mn2(19,ng1)
        REAL(KIND=wp) :: selfref(10,ng1)
        REAL(KIND=wp) :: forref(4,ng1)
        PUBLIC read_externs_rrlw_kg01

    ! read interface
    PUBLIC kgen_read_var
    interface kgen_read_var
        module procedure read_var_real_wp_dim1
        module procedure read_var_real_wp_dim2
    end interface kgen_read_var

    CONTAINS

    ! module extern variables

    SUBROUTINE read_externs_rrlw_kg01(kgen_unit)
    integer, intent(in) :: kgen_unit
    READ(UNIT=kgen_unit) fracrefa
    READ(UNIT=kgen_unit) fracrefb
    READ(UNIT=kgen_unit) absa
    READ(UNIT=kgen_unit) absb
    READ(UNIT=kgen_unit) ka_mn2
    READ(UNIT=kgen_unit) kb_mn2
    READ(UNIT=kgen_unit) selfref
    READ(UNIT=kgen_unit) forref
    END SUBROUTINE read_externs_rrlw_kg01


    ! read subroutines
    subroutine read_var_real_wp_dim1(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:), allocatable :: var
        integer, dimension(2,1) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    subroutine read_var_real_wp_dim2(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:), allocatable :: var
        integer, dimension(2,2) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    END MODULE rrlw_kg01

    MODULE rrlw_kg02
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        INTEGER, parameter :: ng2  = 12
        REAL(KIND=wp) :: fracrefa(ng2)
        REAL(KIND=wp) :: fracrefb(ng2)
        REAL(KIND=wp) :: absa(65,ng2)
        REAL(KIND=wp) :: absb(235,ng2)
        REAL(KIND=wp) :: selfref(10,ng2)
        REAL(KIND=wp) :: forref(4,ng2)
        PUBLIC read_externs_rrlw_kg02

    ! read interface
    PUBLIC kgen_read_var
    interface kgen_read_var
        module procedure read_var_real_wp_dim1
        module procedure read_var_real_wp_dim2
    end interface kgen_read_var

    CONTAINS

    ! module extern variables

    SUBROUTINE read_externs_rrlw_kg02(kgen_unit)
    integer, intent(in) :: kgen_unit
    READ(UNIT=kgen_unit) fracrefa
    READ(UNIT=kgen_unit) fracrefb
    READ(UNIT=kgen_unit) absa
    READ(UNIT=kgen_unit) absb
    READ(UNIT=kgen_unit) selfref
    READ(UNIT=kgen_unit) forref
    END SUBROUTINE read_externs_rrlw_kg02


    ! read subroutines
    subroutine read_var_real_wp_dim1(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:), allocatable :: var
        integer, dimension(2,1) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    subroutine read_var_real_wp_dim2(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:), allocatable :: var
        integer, dimension(2,2) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    END MODULE rrlw_kg02

    MODULE rrlw_kg03
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        INTEGER, parameter :: ng3  = 16
        REAL(KIND=wp) :: fracrefa(ng3,9)
        REAL(KIND=wp) :: fracrefb(ng3,5)
        REAL(KIND=wp) :: absa(585,ng3)
        REAL(KIND=wp) :: absb(1175,ng3)
        REAL(KIND=wp) :: ka_mn2o(9,19,ng3)
        REAL(KIND=wp) :: kb_mn2o(5,19,ng3)
        REAL(KIND=wp) :: selfref(10,ng3)
        REAL(KIND=wp) :: forref(4,ng3)
        PUBLIC read_externs_rrlw_kg03

    ! read interface
    PUBLIC kgen_read_var
    interface kgen_read_var
        module procedure read_var_real_wp_dim2
        module procedure read_var_real_wp_dim3
    end interface kgen_read_var

    CONTAINS

    ! module extern variables

    SUBROUTINE read_externs_rrlw_kg03(kgen_unit)
    integer, intent(in) :: kgen_unit
    READ(UNIT=kgen_unit) fracrefa
    READ(UNIT=kgen_unit) fracrefb
    READ(UNIT=kgen_unit) absa
    READ(UNIT=kgen_unit) absb
    READ(UNIT=kgen_unit) ka_mn2o
    READ(UNIT=kgen_unit) kb_mn2o
    READ(UNIT=kgen_unit) selfref
    READ(UNIT=kgen_unit) forref
    END SUBROUTINE read_externs_rrlw_kg03


    ! read subroutines
    subroutine read_var_real_wp_dim2(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:), allocatable :: var
        integer, dimension(2,2) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    subroutine read_var_real_wp_dim3(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:,:), allocatable :: var
        integer, dimension(2,3) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            READ(UNIT = kgen_unit) kgen_bound(1, 3)
            READ(UNIT = kgen_unit) kgen_bound(2, 3)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    END MODULE rrlw_kg03

    MODULE rrlw_kg04
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        INTEGER, parameter :: ng4  = 14
        REAL(KIND=wp) :: fracrefa(ng4,9)
        REAL(KIND=wp) :: fracrefb(ng4,5)
        REAL(KIND=wp) :: absa(585,ng4)
        REAL(KIND=wp) :: absb(1175,ng4)
        REAL(KIND=wp) :: selfref(10,ng4)
        REAL(KIND=wp) :: forref(4,ng4)
        PUBLIC read_externs_rrlw_kg04

    ! read interface
    PUBLIC kgen_read_var
    interface kgen_read_var
        module procedure read_var_real_wp_dim2
    end interface kgen_read_var

    CONTAINS

    ! module extern variables

    SUBROUTINE read_externs_rrlw_kg04(kgen_unit)
    integer, intent(in) :: kgen_unit
    READ(UNIT=kgen_unit) fracrefa
    READ(UNIT=kgen_unit) fracrefb
    READ(UNIT=kgen_unit) absa
    READ(UNIT=kgen_unit) absb
    READ(UNIT=kgen_unit) selfref
    READ(UNIT=kgen_unit) forref
    END SUBROUTINE read_externs_rrlw_kg04


    ! read subroutines
    subroutine read_var_real_wp_dim2(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:), allocatable :: var
        integer, dimension(2,2) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    END MODULE rrlw_kg04

    MODULE rrlw_kg05
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        INTEGER, parameter :: ng5  = 16
        REAL(KIND=wp) :: fracrefa(ng5,9)
        REAL(KIND=wp) :: fracrefb(ng5,5)
        REAL(KIND=wp) :: absa(585,ng5)
        REAL(KIND=wp) :: absb(1175,ng5)
        REAL(KIND=wp) :: ka_mo3(9,19,ng5)
        REAL(KIND=wp) :: selfref(10,ng5)
        REAL(KIND=wp) :: forref(4,ng5)
        REAL(KIND=wp) :: ccl4(ng5)
        PUBLIC read_externs_rrlw_kg05

    ! read interface
    PUBLIC kgen_read_var
    interface kgen_read_var
        module procedure read_var_real_wp_dim2
        module procedure read_var_real_wp_dim3
        module procedure read_var_real_wp_dim1
    end interface kgen_read_var

    CONTAINS

    ! module extern variables

    SUBROUTINE read_externs_rrlw_kg05(kgen_unit)
    integer, intent(in) :: kgen_unit
    READ(UNIT=kgen_unit) fracrefa
    READ(UNIT=kgen_unit) fracrefb
    READ(UNIT=kgen_unit) absa
    READ(UNIT=kgen_unit) absb
    READ(UNIT=kgen_unit) ka_mo3
    READ(UNIT=kgen_unit) selfref
    READ(UNIT=kgen_unit) forref
    READ(UNIT=kgen_unit) ccl4
    END SUBROUTINE read_externs_rrlw_kg05


    ! read subroutines
    subroutine read_var_real_wp_dim2(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:), allocatable :: var
        integer, dimension(2,2) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    subroutine read_var_real_wp_dim3(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:,:), allocatable :: var
        integer, dimension(2,3) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            READ(UNIT = kgen_unit) kgen_bound(1, 3)
            READ(UNIT = kgen_unit) kgen_bound(2, 3)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    subroutine read_var_real_wp_dim1(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:), allocatable :: var
        integer, dimension(2,1) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    END MODULE rrlw_kg05

    MODULE rrlw_kg06
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        INTEGER, parameter :: ng6  = 8
        REAL(KIND=wp), dimension(ng6) :: fracrefa
        REAL(KIND=wp) :: absa(65,ng6)
        REAL(KIND=wp) :: ka_mco2(19,ng6)
        REAL(KIND=wp) :: selfref(10,ng6)
        REAL(KIND=wp) :: forref(4,ng6)
        REAL(KIND=wp), dimension(ng6) :: cfc11adj
        REAL(KIND=wp), dimension(ng6) :: cfc12
        PUBLIC read_externs_rrlw_kg06

    ! read interface
    PUBLIC kgen_read_var
    interface kgen_read_var
        module procedure read_var_real_wp_dim1
        module procedure read_var_real_wp_dim2
    end interface kgen_read_var

    CONTAINS

    ! module extern variables

    SUBROUTINE read_externs_rrlw_kg06(kgen_unit)
    integer, intent(in) :: kgen_unit
    READ(UNIT=kgen_unit) fracrefa
    READ(UNIT=kgen_unit) absa
    READ(UNIT=kgen_unit) ka_mco2
    READ(UNIT=kgen_unit) selfref
    READ(UNIT=kgen_unit) forref
    READ(UNIT=kgen_unit) cfc11adj
    READ(UNIT=kgen_unit) cfc12
    END SUBROUTINE read_externs_rrlw_kg06


    ! read subroutines
    subroutine read_var_real_wp_dim1(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:), allocatable :: var
        integer, dimension(2,1) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    subroutine read_var_real_wp_dim2(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:), allocatable :: var
        integer, dimension(2,2) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    END MODULE rrlw_kg06

    MODULE rrlw_kg07
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        INTEGER, parameter :: ng7  = 12
        REAL(KIND=wp), dimension(ng7) :: fracrefb
        REAL(KIND=wp) :: fracrefa(ng7,9)
        REAL(KIND=wp) :: absa(585,ng7)
        REAL(KIND=wp) :: absb(235,ng7)
        REAL(KIND=wp) :: ka_mco2(9,19,ng7)
        REAL(KIND=wp) :: kb_mco2(19,ng7)
        REAL(KIND=wp) :: selfref(10,ng7)
        REAL(KIND=wp) :: forref(4,ng7)
        PUBLIC read_externs_rrlw_kg07

    ! read interface
    PUBLIC kgen_read_var
    interface kgen_read_var
        module procedure read_var_real_wp_dim1
        module procedure read_var_real_wp_dim2
        module procedure read_var_real_wp_dim3
    end interface kgen_read_var

    CONTAINS

    ! module extern variables

    SUBROUTINE read_externs_rrlw_kg07(kgen_unit)
    integer, intent(in) :: kgen_unit
    READ(UNIT=kgen_unit) fracrefb
    READ(UNIT=kgen_unit) fracrefa
    READ(UNIT=kgen_unit) absa
    READ(UNIT=kgen_unit) absb
    READ(UNIT=kgen_unit) ka_mco2
    READ(UNIT=kgen_unit) kb_mco2
    READ(UNIT=kgen_unit) selfref
    READ(UNIT=kgen_unit) forref
    END SUBROUTINE read_externs_rrlw_kg07


    ! read subroutines
    subroutine read_var_real_wp_dim1(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:), allocatable :: var
        integer, dimension(2,1) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    subroutine read_var_real_wp_dim2(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:), allocatable :: var
        integer, dimension(2,2) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    subroutine read_var_real_wp_dim3(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:,:), allocatable :: var
        integer, dimension(2,3) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            READ(UNIT = kgen_unit) kgen_bound(1, 3)
            READ(UNIT = kgen_unit) kgen_bound(2, 3)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    END MODULE rrlw_kg07

    MODULE rrlw_kg08
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        INTEGER, parameter :: ng8  = 8
        REAL(KIND=wp), dimension(ng8) :: fracrefa
        REAL(KIND=wp), dimension(ng8) :: fracrefb
        REAL(KIND=wp), dimension(ng8) :: cfc12
        REAL(KIND=wp), dimension(ng8) :: cfc22adj
        REAL(KIND=wp) :: absa(65,ng8)
        REAL(KIND=wp) :: absb(235,ng8)
        REAL(KIND=wp) :: ka_mco2(19,ng8)
        REAL(KIND=wp) :: ka_mn2o(19,ng8)
        REAL(KIND=wp) :: ka_mo3(19,ng8)
        REAL(KIND=wp) :: kb_mco2(19,ng8)
        REAL(KIND=wp) :: kb_mn2o(19,ng8)
        REAL(KIND=wp) :: selfref(10,ng8)
        REAL(KIND=wp) :: forref(4,ng8)
        PUBLIC read_externs_rrlw_kg08

    ! read interface
    PUBLIC kgen_read_var
    interface kgen_read_var
        module procedure read_var_real_wp_dim1
        module procedure read_var_real_wp_dim2
    end interface kgen_read_var

    CONTAINS

    ! module extern variables

    SUBROUTINE read_externs_rrlw_kg08(kgen_unit)
    integer, intent(in) :: kgen_unit
    READ(UNIT=kgen_unit) fracrefa
    READ(UNIT=kgen_unit) fracrefb
    READ(UNIT=kgen_unit) cfc12
    READ(UNIT=kgen_unit) cfc22adj
    READ(UNIT=kgen_unit) absa
    READ(UNIT=kgen_unit) absb
    READ(UNIT=kgen_unit) ka_mco2
    READ(UNIT=kgen_unit) ka_mn2o
    READ(UNIT=kgen_unit) ka_mo3
    READ(UNIT=kgen_unit) kb_mco2
    READ(UNIT=kgen_unit) kb_mn2o
    READ(UNIT=kgen_unit) selfref
    READ(UNIT=kgen_unit) forref
    END SUBROUTINE read_externs_rrlw_kg08


    ! read subroutines
    subroutine read_var_real_wp_dim1(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:), allocatable :: var
        integer, dimension(2,1) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    subroutine read_var_real_wp_dim2(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:), allocatable :: var
        integer, dimension(2,2) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    END MODULE rrlw_kg08

    MODULE rrlw_kg09
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        INTEGER, parameter :: ng9  = 12
        REAL(KIND=wp), dimension(ng9) :: fracrefb
        REAL(KIND=wp) :: fracrefa(ng9,9)
        REAL(KIND=wp) :: absa(585,ng9)
        REAL(KIND=wp) :: absb(235,ng9)
        REAL(KIND=wp) :: ka_mn2o(9,19,ng9)
        REAL(KIND=wp) :: kb_mn2o(19,ng9)
        REAL(KIND=wp) :: selfref(10,ng9)
        REAL(KIND=wp) :: forref(4,ng9)
        PUBLIC read_externs_rrlw_kg09

    ! read interface
    PUBLIC kgen_read_var
    interface kgen_read_var
        module procedure read_var_real_wp_dim1
        module procedure read_var_real_wp_dim2
        module procedure read_var_real_wp_dim3
    end interface kgen_read_var

    CONTAINS

    ! module extern variables

    SUBROUTINE read_externs_rrlw_kg09(kgen_unit)
    integer, intent(in) :: kgen_unit
    READ(UNIT=kgen_unit) fracrefb
    READ(UNIT=kgen_unit) fracrefa
    READ(UNIT=kgen_unit) absa
    READ(UNIT=kgen_unit) absb
    READ(UNIT=kgen_unit) ka_mn2o
    READ(UNIT=kgen_unit) kb_mn2o
    READ(UNIT=kgen_unit) selfref
    READ(UNIT=kgen_unit) forref
    END SUBROUTINE read_externs_rrlw_kg09


    ! read subroutines
    subroutine read_var_real_wp_dim1(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:), allocatable :: var
        integer, dimension(2,1) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    subroutine read_var_real_wp_dim2(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:), allocatable :: var
        integer, dimension(2,2) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    subroutine read_var_real_wp_dim3(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:,:), allocatable :: var
        integer, dimension(2,3) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            READ(UNIT = kgen_unit) kgen_bound(1, 3)
            READ(UNIT = kgen_unit) kgen_bound(2, 3)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    END MODULE rrlw_kg09

    MODULE rrlw_kg10
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        INTEGER, parameter :: ng10 = 6
        REAL(KIND=wp), dimension(ng10) :: fracrefa
        REAL(KIND=wp), dimension(ng10) :: fracrefb
        REAL(KIND=wp) :: absa(65,ng10)
        REAL(KIND=wp) :: absb(235,ng10)
        REAL(KIND=wp) :: selfref(10,ng10)
        REAL(KIND=wp) :: forref(4,ng10)
        PUBLIC read_externs_rrlw_kg10

    ! read interface
    PUBLIC kgen_read_var
    interface kgen_read_var
        module procedure read_var_real_wp_dim1
        module procedure read_var_real_wp_dim2
    end interface kgen_read_var

    CONTAINS

    ! module extern variables

    SUBROUTINE read_externs_rrlw_kg10(kgen_unit)
    integer, intent(in) :: kgen_unit
    READ(UNIT=kgen_unit) fracrefa
    READ(UNIT=kgen_unit) fracrefb
    READ(UNIT=kgen_unit) absa
    READ(UNIT=kgen_unit) absb
    READ(UNIT=kgen_unit) selfref
    READ(UNIT=kgen_unit) forref
    END SUBROUTINE read_externs_rrlw_kg10


    ! read subroutines
    subroutine read_var_real_wp_dim1(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:), allocatable :: var
        integer, dimension(2,1) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    subroutine read_var_real_wp_dim2(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:), allocatable :: var
        integer, dimension(2,2) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    END MODULE rrlw_kg10

    MODULE rrlw_kg11
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        INTEGER, parameter :: ng11 = 8
        REAL(KIND=wp), dimension(ng11) :: fracrefa
        REAL(KIND=wp), dimension(ng11) :: fracrefb
        REAL(KIND=wp) :: absa(65,ng11)
        REAL(KIND=wp) :: absb(235,ng11)
        REAL(KIND=wp) :: ka_mo2(19,ng11)
        REAL(KIND=wp) :: kb_mo2(19,ng11)
        REAL(KIND=wp) :: selfref(10,ng11)
        REAL(KIND=wp) :: forref(4,ng11)
        PUBLIC read_externs_rrlw_kg11

    ! read interface
    PUBLIC kgen_read_var
    interface kgen_read_var
        module procedure read_var_real_wp_dim1
        module procedure read_var_real_wp_dim2
    end interface kgen_read_var

    CONTAINS

    ! module extern variables

    SUBROUTINE read_externs_rrlw_kg11(kgen_unit)
    integer, intent(in) :: kgen_unit
    READ(UNIT=kgen_unit) fracrefa
    READ(UNIT=kgen_unit) fracrefb
    READ(UNIT=kgen_unit) absa
    READ(UNIT=kgen_unit) absb
    READ(UNIT=kgen_unit) ka_mo2
    READ(UNIT=kgen_unit) kb_mo2
    READ(UNIT=kgen_unit) selfref
    READ(UNIT=kgen_unit) forref
    END SUBROUTINE read_externs_rrlw_kg11


    ! read subroutines
    subroutine read_var_real_wp_dim1(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:), allocatable :: var
        integer, dimension(2,1) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    subroutine read_var_real_wp_dim2(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:), allocatable :: var
        integer, dimension(2,2) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    END MODULE rrlw_kg11

    MODULE rrlw_kg12
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        INTEGER, parameter :: ng12 = 8
        REAL(KIND=wp) :: fracrefa(ng12,9)
        REAL(KIND=wp) :: absa(585,ng12)
        REAL(KIND=wp) :: selfref(10,ng12)
        REAL(KIND=wp) :: forref(4,ng12)
        PUBLIC read_externs_rrlw_kg12

    ! read interface
    PUBLIC kgen_read_var
    interface kgen_read_var
        module procedure read_var_real_wp_dim2
    end interface kgen_read_var

    CONTAINS

    ! module extern variables

    SUBROUTINE read_externs_rrlw_kg12(kgen_unit)
    integer, intent(in) :: kgen_unit
    READ(UNIT=kgen_unit) fracrefa
    READ(UNIT=kgen_unit) absa
    READ(UNIT=kgen_unit) selfref
    READ(UNIT=kgen_unit) forref
    END SUBROUTINE read_externs_rrlw_kg12


    ! read subroutines
    subroutine read_var_real_wp_dim2(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:), allocatable :: var
        integer, dimension(2,2) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    END MODULE rrlw_kg12

    MODULE rrlw_kg13
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        INTEGER, parameter :: ng13 = 4
        REAL(KIND=wp), dimension(ng13) :: fracrefb
        REAL(KIND=wp) :: fracrefa(ng13,9)
        REAL(KIND=wp) :: absa(585,ng13)
        REAL(KIND=wp) :: ka_mco2(9,19,ng13)
        REAL(KIND=wp) :: ka_mco(9,19,ng13)
        REAL(KIND=wp) :: kb_mo3(19,ng13)
        REAL(KIND=wp) :: selfref(10,ng13)
        REAL(KIND=wp) :: forref(4,ng13)
        PUBLIC read_externs_rrlw_kg13

    ! read interface
    PUBLIC kgen_read_var
    interface kgen_read_var
        module procedure read_var_real_wp_dim1
        module procedure read_var_real_wp_dim2
        module procedure read_var_real_wp_dim3
    end interface kgen_read_var

    CONTAINS

    ! module extern variables

    SUBROUTINE read_externs_rrlw_kg13(kgen_unit)
    integer, intent(in) :: kgen_unit
    READ(UNIT=kgen_unit) fracrefb
    READ(UNIT=kgen_unit) fracrefa
    READ(UNIT=kgen_unit) absa
    READ(UNIT=kgen_unit) ka_mco2
    READ(UNIT=kgen_unit) ka_mco
    READ(UNIT=kgen_unit) kb_mo3
    READ(UNIT=kgen_unit) selfref
    READ(UNIT=kgen_unit) forref
    END SUBROUTINE read_externs_rrlw_kg13


    ! read subroutines
    subroutine read_var_real_wp_dim1(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:), allocatable :: var
        integer, dimension(2,1) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    subroutine read_var_real_wp_dim2(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:), allocatable :: var
        integer, dimension(2,2) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    subroutine read_var_real_wp_dim3(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:,:), allocatable :: var
        integer, dimension(2,3) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            READ(UNIT = kgen_unit) kgen_bound(1, 3)
            READ(UNIT = kgen_unit) kgen_bound(2, 3)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    END MODULE rrlw_kg13

    MODULE rrlw_kg14
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        INTEGER, parameter :: ng14 = 2
        REAL(KIND=wp), dimension(ng14) :: fracrefa
        REAL(KIND=wp), dimension(ng14) :: fracrefb
        REAL(KIND=wp) :: absa(65,ng14)
        REAL(KIND=wp) :: absb(235,ng14)
        REAL(KIND=wp) :: selfref(10,ng14)
        REAL(KIND=wp) :: forref(4,ng14)
        PUBLIC read_externs_rrlw_kg14

    ! read interface
    PUBLIC kgen_read_var
    interface kgen_read_var
        module procedure read_var_real_wp_dim1
        module procedure read_var_real_wp_dim2
    end interface kgen_read_var

    CONTAINS

    ! module extern variables

    SUBROUTINE read_externs_rrlw_kg14(kgen_unit)
    integer, intent(in) :: kgen_unit
    READ(UNIT=kgen_unit) fracrefa
    READ(UNIT=kgen_unit) fracrefb
    READ(UNIT=kgen_unit) absa
    READ(UNIT=kgen_unit) absb
    READ(UNIT=kgen_unit) selfref
    READ(UNIT=kgen_unit) forref
    END SUBROUTINE read_externs_rrlw_kg14


    ! read subroutines
    subroutine read_var_real_wp_dim1(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:), allocatable :: var
        integer, dimension(2,1) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    subroutine read_var_real_wp_dim2(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:), allocatable :: var
        integer, dimension(2,2) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    END MODULE rrlw_kg14

    MODULE rrlw_kg15
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        INTEGER, parameter :: ng15 = 2
        REAL(KIND=wp) :: fracrefa(ng15,9)
        REAL(KIND=wp) :: absa(585,ng15)
        REAL(KIND=wp) :: ka_mn2(9,19,ng15)
        REAL(KIND=wp) :: selfref(10,ng15)
        REAL(KIND=wp) :: forref(4,ng15)
        PUBLIC read_externs_rrlw_kg15

    ! read interface
    PUBLIC kgen_read_var
    interface kgen_read_var
        module procedure read_var_real_wp_dim2
        module procedure read_var_real_wp_dim3
    end interface kgen_read_var

    CONTAINS

    ! module extern variables

    SUBROUTINE read_externs_rrlw_kg15(kgen_unit)
    integer, intent(in) :: kgen_unit
    READ(UNIT=kgen_unit) fracrefa
    READ(UNIT=kgen_unit) absa
    READ(UNIT=kgen_unit) ka_mn2
    READ(UNIT=kgen_unit) selfref
    READ(UNIT=kgen_unit) forref
    END SUBROUTINE read_externs_rrlw_kg15


    ! read subroutines
    subroutine read_var_real_wp_dim2(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:), allocatable :: var
        integer, dimension(2,2) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    subroutine read_var_real_wp_dim3(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:,:), allocatable :: var
        integer, dimension(2,3) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            READ(UNIT = kgen_unit) kgen_bound(1, 3)
            READ(UNIT = kgen_unit) kgen_bound(2, 3)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1, kgen_bound(2, 3) - kgen_bound(1, 3) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    END MODULE rrlw_kg15

    MODULE rrlw_kg16
        USE mo_kind, ONLY: wp
        IMPLICIT NONE
        INTEGER, parameter :: ng16 = 2
        REAL(KIND=wp), dimension(ng16) :: fracrefb
        REAL(KIND=wp) :: fracrefa(ng16,9)
        REAL(KIND=wp) :: absa(585,ng16)
        REAL(KIND=wp) :: absb(235,ng16)
        REAL(KIND=wp) :: selfref(10,ng16)
        REAL(KIND=wp) :: forref(4,ng16)
        PUBLIC read_externs_rrlw_kg16

    ! read interface
    PUBLIC kgen_read_var
    interface kgen_read_var
        module procedure read_var_real_wp_dim1
        module procedure read_var_real_wp_dim2
    end interface kgen_read_var

    CONTAINS

    ! module extern variables

    SUBROUTINE read_externs_rrlw_kg16(kgen_unit)
    integer, intent(in) :: kgen_unit
    READ(UNIT=kgen_unit) fracrefb
    READ(UNIT=kgen_unit) fracrefa
    READ(UNIT=kgen_unit) absa
    READ(UNIT=kgen_unit) absb
    READ(UNIT=kgen_unit) selfref
    READ(UNIT=kgen_unit) forref
    END SUBROUTINE read_externs_rrlw_kg16


    ! read subroutines
    subroutine read_var_real_wp_dim1(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:), allocatable :: var
        integer, dimension(2,1) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    subroutine read_var_real_wp_dim2(var, kgen_unit)
        integer, intent(in) :: kgen_unit
        real(kind=wp), intent(out), dimension(:,:), allocatable :: var
        integer, dimension(2,2) :: kgen_bound
        logical is_save
        
        READ(UNIT = kgen_unit) is_save
        if ( is_save ) then
            READ(UNIT = kgen_unit) kgen_bound(1, 1)
            READ(UNIT = kgen_unit) kgen_bound(2, 1)
            READ(UNIT = kgen_unit) kgen_bound(1, 2)
            READ(UNIT = kgen_unit) kgen_bound(2, 2)
            ALLOCATE(var(kgen_bound(2, 1) - kgen_bound(1, 1) + 1, kgen_bound(2, 2) - kgen_bound(1, 2) + 1))
            READ(UNIT = kgen_unit) var
        end if
    end subroutine
    END MODULE rrlw_kg16
