! RUN: %flang_fc1 -fdebug-unparse -fopenmp -fopenmp-version=52 %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp -fopenmp-version=52 %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine openmp_declare_target
    integer, save :: x, y
    !CHECK: !$omp declare target device_type(host) enter(x)
    !$omp declare target device_type(host) enter(x)
    !CHECK: !$omp declare target device_type(nohost) enter(x)
    !$omp declare target device_type(nohost) enter(x)
    !CHECK: !$omp declare target device_type(any) enter(x)
    !$omp declare target device_type(any) enter(x)

    !CHECK: !$omp declare target device_type(host) to(x)
    !$omp declare target device_type(host) to(x)
    !CHECK: !$omp declare target device_type(nohost) to(x)
    !$omp declare target device_type(nohost) to(x)
    !CHECK: !$omp declare target device_type(any) to(x)
    !$omp declare target device_type(any) to(x)

    !CHECK: !$omp declare target device_type(host) enter(y) to(x)
    !$omp declare target device_type(host) enter(y) to(x)
    !CHECK: !$omp declare target device_type(nohost) enter(y) to(x)
    !$omp declare target device_type(nohost) enter(y) to(x)
    !CHECK: !$omp declare target device_type(any) enter(y) to(x)
    !$omp declare target device_type(any) enter(y) to(x)
    integer :: a(1024), i
    !CHECK: do
    do i = 1, 1024
        a(i) = i
    !CHECK: end do
    end do

!PARSE-TREE: OpenMPDeclarativeConstruct -> OpenMPDeclareTargetConstruct
!PARSE-TREE: OmpDeclareTargetSpecifier -> OmpDeclareTargetWithClause -> OmpClauseList -> OmpClause -> DeviceType -> OmpDeviceTypeClause -> DeviceTypeDescription = Host
!PARSE-TREE: OmpDeclareTargetSpecifier -> OmpDeclareTargetWithClause -> OmpClauseList -> OmpClause -> DeviceType -> OmpDeviceTypeClause -> DeviceTypeDescription = Nohost
!PARSE-TREE: OmpDeclareTargetSpecifier -> OmpDeclareTargetWithClause -> OmpClauseList -> OmpClause -> DeviceType -> OmpDeviceTypeClause -> DeviceTypeDescription = Any
END subroutine openmp_declare_target
