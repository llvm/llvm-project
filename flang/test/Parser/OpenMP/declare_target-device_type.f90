! RUN: %flang_fc1 -fdebug-unparse -fopenmp %s | FileCheck --ignore-case %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree -fopenmp %s | FileCheck --check-prefix="PARSE-TREE" %s

subroutine openmp_declare_target
    !CHECK: !$omp declare target device_type(host)
    !$omp declare target device_type(host)
    !CHECK: !$omp declare target device_type(nohost)
    !$omp declare target device_type(nohost)
    !CHECK: !$omp declare target device_type(any)
    !$omp declare target device_type(any)
    integer :: a(1024), i
    !CHECK: do
    do i = 1, 1024
        a(i) = i
    !CHECK: end do
    end do

!PARSE-TREE: OpenMPDeclarativeConstruct -> OpenMPDeclareTargetConstruct
!PARSE-TREE: OmpDeclareTargetSpecifier -> OmpDeclareTargetWithClause -> OmpClauseList -> OmpClause -> DeviceType -> OmpDeviceTypeClause -> Type = Host
!PARSE-TREE: OmpDeclareTargetSpecifier -> OmpDeclareTargetWithClause -> OmpClauseList -> OmpClause -> DeviceType -> OmpDeviceTypeClause -> Type = Nohost
!PARSE-TREE: OmpDeclareTargetSpecifier -> OmpDeclareTargetWithClause -> OmpClauseList -> OmpClause -> DeviceType -> OmpDeviceTypeClause -> Type = Any
END subroutine openmp_declare_target
