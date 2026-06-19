! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s
program p
  type t
    integer :: x
  end type

  type(t), pointer :: ptr
  type(t), allocatable :: al
  allocate(ptr, al)

  !$omp target
    ptr%x = ptr%x + 1
    al%x = al%x + 1
  !$omp end target
end program

! CHECK-LABEL: omp.declare_mapper @_QQFt_omp_default_mapper

! CHECK-LABEL: func.func @_QQmain
! The pointer capture should not get an implicit default mapper.
! CHECK: %[[PTR_PTEE_MAP:.*]] = omp.map.info {{.*}}map_clauses(implicit, tofrom){{.*}} {name = ""}
! CHECK: %[[PTR_DESC_MAP:.*]] = omp.map.info {{.*}}members(%[[PTR_PTEE_MAP]]{{.*}}){{.*}} {name = "ptr"}

! The allocatable capture should still use the implicit default mapper.
! CHECK: %[[ALLOC_PTEE_MAP:.*]] = omp.map.info {{.*}}map_clauses(implicit, tofrom){{.*}}mapper(@_QQFt_omp_default_mapper){{.*}} {name = ""}
! CHECK: %[[ALLOC_DESC_MAP:.*]] = omp.map.info {{.*}}members(%[[ALLOC_PTEE_MAP]]{{.*}}){{.*}} {name = "al"}
