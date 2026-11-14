! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=61 %s -o - | FileCheck %s

module polymorphic_derived_type_map
  implicit none

  type :: base_t
    integer :: x
  end type

  type, extends(base_t) :: child_t
    integer :: y
  end type

contains
  subroutine map_polymorphic()
    class(base_t), allocatable :: obj

    allocate(child_t :: obj)

    !$omp target enter data map(to: obj)

    !$omp target map(tofrom: obj)
      select type (obj)
      type is (child_t)
        obj%x = 1
        obj%y = 2
      end select
    !$omp end target

    !$omp target exit data map(from: obj)
  end subroutine
end module

! CHECK: %[[BASE_ADDR:.*]] = fir.box_offset %{{.*}} base_addr
! CHECK: %[[BASE_MAP:.*]] = omp.map.info {{.*}}var_ptr_ptr(%[[BASE_ADDR]]{{.*}})
! CHECK: %[[DESC_MAP:.*]] = omp.map.info {{.*}}members(%[[BASE_MAP]] : {{\[}}0{{\]}}
! CHECK: %[[BASE_ATTACH:.*]] = omp.map.info var_ptr({{.*}}) {{.*}}map_clauses(attach, ref_ptr, ref_ptee){{.*}}var_ptr_ptr(%[[BASE_ADDR]]
! CHECK: %[[TYPE_DESC:.*]] = fir.box_offset %{{.*}} derived_type
! CHECK-NOT: omp.map.info {{.*}}var_ptr_ptr(%[[TYPE_DESC]]{{.*}}!fir.type<_QM__fortran_type_infoTderivedtype>
! CHECK: %[[TYPE_DESC_ATTACH:.*]] = omp.map.info var_ptr(%[[TYPE_DESC]]{{.*}}!fir.llvm_ptr<i8>) {{.*}}map_clauses(attach, ref_ptr, ref_ptee){{.*}}var_ptr_ptr(%[[TYPE_DESC]]{{.*}}!fir.llvm_ptr<i8>)
! CHECK: omp.target_enter_data map_entries(%[[DESC_MAP]], %[[BASE_ATTACH]], %[[TYPE_DESC_ATTACH]], %[[BASE_MAP]]

! CHECK: %[[TGT_BASE_ADDR:.*]] = fir.box_offset %{{.*}} base_addr
! CHECK: %[[TGT_BASE_MAP:.*]] = omp.map.info {{.*}}var_ptr_ptr(%[[TGT_BASE_ADDR]]{{.*}})
! CHECK: %[[TGT_DESC_MAP:.*]] = omp.map.info {{.*}}members(%[[TGT_BASE_MAP]] : {{\[}}0{{\]}}
! CHECK: %[[TGT_BASE_ATTACH:.*]] = omp.map.info var_ptr({{.*}}) {{.*}}map_clauses(attach, ref_ptr, ref_ptee){{.*}}var_ptr_ptr(%[[TGT_BASE_ADDR]]
! CHECK: %[[TGT_TYPE_DESC:.*]] = fir.box_offset %{{.*}} derived_type
! CHECK-NOT: omp.map.info {{.*}}var_ptr_ptr(%[[TGT_TYPE_DESC]]{{.*}}!fir.type<_QM__fortran_type_infoTderivedtype>
! CHECK: %[[TGT_TYPE_DESC_ATTACH:.*]] = omp.map.info var_ptr(%[[TGT_TYPE_DESC]]{{.*}}!fir.llvm_ptr<i8>) {{.*}}map_clauses(attach, ref_ptr, ref_ptee){{.*}}var_ptr_ptr(%[[TGT_TYPE_DESC]]{{.*}}!fir.llvm_ptr<i8>)
! CHECK: omp.target {{.*}}map_entries(%[[TGT_DESC_MAP]] -> %{{.*}}, %[[TGT_BASE_ATTACH]] -> %{{.*}}, %[[TGT_TYPE_DESC_ATTACH]] -> %{{.*}}, %[[TGT_BASE_MAP]] -> %{{.*}}
