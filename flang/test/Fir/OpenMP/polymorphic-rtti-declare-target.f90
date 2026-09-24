! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=61 %s -o - | fir-opt --fir-polymorphic-op | FileCheck %s

! Verify that OpenMP 6.1 polymorphic offload lowering makes runtime type
! descriptors and type-bound dispatch targets available to the device, and maps
! the dynamic type descriptor pointer from the descriptor addendum.

module poly_rtti_declare_target
  type :: base_t
    integer :: x
  contains
    procedure :: set_x
  end type

  type, extends(base_t) :: child_t
    integer :: y
  contains
    procedure :: set_x => child_set_x
  end type

contains
  subroutine set_x(this)
    class(base_t), intent(inout) :: this
    this%x = 1
  end subroutine

  subroutine child_set_x(this)
    class(child_t), intent(inout) :: this
    this%x = 2
    this%y = 3
  end subroutine

  subroutine run()
    class(base_t), allocatable :: obj

    allocate(child_t :: obj)

    !$omp target map(tofrom: obj)
      call obj%set_x()
    !$omp end target
  end subroutine
end module

! CHECK-LABEL: func.func @_QMpoly_rtti_declare_targetPset_x(
! CHECK-SAME: attributes {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>}

! CHECK-LABEL: func.func @_QMpoly_rtti_declare_targetPchild_set_x(
! CHECK-SAME: attributes {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>}

! CHECK: %[[BASE_ADDR:.*]] = fir.box_offset %{{.*}} base_addr
! CHECK: %[[BASE_MAP:.*]] = omp.map.info {{.*}}var_ptr_ptr(%[[BASE_ADDR]]{{.*}})
! CHECK: %[[DESC_MAP:.*]] = omp.map.info {{.*}}members(%[[BASE_MAP]] : {{\[}}0{{\]}}
! CHECK: %[[BASE_ATTACH:.*]] = omp.map.info var_ptr({{.*}}) {{.*}}map_clauses(attach, ref_ptr, ref_ptee){{.*}}var_ptr_ptr(%[[BASE_ADDR]]
! CHECK: %[[TYPE_DESC:.*]] = fir.box_offset %{{.*}} derived_type
! CHECK: %[[TYPE_DESC_ATTACH:.*]] = omp.map.info var_ptr(%[[TYPE_DESC]]{{.*}}!fir.llvm_ptr<i8>) {{.*}}map_clauses(attach, ref_ptr, ref_ptee){{.*}}var_ptr_ptr(%[[TYPE_DESC]]{{.*}}!fir.llvm_ptr<i8>)
! CHECK: omp.target {{.*}}map_entries(%[[DESC_MAP]] -> %{{.*}}, %[[BASE_ATTACH]] -> %{{.*}}, %[[TYPE_DESC_ATTACH]] -> %{{.*}}, %[[BASE_MAP]] -> %{{.*}}

! CHECK: fir.global linkonce_odr @_QMpoly_rtti_declare_targetE.dt.base_t {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>} constant target
! CHECK: fir.global linkonce_odr @_QMpoly_rtti_declare_targetE.dt.child_t {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>} constant target
