! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=61 %s -o - | fir-opt --omp-map-info-finalization --omp-polymorphic-rtti-declare-target --fir-polymorphic-op | FileCheck %s

! Verify that OpenMP 6.1 polymorphic offload lowering makes runtime type
! descriptors and type-bound dispatch targets available to the device when
! device code references the RTTI.

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

module poly_rtti_host_only
  type :: host_only_t
    integer :: z
  contains
    procedure :: host_query
  end type

contains
  subroutine host_query(this)
    class(host_only_t), intent(inout) :: this
    this%z = 4
  end subroutine

  subroutine host_only()
    type(host_only_t) :: h

    call h%host_query()
  end subroutine

  subroutine mixed_unrelated_target()
    type(host_only_t) :: h
    integer :: i

    call h%host_query()

    !$omp target map(tofrom: i)
      i = i + 1
    !$omp end target
  end subroutine
end module

module poly_rtti_explicit_dt
  type :: explicit_t
    integer :: w
  contains
    procedure :: explicit_set
  end type

  !$omp declare target(explicit_use)

contains
  subroutine explicit_set(this)
    class(explicit_t), intent(inout) :: this
    this%w = 5
  end subroutine

  subroutine explicit_use(obj)
    class(explicit_t), intent(inout) :: obj

    call obj%explicit_set()
  end subroutine
end module

! CHECK-LABEL: func.func @_QMpoly_rtti_declare_targetPset_x(
! CHECK-SAME: attributes {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>}

! CHECK-LABEL: func.func @_QMpoly_rtti_declare_targetPchild_set_x(
! CHECK-SAME: attributes {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>}

! CHECK: fir.box_offset %{{.*}} base_addr
! CHECK: omp.map.info {{.*}}var_ptr_ptr(%{{.*}}{{.*}})
! CHECK: omp.map.info {{.*}}members(%{{.*}} : {{\[}}0{{\]}}
! CHECK: omp.map.info var_ptr({{.*}}) {{.*}}map_clauses(attach, ref_ptr, ref_ptee){{.*}}var_ptr_ptr(%{{.*}}
! CHECK: %[[TYPE_DESC:.*]] = fir.box_offset %{{.*}} derived_type
! CHECK: %[[TYPE_DESC_ATTACH:.*]] = omp.map.info var_ptr(%[[TYPE_DESC]]{{.*}}!fir.llvm_ptr<i8>) {{.*}}map_clauses(attach, ref_ptr, ref_ptee){{.*}}var_ptr_ptr(%[[TYPE_DESC]]{{.*}}!fir.llvm_ptr<i8>)
! CHECK: omp.target {{.*}}map_entries({{.*}}%[[TYPE_DESC_ATTACH]] -> %{{.*}}

! CHECK-LABEL: func.func @_QMpoly_rtti_host_onlyPhost_query(
! CHECK-NOT: omp.declare_target
! CHECK-SAME: ) {

! CHECK-LABEL: func.func @_QMpoly_rtti_explicit_dtPexplicit_set(
! CHECK-SAME: attributes {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to, implicit = true>}

! CHECK: fir.global linkonce_odr @_QMpoly_rtti_declare_targetE.dt.base_t {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>} constant target
! CHECK: fir.global linkonce_odr @_QMpoly_rtti_declare_targetE.dt.child_t {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>} constant target
! CHECK: fir.global linkonce_odr @_QMpoly_rtti_host_onlyE.dt.host_only_t constant target
! CHECK: fir.global linkonce_odr @_QMpoly_rtti_explicit_dtE.dt.explicit_t {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = to>} constant target
