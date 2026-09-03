! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s

subroutine nested_dt()
   implicit none

   type :: buffer
     integer :: i
     real :: data(8)
   end type buffer

   type :: array_dt
      type(buffer) :: buf
   end type array_dt

   integer :: i
   type(array_dt) :: ad

   do concurrent(i=1:8)
      ad%buf%data(i) = real(i)
   end do
end subroutine nested_dt

subroutine nested_alloc_dt()
   implicit none

   type :: alloc_buffer
     integer :: i
     real, allocatable :: data(:)
   end type alloc_buffer

   type :: alloc_array_dt
      type(alloc_buffer) :: buf
   end type alloc_array_dt

   integer :: i
   type(alloc_array_dt) :: aad

   allocate(aad%buf%data(8), source=0.0)

   do concurrent(i=1:8)
      aad%buf%data(i) = real(i)
   end do
end subroutine nested_alloc_dt


! CHECK: omp.declare_mapper @[[BUF_MAPPER:_QFnested_alloc_dtTalloc_buffer_omp_default_mapper]] : !fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}> {
! CHECK-NEXT: ^bb0(%[[BUF_ARG:.*]]: !fir.ref<!fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>):
! CHECK-NEXT: %[[BUF_DECL:.*]]:2 = hlfir.declare %[[BUF_ARG]] {uniq_name = ""}
! CHECK: %{{.*}} = fir.field_index i, !fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>
! CHECK: %[[I_COORD:.*]] = fir.coordinate_of %[[BUF_DECL]]#0, i : (!fir.ref<!fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.ref<i32>
! CHECK: %[[BUF_I_MAP:.*]] = omp.map.info var_ptr(%[[I_COORD]] : !fir.ref<i32>, i32) map_clauses(implicit, tofrom) capture(ByRef) name("") -> !fir.ref<i32>
! CHECK: %{{.*}} = fir.field_index data, !fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>
! CHECK: %[[DATA_COORD:.*]] = fir.coordinate_of %[[BUF_DECL]]#0, data : (!fir.ref<!fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK: %[[DATA_BOUNDS:.*]] = omp.map.bounds lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}}) stride_in_bytes(true)
! CHECK: %[[DATA_BASE_PTR:.*]] = fir.box_offset %[[DATA_COORD]] base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>
! CHECK: %[[BUF_DATA_MAP:.*]] = omp.map.info var_ptr(%[[DATA_COORD]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(implicit, tofrom) capture(ByRef) var_ptr_ptr(%[[DATA_BASE_PTR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>, f32) bounds(%[[DATA_BOUNDS]]) name("") -> !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>
! CHECK: %[[BUF_DESC_MAP:.*]] = omp.map.info var_ptr(%[[DATA_COORD]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(always, implicit, to) capture(ByRef) name("") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK: %[[BUF_ATTACH_MAP:.*]] = omp.map.info var_ptr(%[[DATA_COORD]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr(%[[DATA_BASE_PTR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>, f32) bounds(%[[DATA_BOUNDS]]) name("") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK: %[[BUF_PARENT_MAP:.*]] = omp.map.info var_ptr(%[[BUF_DECL]]#1 : !fir.ref<!fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>) map_clauses(implicit) capture(ByRef) members(%[[BUF_I_MAP]], %[[BUF_DESC_MAP]], %[[BUF_DATA_MAP]] : [0], [1], [1, 0] : !fir.ref<i32>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>) name("") partial_map(true) -> !fir.ref<!fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
! CHECK: omp.declare_mapper.info map_entries(%[[BUF_PARENT_MAP]], %[[BUF_I_MAP]], %[[BUF_DESC_MAP]], %[[BUF_ATTACH_MAP]], %[[BUF_DATA_MAP]] : !fir.ref<!fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.ref<i32>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>)


! CHECK: omp.declare_mapper @[[AAD_MAPPER:_QFnested_alloc_dtTalloc_array_dt_omp_default_mapper]] : !fir.type<_QFnested_alloc_dtTalloc_array_dt{buf:!fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>}> {
! CHECK-NEXT: ^bb0(%[[AAD_ARG:.*]]: !fir.ref<!fir.type<_QFnested_alloc_dtTalloc_array_dt{buf:!fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>}>>):
! CHECK-NEXT: %[[AAD_DECL:.*]]:2 = hlfir.declare %[[AAD_ARG]] {uniq_name = ""}
! CHECK: %[[AAD_BUF_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>, !fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>) map_clauses(implicit, tofrom) capture(ByRef) mapper(@[[BUF_MAPPER]]) name("") -> !fir.ref<!fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
! CHECK: %[[AAD_PARENT_MAP:.*]] = omp.map.info var_ptr(%[[AAD_DECL]]#1 : !fir.ref<!fir.type<_QFnested_alloc_dtTalloc_array_dt{buf:!fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>}>>, !fir.type<_QFnested_alloc_dtTalloc_array_dt{buf:!fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>}>) map_clauses(implicit) capture(ByRef) members(%[[AAD_BUF_MAP]] : [0] : !fir.ref<!fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) name("") partial_map(true) -> !fir.ref<!fir.type<_QFnested_alloc_dtTalloc_array_dt{buf:!fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>}>>
! CHECK: omp.declare_mapper.info map_entries(%[[AAD_PARENT_MAP]], %[[AAD_BUF_MAP]] : !fir.ref<!fir.type<_QFnested_alloc_dtTalloc_array_dt{buf:!fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>}>>, !fir.ref<!fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>)

! CHECK-LABEL: func.func @_QPnested_dt() {
! CHECK: %[[AD_HOST:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFnested_dtEad"} : (!fir.ref<!fir.type<_QFnested_dtTarray_dt{buf:!fir.type<_QFnested_dtTbuffer{i:i32,data:!fir.array<8xf32>}>}>>) -> {{.*}}
! CHECK: %[[DT_LB:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<index>, index) map_clauses(implicit) capture(ByCopy) name("") -> !fir.ref<index>
! CHECK: %[[DT_UB:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<index>, index) map_clauses(implicit) capture(ByCopy) name("") -> !fir.ref<index>
! CHECK: %[[DT_ST:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<index>, index) map_clauses(implicit) capture(ByCopy) name("") -> !fir.ref<index>
! CHECK: %[[DT_I:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) name("_QFnested_dtEi") -> !fir.ref<i32>
! CHECK: %[[AD_MAP:.*]] = omp.map.info var_ptr(%[[AD_HOST]]#1 : !fir.ref<!fir.type<_QFnested_dtTarray_dt{buf:!fir.type<_QFnested_dtTbuffer{i:i32,data:!fir.array<8xf32>}>}>>, !fir.type<_QFnested_dtTarray_dt{buf:!fir.type<_QFnested_dtTbuffer{i:i32,data:!fir.array<8xf32>}>}>) map_clauses(implicit, tofrom) capture(ByRef) name("_QFnested_dtEad") -> !fir.ref<!fir.type<_QFnested_dtTarray_dt{buf:!fir.type<_QFnested_dtTbuffer{i:i32,data:!fir.array<8xf32>}>}>>
! CHECK: omp.target kernel_type(spmd) host_eval(%{{.*}} -> %[[DT_A0:.*]], %{{.*}} -> %[[DT_A1:.*]], %{{.*}} -> %[[DT_A2:.*]] : index, index, index) map_entries(%[[DT_LB]] -> %[[DT_MA3:.*]], %[[DT_UB]] -> %[[DT_MA4:.*]], %[[DT_ST]] -> %[[DT_MA5:.*]], %[[DT_I]] -> %[[DT_MA6:.*]], %[[AD_MAP]] -> %[[AD_ARG:.*]] : !fir.ref<index>, !fir.ref<index>, !fir.ref<index>, !fir.ref<i32>, !fir.ref<!fir.type<_QFnested_dtTarray_dt{buf:!fir.type<_QFnested_dtTbuffer{i:i32,data:!fir.array<8xf32>}>}>>) {

! CHECK-LABEL: func.func @_QPnested_alloc_dt() {
! CHECK: %[[AAD_HOST:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFnested_alloc_dtEaad"} : (!fir.ref<!fir.type<_QFnested_alloc_dtTalloc_array_dt{buf:!fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>}>>) -> {{.*}}
! CHECK: %[[AL_LB:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<index>, index) map_clauses(implicit) capture(ByCopy) name("") -> !fir.ref<index>
! CHECK: %[[AL_UB:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<index>, index) map_clauses(implicit) capture(ByCopy) name("") -> !fir.ref<index>
! CHECK: %[[AL_ST:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<index>, index) map_clauses(implicit) capture(ByCopy) name("") -> !fir.ref<index>
! CHECK: %[[AL_I:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<i32>, i32) map_clauses(implicit) capture(ByCopy) name("_QFnested_alloc_dtEi") -> !fir.ref<i32>
! CHECK: %[[AAD_MAP:.*]] = omp.map.info var_ptr(%[[AAD_HOST]]#1 : !fir.ref<!fir.type<_QFnested_alloc_dtTalloc_array_dt{buf:!fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>}>>, !fir.type<_QFnested_alloc_dtTalloc_array_dt{buf:!fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>}>) map_clauses(implicit, tofrom) capture(ByRef) mapper(@[[AAD_MAPPER]]) name("_QFnested_alloc_dtEaad") -> !fir.ref<!fir.type<_QFnested_alloc_dtTalloc_array_dt{buf:!fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>}>>
! CHECK: omp.target kernel_type(spmd) host_eval(%{{.*}} -> %[[AL_A0:.*]], %{{.*}} -> %[[AL_A1:.*]], %{{.*}} -> %[[AL_A2:.*]] : index, index, index) map_entries(%[[AL_LB]] -> %[[AL_MA3:.*]], %[[AL_UB]] -> %[[AL_MA4:.*]], %[[AL_ST]] -> %[[AL_MA5:.*]], %[[AL_I]] -> %[[AL_MA6:.*]], %[[AAD_MAP]] -> %[[AAD_ARG:.*]] : !fir.ref<index>, !fir.ref<index>, !fir.ref<index>, !fir.ref<i32>, !fir.ref<!fir.type<_QFnested_alloc_dtTalloc_array_dt{buf:!fir.type<_QFnested_alloc_dtTalloc_buffer{i:i32,data:!fir.box<!fir.heap<!fir.array<?xf32>>>}>}>>) {
