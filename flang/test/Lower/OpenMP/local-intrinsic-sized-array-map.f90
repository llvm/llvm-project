!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s --check-prefixes="HLFIRDIALECT"

!HLFIRDIALECT: func.func @_QPlocal_variable_intrinsic_size(%[[ARG0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "a"}) {
!HLFIRDIALECT:   %[[SZ_DATA:.*]] = fir.alloca index
!HLFIRDIALECT:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope {{.*}} {uniq_name = "_QFlocal_variable_intrinsic_sizeEa"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
!HLFIRDIALECT:   %[[DIMENSIONS:.*]]:3 = fir.box_dims %[[DECLARE]]#0, %{{.*}} : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
!HLFIRDIALECT:   fir.store %[[DIMENSIONS]]#1 to %[[SZ_DATA]] : !fir.ref<index>
!HLFIRDIALECT:   %[[SIZE_SEL:.*]] = arith.select {{.*}}, {{.*}}, {{.*}} : index
!HLFIRDIALECT:   %[[B_ALLOCA:.*]] = fir.alloca !fir.array<?xf32>, %[[SIZE_SEL]] {bindc_name = "b", uniq_name = "_QFlocal_variable_intrinsic_sizeEb"}
!HLFIRDIALECT:   %[[B_SHAPE:.*]] = fir.shape %[[SIZE_SEL]] : (index) -> !fir.shape<1>
!HLFIRDIALECT:   %[[B_DECLARE:.*]]:2 = hlfir.declare %[[B_ALLOCA]](%[[B_SHAPE]]) {uniq_name = "_QFlocal_variable_intrinsic_sizeEb"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
!HLFIRDIALECT:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound({{.*}} : index) upper_bound({{.*}} : index) extent({{.*}} : index) stride({{.*}} : index) start_idx({{.*}} : index) {stride_in_bytes = true}
!HLFIRDIALECT:   %[[MAP_DATA_B:.*]] = omp.map.info var_ptr(%[[B_DECLARE]]#1 : !fir.ref<!fir.array<?xf32>>, f32) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<?xf32>> {name = "b"}
!HLFIRDIALECT:   %[[MAP_DATA_SZ:.*]] = omp.map.info var_ptr(%[[SZ_DATA]] : !fir.ref<index>, index) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !fir.ref<index> {name = ""}
!HLFIRDIALECT:   omp.target map_entries(%[[MAP_DATA_B]] -> %[[ARG1:.*]], %[[MAP_DATA_SZ]] -> %[[ARG2:.*]] : !fir.ref<!fir.array<?xf32>>, !fir.ref<index>) {
!HLFIRDIALECT:      %[[SZ_LD:.*]] = fir.load %[[ARG2]] : !fir.ref<index>
!HLFIRDIALECT:      %[[SZ_CONV:.*]] = fir.convert %[[SZ_LD]] : (index) -> i64
!HLFIRDIALECT:      %[[SZ_CONV2:.*]] = fir.convert %[[SZ_CONV]] : (i64) -> index
!HLFIRDIALECT:      %[[SEL_SZ:.*]] = arith.cmpi sgt, %[[SZ_CONV2]], %{{.*}} : index
!HLFIRDIALECT:      %[[SEL_SZ2:.*]] = arith.select %[[SEL_SZ]], %[[SZ_CONV2]], %{{.*}} : index
!HLFIRDIALECT:      %[[SHAPE:.*]] = fir.shape %[[SEL_SZ2]] : (index) -> !fir.shape<1>
!HLFIRDIALECT:      %{{.*}} = hlfir.declare %[[ARG1]](%[[SHAPE]]) {uniq_name = "_QFlocal_variable_intrinsic_sizeEb"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)

subroutine local_variable_intrinsic_size(a)
    implicit none
    real, dimension(:) :: a
    real, dimension(size(a, 1)) :: b

!$omp target map(tofrom: b)
        b(5) = 5
!$omp end target
end subroutine
