!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK: fir.global common @var_common_(dense<0> : vector<8xi8>) {{.*}} : !fir.array<8xi8>
!CHECK: fir.global common @var_common_link_(dense<0> : vector<8xi8>) {{{.*}} omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : !fir.array<8xi8>

!CHECK-LABEL: func.func @_QPmap_full_block
!CHECK: %[[CB_ADDR:.*]] = fir.address_of(@var_common_) : !fir.ref<!fir.array<8xi8>>
!CHECK: %[[MAP:.*]] = omp.map.info var_ptr(%[[CB_ADDR]] : !fir.ref<!fir.array<8xi8>>, !fir.array<8xi8>) map_clauses(tofrom) capture(ByRef) -> !fir.ref<!fir.array<8xi8>> {name = "var_common"}
!CHECK: omp.target map_entries(%[[MAP]] -> %[[MAP_ARG:.*]] : !fir.ref<!fir.array<8xi8>>) {
!CHECK:    %[[CONV:.*]] = fir.convert %[[MAP_ARG]] : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK:    %[[INDEX:.*]] = arith.constant 0 : index
!CHECK:    %[[COORD:.*]] = fir.coordinate_of %[[CONV]], %[[INDEX]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK:    %[[CONV2:.*]] = fir.convert %[[COORD]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK:    %[[CB_MEMBER_1:.*]]:2 = hlfir.declare %[[CONV2]] {uniq_name = "_QFmap_full_blockEvar1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[CONV3:.*]] = fir.convert %[[MAP_ARG]] : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK:    %[[INDEX2:.*]] = arith.constant 4 : index
!CHECK:    %[[COORD2:.*]] = fir.coordinate_of %[[CONV3]], %[[INDEX2]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK:    %[[CONV4:.*]] = fir.convert %[[COORD2]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK:    %[[CB_MEMBER_2:.*]]:2 = hlfir.declare %[[CONV4]] {uniq_name = "_QFmap_full_blockEvar2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
subroutine map_full_block
    implicit none
    common /var_common/ var1, var2
    integer :: var1, var2
!$omp target map(tofrom: /var_common/)
    var1 = var1 + 20
    var2 = var2 + 30
!$omp end target
end

!CHECK-LABEL: @_QPmap_mix_of_members
!CHECK: %[[COMMON_BLOCK:.*]] = fir.address_of(@var_common_) : !fir.ref<!fir.array<8xi8>>
!CHECK: %[[CB_CONV:.*]] = fir.convert %[[COMMON_BLOCK]] : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK: %[[INDEX:.*]] = arith.constant 0 : index
!CHECK: %[[COORD:.*]] = fir.coordinate_of %[[CB_CONV]], %[[INDEX]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK: %[[CONV:.*]] = fir.convert %[[COORD]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK: %[[CB_MEMBER_1:.*]]:2 = hlfir.declare %[[CONV]] {uniq_name = "_QFmap_mix_of_membersEvar1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[CB_CONV:.*]] = fir.convert %[[COMMON_BLOCK]] : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK: %[[INDEX:.*]] = arith.constant 4 : index
!CHECK: %[[COORD:.*]] = fir.coordinate_of %[[CB_CONV]], %[[INDEX]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK: %[[CONV:.*]] = fir.convert %[[COORD]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK: %[[CB_MEMBER_2:.*]]:2 = hlfir.declare %[[CONV]] {uniq_name = "_QFmap_mix_of_membersEvar2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[MAP_EXP:.*]] = omp.map.info var_ptr(%[[CB_MEMBER_2]]#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "var2"}
!CHECK: %[[MAP_IMP:.*]] = omp.map.info var_ptr(%[[CB_MEMBER_1]]#1 : !fir.ref<i32>, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !fir.ref<i32> {name = "var1"}
!CHECK: omp.target map_entries(%[[MAP_EXP]] -> %[[ARG_EXP:.*]], %[[MAP_IMP]] -> %[[ARG_IMP:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
!CHECK:  %[[EXP_MEMBER:.*]]:2 = hlfir.declare %[[ARG_EXP]] {uniq_name = "_QFmap_mix_of_membersEvar2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:  %[[IMP_MEMBER:.*]]:2 = hlfir.declare %[[ARG_IMP]] {uniq_name = "_QFmap_mix_of_membersEvar1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
subroutine map_mix_of_members
    implicit none
    common /var_common/ var1, var2
    integer :: var1, var2

!$omp target map(tofrom: var2)
  var2 = var1
!$omp end target
end

!CHECK-LABEL: @_QQmain
!CHECK: %[[DECL_TAR_CB:.*]] = fir.address_of(@var_common_link_) : !fir.ref<!fir.array<8xi8>>
!CHECK: %[[MAP_DECL_TAR_CB:.*]] = omp.map.info var_ptr(%[[DECL_TAR_CB]] : !fir.ref<!fir.array<8xi8>>, !fir.array<8xi8>) map_clauses(tofrom) capture(ByRef) -> !fir.ref<!fir.array<8xi8>> {name = "var_common_link"}
!CHECK: omp.target map_entries(%[[MAP_DECL_TAR_CB]] -> %[[MAP_DECL_TAR_ARG:.*]] : !fir.ref<!fir.array<8xi8>>) {
!CHECK:  %[[CONV:.*]] = fir.convert %[[MAP_DECL_TAR_ARG]] : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK:  %[[INDEX:.*]] = arith.constant 0 : index
!CHECK:  %[[COORD:.*]] = fir.coordinate_of %[[CONV]], %[[INDEX]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK:  %[[CONV:.*]] = fir.convert %[[COORD]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK:  %[[MEMBER_ONE:.*]]:2 = hlfir.declare %[[CONV]] {uniq_name = "_QFElink1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:  %[[CONV:.*]] = fir.convert %[[MAP_DECL_TAR_ARG]] : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
!CHECK:  %[[INDEX:.*]] = arith.constant 4 : index
!CHECK:  %[[COORD:.*]] = fir.coordinate_of %[[CONV]], %[[INDEX]] : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
!CHECK:  %[[CONV:.*]] = fir.convert %[[COORD]] : (!fir.ref<i8>) -> !fir.ref<i32>
!CHECK:  %[[MEMBER_TWO:.*]]:2 = hlfir.declare %[[CONV]] {uniq_name = "_QFElink2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
program main
    implicit none
    common /var_common_link/ link1, link2
    integer :: link1, link2
    !$omp declare target link(/var_common_link/)

!$omp target map(tofrom: /var_common_link/)
    link1 = link2 + 20
!$omp end target
end program
