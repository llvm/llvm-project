! This test checks lowering of OpenMP parallel Directive with
! `PRIVATE` clause present.

! REQUIRES: shell
! RUN: bbc --use-desc-for-alloc=false -fopenmp -emit-hlfir %s -o - \
! RUN: | FileCheck %s --check-prefix=FIRDialect

!FIRDialect: omp.private {type = private} @[[DERIVED_PRIVATIZER:_QFprivate_clause_derived_typeEt_private_ref_rec__QFprivate_clause_derived_typeTmy_type]] : !fir.ref<!fir.type<_QFprivate_clause_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}>> alloc {
!FIRDialect:   ^bb0(%{{.*}}: !fir.ref<!fir.type<_QFprivate_clause_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}>>):
!FIRDialect:     %[[PRIV_ALLOC:.*]] = fir.alloca !fir.type<_QFprivate_clause_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}> {bindc_name = "t", pinned, uniq_name = "_QFprivate_clause_derived_typeEt"}
!FIRDialect:     %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOC]] {uniq_name = "_QFprivate_clause_derived_typeEt"} : (!fir.ref<!fir.type<_QFprivate_clause_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}>>) -> (!fir.ref<!fir.type<_QFprivate_clause_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}>>, !fir.ref<!fir.type<_QFprivate_clause_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}>>)
!FIRDialect:     omp.yield(%[[PRIV_DECL]]#0 : !fir.ref<!fir.type<_QFprivate_clause_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}>>)
!FIRDialect: }

!FIRDialect: func @_QPprivate_clause(%[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "arg1"}, %[[ARG2:.*]]: !fir.ref<!fir.array<10xi32>> {fir.bindc_name = "arg2"}, %[[ARG3:.*]]: !fir.boxchar<1> {fir.bindc_name = "arg3"}, %[[ARG4:.*]]: !fir.boxchar<1> {fir.bindc_name = "arg4"}) {
!FIRDialect-DAG: %[[ALPHA:.*]] = fir.alloca i32 {bindc_name = "alpha", uniq_name = "{{.*}}alpha"}
!FIRDialect-DAG: %[[ALPHA_DECL:.*]]:2 = hlfir.declare %[[ALPHA]] {uniq_name = "{{.*}}alpha"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!FIRDialect-DAG: %[[ALPHA_ARRAY:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "alpha_array", uniq_name = "{{.*}}alpha_array"}
!FIRDialect-DAG: %[[ALPHA_ARRAY_DECL:.*]]:2 = hlfir.declare %[[ALPHA_ARRAY]]({{.*}}) {uniq_name = "{{.*}}alpha_array"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
!FIRDialect-DAG: %[[BETA:.*]] = fir.alloca !fir.char<1,5> {bindc_name = "beta", uniq_name = "{{.*}}beta"}
!FIRDialect-DAG: %[[BETA_DECL:.*]]:2 = hlfir.declare %[[BETA]] typeparams {{.*}} {uniq_name = "{{.*}}beta"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
!FIRDialect-DAG: %[[BETA_ARRAY:.*]] = fir.alloca !fir.array<10x!fir.char<1,5>> {bindc_name = "beta_array", uniq_name = "{{.*}}beta_array"}
!FIRDialect-DAG: %[[BETA_ARRAY_DECL:.*]]:2 = hlfir.declare %[[BETA_ARRAY]]({{.*}}) typeparams {{.*}} {uniq_name = "{{.*}}beta_array"} : (!fir.ref<!fir.array<10x!fir.char<1,5>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<10x!fir.char<1,5>>>, !fir.ref<!fir.array<10x!fir.char<1,5>>>)

!FIRDialect-DAG: omp.parallel private(@{{.*}} %{{.*}}#0 -> %[[ALPHA_PVT:.*]], @{{.*}} %{{.*}}#0 -> %[[ALPHA_ARRAY_PVT:.*]], @{{.*}} %{{.*}}#0 -> %[[BETA_PVT:.*]], @{{.*}} %{{.*}}#0 -> %[[BETA_ARRAY_PVT:.*]], @{{.*}} %{{.*}}#0 -> %[[ARG1_PVT:.*]], @{{.*}} %{{.*}}#0 -> %[[ARG2_PVT:.*]], @{{.*}} %{{.*}}#0 -> %[[ARG3_PVT:.*]], @{{.*}} %{{.*}}#0 -> %[[ARG4_PVT:.*]] : {{.*}}) {
!FIRDialect-DAG:  %[[ALPHA_PVT_DECL:.*]]:2 = hlfir.declare %[[ALPHA_PVT]] {uniq_name = "{{.*}}alpha"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!FIRDialect-DAG:  %[[ALPHA_ARRAY_PVT_DECL:.*]]:2 = hlfir.declare %[[ALPHA_ARRAY_PVT]]({{.*}}) {uniq_name = "{{.*}}alpha_array"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
!FIRDialect-DAG:  %[[BETA_PVT_DECL:.*]]:2 = hlfir.declare %[[BETA_PVT]] typeparams {{.*}} {uniq_name = "{{.*}}beta"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
!FIRDialect-DAG:  %[[BETA_ARRAY_PVT_DECL:.*]]:2 = hlfir.declare %[[BETA_ARRAY_PVT]]({{.*}}) typeparams {{.*}} {uniq_name = "{{.*}}beta_array"} : (!fir.ref<!fir.array<10x!fir.char<1,5>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<10x!fir.char<1,5>>>, !fir.ref<!fir.array<10x!fir.char<1,5>>>)
!FIRDialect-DAG:  %[[ARG1_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG1_PVT]] {uniq_name = "{{.*}}arg1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!FIRDialect-DAG:  %[[ARG2_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG2_PVT]]({{.*}}) {uniq_name = "{{.*}}arg2"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
!FIRDialect-DAG:  %[[ARG3_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG3_PVT]] typeparams {{.*}} {uniq_name = "{{.*}}arg3"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
!FIRDialect-DAG:  %[[ARG4_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG4_PVT]]({{.*}}) typeparams {{.*}} {uniq_name = "{{.*}}arg4"} : (!fir.ref<!fir.array<10x!fir.char<1,5>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<10x!fir.char<1,5>>>, !fir.ref<!fir.array<10x!fir.char<1,5>>>)
!FIRDialect:     omp.terminator
!FIRDialect:  }

subroutine private_clause(arg1, arg2, arg3, arg4)

        integer :: arg1, arg2(10)
        integer :: alpha, alpha_array(10)
        character(5) :: arg3, arg4(10)
        character(5) :: beta, beta_array(10)

!$OMP PARALLEL PRIVATE(alpha, alpha_array, beta, beta_array, arg1, arg2, arg3, arg4)
        alpha = 1
        alpha_array = 4
        beta = "hi"
        beta_array = "hi"
        arg1 = 2
        arg2 = 3
        arg3 = "world"
        arg4 = "world"
!$OMP END PARALLEL

end subroutine

!FIRDialect: func @_QPprivate_clause_scalar() {
!FIRDialect-DAG:  %[[C:.*]] = fir.alloca complex<f32> {bindc_name = "c", uniq_name = "_QFprivate_clause_scalarEc"}
!FIRDialect-DAG:  %[[C_DECL:.*]]:2 = hlfir.declare %[[C]] {uniq_name = "_QFprivate_clause_scalarEc"} : (!fir.ref<complex<f32>>) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
!FIRDialect-DAG:  %[[I1:.*]] = fir.alloca i8 {bindc_name = "i1", uniq_name = "_QFprivate_clause_scalarEi1"}
!FIRDialect-DAG:  %[[I1_DECL:.*]]:2 = hlfir.declare %[[I1]] {uniq_name = "_QFprivate_clause_scalarEi1"} : (!fir.ref<i8>) -> (!fir.ref<i8>, !fir.ref<i8>)
!FIRDialect-DAG:  %[[I16:.*]] = fir.alloca i128 {bindc_name = "i16", uniq_name = "_QFprivate_clause_scalarEi16"}
!FIRDialect-DAG:  %[[I16_DECL:.*]]:2 = hlfir.declare %[[I16]] {uniq_name = "_QFprivate_clause_scalarEi16"} : (!fir.ref<i128>) -> (!fir.ref<i128>, !fir.ref<i128>)
!FIRDialect-DAG:  %[[I2:.*]] = fir.alloca i16 {bindc_name = "i2", uniq_name = "_QFprivate_clause_scalarEi2"}
!FIRDialect-DAG:  %[[I2_DECL:.*]]:2 = hlfir.declare %[[I2]] {uniq_name = "_QFprivate_clause_scalarEi2"} : (!fir.ref<i16>) -> (!fir.ref<i16>, !fir.ref<i16>)
!FIRDialect-DAG:  %[[I4:.*]] = fir.alloca i32 {bindc_name = "i4", uniq_name = "_QFprivate_clause_scalarEi4"}
!FIRDialect-DAG:  %[[I4_DECL:.*]]:2 = hlfir.declare %[[I4]] {uniq_name = "_QFprivate_clause_scalarEi4"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!FIRDialect-DAG:  %[[I8:.*]] = fir.alloca i64 {bindc_name = "i8", uniq_name = "_QFprivate_clause_scalarEi8"}
!FIRDialect-DAG:  %[[I8_DECL:.*]]:2 = hlfir.declare %[[I8]] {uniq_name = "_QFprivate_clause_scalarEi8"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
!FIRDialect-DAG:  %[[L:.*]] = fir.alloca !fir.logical<4> {bindc_name = "l", uniq_name = "_QFprivate_clause_scalarEl"}
!FIRDialect-DAG:  %[[L_DECL:.*]]:2 = hlfir.declare %[[L]] {uniq_name = "_QFprivate_clause_scalarEl"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!FIRDialect-DAG:  %[[R:.*]] = fir.alloca f32 {bindc_name = "r", uniq_name = "_QFprivate_clause_scalarEr"}
!FIRDialect-DAG:  %[[R_DECL:.*]]:2 = hlfir.declare %[[R]] {uniq_name = "_QFprivate_clause_scalarEr"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)

!FIRDialect-DAG: omp.parallel private(@{{.*}} %{{.*}}#0 -> %[[I1_PVT:.*]], @{{.*}} %{{.*}}#0 -> %[[I2_PVT:.*]], @{{.*}} %{{.*}}#0 -> %[[I4_PVT:.*]], @{{.*}} %{{.*}}#0 -> %[[I8_PVT:.*]], @{{.*}} %{{.*}}#0 -> %[[I16_PVT:.*]], @{{.*}} %{{.*}}#0 -> %[[C_PVT:.*]], @{{.*}} %{{.*}}#0 -> %[[L_PVT:.*]], @{{.*}} %{{.*}}#0 -> %[[R_PVT:.*]] : {{.*}}) {
!FIRDialect-DAG:    %[[I1_PVT_DECL:.*]]:2 = hlfir.declare %[[I1_PVT]] {uniq_name = "_QFprivate_clause_scalarEi1"} : (!fir.ref<i8>) -> (!fir.ref<i8>, !fir.ref<i8>)
!FIRDialect-DAG:    %[[I2_PVT_DECL:.*]]:2 = hlfir.declare %[[I2_PVT]] {uniq_name = "_QFprivate_clause_scalarEi2"} : (!fir.ref<i16>) -> (!fir.ref<i16>, !fir.ref<i16>)
!FIRDialect-DAG:    %[[I4_PVT_DECL:.*]]:2 = hlfir.declare %[[I4_PVT]] {uniq_name = "_QFprivate_clause_scalarEi4"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!FIRDialect-DAG:    %[[I8_PVT_DECL:.*]]:2 = hlfir.declare %[[I8_PVT]] {uniq_name = "_QFprivate_clause_scalarEi8"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
!FIRDialect-DAG:    %[[I16_PVT_DECL:.*]]:2 = hlfir.declare %[[I16_PVT]] {uniq_name = "_QFprivate_clause_scalarEi16"} : (!fir.ref<i128>) -> (!fir.ref<i128>, !fir.ref<i128>)
!FIRDialect-DAG:    %[[C_PVT_DECL:.*]]:2 = hlfir.declare %[[C_PVT]] {uniq_name = "_QFprivate_clause_scalarEc"} : (!fir.ref<complex<f32>>) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
!FIRDialect-DAG:    %[[L_PVT_DECL:.*]]:2 = hlfir.declare %[[L_PVT]] {uniq_name = "_QFprivate_clause_scalarEl"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
!FIRDialect-DAG:    %[[R_PVT_DECL:.*]]:2 = hlfir.declare %[[R_PVT]] {uniq_name = "_QFprivate_clause_scalarEr"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
!FIRDialect:        omp.terminator
!FIRDialect:      }
subroutine private_clause_scalar()

        integer(kind=1) :: i1
        integer(kind=2) :: i2
        integer(kind=4) :: i4
        integer(kind=8) :: i8
        integer(kind=16) :: i16
        complex :: c
        logical :: l
        real :: r

!$OMP PARALLEL PRIVATE(i1, i2, i4, i8, i16, c, l, r)
        print *, i1, i2, i4, i8, i16, c, l, r
!$OMP END PARALLEL

end subroutine

!FIRDialect: func @_QPprivate_clause_derived_type() {
!FIRDialect: %[[T:.*]] = fir.alloca !fir.type<_QFprivate_clause_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}> {bindc_name = "t", uniq_name = "{{.*}}Et"}
!FIRDialect: %[[T_DECL:.*]]:2 = hlfir.declare %[[T]] {uniq_name = "{{.*}}Et"} : (!fir.ref<!fir.type<_QFprivate_clause_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}>>) -> (!fir.ref<!fir.type<_QFprivate_clause_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}>>, !fir.ref<!fir.type<_QFprivate_clause_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}>>)

!FIRDialect:   omp.parallel private(@[[DERIVED_PRIVATIZER]] %{{.*}}#0 -> %[[T:.*]] : {{.*}}) {
!FIRDialect: %[[T_DECL:.*]]:2 = hlfir.declare %[[T]] {uniq_name = "{{.*}}Et"} : (!fir.ref<!fir.type<_QFprivate_clause_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}>>) -> (!fir.ref<!fir.type<_QFprivate_clause_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}>>, !fir.ref<!fir.type<_QFprivate_clause_derived_typeTmy_type{t_i:i32,t_arr:!fir.array<5xi32>}>>)

subroutine private_clause_derived_type()

        type my_type
          integer :: t_i
          integer :: t_arr(5)
        end type my_type
        type(my_type) :: t

!$OMP PARALLEL PRIVATE(t)
        print *, t%t_i
!$OMP END PARALLEL

end subroutine

!FIRDialect: func @_QPprivate_clause_allocatable() {
!FIRDialect-DAG:  %[[X:.*]] = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "x", uniq_name = "{{.*}}Ex"}
!FIRDialect-DAG:  %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "{{.*}}Ex"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
!FIRDialect-DAG:  %[[X2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "x2", uniq_name = "{{.*}}Ex2"}
!FIRDialect-DAG:  %[[X2_DECL:.*]]:2 = hlfir.declare %[[X2]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "{{.*}}Ex2"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
!FIRDialect-DAG:  %[[X3:.*]] = fir.address_of(@{{.*}}Ex3) : !fir.ref<!fir.box<!fir.heap<i32>>>
!FIRDialect-DAG:  %[[X3_DECL:.*]]:2 = hlfir.declare %[[X3]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "{{.*}}Ex3"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
!FIRDialect-DAG:  %[[X4:.*]] = fir.address_of(@{{.*}}Ex4) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-DAG:  %[[X4_DECL:.*]]:2 = hlfir.declare %[[X4]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "{{.*}}Ex4"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)

!FIRDialect:   omp.parallel private(@{{.*}} %{{.*}}#0 -> %[[X_PVT:.*]], @{{.*}} %{{.*}}#0 -> %[[X2_PVT:.*]], @{{.*}} %{{.*}}#0 -> %[[X3_PVT:.*]], @{{.*}} %{{.*}}#0 -> %[[X4_PVT:.*]] : {{.*}}) {
!FIRDialect-DAG:    %[[X_PVT_DECL:.*]]:2 = hlfir.declare %[[X_PVT]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "{{.*}}Ex"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
!FIRDialect-DAG:    %[[X2_PVT_DECL:.*]]:2 = hlfir.declare %[[X2_PVT]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "{{.*}}Ex2"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
!FIRDialect-DAG:    %[[X3_PVT_DECL:.*]]:2 = hlfir.declare %[[X3_PVT]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "{{.*}}Ex3"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
!FIRDialect-DAG:    %[[X4_PVT_DECL:.*]]:2 = hlfir.declare %[[X4_PVT]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "{{.*}}Ex4"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
!FIRDialect:        omp.terminator
!FIRDialect:      }

subroutine private_clause_allocatable()

        integer, allocatable :: x, x2(:)
        integer, allocatable, save :: x3, x4(:)

        print *, x, x2, x3, x4

!$OMP PARALLEL PRIVATE(x, x2, x3, x4)
        print *, x, x2, x3, x4
!$OMP END PARALLEL

end subroutine


!FIRDialect: func @_QPprivate_clause_real_call_allocatable() {
!FIRDialect-DAG: %[[X5:.*]] = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "x5", uniq_name = "{{.*}}Ex5"}
!FIRDialect-DAG: {{.*}} = fir.zero_bits !fir.heap<f32>
!FIRDialect-DAG: {{.*}} = fir.embox %1 : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
!FIRDialect-DAG: fir.store %2 to %[[X5]] : !fir.ref<!fir.box<!fir.heap<f32>>>
!FIRDialect-DAG: %[[X5_DECL:.*]]:2 = hlfir.declare %[[X5]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFprivate_clause_real_call_allocatableEx5"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
!FIRDialect-DAG: omp.parallel private(@{{.*}} %{{.*}}#0 -> %[[X5_PVT:.*]] : {{.*}}) {

!FIRDialect-DAG: %[[X5_PVT_DECL:.*]]:2 = hlfir.declare %[[X5_PVT]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFprivate_clause_real_call_allocatableEx5"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)
!FIRDialect-DAG: fir.call @_QFprivate_clause_real_call_allocatablePhelper_private_clause_real_call_allocatable(%[[X5_PVT_DECL]]#0) fastmath<contract> : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> ()
!FIRDialect-DAG:   omp.terminator
!FIRDialect-DAG:   }
!FIRDialect-DAG:   return
!FIRDialect-DAG: }


subroutine private_clause_real_call_allocatable
        real, allocatable :: x5
        !$omp parallel private(x5)
            call helper_private_clause_real_call_allocatable(x5)
        !$omp end parallel
    contains
        subroutine helper_private_clause_real_call_allocatable(x6)
            real, allocatable :: x6
            print *, allocated(x6)
        end subroutine
end subroutine

!FIRDialect:  func.func @_QPincrement_list_items(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.type<_QFincrement_list_itemsTnode{payload:i32,next:!fir.box<!fir.ptr<!fir.type<_QFincrement_list_itemsTnode>>>}>>>> {fir.bindc_name = "head"}) {
!FIRDialect:    %[[P:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.type<_QFincrement_list_itemsTnode{payload:i32,next:!fir.box<!fir.ptr<!fir.type<_QFincrement_list_itemsTnode>>>}>>> {bindc_name = "p", uniq_name = "_QFincrement_list_itemsEp"}
!FIRDialect:    %[[P_DECL:.*]]:2 = hlfir.declare %[[P]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFincrement_list_itemsEp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.type<_QFincrement_list_itemsTnode{payload:i32,next:!fir.box<!fir.ptr<!fir.type<_QFincrement_list_itemsTnode>>>}>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.type<_QFincrement_list_itemsTnode{payload:i32,next:!fir.box<!fir.ptr<!fir.type<_QFincrement_list_itemsTnode>>>}>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.type<_QFincrement_list_itemsTnode{payload:i32,next:!fir.box<!fir.ptr<!fir.type<_QFincrement_list_itemsTnode>>>}>>>>)
!FIRDialect: omp.parallel private(@{{.*}} %{{.*}}#0 -> %[[P_PVT:.*]] : {{.*}}) {
!FIRDialect:      %[[P_PVT_DECL:.*]]:2 = hlfir.declare %[[P_PVT]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFincrement_list_itemsEp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.type<_QFincrement_list_itemsTnode{payload:i32,next:!fir.box<!fir.ptr<!fir.type<_QFincrement_list_itemsTnode>>>}>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.type<_QFincrement_list_itemsTnode{payload:i32,next:!fir.box<!fir.ptr<!fir.type<_QFincrement_list_itemsTnode>>>}>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.type<_QFincrement_list_itemsTnode{payload:i32,next:!fir.box<!fir.ptr<!fir.type<_QFincrement_list_itemsTnode>>>}>>>>)
!FIRDialect:      omp.single   {

!FIRDialect:         omp.terminator
!FIRDialect:       omp.terminator
!FIRDialect:    return

subroutine increment_list_items (head)
  type node
     integer :: payload
     type (node), pointer :: next
  end type node

  type (node), pointer :: head
  type (node), pointer :: p
!$omp parallel private(p)
!$omp single
  p => head
  do
     p => p%next
     if ( associated (p) .eqv. .false. ) exit
  end do
!$omp end single
!$omp end parallel
end subroutine increment_list_items

!FIRDialect:  func.func @_QPparallel_pointer() {
!FIRDialect-DAG: %[[Y1:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "y1", uniq_name = "_QFparallel_pointerEy1"}
!FIRDialect:     fir.store %2 to %[[Y1]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!FIRDialect-DAG: %[[Y1_DECL:.*]]:2 = hlfir.declare %[[Y1]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFparallel_pointerEy1"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!FIRDialect-DAG: %[[Y2:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "y2", uniq_name = "_QFparallel_pointerEy2"}

!FIRDialect:     fir.store %7 to %[[Y2]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!FIRDialect-DAG: %[[Y2_DECL:.*]]:2 = hlfir.declare %[[Y2]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFparallel_pointerEy2"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
!FIRDialect-DAG: %[[Z1:.*]] = fir.alloca i32 {bindc_name = "z1", fir.target, uniq_name = "_QFparallel_pointerEz1"}
!FIRDialect-DAG: %[[Z1_DECL:.*]]:2 = hlfir.declare %[[Z1]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFparallel_pointerEz1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!FIRDialect-DAG:  %[[Z2:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "z2", fir.target, uniq_name = "_QFparallel_pointerEz2"}
!FIRDialect-DAG:  %[[Z2_DECL:.*]]:2 = hlfir.declare %[[Z2]](%12) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFparallel_pointerEz2"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
!FIRDialect:      omp.parallel private(@{{.*}} %{{.*}}#0 -> %[[Y1_PVT:.*]], @{{.*}} %{{.*}}#0 -> %[[Y2_PVT:.*]] : {{.*}}) {
!FIRDialect-DAG:    %[[Y1_PVT_DECL:.*]]:2 = hlfir.declare %[[Y1_PVT]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFparallel_pointerEy1"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
!FIRDialect-DAG:    %[[Y2_PVT_DECL:.*]]:2 = hlfir.declare %[[Y2_PVT]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFparallel_pointerEy2"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
!FIRDialect-DAG:    %[[PP18:.*]] = fir.embox %[[Z1_DECL]]#1 : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
!FIRDialect:       fir.store %[[PP18]] to %[[Y1_PVT_DECL]]#1 : !fir.ref<!fir.box<!fir.ptr<i32>>>
!FIRDialect-DAG:    %[[PP19:.*]] = fir.shape %c10 : (index) -> !fir.shape<1>
!FIRDialect-DAG:    %[[PP20:.*]] = fir.embox %[[Z2_DECL]]#1(%[[PP19]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
!FIRDialect:        fir.store %[[PP20]] to %[[Y2_PVT_DECL]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!FIRDialect:       omp.terminator
!FIRDialect:     }
!FIRDialect:   return
!FIRDialect: }

subroutine parallel_pointer()
    integer, pointer :: y1, y2(:)
    integer, target :: z1, z2(10)

!$omp parallel private(y1, y2)
  y1=>z1
  y2=>z2
!$omp end parallel
end subroutine parallel_pointer


!FIRDialect-LABEL: func @_QPsimple_loop_1()
subroutine simple_loop_1
  integer :: i
  real, allocatable :: r;
   !FIRDialect:    omp.parallel private(@{{.*}} %{{.*}}#0 -> %[[R:.*]] : {{.*}}) {
  !$OMP PARALLEL PRIVATE(r)
  ! FIRDialect:      %[[R_DECL:.*]]:2 = hlfir.declare %[[R]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsimple_loop_1Er"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)

  ! FIRDialect:      %[[ALLOCA_IV:.*]] = fir.alloca i32 {{{.*}}, pinned, {{.*}}}
  ! FIRDialect:      %[[ALLOCA_IV_DECL:.*]]:2 = hlfir.declare %[[ALLOCA_IV]] {uniq_name = "_QFsimple_loop_1Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  ! FIRDialect:      %[[WS_LB:.*]] = arith.constant 1 : i32
  ! FIRDialect:      %[[WS_UB:.*]] = arith.constant 9 : i32
  ! FIRDialect:      %[[WS_STEP:.*]] = arith.constant 1 : i32

  ! FIRDialect:      omp.wsloop {
  ! FIRDialect-NEXT: omp.loop_nest (%[[I:.*]]) : i32 = (%[[WS_LB]]) to (%[[WS_UB]]) inclusive step (%[[WS_STEP]]) {
  !$OMP DO
  do i=1, 9
  ! FIRDialect:      fir.store %[[I]] to %[[ALLOCA_IV_DECL]]#1 : !fir.ref<i32>
  ! FIRDialect:      %[[LOAD_IV:.*]] = fir.load %[[ALLOCA_IV_DECL]]#0 : !fir.ref<i32>
  ! FIRDialect:      fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) {{.*}} : (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  ! FIRDialect:      omp.yield
  !$OMP END DO
  ! FIRDialect:  omp.terminator
  !$OMP END PARALLEL
end subroutine

!FIRDialect-LABEL: func @_QPsimple_loop_2()
subroutine simple_loop_2
  integer :: i
  real, allocatable :: r;
  ! FIRDialect:  omp.parallel
  !$OMP PARALLEL
  ! FIRDialect:      [[R:%.*]] = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "r", pinned, uniq_name = "{{.*}}Er"}
  ! FIRDialect:      fir.store {{%.*}} to [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:      fir.store {{%.*}} to [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:      %[[R_DECL:.*]]:2 = hlfir.declare [[R]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "{{.*}}Er"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)

  ! FIRDialect:      %[[ALLOCA_IV:.*]] = fir.alloca i32 {{{.*}}, pinned, {{.*}}}
  ! FIRDialect:      %[[ALLOCA_IV_DECL:.*]]:2 = hlfir.declare %[[ALLOCA_IV]] {uniq_name = "{{.*}}Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  ! FIRDialect:      %[[WS_LB:.*]] = arith.constant 1 : i32
  ! FIRDialect:      %[[WS_UB:.*]] = arith.constant 9 : i32
  ! FIRDialect:      %[[WS_STEP:.*]] = arith.constant 1 : i32

  ! FIRDialect:      omp.wsloop {
  ! FIRDialect-NEXT: omp.loop_nest (%[[I:.*]]) : i32 = (%[[WS_LB]]) to (%[[WS_UB]]) inclusive step (%[[WS_STEP]]) {
  !$OMP DO PRIVATE(r)
  do i=1, 9
  ! FIRDialect:     fir.store %[[I]] to %[[ALLOCA_IV_DECL]]#1 : !fir.ref<i32>
  ! FIRDialect:     %[[LOAD_IV:.*]] = fir.load %[[ALLOCA_IV_DECL]]#0 : !fir.ref<i32>
  ! FIRDialect:     fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  ! FIRDialect:     omp.yield
  ! FIRDialect:     {{%.*}} = fir.load %[[R_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:     fir.if {{%.*}} {
  ! FIRDialect:     [[LD:%.*]] = fir.load %[[R_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:     [[AD:%.*]] = fir.box_addr [[LD]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  ! FIRDialect:     fir.freemem [[AD]] : !fir.heap<f32>
  ! FIRDialect:     fir.store {{%.*}} to %[[R_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  !$OMP END DO
  ! FIRDialect:  omp.terminator
  !$OMP END PARALLEL
end subroutine

!FIRDialect-LABEL: func @_QPsimple_loop_3()
subroutine simple_loop_3
  integer :: i
  real, allocatable :: r;
  ! FIRDialect:  omp.parallel

  ! FIRDialect:      [[R:%.*]] = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "r", pinned, uniq_name = "{{.*}}Er"}
  ! FIRDialect:      fir.store {{%.*}} to [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:      fir.store {{%.*}} to [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:      [[R_DECL:%.*]]:2 = hlfir.declare [[R]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "{{.*}}Er"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)

  ! FIRDialect:      %[[ALLOCA_IV:.*]] = fir.alloca i32 {{{.*}}, pinned, {{.*}}}
  ! FIRDialect:      %[[ALLOCA_IV_DECL:.*]]:2 = hlfir.declare %[[ALLOCA_IV]] {uniq_name = "{{.*}}Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

  ! FIRDialect:      %[[WS_LB:.*]] = arith.constant 1 : i32
  ! FIRDialect:      %[[WS_UB:.*]] = arith.constant 9 : i32
  ! FIRDialect:      %[[WS_STEP:.*]] = arith.constant 1 : i32

  ! FIRDialect:      omp.wsloop {
  ! FIRDialect-NEXT: omp.loop_nest (%[[I:.*]]) : i32 = (%[[WS_LB]]) to (%[[WS_UB]]) inclusive step (%[[WS_STEP]]) {
  !$OMP PARALLEL DO PRIVATE(r)
  do i=1, 9
  ! FIRDialect:      fir.store %[[I]] to %[[ALLOCA_IV_DECL:.*]]#1 : !fir.ref<i32>
  ! FIRDialect:      %[[LOAD_IV:.*]] = fir.load %[[ALLOCA_IV_DECL]]#0 : !fir.ref<i32>
  ! FIRDialect:      fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  ! FIRDialect:      omp.yield
  ! FIRDialect:      {{%.*}} = fir.load [[R_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:      fir.if {{%.*}} {
  ! FIRDialect:      [[LD:%.*]] = fir.load [[R_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:      [[AD:%.*]] = fir.box_addr [[LD]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  ! FIRDialect:      fir.freemem [[AD]] : !fir.heap<f32>
  ! FIRDialect:      fir.store {{%.*}} to [[R_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  !$OMP END PARALLEL DO
  ! FIRDialect:  omp.terminator
end subroutine

!CHECK-LABEL: func @_QPsimd_loop_1()
subroutine simd_loop_1
  integer :: i
  real, allocatable :: r;
  ! FIRDialect:     [[R:%.*]] = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "r", pinned, uniq_name = "{{.*}}Er"}
  ! FIRDialect:     fir.store {{%.*}} to [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:     fir.store {{%.*}} to [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:     [[R_DECL:%.*]]:2 = hlfir.declare [[R]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "{{.*}}r"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> (!fir.ref<!fir.box<!fir.heap<f32>>>, !fir.ref<!fir.box<!fir.heap<f32>>>)

  ! FIRDialect:     %[[LB:.*]] = arith.constant 1 : i32
  ! FIRDialect:     %[[UB:.*]] = arith.constant 9 : i32
  ! FIRDialect:     %[[STEP:.*]] = arith.constant 1 : i32

  ! FIRDialect: omp.simd {
  ! FIRDialect-NEXT: omp.loop_nest (%[[I:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
  !$OMP SIMD PRIVATE(r)
  do i=1, 9
  ! FIRDialect:     fir.store %[[I]] to %[[LOCAL:.*]]#1 : !fir.ref<i32>
  ! FIRDialect:     %[[LOAD_IV:.*]] = fir.load %[[LOCAL]]#0 : !fir.ref<i32>
  ! FIRDialect:     fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  !$OMP END SIMD
  ! FIRDialect:     omp.yield
  ! FIRDialect:     {{%.*}} = fir.load [[R_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:     fir.if {{%.*}} {
  ! FIRDialect:     [[LD:%.*]] = fir.load [[R_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:     [[AD:%.*]] = fir.box_addr [[LD]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  ! FIRDialect:     fir.freemem [[AD]] : !fir.heap<f32>
  ! FIRDialect:     fir.store {{%.*}} to [[R_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<f32>>>
end subroutine
