! This test checks lowering of OpenMP parallel Directive with
! `PRIVATE` clause present.

! REQUIRES: shell
! RUN: bbc -fopenmp -emit-fir %s -o - | \
! RUN:   FileCheck %s --check-prefix=FIRDialect

!FIRDialect: func @_QPprivate_clause(%[[ARG1:.*]]: !fir.ref<i32>{{.*}}, %[[ARG2:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[ARG3:.*]]: !fir.boxchar<1>{{.*}}, %[[ARG4:.*]]: !fir.boxchar<1>{{.*}}) {
!FIRDialect-DAG: %[[ALPHA:.*]] = fir.alloca i32 {{{.*}}, uniq_name = "{{.*}}Ealpha"}
!FIRDialect-DAG: %[[ALPHA_ARRAY:.*]] = fir.alloca !fir.array<10xi32> {{{.*}}, uniq_name = "{{.*}}Ealpha_array"}
!FIRDialect-DAG: %[[BETA:.*]] = fir.alloca !fir.char<1,5> {{{.*}}, uniq_name = "{{.*}}Ebeta"}
!FIRDialect-DAG: %[[BETA_ARRAY:.*]] = fir.alloca !fir.array<10x!fir.char<1,5>> {{{.*}}, uniq_name = "{{.*}}Ebeta_array"}

!FIRDialect-DAG:  omp.parallel {
!FIRDialect-DAG: %[[ALPHA_PRIVATE:.*]] = fir.alloca i32 {{{.*}}, pinned, uniq_name = "{{.*}}Ealpha"}
!FIRDialect-DAG: %[[ALPHA_ARRAY_PRIVATE:.*]] = fir.alloca !fir.array<10xi32> {{{.*}}, pinned, uniq_name = "{{.*}}Ealpha_array"}
!FIRDialect-DAG: %[[BETA_PRIVATE:.*]] = fir.alloca !fir.char<1,5> {{{.*}}, pinned, uniq_name = "{{.*}}Ebeta"}
!FIRDialect-DAG: %[[BETA_ARRAY_PRIVATE:.*]] = fir.alloca !fir.array<10x!fir.char<1,5>> {{{.*}}, pinned, uniq_name = "{{.*}}Ebeta_array"}
!FIRDialect-DAG: %[[ARG1_PRIVATE:.*]] = fir.alloca i32 {{{.*}}, pinned, uniq_name = "{{.*}}Earg1"}
!FIRDialect-DAG: %[[ARG2_ARRAY_PRIVATE:.*]] = fir.alloca !fir.array<10xi32> {{{.*}}, pinned, uniq_name = "{{.*}}Earg2"}
!FIRDialect-DAG: %[[ARG3_PRIVATE:.*]] = fir.alloca !fir.char<1,5> {{{.*}}, pinned, uniq_name = "{{.*}}Earg3"}
!FIRDialect-DAG: %[[ARG4_ARRAY_PRIVATE:.*]] = fir.alloca !fir.array<10x!fir.char<1,5>> {{{.*}}, pinned, uniq_name = "{{.*}}Earg4"}
!FIRDialect:    omp.terminator
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
!FIRDialect-DAG:   {{.*}} = fir.alloca !fir.complex<4> {bindc_name = "c", uniq_name = "{{.*}}Ec"}
!FIRDialect-DAG:   {{.*}} = fir.alloca i8 {bindc_name = "i1", uniq_name = "{{.*}}Ei1"}
!FIRDialect-DAG:   {{.*}} = fir.alloca i128 {bindc_name = "i16", uniq_name = "{{.*}}Ei16"}
!FIRDialect-DAG:   {{.*}} = fir.alloca i16 {bindc_name = "i2", uniq_name = "{{.*}}Ei2"}
!FIRDialect-DAG:   {{.*}} = fir.alloca i32 {bindc_name = "i4", uniq_name = "{{.*}}Ei4"}
!FIRDialect-DAG:   {{.*}} = fir.alloca i64 {bindc_name = "i8", uniq_name = "{{.*}}Ei8"}
!FIRDialect-DAG:   {{.*}} = fir.alloca !fir.logical<4> {bindc_name = "l", uniq_name = "{{.*}}El"}
!FIRDialect-DAG:   {{.*}} = fir.alloca f32 {bindc_name = "r", uniq_name = "{{.*}}Er"}

!FIRDialect:   omp.parallel {
!FIRDialect-DAG:     {{.*}} = fir.alloca i8 {bindc_name = "i1", pinned, uniq_name = "{{.*}}Ei1"}
!FIRDialect-DAG:     {{.*}} = fir.alloca i16 {bindc_name = "i2", pinned, uniq_name = "{{.*}}Ei2"}
!FIRDialect-DAG:     {{.*}} = fir.alloca i32 {bindc_name = "i4", pinned, uniq_name = "{{.*}}Ei4"}
!FIRDialect-DAG:     {{.*}} = fir.alloca i64 {bindc_name = "i8", pinned, uniq_name = "{{.*}}Ei8"}
!FIRDialect-DAG:     {{.*}} = fir.alloca i128 {bindc_name = "i16", pinned, uniq_name = "{{.*}}Ei16"}
!FIRDialect-DAG:     {{.*}} = fir.alloca !fir.complex<4> {bindc_name = "c", pinned, uniq_name = "{{.*}}Ec"}
!FIRDialect-DAG:     {{.*}} = fir.alloca !fir.logical<4> {bindc_name = "l", pinned, uniq_name = "{{.*}}El"}
!FIRDialect-DAG:     {{.*}} = fir.alloca f32 {bindc_name = "r", pinned, uniq_name = "{{.*}}Er"}

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
!FIRDialect:   {{.*}} = fir.alloca !fir.type<{{.*}}{t_i:i32,t_arr:!fir.array<5xi32>}> {bindc_name = "t", uniq_name = "{{.*}}Et"}

!FIRDialect:   omp.parallel {
!FIRDialect:     {{.*}} = fir.alloca !fir.type<{{.*}}{t_i:i32,t_arr:!fir.array<5xi32>}> {bindc_name = "t", pinned, uniq_name = "{{.*}}Et"}

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
!FIRDialect-DAG:  {{.*}} = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "x", uniq_name = "{{.*}}Ex"}
!FIRDialect-DAG:  {{.*}} = fir.alloca !fir.heap<i32> {uniq_name = "{{.*}}Ex.addr"}
!FIRDialect-DAG:  {{.*}} = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "x2", uniq_name = "{{.*}}Ex2"}
!FIRDialect-DAG:  {{.*}} = fir.alloca !fir.heap<!fir.array<?xi32>> {uniq_name = "{{.*}}Ex2.addr"}
!FIRDialect-DAG:  {{.*}} = fir.address_of(@{{.*}}Ex3) : !fir.ref<!fir.box<!fir.heap<i32>>>
!FIRDialect-DAG:  [[TMP9:%.*]] = fir.address_of(@{{.*}}Ex4) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

!FIRDialect:   omp.parallel {
!FIRDialect-DAG:    [[TMP37:%.*]] = fir.alloca i32 {bindc_name = "x", pinned, uniq_name = "{{.*}}Ex"}
!FIRDialect-DAG:    [[TMP40:%.*]] = fir.alloca !fir.array<?xi32>, {{.*}} {bindc_name = "x2", pinned, uniq_name = "{{.*}}Ex2"}
!FIRDialect-DAG:    [[TMP41:%.*]] = fir.alloca i32 {bindc_name = "x3", pinned, uniq_name = "{{.*}}Ex3"}
!FIRDialect-DAG:    [[TMP42:%.*]] = fir.load [[TMP9]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-DAG:    [[TMP43:%.*]]:3 = fir.box_dims [[TMP42]], {{.*}} : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-DAG:    [[TMP44:%.*]] = fir.alloca !fir.array<?xi32>, [[TMP43]]#1 {bindc_name = "x4", pinned, uniq_name = "{{.*}}Ex4"}
!FIRDialect-DAG:    [[TMP52:%.*]] = fir.embox [[TMP40]]({{.*}}) : (!fir.ref<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xi32>>
!FIRDialect-DAG:    {{.*}} = fir.convert [[TMP52]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
!FIRDialect-DAG:    [[TMP58:%.*]] = fir.shape_shift [[TMP43]]#0, [[TMP43]]#1 : (index, index) -> !fir.shapeshift<1>
!FIRDialect-DAG:    [[TMP59:%.*]] = fir.embox [[TMP44]]([[TMP58]]) : (!fir.ref<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xi32>>
!FIRDialect-DAG:    {{.*}} = fir.convert [[TMP59]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>

subroutine private_clause_allocatable()

        integer, allocatable :: x, x2(:)
        integer, allocatable, save :: x3, x4(:)

        print *, x, x2, x3, x4

!$OMP PARALLEL PRIVATE(x, x2, x3, x4)
        print *, x, x2, x3, x4
!$OMP END PARALLEL

end subroutine
