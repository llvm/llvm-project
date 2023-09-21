! This test checks lowering of OpenMP parallel Directive with
! `PRIVATE` clause present.

! REQUIRES: shell
! RUN: bbc --use-desc-for-alloc=false -fopenmp -emit-fir %s -o - | \
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
!FIRDialect-DAG:  [[TMP8:%.*]] = fir.address_of(@{{.*}}Ex4) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

!FIRDialect:   omp.parallel {
!FIRDialect-DAG:    [[TMP35:%.*]] = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "x", pinned, uniq_name = "{{.*}}Ex"}
!FIRDialect-DAG:    [[TMP39:%.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "x2", pinned, uniq_name = "{{.*}}Ex2"}
!FIRDialect-DAG:    [[TMP45:%.*]] = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "x3", pinned, uniq_name = "{{.*}}Ex3"}

!FIRDialect-DAG:    [[TMP51:%.*]] = fir.load [[TMP8]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-DAG:    [[TMP97:%.*]] = fir.load [[TMP8]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!FIRDialect-DAG:    [[TMP98:%.*]]:3 = fir.box_dims [[TMP97]], {{.*}} : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!FIRDialect-DAG:    [[TMP50:%.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "x4", pinned, uniq_name = "{{.*}}Ex4"}

! FIRDialect-DAG:    [[TMP101:%.*]] = fir.allocmem !fir.array<?xi32>, {{.*}} {fir.must_be_heap = true, uniq_name = "{{.*}}Ex4.alloc"}
! FIRDialect-DAG:    [[TMP102:%.*]] = fir.shape_shift {{.*}}#0, {{.*}} : (index, index) -> !fir.shapeshift<1>
! FIRDialect-DAG:    [[TMP103:%.*]] = fir.embox [[TMP101]]([[TMP102]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! FIRDialect-DAG:  fir.store [[TMP103]] to [[TMP50]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>


subroutine private_clause_allocatable()

        integer, allocatable :: x, x2(:)
        integer, allocatable, save :: x3, x4(:)

        print *, x, x2, x3, x4

!$OMP PARALLEL PRIVATE(x, x2, x3, x4)
        print *, x, x2, x3, x4
!$OMP END PARALLEL

end subroutine


!FIRDialect: func @_QPprivate_clause_real_call_allocatable() {
!FIRDialect-DAG: {{.*}} = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "x5", uniq_name = "{{.*}}Ex5"}
!FIRDialect-DAG: {{.*}} = fir.zero_bits !fir.heap<f32>
!FIRDialect-DAG: {{.*}} = fir.embox %1 : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
!FIRDialect-DAG: fir.store %2 to %0 : !fir.ref<!fir.box<!fir.heap<f32>>>
!FIRDialect-DAG: omp.parallel   {
!FIRDialect-DAG:  [[TMP203:%.*]] = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "x5", pinned, uniq_name = "{{.*}}Ex5"}

!FIRDialect-DAG: fir.if %{{.*}} {

!FIRDialect-DAG:   fir.store %{{.*}} to [[TMP203]] : !fir.ref<!fir.box<!fir.heap<f32>>>
!FIRDialect-DAG: } else {

!FIRDialect-DAG:   fir.store %{{.*}} to [[TMP203]] : !fir.ref<!fir.box<!fir.heap<f32>>>
!FIRDialect-DAG: }
!FIRDialect-DAG: fir.call @_QFprivate_clause_real_call_allocatablePhelper_private_clause_real_call_allocatable([[TMP203]]) fastmath<contract> : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> ()
!FIRDialect-DAG: %{{.*}} = fir.load [[TMP203]] : !fir.ref<!fir.box<!fir.heap<f32>>>

!FIRDialect-DAG: fir.if %{{.*}} {
!FIRDialect-DAG:   %{{.*}} = fir.load [[TMP203]] : !fir.ref<!fir.box<!fir.heap<f32>>>

!FIRDialect-DAG:     fir.store %{{.*}} to [[TMP203]] : !fir.ref<!fir.box<!fir.heap<f32>>>
!FIRDialect-DAG:   }
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
!FIRDialect:    {{%.*}} = fir.alloca !fir.box<!fir.ptr<!fir.type<_QFincrement_list_itemsTnode{payload:i32,next:!fir.box<!fir.ptr<!fir.type<_QFincrement_list_itemsTnode>>>}>>> {bindc_name = "p", uniq_name = "_QFincrement_list_itemsEp"}
!FIRDialect:    omp.parallel   {
!FIRDialect:      {{%.*}} = fir.alloca !fir.box<!fir.ptr<!fir.type<_QFincrement_list_itemsTnode{payload:i32,next:!fir.box<!fir.ptr<!fir.type<_QFincrement_list_itemsTnode>>>}>>> {bindc_name = "p", pinned, uniq_name = "_QFincrement_list_itemsEp"}
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
!FIRDialect-DAG: [[PP0:%.*]]  = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "y1", uniq_name = "{{.*}}Ey1"}
!FIRDialect-DAG: [[PP1:%.*]]  = fir.alloca !fir.ptr<i32> {uniq_name = "{{.*}}Ey1.addr"}
!FIRDialect-DAG: [[PP2:%.*]]  = fir.zero_bits !fir.ptr<i32>
!FIRDialect:     fir.store [[PP2]] to [[PP1]] : !fir.ref<!fir.ptr<i32>>
!FIRDialect-DAG: [[PP3:%.*]]  = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "y2", uniq_name = "{{.*}}Ey2"}

!FIRDialect:     fir.store %6 to %3 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
!FIRDialect-DAG: [[PP7:%.*]] = fir.alloca i32 {bindc_name = "z1", fir.target, uniq_name = "{{.*}}Ez1"}

!FIRDialect-DAG: [[PP8:%.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "z2", fir.target, uniq_name = "{{.*}}Ez2"}
!FIRDialect:     omp.parallel   {
!FIRDialect-DAG:   [[PP9:%.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "y1", pinned, uniq_name = "{{.*}}Ey1"}
!FIRDialect-DAG:   [[PP10:%.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "y2", pinned, uniq_name = "{{.*}}Ey2"}
!FIRDialect-DAG:   [[PP11:%.*]] = fir.embox [[PP7]] : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
!FIRDialect:       fir.store [[PP11]] to [[PP9]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
!FIRDialect-DAG:   [[PP12:%.*]] = fir.shape %c{{.*}} : (index) -> !fir.shape<1>
!FIRDialect-DAG:   [[PP13:%.*]] = fir.embox [[PP8]]([[PP12]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
!FIRDialect:       fir.store %13 to [[PP10]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
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
  ! FIRDialect:  omp.parallel
  !$OMP PARALLEL PRIVATE(r)
  ! FIRDialect:     %[[ALLOCA_IV:.*]] = fir.alloca i32 {{{.*}}, pinned}

  ! FIRDialect:     [[R:%.*]] = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "r", pinned, uniq_name = "{{.*}}Er"}
  ! FIRDialect:     fir.store {{%.*}} to [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:     fir.store {{%.*}} to [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>

  ! FIRDialect:     %[[WS_LB:.*]] = arith.constant 1 : i32
  ! FIRDialect:     %[[WS_UB:.*]] = arith.constant 9 : i32
  ! FIRDialect:     %[[WS_STEP:.*]] = arith.constant 1 : i32

  ! FIRDialect:     omp.wsloop for (%[[I:.*]]) : i32 = (%[[WS_LB]]) to (%[[WS_UB]]) inclusive step (%[[WS_STEP]])
  !$OMP DO
  do i=1, 9
  ! FIRDialect:     fir.store %[[I]] to %[[ALLOCA_IV:.*]] : !fir.ref<i32>
  ! FIRDialect:     %[[LOAD_IV:.*]] = fir.load %[[ALLOCA_IV]] : !fir.ref<i32>
  ! FIRDialect:     fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  ! FIRDialect:     omp.yield
  ! FIRDialect:     {{%.*}} = fir.load [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:     fir.if {{%.*}} {
  ! FIRDialect:     [[LD:%.*]] = fir.load [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:     [[AD:%.*]] = fir.box_addr [[LD]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  ! FIRDialect:     fir.freemem [[AD]] : !fir.heap<f32>
  ! FIRDialect:     fir.store {{%.*}} to [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>
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
  ! FIRDialect:     %[[ALLOCA_IV:.*]] = fir.alloca i32 {{{.*}}, pinned}

  ! FIRDialect:     [[R:%.*]] = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "r", pinned, uniq_name = "{{.*}}Er"}
  ! FIRDialect:     fir.store {{%.*}} to [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:     fir.store {{%.*}} to [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>

  ! FIRDialect:     %[[WS_LB:.*]] = arith.constant 1 : i32
  ! FIRDialect:     %[[WS_UB:.*]] = arith.constant 9 : i32
  ! FIRDialect:     %[[WS_STEP:.*]] = arith.constant 1 : i32

  ! FIRDialect:     omp.wsloop for (%[[I:.*]]) : i32 = (%[[WS_LB]]) to (%[[WS_UB]]) inclusive step (%[[WS_STEP]])
  !$OMP DO PRIVATE(r)
  do i=1, 9
  ! FIRDialect:     fir.store %[[I]] to %[[ALLOCA_IV:.*]] : !fir.ref<i32>
  ! FIRDialect:     %[[LOAD_IV:.*]] = fir.load %[[ALLOCA_IV]] : !fir.ref<i32>
  ! FIRDialect:     fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  ! FIRDialect:     omp.yield
  ! FIRDialect:     {{%.*}} = fir.load [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:     fir.if {{%.*}} {
  ! FIRDialect:     [[LD:%.*]] = fir.load [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:     [[AD:%.*]] = fir.box_addr [[LD]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  ! FIRDialect:     fir.freemem [[AD]] : !fir.heap<f32>
  ! FIRDialect:     fir.store {{%.*}} to [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  !$OMP END DO
  ! FIRDialect:  omp.terminator
  !$OMP END PARALLEL
end subroutine

!FIRDialect-LABEL: func @_QPsimple_loop_3()
subroutine simple_loop_3
  integer :: i
  real, allocatable :: r;
  ! FIRDialect:  omp.parallel
  ! FIRDialect:     %[[ALLOCA_IV:.*]] = fir.alloca i32 {{{.*}}, pinned}

  ! FIRDialect:     [[R:%.*]] = fir.alloca !fir.box<!fir.heap<f32>> {bindc_name = "r", pinned, uniq_name = "{{.*}}Er"}
  ! FIRDialect:     fir.store {{%.*}} to [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:     fir.store {{%.*}} to [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>

  ! FIRDialect:     %[[WS_LB:.*]] = arith.constant 1 : i32
  ! FIRDialect:     %[[WS_UB:.*]] = arith.constant 9 : i32
  ! FIRDialect:     %[[WS_STEP:.*]] = arith.constant 1 : i32

  ! FIRDialect:     omp.wsloop for (%[[I:.*]]) : i32 = (%[[WS_LB]]) to (%[[WS_UB]]) inclusive step (%[[WS_STEP]])
  !$OMP PARALLEL DO PRIVATE(r)
  do i=1, 9
  ! FIRDialect:     fir.store %[[I]] to %[[ALLOCA_IV:.*]] : !fir.ref<i32>
  ! FIRDialect:     %[[LOAD_IV:.*]] = fir.load %[[ALLOCA_IV]] : !fir.ref<i32>
  ! FIRDialect:     fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  ! FIRDialect:     omp.yield
  ! FIRDialect:     {{%.*}} = fir.load [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:     fir.if {{%.*}} {
  ! FIRDialect:     [[LD:%.*]] = fir.load [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:     [[AD:%.*]] = fir.box_addr [[LD]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  ! FIRDialect:     fir.freemem [[AD]] : !fir.heap<f32>
  ! FIRDialect:     fir.store {{%.*}} to [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>
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

  ! FIRDialect:     %[[LB:.*]] = arith.constant 1 : i32
  ! FIRDialect:     %[[UB:.*]] = arith.constant 9 : i32
  ! FIRDialect:     %[[STEP:.*]] = arith.constant 1 : i32

  ! FIRDialect: omp.simdloop for (%[[I:.*]]) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
  !$OMP SIMD PRIVATE(r)
  do i=1, 9
  ! FIRDialect:     fir.store %[[I]] to %[[LOCAL:.*]] : !fir.ref<i32>
  ! FIRDialect:     %[[LOAD_IV:.*]] = fir.load %[[LOCAL]] : !fir.ref<i32>
  ! FIRDialect:     fir.call @_FortranAioOutputInteger32({{.*}}, %[[LOAD_IV]]) {{.*}}: (!fir.ref<i8>, i32) -> i1
    print*, i
  end do
  !$OMP END SIMD
  ! FIRDialect:     omp.yield
  ! FIRDialect:     {{%.*}} = fir.load [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:     fir.if {{%.*}} {
  ! FIRDialect:     [[LD:%.*]] = fir.load [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>
  ! FIRDialect:     [[AD:%.*]] = fir.box_addr [[LD]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  ! FIRDialect:     fir.freemem [[AD]] : !fir.heap<f32>
  ! FIRDialect:     fir.store {{%.*}} to [[R]] : !fir.ref<!fir.box<!fir.heap<f32>>>
end subroutine
