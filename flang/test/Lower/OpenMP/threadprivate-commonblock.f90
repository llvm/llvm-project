! This test checks lowering of OpenMP Threadprivate Directive.
! Test for common block.

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module test
  integer:: a
  real :: b(2)
  complex, pointer :: c, d(:)
  character(5) :: e, f(2)
  common /blk/ a, b, c, d, e, f

  !$omp threadprivate(/blk/)

!CHECK: fir.global common @blk_(dense<0> : vector<103xi8>) {alignment = 8 : i64} : !fir.array<103xi8>

contains
  subroutine sub()
  !CHECK-DAG:  %[[CBLK_ADDR:.*]] = fir.address_of(@blk_) : !fir.ref<!fir.array<103xi8>>
  !CHECK-DAG:  %[[CBLK_ADDR_CVT:.*]] = fir.convert %[[CBLK_ADDR]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
  !CHECK-DAG:  %[[A_ADDR:.*]] = fir.coordinate_of %[[CBLK_ADDR_CVT]], %c0 : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  !CHECK-DAG:  %[[A_ADDR_CVT:.*]] = fir.convert %[[A_ADDR]] : (!fir.ref<i8>) -> !fir.ref<i32>
  !CHECK-DAG:  %[[A_VAL:.*]]:2 = hlfir.declare %[[A_ADDR_CVT]] {uniq_name = "_QMtestEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  !CHECK-DAG:  %[[OMP_CBLK:.*]] = omp.threadprivate %[[CBLK_ADDR]] : !fir.ref<!fir.array<103xi8>> -> !fir.ref<!fir.array<103xi8>>
  !CHECK-DAG:  %[[OMP_CBLK_ADDR:.*]] = fir.convert %[[OMP_CBLK]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
  !CHECK-DAG:  %[[A_ADDR:.*]] = fir.coordinate_of %[[OMP_CBLK_ADDR]], {{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  !CHECK-DAG:  %[[A_ADDR_CVT:.*]] = fir.convert %[[A_ADDR]] : (!fir.ref<i8>) -> !fir.ref<i32>
  !CHECK-DAG:  %[[A_DECL:.*]]:2 = hlfir.declare %[[A_ADDR_CVT]] {uniq_name = "_QMtestEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  !CHECK-DAG:  %[[OMP_CBLK_ADDR:.*]] = fir.convert %[[OMP_CBLK]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
  !CHECK-DAG:  %[[B_ADDR:.*]] = fir.coordinate_of %[[OMP_CBLK_ADDR]], {{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  !CHECK-DAG:  %[[B_ADDR_CVT:.*]] = fir.convert %[[B_ADDR]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<2xf32>>
  !CHECK-DAG:  %[[B_DECL:.*]]:2 = hlfir.declare %[[B_ADDR_CVT]]({{.*}}) {uniq_name = "_QMtestEb"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xf32>>, !fir.ref<!fir.array<2xf32>>)
  !CHECK-DAG:  %[[OMP_CBLK_ADDR:.*]] = fir.convert %[[OMP_CBLK]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
  !CHECK-DAG:  %[[C_ADDR:.*]] = fir.coordinate_of %[[OMP_CBLK_ADDR]], {{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  !CHECK-DAG:  %[[C_ADDR_CVT:.*]] = fir.convert %[[C_ADDR]] : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<complex<f32>>>>
  !CHECK-DAG:  %[[C_DECL:.*]]:2 = hlfir.declare %[[C_ADDR_CVT]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMtestEc"} : (!fir.ref<!fir.box<!fir.ptr<complex<f32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<complex<f32>>>>, !fir.ref<!fir.box<!fir.ptr<complex<f32>>>>)
  !CHECK-DAG:  %[[OMP_CBLK_ADDR:.*]] = fir.convert %[[OMP_CBLK]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
  !CHECK-DAG:  %[[D_ADDR:.*]] = fir.coordinate_of %[[OMP_CBLK_ADDR]], {{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  !CHECK-DAG:  %[[D_ADDR_CVT:.*]] = fir.convert %[[D_ADDR]] : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xcomplex<f32>>>>>
  !CHECK-DAG:  %[[D_DECL:.*]]:2 = hlfir.declare %[[D_ADDR_CVT]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMtestEd"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xcomplex<f32>>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xcomplex<f32>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xcomplex<f32>>>>>)
  !CHECK-DAG:  %[[OMP_CBLK_ADDR:.*]] = fir.convert %[[OMP_CBLK]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
  !CHECK-DAG:  %[[E_ADDR:.*]] = fir.coordinate_of %[[OMP_CBLK_ADDR]], {{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  !CHECK-DAG:  %[[E_ADDR_CVT:.*]] = fir.convert %[[E_ADDR]] : (!fir.ref<i8>) -> !fir.ref<!fir.char<1,5>>
  !CHECK-DAG:  %[[E_DECL:.*]]:2 = hlfir.declare %[[E_ADDR_CVT]] typeparams {{.*}} {uniq_name = "_QMtestEe"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
  !CHECK-DAG:  %[[OMP_CBLK_ADDR:.*]] = fir.convert %[[OMP_CBLK]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
  !CHECK-DAG:  %[[F_ADDR:.*]] = fir.coordinate_of %[[OMP_CBLK_ADDR]], {{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
  !CHECK-DAG:  %[[F_ADDR_CVT:.*]] = fir.convert %[[F_ADDR]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<2x!fir.char<1,5>>>
  !CHECK-DAG:  %[[F_DECL:.*]]:2 = hlfir.declare %[[F_ADDR_CVT]]({{.*}}) typeparams {{.*}} {uniq_name = "_QMtestEf"} : (!fir.ref<!fir.array<2x!fir.char<1,5>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<2x!fir.char<1,5>>>, !fir.ref<!fir.array<2x!fir.char<1,5>>>)
  !CHECK-DAG:  {{.*}} = fir.load %[[A_DECL]]#0 : !fir.ref<i32>
  !CHECK-DAG:  {{.*}} = fir.embox %[[B_DECL]]#1({{.*}}) : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf32>>
  !CHECK-DAG:  {{.*}} = fir.load %[[C_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<complex<f32>>>>
  !CHECK-DAG:  {{.*}} = fir.load %[[D_DECL]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xcomplex<f32>>>>>
  !CHECK-DAG:  {{.*}} = fir.convert %[[E_DECL]]#1 : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
  !CHECK-DAG:  {{.*}} = fir.embox %[[F_DECL]]#1({{.*}}) : (!fir.ref<!fir.array<2x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.char<1,5>>>
    print *, a, b, c, d, e, f

    !$omp parallel
    !CHECK-DAG: omp.parallel   {
    !CHECK-DAG:  %[[TP_PARALLEL:.*]] = omp.threadprivate %[[CBLK_ADDR]] : !fir.ref<!fir.array<103xi8>> -> !fir.ref<!fir.array<103xi8>>
    !CHECK-DAG:  %[[TP_PARALLEL_ADDR:.*]] = fir.convert %[[TP_PARALLEL]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
    !CHECK-DAG:  %[[TP_A_ADDR:.*]] = fir.coordinate_of %[[TP_PARALLEL_ADDR]], {{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    !CHECK-DAG:  %[[TP_A_ADDR_CVT:.*]] = fir.convert %[[TP_A_ADDR]] : (!fir.ref<i8>) -> !fir.ref<i32>
    !CHECK-DAG:  %[[TP_A_DECL:.*]]:2 = hlfir.declare %[[TP_A_ADDR_CVT]] {uniq_name = "_QMtestEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    !CHECK-DAG:  %[[TP_PARALLEL_ADDR:.*]] = fir.convert %[[TP_PARALLEL]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
    !CHECK-DAG:  %[[TP_B_ADDR:.*]] = fir.coordinate_of %[[TP_PARALLEL_ADDR]], {{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    !CHECK-DAG:  %[[TP_B_ADDR_CVT:.*]] = fir.convert %[[TP_B_ADDR]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<2xf32>>
    !CHECK-DAG:  %[[TP_B_DECL:.*]]:2 = hlfir.declare %[[TP_B_ADDR_CVT]](%92) {uniq_name = "_QMtestEb"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xf32>>, !fir.ref<!fir.array<2xf32>>)
    !CHECK-DAG:  %[[TP_PARALLEL_ADDR:.*]] = fir.convert %[[TP_PARALLEL]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
    !CHECK-DAG:  %[[TP_C_ADDR:.*]] = fir.coordinate_of %[[TP_PARALLEL_ADDR]], {{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    !CHECK-DAG:  %[[TP_C_ADDR_CVT:.*]] = fir.convert %[[TP_C_ADDR]] : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<complex<f32>>>>
    !CHECK-DAG:  %[[TP_C_DECL:.*]]:2 = hlfir.declare %[[TP_C_ADDR_CVT]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMtestEc"} : (!fir.ref<!fir.box<!fir.ptr<complex<f32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<complex<f32>>>>, !fir.ref<!fir.box<!fir.ptr<complex<f32>>>>)
    !CHECK-DAG:  %[[TP_PARALLEL_ADDR:.*]] = fir.convert %[[TP_PARALLEL]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
    !CHECK-DAG:  %[[TP_D_ADDR:.*]] = fir.coordinate_of %[[TP_PARALLEL_ADDR]], {{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    !CHECK-DAG:  %[[TP_D_ADDR_CVT:.*]] = fir.convert %[[TP_D_ADDR]] : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xcomplex<f32>>>>>
    !CHECK-DAG:  %[[TP_D_DECL:.*]]:2 = hlfir.declare %[[TP_D_ADDR_CVT]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMtestEd"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xcomplex<f32>>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xcomplex<f32>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xcomplex<f32>>>>>)
    !CHECK-DAG:  %[[TP_PARALLEL_ADDR:.*]] = fir.convert %[[TP_PARALLEL]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
    !CHECK-DAG:  %[[TP_E_ADDR:.*]] = fir.coordinate_of %[[TP_PARALLEL_ADDR]], {{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    !CHECK-DAG:  %[[TP_E_ADDR_CVT:.*]] = fir.convert %[[TP_E_ADDR]] : (!fir.ref<i8>) -> !fir.ref<!fir.char<1,5>>
    !CHECK-DAG:  %[[TP_E_DECL:.*]]:2 = hlfir.declare %[[TP_E_ADDR_CVT]] typeparams {{.*}} {uniq_name = "_QMtestEe"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
    !CHECK-DAG:  %[[TP_PARALLEL_ADDR:.*]] = fir.convert %[[TP_PARALLEL]] : (!fir.ref<!fir.array<103xi8>>) -> !fir.ref<!fir.array<?xi8>>
    !CHECK-DAG:  %[[TP_F_ADDR:.*]] = fir.coordinate_of %[[TP_PARALLEL_ADDR]], %{{.*}} : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    !CHECK-DAG:  %[[TP_F_ADDR_CVT:.*]] = fir.convert %[[TP_F_ADDR]] : (!fir.ref<i8>) -> !fir.ref<!fir.array<2x!fir.char<1,5>>>
    !CHECK-DAG:  %[[TP_F_DECL:.*]]:2 = hlfir.declare %[[TP_F_ADDR_CVT]]({{.*}}) typeparams {{.*}} {uniq_name = "_QMtestEf"} : (!fir.ref<!fir.array<2x!fir.char<1,5>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<2x!fir.char<1,5>>>, !fir.ref<!fir.array<2x!fir.char<1,5>>>)
    !CHECK-DAG:  {{.*}} = fir.load %[[TP_A_DECL]]#0 : !fir.ref<i32>
    !CHECK-DAG:  {{.*}} = fir.embox %[[TP_B_DECL]]#1({{.*}}) : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf32>>
    !CHECK-DAG:  {{.*}} = fir.load %[[TP_C_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<complex<f32>>>>
    !CHECK-DAG:  {{.*}} = fir.load %[[TP_D_DECL]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xcomplex<f32>>>>>
    !CHECK-DAG:  {{.*}} = fir.convert %[[TP_E_DECL]]#1 : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
    !CHECK-DAG:  {{.*}} = fir.embox %[[TP_F_DECL]]#1(%{{.*}}) : (!fir.ref<!fir.array<2x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.char<1,5>>>
    !CHECK-DAG: omp.terminator
    !CHECK-DAG: }
    print *, a, b, c, d, e, f
    !$omp end parallel

  !CHECK-DAG:  %{{.*}} = fir.load %[[A_DECL]]#0 : !fir.ref<i32>
  !CHECK-DAG:  %{{.*}} = fir.embox %[[B_DECL]]#1(%63) : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xf32>>
  !CHECK-DAG:  %{{.*}} = fir.load %[[C_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<complex<f32>>>>
  !CHECK-DAG:  %{{.*}} = fir.load %[[D_DECL]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xcomplex<f32>>>>>
  !CHECK-DAG:  %{{.*}} = fir.convert %[[E_DECL]]#1 : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
  !CHECK-DAG:  %{{.*}} = fir.embox %[[F_DECL]]#1(%79) : (!fir.ref<!fir.array<2x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<2x!fir.char<1,5>>>
    print *, a, b, c, d, e, f

  end
end
