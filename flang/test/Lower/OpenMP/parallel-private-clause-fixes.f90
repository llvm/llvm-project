! This test checks a few bug fixes in the PRIVATE clause lowering

! RUN: bbc -fopenmp -emit-hlfir %s -o - \
! RUN: | FileCheck %s

! CHECK:  omp.private {type = private} @[[BOX_HEAP_CHAR_PRIVATIZER:_QFsub01Eaaa_private_ref_box_heap_c8xU]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>> alloc {
! CHECK:  ^bb0(%[[ORIG_REF:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>):
! CHECK:             %[[VAL_4:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>> {bindc_name = "aaa", pinned, uniq_name = "_QFsub01Eaaa"}
! CHECK:             %[[VAL_5:.*]] = fir.load %[[ORIG_REF]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:             %[[VAL_6:.*]] = fir.box_addr %[[VAL_5]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:             %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (!fir.heap<!fir.char<1,?>>) -> i64
! CHECK:             %[[VAL_8:.*]] = arith.constant 0 : i64
! CHECK:             %[[VAL_9:.*]] = arith.cmpi ne, %[[VAL_7]], %[[VAL_8]] : i64
! CHECK:             fir.if %[[VAL_9]] {
! CHECK:               %[[ELEM_SIZE:.*]] = fir.box_elesize %{{.*}} : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
! CHECK:               %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:               %[[VAL_11:.*]] = arith.cmpi sgt, %[[ELEM_SIZE]], %[[VAL_10]] : index
! CHECK:               %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[ELEM_SIZE]], %[[VAL_10]] : index
! CHECK:               %[[VAL_13:.*]] = fir.allocmem !fir.char<1,?>(%[[VAL_12]] : index) {fir.must_be_heap = true, uniq_name = "_QFsub01Eaaa.alloc"}
! CHECK:               %[[VAL_14:.*]] = fir.embox %[[VAL_13]] typeparams %[[VAL_12]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:               fir.store %[[VAL_14]] to %[[VAL_4]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:             } else {
! CHECK:               %[[VAL_15:.*]] = fir.zero_bits !fir.heap<!fir.char<1,?>>
! CHECK:               %[[VAL_16:.*]] = arith.constant 0 : index
! CHECK:               %[[VAL_17:.*]] = fir.embox %[[VAL_15]] typeparams %[[VAL_16]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:               fir.store %[[VAL_17]] to %[[VAL_4]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:             }
! CHECK:             %[[VAL_18:.*]]:2 = hlfir.declare %[[VAL_4]] {fortran_attrs = #{{.*}}<allocatable>, uniq_name = "_QFsub01Eaaa"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>)
!CHECK:              omp.yield(%[[VAL_18]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>)
!CHECK:  } dealloc {
!CHECK:  ^bb0(%[[ORIG_REF:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>):
! CHECK:             %[[VAL_19:.*]] = fir.load %[[ORIG_REF]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:             %[[VAL_20:.*]] = fir.box_addr %[[VAL_19]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (!fir.heap<!fir.char<1,?>>) -> i64
! CHECK:             %[[VAL_22:.*]] = arith.constant 0 : i64
! CHECK:             %[[VAL_23:.*]] = arith.cmpi ne, %[[VAL_21]], %[[VAL_22]] : i64
! CHECK:             fir.if %[[VAL_23]] {
! CHECK:               %[[VAL_24:.*]] = fir.load %[[ORIG_REF]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:               %[[VAL_25:.*]] = fir.box_addr %[[VAL_24]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:               fir.freemem %[[VAL_25]] : !fir.heap<!fir.char<1,?>>
! CHECK:               %[[VAL_26:.*]] = fir.zero_bits !fir.heap<!fir.char<1,?>>
! CHECK:               %[[VAL_27:.*]] = arith.constant 0 : index
! CHECK:               %[[VAL_28:.*]] = fir.embox %[[VAL_26]] typeparams %[[VAL_27]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:               fir.store %[[VAL_28]] to %[[ORIG_REF]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
!CHECK:    }
!CHECK:    omp.yield
!CHECK:  }

! CHECK-LABEL: @_QPmultiple_private_fix(
! CHECK-SAME:  %[[GAMA:.*]]: !fir.ref<i32> {fir.bindc_name = "gama"}
! CHECK-DAG:         %[[GAMA_DECL:.*]]:2 = hlfir.declare %[[GAMA]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFmultiple_private_fixEgama"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-DAG:         %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFmultiple_private_fixEi"}
! CHECK-DAG:         %[[I_DECL:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFmultiple_private_fixEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-DAG:         %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFmultiple_private_fixEj"}
! CHECK-DAG:         %[[J_DECL:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFmultiple_private_fixEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-DAG:         %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFmultiple_private_fixEx"}
! CHECK-DAG:         %[[X_DECL:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFmultiple_private_fixEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:         omp.parallel {
! CHECK-DAG:           %[[PRIV_I:.*]] = fir.alloca i32 {bindc_name = "i", pinned, {{.*}}}
! CHECK-DAG:           %[[PRIV_I_DECL:.*]]:2 = hlfir.declare %[[PRIV_I]] {uniq_name = "_QFmultiple_private_fixEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-DAG:           %[[PRIV_J:.*]] = fir.alloca i32 {bindc_name = "j", pinned, uniq_name = "_QFmultiple_private_fixEj"}
! CHECK-DAG:           %[[PRIV_J_DECL:.*]]:2 = hlfir.declare %[[PRIV_J]] {uniq_name = "_QFmultiple_private_fixEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-DAG:           %[[PRIV_X:.*]] = fir.alloca i32 {bindc_name = "x", pinned, {{.*}}}
! CHECK-DAG:           %[[PRIV_X_DECL:.*]]:2 = hlfir.declare %[[PRIV_X]] {uniq_name = "_QFmultiple_private_fixEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[ONE:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_3:.*]] = fir.load %[[GAMA_DECL]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i32
! CHECK:           omp.wsloop {
! CHECK-NEXT:        omp.loop_nest (%[[VAL_6:.*]]) : i32 = (%[[ONE]]) to (%[[VAL_3]]) inclusive step (%[[VAL_5]]) {
! CHECK:               fir.store %[[VAL_6]] to %[[PRIV_I_DECL]]#1 : !fir.ref<i32>
! CHECK:               %[[VAL_7:.*]] = arith.constant 1 : i32
! CHECK:               %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
! CHECK:               %[[VAL_9:.*]] = fir.load %[[GAMA_DECL]]#0 : !fir.ref<i32>
! CHECK:               %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:               %[[VAL_11:.*]] = arith.constant 1 : index
! CHECK:               %[[LB:.*]] = fir.convert %[[VAL_8]] : (index) -> i32
! CHECK:               %[[VAL_12:.*]]:2 = fir.do_loop %[[VAL_13:[^ ]*]] =
! CHECK-SAME:              %[[VAL_8]] to %[[VAL_10]] step %[[VAL_11]]
! CHECK-SAME:              iter_args(%[[IV:.*]] = %[[LB]]) -> (index, i32) {
! CHECK:                 fir.store %[[IV]] to %[[PRIV_J_DECL]]#1 : !fir.ref<i32>
! CHECK:                 %[[LOAD:.*]] = fir.load %[[PRIV_I_DECL]]#0 : !fir.ref<i32>
! CHECK:                 %[[VAL_15:.*]] = fir.load %[[PRIV_J_DECL]]#0 : !fir.ref<i32>
! CHECK:                 %[[VAL_16:.*]] = arith.addi %[[LOAD]], %[[VAL_15]] : i32
! CHECK:                 hlfir.assign %[[VAL_16]] to %[[PRIV_X_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:                 %[[VAL_17:.*]] = arith.addi %[[VAL_13]], %[[VAL_11]] : index
! CHECK:                 %[[STEPCAST:.*]] = fir.convert %[[VAL_11]] : (index) -> i32
! CHECK:                 %[[IVLOAD:.*]] = fir.load %[[PRIV_J_DECL]]#1 : !fir.ref<i32>
! CHECK:                 %[[IVINC:.*]] = arith.addi %[[IVLOAD]], %[[STEPCAST]]
! CHECK:                 fir.result %[[VAL_17]], %[[IVINC]] : index, i32
! CHECK:               }
! CHECK:               fir.store %[[VAL_12]]#1 to %[[PRIV_J_DECL]]#1 : !fir.ref<i32>
! CHECK:               omp.yield
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           omp.terminator
! CHECK:         }
! CHECK:         return
subroutine multiple_private_fix(gama)
        integer :: i, j, x, gama
!$OMP PARALLEL DO PRIVATE(j,x)
        do i = 1, gama
          do j = 1, gama
            x = i + j
          end do
        end do
!$OMP END PARALLEL DO
end subroutine

! CHECK-LABEL: multiple_private_fix2
! CHECK:  %[[X1:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFmultiple_private_fix2Ex"}
! CHECK:  %[[X1_DECL:.*]]:2 = hlfir.declare %[[X1]] {uniq_name = "_QFmultiple_private_fix2Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  omp.parallel private({{.*}} {{.*}}#0 -> %[[X2:.*]] : {{.*}}) {
! CHECK:    %[[X2_DECL:.*]]:2 = hlfir.declare %[[X2]] {uniq_name = "_QFmultiple_private_fix2Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:    omp.parallel private({{.*}} {{.*}}#0 -> %[[X3:.*]] : {{.*}}) {
! CHECK:      %[[X3_DECL:.*]]:2 = hlfir.declare %[[X3]] {uniq_name = "_QFmultiple_private_fix2Ex"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:      %[[C3:.*]] = arith.constant 1 : i32
! CHECK:      hlfir.assign %[[C3]] to %[[X3_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:      omp.terminator
! CHECK:    }
! CHECK:      %[[C2:.*]] = arith.constant 1 : i32
! CHECK:      hlfir.assign %[[C2]] to %[[X2_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:    omp.terminator
! CHECK:  }
! CHECK:      %[[C1:.*]] = arith.constant 1 : i32
! CHECK:      hlfir.assign %[[C1]] to %[[X1_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:  return
subroutine multiple_private_fix2()
   integer :: x
   !$omp parallel private(x)
   !$omp parallel private(x)
      x = 1
   !$omp end parallel
      x = 1
   !$omp end parallel
      x = 1
end subroutine


! CHECK-LABEL:   func.func @_QPsub01(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>> {fir.bindc_name = "aaa"}) {
! CHECK:           %[[VAL_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:           %[[VAL_2:.*]] = fir.box_elesize %[[VAL_1]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] typeparams %[[VAL_2]] dummy_scope %{{[0-9]+}} {fortran_attrs = #{{.*}}<allocatable>, uniq_name = "_QFsub01Eaaa"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, index, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>)
! CHECK:           omp.parallel private(@[[BOX_HEAP_CHAR_PRIVATIZER]] %[[VAL_3]]#0 -> %{{.*}} : {{.*}}) {
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine sub01(aaa)
  character(*),allocatable :: aaa
  !$omp parallel private(aaa)
  !$omp end parallel
end subroutine

! CHECK-LABEL:   func.func @_QPsub02(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>> {fir.bindc_name = "bbb"}) {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #{{.*}}<allocatable>, uniq_name = "_QFsub02Ebbb"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>, !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>)
! CHECK:           omp.parallel private(@{{.*}} %[[VAL_1]]#0 -> %[[PRIV_ARG:.*]] : {{.*}}) {
! CHECK:             %{{.*}}:2 = hlfir.declare %[[PRIV_ARG]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFsub02Ebbb"} : (!fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>)
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           return
! CHECK:         }

subroutine sub02(bbb)
  character(:),allocatable :: bbb
  !$omp parallel private(bbb)
  !$omp end parallel
end subroutine sub02

