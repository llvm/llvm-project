!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s --check-prefix HLFIR
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s --check-prefix FIR
!RUN: %flang_fc1 -emit-llvm -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s --check-prefix LLVM
!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -o - | fir-opt --lower-workshare --allow-unregistered-dialect -o - | FileCheck %s --check-prefix FIROPT

! Test that parallel workshare with firstprivate(P) where P is a pointer
! correctly places stores through the pointer target in omp.single rather
! than parallelizing them. The pointer descriptor is thread-local (firstprivate),
! but the target data is shared memory.

subroutine test_workshare_firstprivate_pointer(P)
  integer, pointer, intent(in) :: P(:)
  integer :: i
  !$omp parallel workshare firstprivate(P)
  forall (i = 1:SIZE(P)) P(i) = i
  !$omp end parallel workshare
end subroutine

! HLFIR-LABEL: {{.*}}test_workshare_firstprivate_pointer{{.*}} {
! HLFIR:     %[[ORIG_P:.*]]:2 = hlfir.declare %{{.*}} {{.*}}uniq_name = "_QFtest_workshare_firstprivate_pointerEp"
! HLFIR-LABEL:     omp.parallel {
! HLFIR-LABEL:       omp.workshare {
! The firstprivate copy: alloca, zero-init, declare, then copy from original
! HLFIR:         %[[FP_ALLOCA:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
! HLFIR:         fir.store %{{.*}} to %[[FP_ALLOCA]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! HLFIR:         %[[FP_DECL:.*]]:2 = hlfir.declare %[[FP_ALLOCA]] {{{.*}}uniq_name = "_QFtest_workshare_firstprivate_pointerEp"}
! HLFIR:         %[[ORIG_VAL:.*]] = fir.load %[[ORIG_P]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! HLFIR:         fir.store %[[ORIG_VAL]] to %[[FP_DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! HLFIR:         hlfir.forall
! HLFIR:         omp.terminator
! HLFIR:       }
! HLFIR:       omp.terminator
! HLFIR:     }
! HLFIR:     return

! After workshare lowering, the forall body (which stores through the pointer
! target) must be inside omp.single, not parallelized.
! FIR: {{.*}}test_workshare_firstprivate_pointer
! FIR-SAME: (%[[ARG0:.*]]: {{.*}}) {
! FIR: %[[C1:.*]] = arith.constant 1 : index
! FIR: %[[C1_I32:.*]] = arith.constant 1 : i32
! FIR: %[[C0:.*]] = arith.constant 0 : index
! FIR: %[[DSCOPE:.*]] = fir.dummy_scope{{.*}}
! FIR: %[[P_DECL:.*]] = fir.declare %[[ARG0]]{{.*}}fortran_attrs = #fir.var_attrs<intent_in, pointer>{{.*}}
! FIR: omp.parallel {
! Thread-private storage for firstprivate pointer descriptor.
! FIR: %[[P_PRIV:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {
! FIR-SAME: bindc_name = "p"
! FIR-SAME: pinned
! FIR: omp.single copyprivate(%[[P_PRIV]]{{.*}} {
! FIR: %[[ZERO_PTR:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
! FIR: %[[SHAPE:.*]] = fir.shape %[[C0]]
! FIR: %[[EMPTY_BOX:.*]] = fir.embox %[[ZERO_PTR]](%[[SHAPE]])
! FIR: fir.store %[[EMPTY_BOX]] to %[[P_PRIV]]
! FIR-SAME: : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! FIR: %[[P_FP_DECL:.*]] = fir.declare %[[P_PRIV]]
! FIR-SAME: fortran_attrs = #fir.var_attrs<intent_in, pointer>
! FIR: %[[ORIG_BOX:.*]] = fir.load %[[P_DECL]]
! FIR-SAME: : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! FIR: fir.store %[[ORIG_BOX]] to %[[P_FP_DECL]]
! FIR-SAME: : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! FIR: %[[P_PRIVATE:.*]] = fir.load %[[P_FP_DECL]]
! FIR: %[[P_SIZE:.*]]:3 = fir.box_dims %[[P_PRIVATE]], %[[C0]]
! FIR: %[[SIZE_TMP1:.*]] = fir.convert %[[P_SIZE]]#1
! FIR: %[[SIZE_TMP2:.*]] = fir.convert %[[SIZE_TMP1]]
! FIR: %[[LOOP_LB:.*]] = fir.convert %[[C1_I32]]
! FIR: %[[LOOP_UB:.*]] = fir.convert %[[SIZE_TMP2]]
! FIR: fir.do_loop %[[IV:.*]] = %[[LOOP_LB]] to %[[LOOP_UB]] step %[[C1]] {
! FIR: %[[IV_VAL:.*]] = fir.convert %[[IV]]
! FIR: fir.store %[[IV_VAL]] to %[[I_PRIV:.*]] : !fir.ref<i32>
! FIR: %[[RHS_STORE_VAL:.*]] = fir.load %[[I_PRIV]] : !fir.ref<i32>
! FIR: %[[P_CUR:.*]] = fir.load %[[P_FP_DECL]]
! FIR: %[[LHS_ELEM_ADDR:.*]] = fir.array_coor %[[P_CUR]]
! FIR: fir.store %[[RHS_STORE_VAL]] to %[[LHS_ELEM_ADDR]] : !fir.ref<i32>
! FIR: omp.terminator
! FIR: }
! FIR: omp.barrier
! FIR: omp.terminator
! FIR: }
! FIR: return

! FIROPT: func.func @_QPtest_workshare_firstprivate_pointer(
! FIROPT-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>

! FIROPT: %[[SCOPE:.*]] = fir.dummy_scope : !fir.dscope
! FIROPT: %[[I_ALLOC:.*]] = fir.alloca i32
! FIROPT: %[[I_DECL:.*]]:2 = hlfir.declare %[[I_ALLOC]]

! FIROPT: %[[P_DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[SCOPE]] arg 1
! FIROPT-SAME: fortran_attrs = #fir.var_attrs<intent_in, pointer>

! FIROPT: omp.parallel {

! FIROPT: %[[P_PRIV:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "p", pinned

! FIROPT: omp.single copyprivate(%[[P_PRIV]] -> @_workshare_copy_box_ptr_Uxi32 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) nowait {

! FIROPT: %[[ZERO_PTR:.*]] = fir.zero_bits !fir.ptr<!fir.array<?xi32>>
! FIROPT: %[[C0:.*]] = arith.constant 0 : index
! FIROPT: %[[SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
! FIROPT: %[[EMBOX:.*]] = fir.embox %[[ZERO_PTR]](%[[SHAPE]])
! FIROPT: fir.store %[[EMBOX]] to %[[P_PRIV]]

! FIROPT: %[[P_FP:.*]]:2 = hlfir.declare %[[P_PRIV]]
! FIROPT-SAME: fortran_attrs = #fir.var_attrs<intent_in, pointer>

! FIROPT: %[[LOAD_ORIG:.*]] = fir.load %[[P_DECL]]#0
! FIROPT: fir.store %[[LOAD_ORIG]] to %[[P_FP]]#0

! FIROPT: %[[C1:.*]] = arith.constant 1 : i32

! FIROPT: %[[LOAD_PRIV:.*]] = fir.load %[[P_FP]]#0
! FIROPT: %[[C0_2:.*]] = arith.constant 0 : index
! FIROPT: %[[DIMS:.*]]:3 = fir.box_dims %[[LOAD_PRIV]], %[[C0_2]]

! FIROPT: %[[EXT64:.*]] = fir.convert %[[DIMS]]#1 : (index) -> i64
! FIROPT: %[[EXT32:.*]] = fir.convert %[[EXT64]] : (i64) -> i32

! FIROPT: hlfir.forall lb {
! FIROPT: hlfir.yield %[[C1]] : i32
! FIROPT: } ub {
! FIROPT: hlfir.yield %[[EXT32]] : i32
! FIROPT: }  (%[[IV:.*]]: i32) {

! FIROPT: %[[IDX:.*]] = hlfir.forall_index "i" %[[IV]] : (i32) -> !fir.ref<i32>

! FIROPT: hlfir.region_assign {

! FIROPT: %[[IDX_VAL:.*]] = fir.load %[[IDX]] : !fir.ref<i32>
! FIROPT: hlfir.yield %[[IDX_VAL]] : i32

! FIROPT: } to {

! FIROPT: %[[LOAD_BOX:.*]] = fir.load %[[P_FP]]#0
! FIROPT: %[[LOAD_I:.*]] = fir.load %[[IDX]] : !fir.ref<i32>
! FIROPT: %[[IDX64:.*]] = fir.convert %[[LOAD_I]] : (i32) -> i64

! FIROPT: %[[DESIG:.*]] = hlfir.designate %[[LOAD_BOX]] (%[[IDX64]])
! FIROPT-SAME: -> !fir.ref<i32>

! FIROPT: hlfir.yield %[[DESIG]] : !fir.ref<i32>

! FIROPT: }
! FIROPT: }

! FIROPT: omp.terminator
! FIROPT: }

! FIROPT: %[[POST_DECL:.*]]:2 = hlfir.declare %[[P_PRIV]]
! FIROPT-SAME: fortran_attrs = #fir.var_attrs<intent_in, pointer>

! FIROPT: omp.barrier
! FIROPT: omp.terminator
! FIROPT: }

! At LLVM IR level, verify the OpenMP fork call exists and the loop body
! is inside the outlined function.
! LLVM:       call void {{.*}}__kmpc_fork_call({{.*}}@test_workshare_firstprivate_pointer_..omp_par{{.*}})
! LLVM: {{.*}}test_workshare_firstprivate_pointer_..omp_par{{.*}}
! LLVM-LABEL: omp.par.region{{[0-9]+}}:
! LLVM:       call i32 @__kmpc_single
! LLVM:       icmp ne i32
! LLVM-LABEL: omp_region.end:
! LLVM:       call void @__kmpc_copyprivate
! LLVM:       call void {{.*}}__kmpc_barrier
! LLVM-LABEL: omp.single.region:
! LLVM:       call void @llvm.memcpy{{.*}}
! LLVM:       getelementptr {{.*}} i32 0, i32 7
! LLVM:       load i64{{.*}}
! LLVM-LABEL: omp_region.finalize:
! LLVM:       call void @__kmpc_end_single
! LLVM:       store i32 %{{.*}}, ptr %{{.*}}
! LLVM:       getelementptr nusw nuw i8
! LLVM:       ret void

! Test for "workshare firstprivate(z)" where z is an array.
! Check code to correctly broadcast the address of the firstprivate
! copy to all threads, instead of using a broken load/store copyprivate
! that only copies a single element for dynamically-sized arrays.

subroutine test_workshare_firstprivate_array(a, z, n)
  integer(4) :: n
  integer(4), dimension(n) :: z, a
  !$omp parallel workshare firstprivate(z)
  a = z + 1
  !$omp end parallel workshare
end subroutine

! After workshare lowering, the dynamic alloca for the firstprivate copy
! must be inside omp.single, with its address broadcast via a !fir.box
! indirection alloca + copyprivate.
! FIR-LABEL:     {{.*}}test_workshare_firstprivate_array(
! FIR:           %[[C1:.*]] = arith.constant 1 : i32
! FIR-LABEL:     omp.parallel {

! The box indirection alloca is hoisted for copyprivate
! FIR:       omp.single copyprivate(%[[BOX_INDIRECT:.*]] -> @_workshare_copy_box_Uxi32{{.*}}) {

! The dynamic alloca (firstprivate copy) is inside the single block
! FIR:         %[[FP_ARRAY:.*]] = fir.alloca{{.*}}

! Runtime shape construction for the firstprivate array.
! FIR:         %[[SHAPE:.*]] = fir.shape %{{.*}}
! FIR:         %[[BOX_VAL:.*]] = fir.embox %[[FP_ARRAY]](%[[SHAPE]]){{.*}}fir.array<?xi32>{{.*}}
! FIR:         fir.store %[[BOX_VAL]] to %[[BOX_INDIRECT]] {{.*}}fir.array<?xi32>{{.*}}

! The initialization of the firstprivate copy
! FIR:         fir.call @_FortranAAssign
! FIR:         omp.terminator
! FIR:       }
! After single, the box is loaded and the address extracted
! FIR:       %[[LOADED_BOX:.*]] = fir.load %[[BOX_INDIRECT]]{{.*}}fir.array<?xi32>{{.*}}
! FIR:       %[[ARRAY_ADDR:.*]] = fir.box_addr %[[LOADED_BOX]]{{.*}}fir.array<?xi32>>{{.*}}
! The workshared loop uses the broadcast address
! FIR:       omp.wsloop {
! FIR:         %[[SRC_ELEM:.*]] = fir.array_coor %{{.*}}(%{{.*}}) %{{.*}}
! FIR:         %[[SRC_VAL:.*]] = fir.load %[[SRC_ELEM]]
! FIR:         %[[ADD_RES:.*]] = arith.addi %[[SRC_VAL]], %[[C1]] : i32
! FIR:         %[[DST_ELEM:.*]] = fir.array_coor %{{.*}}(%{{.*}}) %{{.*}}
! FIR:         fir.store %[[ADD_RES]] to %[[DST_ELEM]]
! FIR:       }
! FIR:       omp.barrier
! FIR:       omp.terminator
! FIR:       return
