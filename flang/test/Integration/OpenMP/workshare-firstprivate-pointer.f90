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
! LLVM:       getelementptr {{.*}}i8
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
! is hoisted so each thread gets its own allocation (true firstprivate).
! The array data is broadcast via copyprivate using a box with a
! data-copying function (_FortranAAssign).
! FIR-LABEL:     {{.*}}test_workshare_firstprivate_array(
! FIR:           %[[C1:.*]] = arith.constant 1 : i32
! FIR-LABEL:     omp.parallel {

! The dynamic alloca is hoisted (per-thread allocation)
! FIR:       %[[FP_ARRAY:.*]] = fir.alloca !fir.array<?xi32>
! The box slot and embox are hoisted for copyprivate
! FIR:       %[[BOX_SLOT:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
! FIR:       %[[SHAPE:.*]] = fir.shape %{{.*}}
! FIR:       %[[BOX_VAL:.*]] = fir.embox %[[FP_ARRAY]](%[[SHAPE]]){{.*}}
! FIR:       fir.store %[[BOX_VAL]] to %[[BOX_SLOT]]

! Copyprivate uses box-data copy function to broadcast array contents
! FIR:       omp.single copyprivate(%[[BOX_SLOT]] -> @_workshare_copy_data_box_Uxi32{{.*}}) {

! The initialization of the firstprivate copy (single thread only)
! FIR:         fir.call @_FortranAAssign
! FIR:         omp.terminator
! FIR:       }
! The workshared loop uses the per-thread allocation directly
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

subroutine allocatable_example()
  implicit none

  integer, allocatable :: p(:)
  integer :: a(4)

  allocate(p(4))
  p = [1, 2, 3, 4]

  !$omp parallel workshare firstprivate(p)
    a = p + 1
  !$omp end parallel workshare
end subroutine

! HLFIR-LABEL: func.func @_QPallocatable_example() {
! HLFIR: %[[A:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {{.*}}uniq_name = "_QFallocatable_exampleEa"
! HLFIR: %[[ORIG_P:.*]]:2 = hlfir.declare %{{.*}} {{.*}}fortran_attrs = #fir.var_attrs<allocatable>{{.*}}uniq_name = "_QFallocatable_exampleEp"

! Initial allocation/assignment of original p
! HLFIR: fir.allocmem !fir.array<?xi32>
! HLFIR: fir.store %{{.*}} to %[[ORIG_P]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR: hlfir.assign %{{.*}} to %[[ORIG_P]]#0 realloc : {{.*}}, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

! HLFIR: omp.parallel {
! HLFIR:   omp.workshare {

! Firstprivate allocatable descriptor
! HLFIR:     %[[FP_ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {{.*}}bindc_name = "p"{{.*}}

! Allocate/init firstprivate copy depending on original allocation status
! HLFIR:     %[[ORIG_VAL0:.*]] = fir.load %[[ORIG_P]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR:     %[[ORIG_ADDR0:.*]] = fir.box_addr %[[ORIG_VAL0]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! HLFIR:     fir.convert %[[ORIG_ADDR0]]
! HLFIR:     arith.cmpi ne
! HLFIR:     fir.if %{{.*}} {
! HLFIR:       %[[ORIG_VAL1:.*]] = fir.load %[[ORIG_P]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR:       fir.box_dims %[[ORIG_VAL1]]
! HLFIR:       fir.allocmem !fir.array<?xi32>
! HLFIR:       fir.embox
! HLFIR:       fir.store %{{.*}} to %[[FP_ALLOCA]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR:     } else {
! HLFIR:       fir.zero_bits !fir.heap<!fir.array<?xi32>>
! HLFIR:       fir.embox
! HLFIR:       fir.store %{{.*}} to %[[FP_ALLOCA]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR:     }

! Declare firstprivate p
! HLFIR:     %[[FP_DECL:.*]]:2 = hlfir.declare %[[FP_ALLOCA]] {{.*}}fortran_attrs = #fir.var_attrs<allocatable>{{.*}}uniq_name = "_QFallocatable_exampleEp"

! Copy original p into firstprivate p
! HLFIR:     %[[FP_VAL0:.*]] = fir.load %[[FP_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR:     %[[FP_ADDR0:.*]] = fir.box_addr %[[FP_VAL0]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! HLFIR:     fir.convert %[[FP_ADDR0]]
! HLFIR:     arith.cmpi ne
! HLFIR:     fir.if %{{.*}} {
! HLFIR:       %[[ORIG_VAL2:.*]] = fir.load %[[ORIG_P]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR:       hlfir.assign %[[ORIG_VAL2]] to %[[FP_DECL]]#0 realloc : !fir.box<!fir.heap<!fir.array<?xi32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR:     }

! Use firstprivate p in: a = p + 1
! HLFIR:     %[[FP_VAL1:.*]] = fir.load %[[FP_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR:     %[[EXPR:.*]] = hlfir.elemental
! HLFIR:       hlfir.designate %[[FP_VAL1]]
! HLFIR:       fir.load
! HLFIR:       arith.addi
! HLFIR:       hlfir.yield_element
! HLFIR:     hlfir.assign %[[EXPR]] to %[[A]]#0 : !hlfir.expr<?xi32>, !fir.ref<!fir.array<4xi32>>
! HLFIR:     hlfir.destroy %[[EXPR]] : !hlfir.expr<?xi32>

! Cleanup firstprivate p
! HLFIR:     %[[FP_VAL2:.*]] = fir.load %[[FP_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR:     %[[FP_ADDR1:.*]] = fir.box_addr %[[FP_VAL2]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! HLFIR:     fir.convert %[[FP_ADDR1]]
! HLFIR:     arith.cmpi ne
! HLFIR:     fir.if %{{.*}} {
! HLFIR:       %[[FP_VAL3:.*]] = fir.load %[[FP_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR:       %[[FP_ADDR2:.*]] = fir.box_addr %[[FP_VAL3]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! HLFIR:       fir.freemem %[[FP_ADDR2]] : !fir.heap<!fir.array<?xi32>>
! HLFIR:       fir.store %{{.*}} to %[[FP_DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR:     }
! HLFIR:     omp.terminator
! HLFIR:   }
! HLFIR:   omp.terminator
! HLFIR: }

! Final cleanup of original p
! HLFIR: %[[ORIG_VAL3:.*]] = fir.load %[[ORIG_P]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR: %[[ORIG_ADDR1:.*]] = fir.box_addr %[[ORIG_VAL3]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! HLFIR: fir.convert %[[ORIG_ADDR1]]
! HLFIR: arith.cmpi ne
! HLFIR: fir.if %{{.*}} {
! HLFIR:   %[[ORIG_VAL4:.*]] = fir.load %[[ORIG_P]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR:   %[[ORIG_ADDR2:.*]] = fir.box_addr %[[ORIG_VAL4]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! HLFIR:   fir.freemem %[[ORIG_ADDR2]] : !fir.heap<!fir.array<?xi32>>
! HLFIR:   fir.store %{{.*}} to %[[ORIG_P]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! HLFIR: }

! HLFIR: return
! HLFIR: }

! FIR-LABEL: func.func @_QPallocatable_example()

! FIR:           %[[C1_I32:.*]] = arith.constant 1 : i32

! Original allocatable p declaration/allocation
! FIR:           %[[A_DECL:.*]] = fir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFallocatable_exampleEa"}
! FIR:           %[[ORIG_P:.*]] = fir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFallocatable_exampleEp"}
! FIR:           fir.allocmem !fir.array<?xi32>
! FIR:           fir.store %{{.*}} to %[[ORIG_P]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! FIR:           fir.call @_FortranAAssign

! FIR:           omp.parallel {

! Allocas for copyprivate slots
! FIR:             %[[A_BOX_SLOT:.*]] = fir.alloca !fir.box<!fir.array<4xi32>>
! FIR:             %[[FP_BOX_SLOT:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "p", pinned
! FIR:             %[[COPY_BOX_SLOT:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
! FIR:             %[[COPY_HEAP_SLOT:.*]] = fir.alloca !fir.heap<!fir.array<?xi32>>

! Copyprivate with three slots for broadcasting firstprivate data
! FIR:             omp.single copyprivate(%[[FP_BOX_SLOT]] -> @_workshare_copy_box_heap_Uxi32 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %[[COPY_BOX_SLOT]] -> @_workshare_copy_box_heap_Uxi32 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %[[COPY_HEAP_SLOT]] -> @_workshare_copy_heap_Uxi32 : !fir.ref<!fir.heap<!fir.array<?xi32>>>) {

! Check original allocation status and allocate firstprivate copy
! FIR:               %[[ORIG_BOX0:.*]] = fir.load %[[ORIG_P]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! FIR:               fir.box_addr %[[ORIG_BOX0]]
! FIR:               arith.cmpi ne
! FIR:               fir.if %{{.*}} {
! FIR:                 fir.load %[[ORIG_P]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! FIR:                 fir.box_dims
! FIR:                 fir.allocmem !fir.array<?xi32>
! FIR:                 fir.store %{{.*}} to %[[FP_BOX_SLOT]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! FIR:               } else {
! FIR:                 fir.zero_bits !fir.heap<!fir.array<?xi32>>
! FIR:                 fir.store %{{.*}} to %[[FP_BOX_SLOT]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! FIR:               }

! Declare firstprivate p and copy data from original
! FIR:               %[[FP_DECL:.*]] = fir.declare %[[FP_BOX_SLOT]] {fortran_attrs = #fir.var_attrs<allocatable>
! FIR:               fir.load %[[FP_DECL]]
! FIR:               fir.box_addr
! FIR:               arith.cmpi ne
! FIR:               fir.if %{{.*}} {
! FIR:                 fir.call @_FortranAAssign
! FIR:               }

! Store to copyprivate broadcast slots
! FIR:               fir.store %{{.*}} to %[[COPY_BOX_SLOT]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! FIR:               fir.store %{{.*}} to %[[COPY_HEAP_SLOT]] : !fir.ref<!fir.heap<!fir.array<?xi32>>>
! FIR:               omp.terminator
! FIR:             }

! After single: use copyprivate broadcast data for workshared computation
! FIR:             %[[FP_DECL2:.*]] = fir.declare %[[FP_BOX_SLOT]] {fortran_attrs = #fir.var_attrs<allocatable>
! FIR:             %[[COPY_BOX:.*]] = fir.load %[[COPY_BOX_SLOT]]

! Workshared loop: temp = p + 1
! FIR:             omp.wsloop {
! FIR:               omp.loop_nest (%[[I:.*]]) : index
! FIR:                 %[[SRC_VAL:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! FIR:                 %[[ADD_RES:.*]] = arith.addi %[[SRC_VAL]], %[[C1_I32]] : i32
! FIR:                 fir.store %[[ADD_RES]] to %{{.*}} : !fir.ref<i32>
! FIR:                 omp.yield
! FIR:             }

! Assignment of temp to a and cleanup (in omp.single nowait)
! FIR:             omp.single nowait {
! FIR:               fir.call @_FortranAAssign
! FIR:               fir.freemem
! Cleanup firstprivate p
! FIR:               fir.if %{{.*}} {
! FIR:                 fir.freemem
! FIR:               }
! FIR:               omp.terminator
! FIR:             }

! FIR:             omp.barrier
! FIR:             omp.terminator
! FIR:           }

! Cleanup original p
! FIR:           fir.if %{{.*}} {
! FIR:             fir.freemem
! FIR:           }
! FIR:           return

subroutine derived_type_example()
  implicit none

  type :: t
    integer :: x
  end type

  type(t) :: p(4)
  integer :: a(4)

  p%x = [1, 2, 3, 4]

  !$omp parallel workshare firstprivate(p)
    a = p%x + 1
  !$omp end parallel workshare
end subroutine

! FIR-LABEL: func.func @_QPderived_type_example()
! FIR:           %[[C1_I32:.*]] = arith.constant 1 : i32
! FIR:           %[[A_DECL:.*]] = fir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFderived_type_exampleEa"}
! FIR:           %[[ORIG_P:.*]] = fir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFderived_type_exampleEp"}
! FIR:           fir.call @_FortranAAssign
! FIR:           omp.parallel {

! Allocas for copyprivate slots
! FIR:             %[[A_BOX_SLOT:.*]] = fir.alloca !fir.box<!fir.array<4xi32>>
! FIR:             %[[P_BOX_SLOT:.*]] = fir.alloca !fir.box<!fir.array<4x!fir.type<{{.*}}>>>
! FIR:             %[[FP_ARRAY:.*]] = fir.alloca !fir.array<4x!fir.type<{{.*}}>> {bindc_name = "p", pinned
! FIR:             %[[HEAP_SLOT:.*]] = fir.alloca !fir.heap<!fir.array<4xi32>>

! Copyprivate with derived-type copy function and heap copy function
! FIR:             omp.single copyprivate(%[[FP_ARRAY]] -> @_workshare_copy_4xrec__QFderived_type_exampleTt : {{.*}}, %[[HEAP_SLOT]] -> @_workshare_copy_heap_4xi32 : {{.*}}) {

! Declare firstprivate p and copy original data
! FIR:               fir.declare %[[FP_ARRAY]]
! FIR:               fir.call @_FortranAAssign

! Allocate temp array for expression result
! FIR:               fir.allocmem !fir.array<4xi32>
! FIR:               fir.store %{{.*}} to %[[HEAP_SLOT]]
! FIR:               omp.terminator
! FIR:             }

! After single: declare firstprivate p and extract p%x via slice
! FIR:             %[[FP_DECL:.*]] = fir.declare %[[FP_ARRAY]]
! FIR:             fir.field_index x, !fir.type<{{.*}}>
! FIR:             fir.slice
! FIR:             %[[PX_ADDR:.*]] = fir.box_addr

! Load temp array from copyprivate slot
! FIR:             %[[HEAP_VAL:.*]] = fir.load %[[HEAP_SLOT]]
! FIR:             %[[TMP_DECL:.*]] = fir.declare %[[HEAP_VAL]]

! Workshared loop: temp = p%x + 1
! FIR:             omp.wsloop {
! FIR:               omp.loop_nest (%[[I:.*]]) : index
! FIR:                 %[[SRC_ELEM:.*]] = fir.array_coor %[[PX_ADDR]]
! FIR:                 %[[SRC_VAL:.*]] = fir.load %[[SRC_ELEM]] : !fir.ref<i32>
! FIR:                 %[[ADD_RES:.*]] = arith.addi %[[SRC_VAL]], %[[C1_I32]] : i32
! FIR:                 %[[DST_ELEM:.*]] = fir.array_coor %[[TMP_DECL]]
! FIR:                 fir.store %[[ADD_RES]] to %[[DST_ELEM]] : !fir.ref<i32>
! FIR:                 omp.yield

! Assignment of temp to a and cleanup (in omp.single nowait)
! FIR:             omp.single nowait {
! FIR:               fir.call @_FortranAAssign
! FIR:               fir.freemem
! FIR:               omp.terminator
! FIR:             }

! FIR:             omp.barrier
! FIR:             omp.terminator
! FIR:           }

! FIR:           return
