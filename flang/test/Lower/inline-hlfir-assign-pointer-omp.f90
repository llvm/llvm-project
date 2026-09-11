! Test inlining of hlfir.assign for pointer-based array assignment with OpenMP.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | fir-opt --inline-hlfir-assign | FileCheck %s --check-prefix=FIRST_INLINE
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | fir-opt --inline-hlfir-assign --bufferize-hlfir | FileCheck %s --check-prefix=BUFFERIZE
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | fir-opt --inline-hlfir-assign --bufferize-hlfir --inline-hlfir-assign | FileCheck %s --check-prefix=INLINE_HLFIR

subroutine test()
  integer,target,dimension(64):: a, b
  integer,pointer:: sum(:)
  integer :: i

  b = (/(i,i=1,64)/)

  !$omp parallel do private(sum)
  do i=1, 32
    sum => b(i:i+32)
    a(i:i+32) = sum
  end do
  !$omp end parallel do
end subroutine test

!! =============================================================================
!! After 1st InlineHLFIRAssign
!! =============================================================================
! FIRST_INLINE-LABEL: func.func @_QPtest
! FIRST_INLINE: omp.parallel
! FIRST_INLINE:   %[[SUM:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! FIRST_INLINE:   %[[LHS:.*]] = hlfir.designate %{{.*}} (%{{.*}}:%{{.*}}:%{{.*}}) shape %{{.*}} : (!fir.ref<!fir.array<64xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! FIRST_INLINE:   %[[LHS_ADDR:.*]] = fir.box_addr %[[LHS]]
! FIRST_INLINE:   %[[LHS_BASE:.*]] = fir.convert %[[LHS_ADDR]] : {{.*}} -> i64
! FIRST_INLINE:   fir.box_elesize %[[LHS]]
! FIRST_INLINE:   fir.box_dims %[[LHS]], %{{.*}}
! FIRST_INLINE:   %[[LHS_START:.*]] = arith.addi %[[LHS_BASE]], %{{.*}} : i64
! FIRST_INLINE:   %[[LHS_END:.*]] = arith.addi %[[LHS_BASE]], %{{.*}} : i64
! FIRST_INLINE:   %[[RHS_ADDR:.*]] = fir.box_addr %[[SUM]]
! FIRST_INLINE:   %[[RHS_BASE:.*]] = fir.convert %[[RHS_ADDR]] : {{.*}} -> i64
! FIRST_INLINE:   fir.box_elesize %[[SUM]]
! FIRST_INLINE:   fir.box_dims %[[SUM]], %{{.*}}
! FIRST_INLINE:   %[[RHS_START:.*]] = arith.addi %[[RHS_BASE]], %{{.*}} : i64
! FIRST_INLINE:   %[[RHS_END:.*]] = arith.addi %[[RHS_BASE]], %{{.*}} : i64
! FIRST_INLINE:   %[[CMP1:.*]] = arith.cmpi ult, %[[LHS_END]], %[[RHS_START]] : i64
! FIRST_INLINE:   %[[CMP2:.*]] = arith.cmpi ult, %[[RHS_END]], %[[LHS_START]] : i64
! FIRST_INLINE:   %[[DISJOINT:.*]] = arith.ori %[[CMP1]], %[[CMP2]] : i1
! FIRST_INLINE:   fir.if %[[DISJOINT]] {
! FIRST_INLINE:     fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
! FIRST_INLINE:       hlfir.designate %[[SUM]] (%{{.*}})
! FIRST_INLINE:       fir.load
! FIRST_INLINE:       hlfir.designate %[[LHS]] (%{{.*}})
! FIRST_INLINE:       hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ref<i32>
! FIRST_INLINE:     }
! FIRST_INLINE:   } else {
! FIRST_INLINE:     %[[ALLOC:.*]] = fir.allocmem !fir.array<?xi32>, %{{.*}} {bindc_name = ".tmp", uniq_name = ""}
! FIRST_INLINE:     %[[TEMP:.*]]:2 = hlfir.declare %[[ALLOC]](%{{.*}}) {uniq_name = ".tmp"}
! FIRST_INLINE:     fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
! FIRST_INLINE:       hlfir.designate %[[SUM]] (%{{.*}})
! FIRST_INLINE:       fir.load
! FIRST_INLINE:       hlfir.designate %[[TEMP]]#0 (%{{.*}})
! FIRST_INLINE:       hlfir.assign %{{.*}} to %{{.*}} temporary_lhs : i32, !fir.ref<i32>
! FIRST_INLINE:     }
! FIRST_INLINE:     fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
! FIRST_INLINE:       hlfir.designate %[[TEMP]]#0 (%{{.*}})
! FIRST_INLINE:       fir.load
! FIRST_INLINE:       hlfir.designate %[[LHS]] (%{{.*}})
! FIRST_INLINE:       hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ref<i32>
! FIRST_INLINE:     }
! FIRST_INLINE:     fir.freemem %{{.*}}
! FIRST_INLINE:   }


!! =============================================================================
!! After BufferizeHLFIR
!! =============================================================================
! BUFFERIZE-LABEL: func.func @_QPtest
! BUFFERIZE: omp.parallel
! BUFFERIZE:   fir.if %{{.*}} {
! BUFFERIZE:     fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
! BUFFERIZE:       hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ref<i32>
! BUFFERIZE:     }
! BUFFERIZE:   } else {
! BUFFERIZE:     %[[ALLOC:.*]] = fir.allocmem !fir.array<?xi32>, %{{.*}} {bindc_name = ".tmp", uniq_name = ""}
! BUFFERIZE:     %[[TEMP:.*]]:2 = hlfir.declare %[[ALLOC]](%{{.*}}) {uniq_name = ".tmp"}
! BUFFERIZE:     fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
! BUFFERIZE:       hlfir.designate %{{.*}} (%{{.*}})
! BUFFERIZE:       fir.load
! BUFFERIZE:       hlfir.designate %[[TEMP]]#0 (%{{.*}})
! BUFFERIZE:       hlfir.assign %{{.*}} to %{{.*}} temporary_lhs : i32, !fir.ref<i32>
! BUFFERIZE:     }
! BUFFERIZE:     fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
! BUFFERIZE:       hlfir.designate %[[TEMP]]#0 (%{{.*}})
! BUFFERIZE:       fir.load
! BUFFERIZE:       hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ref<i32>
! BUFFERIZE:     }
! BUFFERIZE:     fir.freemem %{{.*}}
! BUFFERIZE:   }

!! =============================================================================
!! After 2nd InlineHLFIRAssign
!! =============================================================================
! INLINE_HLFIR-LABEL: func.func @_QPtest
! INLINE_HLFIR: omp.parallel
! INLINE_HLFIR:   fir.if %{{.*}} {
! INLINE_HLFIR:     fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
! INLINE_HLFIR:       hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ref<i32>
! INLINE_HLFIR:     }
! INLINE_HLFIR:   } else {
! INLINE_HLFIR:     %[[ALLOC2:.*]] = fir.allocmem !fir.array<?xi32>, %{{.*}} {bindc_name = ".tmp", uniq_name = ""}
! INLINE_HLFIR:     %[[TEMP2:.*]]:2 = hlfir.declare %[[ALLOC2]](%{{.*}}) {uniq_name = ".tmp"}
! INLINE_HLFIR:     fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
! INLINE_HLFIR:       hlfir.designate %{{.*}} (%{{.*}})
! INLINE_HLFIR:       fir.load
! INLINE_HLFIR:       hlfir.designate %[[TEMP2]]#0 (%{{.*}})
! INLINE_HLFIR:       hlfir.assign %{{.*}} to %{{.*}} temporary_lhs : i32, !fir.ref<i32>
! INLINE_HLFIR:     }
! INLINE_HLFIR:     fir.do_loop %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} unordered {
! INLINE_HLFIR:       hlfir.designate %[[TEMP2]]#0 (%{{.*}})
! INLINE_HLFIR:       fir.load
! INLINE_HLFIR:       hlfir.designate %{{.*}} (%{{.*}})
! INLINE_HLFIR:       hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ref<i32>
! INLINE_HLFIR:     }
! INLINE_HLFIR:     fir.freemem %{{.*}}
! INLINE_HLFIR:   }

! INLINE_HLFIR-NOT: hlfir.as_expr
! INLINE_HLFIR-NOT: hlfir.apply
! INLINE_HLFIR-NOT: hlfir.destroy
