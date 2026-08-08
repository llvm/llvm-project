! Test inlining of hlfir.assign for FORALL with threadprivate array self-assignment.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | fir-opt --inline-hlfir-assign --lower-hlfir-ordered-assignments --lower-hlfir-intrinsics --bufferize-hlfir | FileCheck %s --check-prefix=BUFFERIZE
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | fir-opt --inline-hlfir-assign --lower-hlfir-ordered-assignments --lower-hlfir-intrinsics --bufferize-hlfir --inline-hlfir-assign | FileCheck %s --check-prefix=INLINE_HLFIR

module m1
  integer, parameter :: k2 = 10, k4 = 100
  integer :: a(2, k2, 2, k4)
  !$omp threadprivate(a)
contains
  subroutine s1()
    integer :: n2, n3, n
    forall (n2=1:2, n3=1:2)
      a(n2,:,n3,:) = a(n2,:,n3,:) + reshape([(n, n=1, k2*k4)], [k2, k4])
    end forall
  end subroutine
end module

!! =============================================================================
!! After BufferizeHLFIR
!! =============================================================================
! BUFFERIZE-LABEL: func.func @_QMm1Ps1

! Value stack creation:
! BUFFERIZE: %[[VSTACK:.*]] = fir.call @_FortranACreateValueStack

! Phase 1: evaluate RHS into temp and push to value stack
! BUFFERIZE: fir.do_loop
! BUFFERIZE:   fir.do_loop
! BUFFERIZE:     fir.allocmem !fir.array<10x100xi32>
! BUFFERIZE:     hlfir.declare
! BUFFERIZE:     fir.do_loop {{.*}} unordered
! BUFFERIZE:       fir.do_loop {{.*}} unordered
! BUFFERIZE:         hlfir.assign %{{.*}} to %{{.*}} temporary_lhs : i32, !fir.ref<i32>
! BUFFERIZE:       }
! BUFFERIZE:     }
! BUFFERIZE:     fir.call @_FortranAPushValue
! BUFFERIZE:     fir.freemem

! BUFFERIZE: fir.do_loop
! BUFFERIZE:   fir.do_loop
! BUFFERIZE:     fir.call @_FortranAValueAt(%[[VSTACK]], %{{.*}}, %{{.*}})
! BUFFERIZE:     %[[RHS:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<10x100xi32>>>>
! BUFFERIZE:     %[[LHS:.*]] = hlfir.designate %{{.*}} (%{{.*}}, %{{.*}}:%{{.*}}:%{{.*}}, %{{.*}}, %{{.*}}:%{{.*}}:%{{.*}}) shape %{{.*}}
! BUFFERIZE:     hlfir.assign %[[RHS]] to %[[LHS]] : !fir.box<!fir.heap<!fir.array<10x100xi32>>>, !fir.box<!fir.array<10x100xi32>>

! BUFFERIZE: fir.call @_FortranADestroyValueStack(%[[VSTACK]])

!! =============================================================================
!! After InlineHLFIRAssign
!! =============================================================================
! INLINE_HLFIR-LABEL: func.func @_QMm1Ps1

! Value stack creation:
! INLINE_HLFIR: %[[VSTACK:.*]] = fir.call @_FortranACreateValueStack

! Phase 2 is now inlined with address-based disjointness check:
! INLINE_HLFIR: fir.call @_FortranAValueAt(%[[VSTACK]], %{{.*}}, %{{.*}})
! INLINE_HLFIR: %[[RHS:.*]] = fir.load %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<10x100xi32>>>>
! INLINE_HLFIR: %[[LHS:.*]] = hlfir.designate %{{.*}} (%{{.*}}, %{{.*}}:%{{.*}}:%{{.*}}, %{{.*}}, %{{.*}}:%{{.*}}:%{{.*}}) shape %{{.*}}

! Address-based disjointness check:
! INLINE_HLFIR: %[[LHS_ADDR:.*]] = fir.box_addr %[[LHS]]
! INLINE_HLFIR: %[[LHS_BASE:.*]] = fir.convert %[[LHS_ADDR]] : {{.*}} -> i64
! INLINE_HLFIR: fir.box_elesize %[[LHS]]
! INLINE_HLFIR: fir.box_dims %[[LHS]], %{{.*}}
! INLINE_HLFIR: fir.box_dims %[[LHS]], %{{.*}}
! INLINE_HLFIR: %[[LHS_START:.*]] = arith.addi %[[LHS_BASE]], %{{.*}} : i64
! INLINE_HLFIR: %[[LHS_END:.*]] = arith.addi %[[LHS_BASE]], %{{.*}} : i64
! INLINE_HLFIR: %[[RHS_ADDR:.*]] = fir.box_addr %[[RHS]]
! INLINE_HLFIR: %[[RHS_BASE:.*]] = fir.convert %[[RHS_ADDR]] : {{.*}} -> i64
! INLINE_HLFIR: fir.box_elesize %[[RHS]]
! INLINE_HLFIR: fir.box_dims %[[RHS]], %{{.*}}
! INLINE_HLFIR: fir.box_dims %[[RHS]], %{{.*}}
! INLINE_HLFIR: %[[RHS_START:.*]] = arith.addi %[[RHS_BASE]], %{{.*}} : i64
! INLINE_HLFIR: %[[RHS_END:.*]] = arith.addi %[[RHS_BASE]], %{{.*}} : i64
! INLINE_HLFIR: %[[CMP1:.*]] = arith.cmpi ult, %[[LHS_END]], %[[RHS_START]] : i64
! INLINE_HLFIR: %[[CMP2:.*]] = arith.cmpi ult, %[[RHS_END]], %[[LHS_START]] : i64
! INLINE_HLFIR: %[[DISJOINT:.*]] = arith.ori %[[CMP1]], %[[CMP2]] : i1

! direct element-wise copy
! INLINE_HLFIR: fir.if %[[DISJOINT]] {
! INLINE_HLFIR:   fir.do_loop {{.*}} unordered
! INLINE_HLFIR:     fir.do_loop {{.*}} unordered
! INLINE_HLFIR:       hlfir.designate %[[RHS]]
! INLINE_HLFIR:       fir.load
! INLINE_HLFIR:       hlfir.designate %[[LHS]]
! INLINE_HLFIR:       hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ref<i32>
! INLINE_HLFIR:     }
! INLINE_HLFIR:   }

! copy through temporary
! INLINE_HLFIR: } else {
! INLINE_HLFIR:   %[[TALLOC:.*]] = fir.allocmem !fir.array<10x100xi32>
! INLINE_HLFIR:   %[[TEMP:.*]]:2 = hlfir.declare %[[TALLOC]](%{{.*}}) {uniq_name = ".tmp"}
! INLINE_HLFIR:   fir.do_loop {{.*}} unordered
! INLINE_HLFIR:     fir.do_loop {{.*}} unordered
! INLINE_HLFIR:       hlfir.designate %[[RHS]]
! INLINE_HLFIR:       fir.load
! INLINE_HLFIR:       hlfir.designate %[[TEMP]]#0
! INLINE_HLFIR:       hlfir.assign %{{.*}} to %{{.*}} temporary_lhs : i32, !fir.ref<i32>
! INLINE_HLFIR:     }
! INLINE_HLFIR:   }
! INLINE_HLFIR:   fir.do_loop {{.*}} unordered
! INLINE_HLFIR:     fir.do_loop {{.*}} unordered
! INLINE_HLFIR:       hlfir.designate %[[TEMP]]#0
! INLINE_HLFIR:       fir.load
! INLINE_HLFIR:       hlfir.designate %[[LHS]]
! INLINE_HLFIR:       hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ref<i32>
! INLINE_HLFIR:     }
! INLINE_HLFIR:   }
! INLINE_HLFIR:   fir.freemem %{{.*}}
! INLINE_HLFIR: }

! INLINE_HLFIR: fir.call @_FortranADestroyValueStack(%[[VSTACK]])

! INLINE_HLFIR-NOT: hlfir.as_expr
! INLINE_HLFIR-NOT: hlfir.apply
! INLINE_HLFIR-NOT: hlfir.destroy
