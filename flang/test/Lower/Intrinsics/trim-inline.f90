! RUN: %flang_fc1 -emit-fir -O1 -o - %s | FileCheck %s

! What TRIM of a character variable becomes at -O1 after bufferization: an
! inline LEN_TRIM scan and one memmove into a heap temporary, with the runtime
! call kept only in the fallback cases. The allocation placement passes, which
! run later, decide whether the temporary moves to the stack
! (see Transforms/stack-arrays-chartrim.f90). The consumer is an external
! procedure so that the copy cannot be elided by inlining.

! A constant length: the temporary is allocated with that constant length so
! that the allocation has no operands, and freed after the call.
! CHECK-LABEL: func.func @_QPfixed(
! CHECK:         %[[SCAN:.*]]:2 = fir.iterate_while
! CHECK:           arith.cmpi eq, %{{.*}}, %c32{{.*}} : i8
! CHECK:         %[[LT:.*]] = arith.select %[[SCAN]]#0, %c0{{.*}}, %{{.*}} : index
! CHECK:         %[[MEM:.*]] = fir.allocmem !fir.char<1,8> {{.*}}.tmp.trim
! CHECK:         %[[DYN:.*]] = fir.convert %[[MEM]] : (!fir.heap<!fir.char<1,8>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:         %[[TMP:.*]] = fir.declare %[[DYN]] typeparams %[[LT]] {{.*}}.tmp.trim
! CHECK:         "llvm.intr.memmove"
! CHECK:         fir.call @_QPsub(
! CHECK:         fir.freemem %{{.*}} : !fir.heap<!fir.char<1,?>>
! CHECK-NOT:     _FortranATrim
! CHECK:         return
subroutine fixed(x)
  character(8) :: x
  call sub(trim(x))
end subroutine

! Assumed length: heap temporary sized LEN_TRIM(x). The same length sizes
! the allocation and the boxchar passed to the consumer, and the same buffer
! is freed after the call.
! CHECK-LABEL: func.func @_QPheap(
! CHECK-NOT:     fir.alloca !fir.char
! CHECK:         %[[SCAN:.*]]:2 = fir.iterate_while
! CHECK:         %[[LT:.*]] = arith.select %[[SCAN]]#0, %c0{{.*}}, %{{.*}} : index
! CHECK:         %[[MEM:.*]] = fir.allocmem !fir.char<1,?>(%[[LT]] : index) {{.*}}.tmp.trim
! CHECK:         %[[TMP:.*]] = fir.declare %[[MEM]] typeparams %[[LT]] {{.*}}.tmp.trim
! CHECK:         %[[BOX:.*]] = fir.emboxchar %[[TMP]], %[[LT]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:         "llvm.intr.memmove"
! CHECK:         %[[REF:.*]] = fir.convert %[[TMP]] : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:         fir.call @_QPsub(%[[BOX]])
! CHECK:         %[[HEAP:.*]] = fir.convert %[[REF]] : (!fir.ref<!fir.char<1,?>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:         fir.freemem %[[HEAP]] : !fir.heap<!fir.char<1,?>>
! CHECK-NOT:     _FortranATrim
! CHECK:         return
subroutine heap(x)
  character(*) :: x
  call sub(trim(x))
end subroutine

! Fallback: a volatile operand keeps the runtime call.
! CHECK-LABEL: func.func @_QPvol(
! CHECK-NOT:     fir.iterate_while
! CHECK-NOT:     .tmp.trim
! CHECK:         fir.call @_FortranATrim(
! CHECK:         fir.call @_QPsub(
subroutine vol(x)
  character(8), volatile :: x
  call sub(trim(x))
end subroutine

! Fallback: an expression operand has no storage to designate and keeps the
! runtime call.
! CHECK-LABEL: func.func @_QPexpr(
! CHECK-NOT:     .tmp.trim
! CHECK:         fir.call @_FortranATrim(
! CHECK:         fir.call @_QPsub(
subroutine expr(x, y)
  character(*) :: x, y
  call sub(trim(x // y))
end subroutine
