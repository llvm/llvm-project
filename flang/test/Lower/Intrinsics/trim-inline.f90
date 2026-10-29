! RUN: %flang_fc1 -emit-fir -O1 -o - %s | FileCheck %s

! What TRIM of a character variable becomes at -O1 after bufferization: an
! inline LEN_TRIM scan and one memmove into a temporary placed by the
! allocation policy, with the runtime call kept only in the fallback cases.
! The consumer is an external procedure so that the copy cannot be elided
! by inlining.

! A constant length within the small-array threshold: constant-size stack
! temporary hoisted to the entry block, nothing freed.
! CHECK-LABEL: func.func @_QPstack(
! CHECK:         fir.alloca !fir.char<1,8> {{.*}}.tmp.trim
! CHECK-NOT:     fir.allocmem
! CHECK:         fir.iterate_while
! CHECK:           arith.cmpi eq, %{{.*}}, %c32{{.*}} : i8
! CHECK:         "llvm.intr.memmove"
! CHECK:         fir.call @_QPsub(
! CHECK-NOT:     fir.freemem
! CHECK-NOT:     _FortranATrim
! CHECK:         return
subroutine stack(x)
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

! A constant length above the small-array threshold also goes to the heap.
! CHECK-LABEL: func.func @_QPbig(
! CHECK-NOT:     fir.alloca !fir.char<1,2048> {{.*}}.tmp.trim
! CHECK:         fir.allocmem !fir.char<1,?>(%{{.*}} : index) {{.*}}.tmp.trim
! CHECK:         fir.freemem
! CHECK-NOT:     _FortranATrim
! CHECK:         return
subroutine big(x)
  character(2048) :: x
  call sub(trim(x))
end subroutine

! Two sites that fit the threshold individually but not together: one stack
! temporary, the other on the heap, so the frame does not grow with the
! number of sites.
! CHECK-LABEL: func.func @_QPbudget(
! CHECK:         fir.alloca !fir.char<1,600> {{.*}}.tmp.trim
! CHECK-NOT:     fir.alloca !fir.char<1,600>
! CHECK:         fir.allocmem !fir.char<1,?>(%{{.*}} : index) {{.*}}.tmp.trim
! CHECK-NOT:     fir.allocmem
! CHECK:         fir.call @_QPsub2(
! CHECK:         fir.freemem
! CHECK-NOT:     _FortranATrim
! CHECK:         return
subroutine budget(x)
  character(600) :: x
  call sub2(trim(x), trim(x))
end subroutine

! The stack temporary is hoisted out of a loop that is lowered to a CFG, so
! it is allocated once per call of the procedure, not once per iteration.
! CHECK-LABEL: func.func @_QPloop(
! CHECK:         fir.alloca !fir.char<1,8> {{.*}}.tmp.trim
! CHECK:         cf.br
! CHECK-NOT:     fir.alloca
! CHECK:         "llvm.intr.memmove"
! CHECK:         fir.call @_QPsub(
! CHECK-NOT:     fir.alloca
! CHECK-NOT:     _FortranATrim
! CHECK:         return
subroutine loop(x, n)
  character(8) :: x
  integer :: n, i
  i = 0
  do while (i < n)
    call sub(trim(x))
    i = i + 1
  end do
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
