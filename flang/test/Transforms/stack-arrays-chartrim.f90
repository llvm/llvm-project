! The TRIM inlining (simplify-hlfir-intrinsics) always allocates its temporary
! on the heap, with a constant length when LEN(x) is constant, and leaves the
! placement to the stack-arrays or allocation-placement pass. Run the lowering
! output through the passes in between to check that bufferization produces a
! fir.freemem the StackArrays analysis can match to the allocmem, so that the
! constant-length temporaries end up on the stack, hoisted to the entry block,
! while the dynamic-length ones stay on the heap. The driver runs check the
! same thing through the real -O2 pipeline. The consumer is an external
! procedure so that the copy cannot be elided by inlining.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - \
! RUN: | fir-opt --simplify-hlfir-intrinsics \
! RUN:           --lower-hlfir-ordered-assignments \
! RUN:           --bufferize-hlfir \
! RUN:           --convert-hlfir-to-fir \
! RUN:           --stack-arrays \
! RUN: | FileCheck %s --check-prefixes=ALL,STACK
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - \
! RUN: | fir-opt --simplify-hlfir-intrinsics \
! RUN:           --lower-hlfir-ordered-assignments \
! RUN:           --bufferize-hlfir \
! RUN:           --convert-hlfir-to-fir \
! RUN:           --allocation-placement \
! RUN: | FileCheck %s --check-prefixes=ALL,PLACE
! RUN: %flang -S -O2 -fopenmp -fstack-arrays \
! RUN:   -mmlir --mlir-disable-threading \
! RUN:   -mmlir --mlir-print-ir-after=allocation-placement \
! RUN:   %s -o /dev/null 2>&1 | FileCheck %s --check-prefixes=ALL,STACK
! RUN: %flang -S -O2 -fopenmp -mllvm -enable-allocation-placement \
! RUN:   -mmlir --mlir-disable-threading \
! RUN:   -mmlir --mlir-print-ir-after=allocation-placement \
! RUN:   %s -o /dev/null 2>&1 | FileCheck %s --check-prefixes=ALL,PLACE

! ALL-LABEL: func.func @_QPfixed(
! ALL:         fir.alloca !fir.char<1,8>
! ALL-NOT:     fir.allocmem
! ALL:         fir.call @_QPsub(
! ALL-NOT:     fir.freemem
! ALL-NOT:     _FortranATrim
! ALL:         return
subroutine fixed(x)
  character(8) :: x
  call sub(trim(x))
end subroutine

! Two sites live at the same time get two stack temporaries.
! ALL-LABEL: func.func @_QPtwo(
! ALL:         fir.alloca !fir.char<1,8>
! ALL:         fir.alloca !fir.char<1,8>
! ALL-NOT:     fir.allocmem
! ALL:         fir.call @_QPsub2(
! ALL-NOT:     fir.freemem
! ALL:         return
subroutine two(x)
  character(8) :: x
  call sub2(trim(x), trim(x))
end subroutine

! The temporary is hoisted out of the loop, so it is allocated once per call
! of the procedure, not once per iteration.
! ALL-LABEL: func.func @_QPloop(
! ALL:         fir.alloca !fir.char<1,8>
! ALL:         fir.do_loop
! ALL-NOT:       fir.alloca
! ALL-NOT:       fir.allocmem
! ALL:           fir.call @_QPsub(
! ALL-NOT:       fir.freemem
! ALL:         return
subroutine loop(x, n)
  character(8) :: x
  integer :: n, i
  do i = 1, n
    call sub(trim(x))
  end do
end subroutine

! Inside a target region the temporary is hoisted no further than the
! region's entry block, so that each execution of the region gets its own.
! ALL-LABEL: func.func @_QPin_target(
! ALL-NOT:     fir.alloca !fir.char<1,8>
! ALL:         omp.target
! ALL-NOT:       fir.allocmem
! ALL:           fir.alloca !fir.char<1,8>
! ALL-NOT:       fir.allocmem
! ALL:           fir.call @_QPsub(
! ALL-NOT:       fir.freemem
! ALL:           omp.terminator
! ALL:         return
subroutine in_target(x)
  character(8) :: x
  !$omp target
  call sub(trim(x))
  !$omp end target
end subroutine

! Above the small-array threshold the default policy keeps the temporary on
! the heap; -fstack-arrays moves it to the stack.
! ALL-LABEL: func.func @_QPbig(
! STACK:       fir.alloca !fir.char<1,2048>
! STACK-NOT:   fir.allocmem
! PLACE-NOT:   fir.alloca !fir.char<1,2048>
! PLACE:       fir.allocmem !fir.char<1,2048>
! PLACE:       fir.freemem
! ALL-NOT:     _FortranATrim
! ALL:         return
subroutine big(x)
  character(2048) :: x
  call sub(trim(x))
end subroutine

! ALL-LABEL: func.func @_QPdynamic(
! ALL-NOT:     fir.alloca !fir.char
! ALL:         fir.allocmem !fir.char<1,?>
! ALL:         fir.call @_QPsub(
! ALL:         fir.freemem
! ALL-NOT:     _FortranATrim
! ALL:         return
subroutine dynamic(x)
  character(*) :: x
  call sub(trim(x))
end subroutine
