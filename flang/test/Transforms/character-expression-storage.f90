! RUN: bbc -emit-fir %s -o - | fir-opt --canonicalize --allocation-placement | FileCheck %s
! RUN: bbc -emit-hlfir %s -o - | fir-opt --inline-elementals | FileCheck %s --check-prefix=INLINE
! RUN: bbc -emit-hlfir %s -o - | fir-opt --lower-hlfir-ordered-assignments | FileCheck %s --check-prefix=WHERE
! RUN: bbc -emit-hlfir %s -o - | fir-opt --inline-elementals --opt-bufferization | FileCheck %s --check-prefix=OPT

! A statement in an unstructured loop must reclaim its character temporaries
! before the backedge, including when the statement is under an IF.
subroutine scalar_loop(s, n)
  character(*) :: s
  integer :: n, i
  i = 0
  do while (i < n)
    i = i + 1
    if (i /= 2) call consume(s(1:i) // '')
  end do
end subroutine
! CHECK-LABEL: func.func @_QPscalar_loop
! CHECK: llvm.intr.stacksave
! CHECK: fir.alloca !fir.char<1,?>
! CHECK: fir.call @_QPconsume
! CHECK: llvm.intr.stackrestore
! CHECK-NOT: fir.allocmem !fir.char
! CHECK: return

! After elemental fusion, the scalar scratch buffers of an element are
! reclaimed before the next iteration.
subroutine concat_array(a, b, c)
  character(*) :: a(:), b(:), c(:)
  c = a // b // '!'
end subroutine
! CHECK-LABEL: func.func @_QPconcat_array
! CHECK: fir.do_loop
! CHECK: llvm.intr.stacksave
! CHECK: fir.alloca !fir.char<1,?>
! CHECK: fir.alloca !fir.char<1,?>
! CHECK: llvm.intr.stackrestore
! CHECK-NOT: fir.allocmem !fir.char
! CHECK: return

! The concatenation becomes an intermediate scalar expression when the
! elementals are fused. It needs cleanup even though it is not yielded.
subroutine compare_array(a, b, c, result)
  character(*) :: a(:), b(:), c(:)
  logical :: result(:)
  result = a // b == c
end subroutine
! CHECK-LABEL: func.func @_QPcompare_array
! CHECK: fir.alloca !fir.char<1,?>
! CHECK: fir.do_loop
! CHECK-NOT: fir.allocmem !fir.char
! CHECK-NOT: fir.alloca !fir.char
! CHECK: return

! Masked assignment recursively inlines elemental bodies through a different
! path. Both the intermediate and final scalar results need cleanup.
subroutine masked_concat(a, b, c, mask)
  character(*) :: a(:), b(:), c(:)
  logical :: mask(:)
  where (mask) c = a // b // '!'
end subroutine
! CHECK-LABEL: func.func @_QPmasked_concat
! CHECK: fir.alloca !fir.char<1,?>
! CHECK-NOT: fir.allocmem !fir.char
! CHECK: return

! Inlining appends the producer cleanup to the consumer cleanup. Both
! temporaries are destroyed exactly once after the final value is consumed.
! INLINE-LABEL: func.func @_QPconcat_array
! INLINE: %[[INNER:.*]] = hlfir.concat
! INLINE: %[[OUTER:.*]] = hlfir.concat %[[INNER]]
! INLINE: hlfir.yield_element %[[OUTER]] : {{.*}} cleanup {
! INLINE-NEXT: hlfir.destroy %[[OUTER]]
! INLINE-NEXT: hlfir.destroy %[[INNER]]
! INLINE-NEXT: }

! The logical consumer does not transfer the character result to its yield.
! INLINE-LABEL: func.func @_QPcompare_array
! INLINE: %[[TEMP:.*]] = hlfir.concat
! INLINE: hlfir.cmpchar
! INLINE: hlfir.yield_element {{.*}} cleanup {
! INLINE-NEXT: hlfir.destroy %[[TEMP]]
! INLINE-NEXT: }

! Exercise recursive inlining directly, without the inline-elementals pass.
! WHERE-LABEL: func.func @_QPmasked_concat
! WHERE: %[[INNER:.*]] = hlfir.concat
! WHERE: %[[OUTER:.*]] = hlfir.concat %[[INNER]]
! WHERE: hlfir.assign %[[OUTER]]
! WHERE-NEXT: hlfir.destroy %[[OUTER]]
! WHERE-NEXT: hlfir.destroy %[[INNER]]
! WHERE-NOT: hlfir.destroy
! WHERE: return

subroutine parenthesized_compare(a, b, c, result)
  character(*) :: a(:), b(:), c(:)
  logical :: result(:)
  result = (a // b) == c
end subroutine
! INLINE-LABEL: func.func @_QPparenthesized_compare
! INLINE: %[[TEMP:.*]] = hlfir.concat
! INLINE: %[[ALIAS:.*]] = hlfir.no_reassoc %[[TEMP]]
! INLINE-NOT: hlfir.destroy
! INLINE: hlfir.cmpchar
! INLINE: hlfir.yield_element {{.*}} cleanup {
! INLINE-NEXT: hlfir.destroy %[[TEMP]]
! INLINE-NEXT: }
! CHECK-LABEL: func.func @_QPparenthesized_compare
! CHECK: fir.alloca !fir.char<1,?>
! CHECK: fir.do_loop
! CHECK-NOT: fir.allocmem !fir.char
! CHECK-NOT: fir.alloca !fir.char
! CHECK: return

subroutine masked_parentheses(a, b, c, mask)
  character(*) :: a(:), b(:), c(:)
  logical :: mask(:)
  where (mask) c = (a // b)
end subroutine
! WHERE-LABEL: func.func @_QPmasked_parentheses
! WHERE: %[[TEMP:.*]] = hlfir.concat
! WHERE: %[[ALIAS:.*]] = hlfir.no_reassoc %[[TEMP]]
! WHERE: hlfir.assign %[[ALIAS]]
! WHERE-NEXT: hlfir.destroy %[[TEMP]]
! WHERE-NOT: hlfir.destroy
! WHERE: return
! CHECK-LABEL: func.func @_QPmasked_parentheses
! CHECK: fir.alloca !fir.char<1,?>
! CHECK-NOT: fir.allocmem !fir.char
! CHECK-NOT: fir.alloca !fir.char
! CHECK: return

! Optimized bufferization must retain intermediate character cleanup after
! assigning the logical element directly into the output array.
! OPT-LABEL: func.func @_QPcompare_array
! OPT-NOT: hlfir.elemental
! OPT: fir.do_loop
! OPT: %[[TEMP:.*]] = hlfir.concat
! OPT: hlfir.cmpchar
! OPT: hlfir.assign
! OPT-NEXT: hlfir.destroy %[[TEMP]]
! OPT-NOT: hlfir.elemental
! OPT: return
