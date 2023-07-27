! RUN: bbc -emit-hlfir %s -o - | FileCheck --check-prefix CHECK-ALL --check-prefix CHECK-HLFIR %s
! RUN: bbc -emit-hlfir %s -o - | fir-opt --lower-hlfir-intrinsics | fir-opt --bufferize-hlfir | fir-opt --convert-hlfir-to-fir | FileCheck --check-prefix CHECK-ALL --check-prefix CHECK-FIR %s
subroutine foo(a, b)
  real :: a(2, 2), b(:, :)
  interface
    elemental subroutine elem_sub(x)
      real, intent(in) :: x
    end subroutine
  end interface
  call elem_sub(matmul(a, b))
end subroutine
! CHECK-ALL-LABEL:   func.func @_QPfoo
! CHECK-ALL:             %[[A_ARG:.*]]: !fir.ref<!fir.array<2x2xf32>> {fir.bindc_name = "a"}
! CHECK-ALL:             %[[B_ARG:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "b"}

! CHECK-HLFIR-DAG:     %[[A_VAR:.*]]:2 = hlfir.declare %[[A_ARG]]
! CHECK-HLFIR-DAG:     %[[B_VAR:.*]]:2 = hlfir.declare %[[B_ARG]]
! CHECK-HLFIR-NEXT:    %[[MUL:.*]] = hlfir.matmul %[[A_VAR]]#0 %[[B_VAR]]#0
! CHECK-HLFIR-NEXT:    %[[SHAPE:.*]] = hlfir.shape_of %[[MUL]] : (!hlfir.expr<2x?xf32>) -> !fir.shape<2>
! CHECK-HLFIR-NEXT:    %[[EXT0:.*]] = arith.constant 2 : index
! CHECK-HLFIR-NEXT:    %[[EXT1:.*]] = hlfir.get_extent %[[SHAPE]] {dim = 1 : index} : (!fir.shape<2>) -> index
! CHECK-HLFIR-NEXT:    %[[C1:.*]] = arith.constant 1 : index
! CHECK-HLFIR-NEXT:    fir.do_loop %[[ARG2:.*]] = %[[C1]] to %[[EXT1]] step %[[C1]] unordered {
! CHECK-HLFIR-NEXT:      fir.do_loop %[[ARG3:.*]] = %[[C1]] to %[[EXT0]] step %[[C1]] unordered {
! CHECK-HLFIR-NEXT:        %[[ELE:.*]] = hlfir.apply %[[MUL]], %[[ARG3]], %[[ARG2]] : (!hlfir.expr<2x?xf32>, index, index) -> f32
! CHECK-HLFIR-NEXT:        %[[ASSOC:.*]]:3 = hlfir.associate %[[ELE]] {uniq_name = "adapt.valuebyref"} : (f32) -> (!fir.ref<f32>, !fir.ref<f32>, i1)
! CHECK-HLFIR-NEXT:        fir.call
! CHECK-HLFIR-NEXT:        hlfir.end_associate
! CHECK-HLFIR-NEXT:      }
! CHECK-HLFIR-NEXT:    }
! CHECK-HLFIR-NEXT:    hlfir.destroy %[[MUL]]

! ...
! CHECK-FIR:           fir.call @_FortranAMatmul
! CHECK-FIR-NEXT:      %[[MUL:.*]] = fir.load %[[MUL_BOX:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK-FIR-NEXT:      %[[C0:.*]] = arith.constant 0 : index
! CHECK-FIR-NEXT:      %[[DIMS0:.*]]:3 = fir.box_dims %[[MUL]], %[[C0]]
! CHECK-FIR-NEXT:      %[[C1:.*]] = arith.constant 1 : index
! CHECK-FIR-NEXT:      %[[DIMS1:.*]]:3 = fir.box_dims %[[MUL]], %[[C1]]
! ...
! CHECK-FIR:           %[[SHAPE:.*]] = fir.shape %[[DIMS0]]#1, %[[DIMS1]]#1
! CHECK-FIR-NEXT:      %[[C2:.*]] = arith.constant 2 : index
! CHECK-FIR-NEXT:      %[[C1_1:.*]] = arith.constant 1 : index
! CHECK-FIR-NEXT:      fir.do_loop %[[ARG2:.*]] = %[[C1_1]] to %[[DIMS1]]#1 step %[[C1_1]] unordered {
! CHECK-FIR-NEXT:        fir.do_loop %[[ARG3:.*]] = %[[C1_1]] to %[[C2]] step %[[C1_1]] unordered {
! ...

! CHECK-ALL:           return
! CHECK-ALL-NEXT:    }
