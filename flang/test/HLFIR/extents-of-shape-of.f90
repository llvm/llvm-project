! RUN: bbc -emit-fir -hlfir %s -o - | FileCheck %s
subroutine foo(a, b)
  real :: a(:, :), b(:, :)
  interface
    elemental subroutine elem_sub(x)
      real, intent(in) :: x
    end subroutine
  end interface
  call elem_sub(matmul(a, b))
end subroutine
! CHECK-LABEL: func.func @_QPfoo
! CHECK:           %[[A_ARG:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "a"}
! CHECK:           %[[B_ARG:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "b"}
! CHECK-DAG:     %[[A_VAR:.*]]:2 = hlfir.declare %[[A_ARG]]
! CHECK-DAG:     %[[B_VAR:.*]]:2 = hlfir.declare %[[B_ARG]]
! CHECK-NEXT:    %[[MUL:.*]] = hlfir.matmul %[[A_VAR]]#0 %[[B_VAR]]#0
! CHECK-NEXT:    %[[SHAPE:.*]] = hlfir.shape_of %[[MUL]] : (!hlfir.expr<?x?xf32>) -> !fir.shape<2>
! CHECK-NEXT:    %[[EXT0:.*]] = hlfir.get_extent %[[SHAPE]] {dim = 0 : index} : (!fir.shape<2>) -> index
! CHECK-NEXT:    %[[EXT1:.*]] = hlfir.get_extent %[[SHAPE]] {dim = 1 : index} : (!fir.shape<2>) -> index
! CHECK-NEXT:    %[[C1:.*]] = arith.constant 1 : index
! CHECK-NEXT:    fir.do_loop %[[ARG2:.*]] = %[[C1]] to %[[EXT1]] step %[[C1]] {
! CHECK-NEXT:      fir.do_loop %[[ARG3:.*]] = %[[C1]] to %[[EXT0]] step %[[C1]] {
! CHECK-NEXT:        %[[ELE:.*]] = hlfir.apply %[[MUL]], %[[ARG3]], %[[ARG2]] : (!hlfir.expr<?x?xf32>, index, index) -> f32
! CHECK-NEXT:        %[[ASSOC:.*]]:3 = hlfir.associate %[[ELE]] {uniq_name = "adapt.valuebyref"} : (f32) -> (!fir.ref<f32>, !fir.ref<f32>, i1)
! CHECK-NEXT:        fir.call
! CHECK-NEXT:        hlfir.end_associate
! CHECK-NEXT:      }
! CHECK-NEXT:    }
! CHECK-NEXT:    hlfir.destroy %[[MUL]]
! CHECK-NEXT:    return
! CHECK-NEXT:  }
