// Test code motion for affine if

// RUN: tco --promote-to-affine --disable-affine-promotion=false --affine-loop-invariant-code-motion --cse %s | FileCheck %s

!arr_d1 = type !fir.ref<!fir.array<?xf32>>
#arr_len = affine_map<()[j1,k1] -> (k1 - j1 + 1)>

// CHECK-LABEL: func @calc
func @calc(%a: !arr_d1, %v: f32) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %len = constant 100 : index
  %dims = fir.shape %len : (index) -> !fir.shape<1>

  fir.do_loop %i = %c1 to %len step %c1 {
    fir.do_loop %j = %c1 to %len step %c1 {
      fir.do_loop %k = %c1 to %len step %c1 {
        %im2 = subi %i, %c2 : index
        %cond = cmpi "sgt", %im2, %c0 : index
        fir.if %cond {
          %a_idx = fir.array_coor %a(%dims) %i
            : (!arr_d1, !fir.shape<1>, index) -> !fir.ref<f32>
          fir.store %v to %a_idx : !fir.ref<f32>
        }
          %aj_idx = fir.array_coor %a(%dims) %j
            : (!arr_d1, !fir.shape<1>, index) -> !fir.ref<f32>
          fir.store %v to %aj_idx : !fir.ref<f32>
          %ak_idx = fir.array_coor %a(%dims) %k
            : (!arr_d1, !fir.shape<1>, index) -> !fir.ref<f32>
          fir.store %v to %ak_idx : !fir.ref<f32>
      }
    }
  }
  // CHECK: affine.for
  // CHECK-NEXT:   subi
  // CHECK-NEXT:   affine.if
  // CHECK-NEXT:     affine.apply
  // CHECK-NEXT:     affine.store
  // CHECK-NEXT:   }
  // CHECK-NEXT:   affine.for
  // CHECK-NEXT:     affine.apply
  // CHECK-NEXT:     affine.store
  // CHECK-NEXT:     affine.for
  // CHECK-NEXT:       affine.apply
  // CHECK-NEXT:       affine.store
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  return
}
