// Test loop fusion after affine promotion

// RUN: tco --promote-to-affine --disable-affine-promotion=false --affine-loop-invariant-code-motion --cse --affine-loop-fusion --simplify-affine-structures --cse --memref-dataflow-opt  %s | FileCheck %s

!arr_d1 = type !fir.ref<!fir.array<?xf32>>
#arr_len = affine_map<()[j1,k1] -> (k1 - j1 + 1)>

// CHECK-LABEL: func @calc
func @calc(%a1: !arr_d1, %a2: !arr_d1, %a3: !arr_d1) {
  %c1 = constant 1 : index
  %c0 = constant 0 : index
  %len = constant 100 : index
  %dims = fir.shape %len : (index) -> !fir.shape<1>
  %siz = affine.apply #arr_len()[%c1,%len]
  %t1 = fir.alloca !fir.array<?xf32>, %siz

  fir.do_loop %i = %c1 to %len step %c1 {
    %a1_idx = fir.array_coor %a1(%dims) %i
            : (!arr_d1, !fir.shape<1>, index) -> !fir.ref<f32>
    %a1_v = fir.load %a1_idx : !fir.ref<f32>

    %a2_idx = fir.array_coor %a2(%dims) %i
            : (!arr_d1, !fir.shape<1>, index) -> !fir.ref<f32>
    %a2_v = fir.load %a2_idx : !fir.ref<f32>

    %v = addf %a1_v, %a2_v : f32
    %t1_idx = fir.array_coor %t1(%dims) %i
            : (!arr_d1, !fir.shape<1>, index) -> !fir.ref<f32>

    fir.store %v to %t1_idx : !fir.ref<f32>
  }
  fir.do_loop %i = %c1 to %len step %c1 {
    %t1_idx = fir.array_coor %t1(%dims) %i
            : (!arr_d1, !fir.shape<1>, index) -> !fir.ref<f32>
    %t1_v = fir.load %t1_idx : !fir.ref<f32>

    %a2_idx = fir.array_coor %a2(%dims) %i
            : (!arr_d1, !fir.shape<1>, index) -> !fir.ref<f32>
    %a2_v = fir.load %a2_idx : !fir.ref<f32>

    %v = mulf %t1_v, %a2_v : f32
    %a3_idx = fir.array_coor %a3(%dims) %i
            : (!arr_d1, !fir.shape<1>, index) -> !fir.ref<f32>

    fir.store %v to %a3_idx : !fir.ref<f32>
  }
// CHECK:  affine.for %{{.*}} = 1 to 101 {
// CHECK-NEXT:    affine.apply affine_map<(d0) -> (d0 - 1)>(%{{.*}})
// CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<?xf32>
// CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<?xf32>
// CHECK-NEXT:    addf
// CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<?xf32>
// CHECK-NEXT:    mulf
// CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<?xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
  return
}
