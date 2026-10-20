// RUN: fir-opt %s | FileCheck %s --check-prefix=PLAIN
// RUN: fir-opt %s -canonicalize | FileCheck %s --check-prefix=CANON

// PLAIN-LABEL: func @fold_shape_extents
// PLAIN:         fir.shape_extents
// CANON-LABEL:   func @fold_shape_extents
// CANON:         fir.fake_use %[[N:arg[0-9]+]] : index
func.func @fold_shape_extents(%n : index) {
  %sh = fir.shape %n : (index) -> !fir.shape<1>
  %e = fir.shape_extents %sh : (!fir.shape<1>) -> (index)
  fir.fake_use %e : index
  return
}

// PLAIN-LABEL: func @shape_extents_2d
// PLAIN:         fir.shape_extents
// CANON-LABEL:   func @shape_extents_2d
// CANON:         fir.fake_use %[[N1:arg[0-9]+]], %[[N2:arg[0-9]+]] : index, index
func.func @shape_extents_2d(%n1 : index, %n2 : index) {
  %sh = fir.shape %n1, %n2 : (index, index) -> !fir.shape<2>
  %e0, %e1 = fir.shape_extents %sh : (!fir.shape<2>) -> (index, index)
  fir.fake_use %e0, %e1 : index, index
  return
}

// PLAIN-LABEL: func @shape_extents_block_arg
// PLAIN:         fir.shape_extents
// CANON-LABEL:   func @shape_extents_block_arg
// CANON:         fir.shape_extents
func.func @shape_extents_block_arg(%pred : i1, %n1 : index, %n2 : index) {
  cf.cond_br %pred, ^bb1, ^bb2
^bb1:
  %sh1 = fir.shape %n1 : (index) -> !fir.shape<1>
  cf.br ^bb3(%sh1 : !fir.shape<1>)
^bb2:
  %sh2 = fir.shape %n2 : (index) -> !fir.shape<1>
  cf.br ^bb3(%sh2 : !fir.shape<1>)
^bb3(%phi : !fir.shape<1>):
  %e = fir.shape_extents %phi : (!fir.shape<1>) -> (index)
  fir.fake_use %e : index
  return
}

// Check for proper insertion of casting when types of 
// fir.shape ops and fir.shape_extents results do not match
// PLAIN-LABEL: func @fold_shape_extents_cast
// PLAIN:         fir.shape_extents
// CANON-LABEL:   func @fold_shape_extents_cast
// CANON-NOT:     fir.shape_extents
// CANON:         fir.convert %{{.*}} : (i64) -> index
// CANON:         fir.fake_use %{{.*}} : index
func.func @fold_shape_extents_cast(%e : i64) {
  %sh = fir.shape %e : (i64) -> !fir.shape<1>
  %ext = fir.shape_extents %sh : (!fir.shape<1>) -> (index)
  fir.fake_use %ext : index
  return
}