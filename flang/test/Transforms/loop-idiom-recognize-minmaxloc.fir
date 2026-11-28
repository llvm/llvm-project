// RUN: fir-opt %s --flang-loop-idiom | FileCheck %s

// Tests for elimination of redundant min/max tracking in equality-mask loops.
#loop_vectorize = #llvm.loop_vectorize<disable = false>
#loop_annotation = #llvm.loop_annotation<vectorize = #loop_vectorize>



// 1D minloc with equality mask.
func.func @test_1d_fir_cmpi_eq(%input: !fir.ref<!fir.array<100xi32>>, %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c2147483647_i32, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_1d_fir_cmpi_eq(
// CHECK:         fir.do_loop {{.*}} unordered iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:           arith.cmpi eq
// CHECK-NOT:       arith.cmpi slt
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i32, i1) {
// CHECK:             %[[FALSE:.*]] = arith.constant false
// CHECK:             fir.result %{{.*}}, %[[FALSE]] : i32, i1
// CHECK:           }
// CHECK-NOT:     arith.cmpi slt
// CHECK-NOT:     arith.select {{.*}} : i32

// Loop attributes must be carried over to the rebuilt loop.
func.func @test_keep_loop_annotation(%input: !fir.ref<!fir.array<100xi32>>, %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c2147483647_i32, %arg2 = %true)
      -> (i32, i32, i1) attributes {loopAnnotation = #loop_annotation, test.attr = 1 : i32} {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_keep_loop_annotation(
// CHECK:         fir.do_loop {{.*}} -> (i32, i1){{.*}}loopAnnotation = #{{.*}}test.attr = 1 : i32
// CHECK-NOT:     arith.cmpi slt

// cmpi slt mask -- no transform.
func.func @test_no_transform_non_eq_mask(%input: !fir.ref<!fir.array<100xi32>>,
                                          %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c2147483647_i32, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi slt, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_non_eq_mask(
// CHECK:         arith.constant 2147483647
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// Mask is a plain i1 argument, not a cmpi -- no transform.
func.func @test_no_transform_mask_not_cmpi(%input: !fir.ref<!fir.array<100xi32>>,
                                            %mask: i1) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c2147483647_i32, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_mask_not_cmpi(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// Invariant target on LHS of cmpi eq.
func.func @test_eq_mask_fir_target_lhs(%input: !fir.ref<!fir.array<100xi32>>, %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c2147483647_i32, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %val, %ld : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_eq_mask_fir_target_lhs(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:           arith.cmpi eq
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i32, i1) {
// CHECK:             %[[FALSE:.*]] = arith.constant false
// CHECK:             fir.result %{{.*}}, %[[FALSE]] : i32, i1
// CHECK:           }
// CHECK-NOT:     arith.cmpi slt
// CHECK-NOT:     arith.select {{.*}} : i32

// Mask compares a different array -- no transform.
func.func @test_eq_mask_fir_different_array(%input: !fir.ref<!fir.array<100xi32>>,
                                             %other: !fir.ref<!fir.array<100xi32>>,
                                             %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c2147483647_i32, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %other(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %icoor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ild = fir.load %icoor : !fir.ref<i32>
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ild, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ild, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_eq_mask_fir_different_array(
// CHECK:         arith.constant 2147483647
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// No mask, no isFirst flag — no transform.
func.func @test_minloc_no_mask(%input: !fir.ref<!fir.array<100xi32>>) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:2 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c2147483647_i32)
      -> (i32, i32) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %cmp = arith.cmpi slt, %ld, %arg1 : i32
    %loc = fir.convert %i : (index) -> i32
    %nloc = arith.select %cmp, %loc, %arg0 : i32
    %nmin = arith.select %cmp, %ld, %arg1 : i32
    fir.result %nloc, %nmin : i32, i32
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_minloc_no_mask(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (i32, i32)
// CHECK:           arith.cmpi slt
// CHECK:           arith.select
// CHECK-NOT:     arith.andi

// isFirst iter arg initialised to false -- no transform.
func.func @test_no_transform_isFirst_init_false(%input: !fir.ref<!fir.array<100xi32>>,
                                                 %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c2147483647_i32, %arg2 = %false)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_isFirst_init_false(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %false) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// Else-block not a pass-through (resets loc to 0) -- no transform.
func.func @test_no_transform_else_not_passthrough(%input: !fir.ref<!fir.array<100xi32>>,
                                                   %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c2147483647_i32, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %c0_i32, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_else_not_passthrough(
// CHECK:         arith.constant 2147483647
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// 3D minloc equality mask.
func.func @test_minloc_3d_eq_mask(%input: !fir.ref<!fir.array<4x5x6xi32>>,
                                   %val: i32) -> (i32, i32, i32) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %c6 = arith.constant 6 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c4, %c5, %c6 : (index, index, index) -> !fir.shape<3>
  %res:5 = fir.do_loop %k = %c1 to %c6 step %c1 unordered
      iter_args(%a0 = %c0_i32, %a1 = %c0_i32, %a2 = %c0_i32, %a3 = %cmax, %a4 = %true)
      -> (i32, i32, i32, i32, i1) {
    %mid:5 = fir.do_loop %j = %c1 to %c5 step %c1 unordered
        iter_args(%b0 = %a0, %b1 = %a1, %b2 = %a2, %b3 = %a3, %b4 = %a4)
        -> (i32, i32, i32, i32, i1) {
      %inn:5 = fir.do_loop %i = %c1 to %c4 step %c1 unordered
          iter_args(%c0 = %b0, %c1_v = %b1, %c2 = %b2, %c3 = %b3, %c4_v = %b4)
          -> (i32, i32, i32, i32, i1) {
        %coor = fir.array_coor %input(%shape) %i, %j, %k
            : (!fir.ref<!fir.array<4x5x6xi32>>, !fir.shape<3>, index, index, index) -> !fir.ref<i32>
        %ld = fir.load %coor : !fir.ref<i32>
        %mask = arith.cmpi eq, %ld, %val : i32
        %if:5 = fir.if %mask -> (i32, i32, i32, i32, i1) {
          %cmp = arith.cmpi slt, %ld, %c3 : i32
          %upd = arith.ori %cmp, %c4_v : i1
          %li = fir.convert %i : (index) -> i32
          %lj = fir.convert %j : (index) -> i32
          %lk = fir.convert %k : (index) -> i32
          %ni = arith.select %upd, %li, %c0 : i32
          %nj = arith.select %upd, %lj, %c1_v : i32
          %nk = arith.select %upd, %lk, %c2 : i32
          %nm = arith.select %upd, %ld, %c3 : i32
          fir.result %ni, %nj, %nk, %nm, %false : i32, i32, i32, i32, i1
        } else {
          fir.result %c0, %c1_v, %c2, %c3, %c4_v : i32, i32, i32, i32, i1
        }
        fir.result %if#0, %if#1, %if#2, %if#3, %if#4 : i32, i32, i32, i32, i1
      }
      fir.result %inn#0, %inn#1, %inn#2, %inn#3, %inn#4 : i32, i32, i32, i32, i1
    }
    fir.result %mid#0, %mid#1, %mid#2, %mid#3, %mid#4 : i32, i32, i32, i32, i1
  }
  return %res#0, %res#1, %res#2 : i32, i32, i32
}
// CHECK-LABEL: func.func @test_minloc_3d_eq_mask(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i32, i1)
// CHECK:           fir.do_loop {{.*}} iter_args(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) -> (i32, i32, i32, i1)
// CHECK:             fir.do_loop {{.*}} iter_args(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) -> (i32, i32, i32, i1)
// CHECK:               arith.cmpi eq
// CHECK:               arith.andi
// CHECK:               fir.if {{.*}} -> (i32, i32, i32, i1) {
// CHECK:                 %[[FALSE:.*]] = arith.constant false
// CHECK:                 fir.result %{{.*}}, %{{.*}}, %{{.*}}, %[[FALSE]] : i32, i32, i32, i1
// CHECK:               }
// CHECK-NOT:     arith.cmpi slt
// CHECK-NOT:     arith.select {{.*}} : i32

// maxloc 2D equality mask.
func.func @test_maxloc_2d_eq_mask(%input: !fir.ref<!fir.array<10x20xi32>>,
                                   %val: i32) -> (i32, i32) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c0_i32 = arith.constant 0 : i32
  %cmin = arith.constant -2147483648 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
  %res:4 = fir.do_loop %j = %c1 to %c20 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c0_i32, %arg2 = %cmin, %arg3 = %true)
      -> (i32, i32, i32, i1) {
    %inner:4 = fir.do_loop %i = %c1 to %c10 step %c1 unordered
        iter_args(%arg4 = %arg0, %arg5 = %arg1, %arg6 = %arg2, %arg7 = %arg3)
        -> (i32, i32, i32, i1) {
      %coor = fir.array_coor %input(%shape) %i, %j
          : (!fir.ref<!fir.array<10x20xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
      %ld = fir.load %coor : !fir.ref<i32>
      %mask = arith.cmpi eq, %ld, %val : i32
      %if:4 = fir.if %mask -> (i32, i32, i32, i1) {
        %cmp = arith.cmpi sgt, %ld, %arg6 : i32
        %upd = arith.ori %cmp, %arg7 : i1
        %loci = fir.convert %i : (index) -> i32
        %locj = fir.convert %j : (index) -> i32
        %nloci = arith.select %upd, %loci, %arg4 : i32
        %nlocj = arith.select %upd, %locj, %arg5 : i32
        %nmax = arith.select %upd, %ld, %arg6 : i32
        fir.result %nloci, %nlocj, %nmax, %false : i32, i32, i32, i1
      } else {
        fir.result %arg4, %arg5, %arg6, %arg7 : i32, i32, i32, i1
      }
      fir.result %if#0, %if#1, %if#2, %if#3 : i32, i32, i32, i1
    }
    fir.result %inner#0, %inner#1, %inner#2, %inner#3 : i32, i32, i32, i1
  }
  return %res#0, %res#1 : i32, i32
}
// CHECK-LABEL: func.func @test_maxloc_2d_eq_mask(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK:           fir.do_loop {{.*}} iter_args(%{{.*}}, %{{.*}}, %{{.*}}) -> (i32, i32, i1)
// CHECK:             arith.cmpi eq
// CHECK:             arith.andi
// CHECK:             fir.if {{.*}} -> (i32, i32, i1) {
// CHECK:               %[[FALSE:.*]] = arith.constant false
// CHECK:               fir.result %{{.*}}, %{{.*}}, %[[FALSE]] : i32, i32, i1
// CHECK:             }
// CHECK-NOT:     arith.cmpi sgt
// CHECK-NOT:     arith.select {{.*}} : i32

// isFirst at position 0, minMax at position 1 -- must transform.
// Backward chain matching finds the pair regardless of position.
func.func @test_transform_isFirst_non_canonical(%input: !fir.ref<!fir.array<100xi32>>,
                                                %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %true, %arg1 = %c2147483647_i32, %arg2 = %c0_i32)
      -> (i1, i32, i32) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i1, i32, i32) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg0 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg2 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %false, %nmin, %nloc : i1, i32, i32
    } else {
      fir.result %arg0, %arg1, %arg2 : i1, i32, i32
    }
    fir.result %if#0, %if#1, %if#2 : i1, i32, i32
  }
  return %res#2 : i32
}
// CHECK-LABEL: func.func @test_transform_isFirst_non_canonical(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %true, %{{.*}} = %{{.*}}) -> (i1, i32)
// CHECK:           arith.cmpi eq
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i1, i32) {
// CHECK:             %[[FALSE:.*]] = arith.constant false
// CHECK:             fir.result %[[FALSE]], %{{.*}} : i1, i32
// CHECK:           }
// CHECK-NOT:     arith.cmpi slt
// CHECK-NOT:     arith.select {{.*}} : i32

// Equality mask with fir.convert wrapping the cmpi -- must transform.
func.func @test_eq_mask_with_convert(%input: !fir.ref<!fir.array<100xi32>>, %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c2147483647_i32, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %mask_i8 = fir.convert %mask : (i1) -> i8
    %mask_i1 = fir.convert %mask_i8 : (i8) -> i1
    %if:3 = fir.if %mask_i1 -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_eq_mask_with_convert(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:           arith.cmpi eq
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i32, i1) {
// CHECK:             %[[FALSE:.*]] = arith.constant false
// CHECK:             fir.result %{{.*}}, %[[FALSE]] : i32, i1
// CHECK:           }
// CHECK-NOT:     arith.cmpi slt
// CHECK-NOT:     arith.select {{.*}} : i32

// Mask compares array element against loop-index-dependent value -- no transform.
func.func @test_no_transform_variant_target(%input: !fir.ref<!fir.array<100xi32>>) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c2147483647_i32, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %idx = fir.convert %i : (index) -> i32
    %mask = arith.cmpi eq, %ld, %idx : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_variant_target(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// minMax result used outside the loop -- no transform.
func.func @test_no_transform_minmax_used_outside(%input: !fir.ref<!fir.array<100xi32>>,
                                                  %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c2147483647_i32, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#1 : i32
}
// CHECK-LABEL: func.func @test_no_transform_minmax_used_outside(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// 2D loop where comparison target is an iter_arg of the outer loop -- must transform.
func.func @test_transform_2d_outer_iterarg_target(
    %input: !fir.ref<!fir.array<10x20xi32>>) -> (i32, i32) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
  // %val is an iter_arg of an outer unrelated loop.
  %outer:3 = fir.do_loop %k = %c1 to %c20 step %c1
      iter_args(%val = %c0_i32, %acc0 = %c0_i32, %acc1 = %c0_i32)
      -> (i32, i32, i32) {
    %res:4 = fir.do_loop %j = %c1 to %c20 step %c1 unordered
        iter_args(%arg0 = %c0_i32, %arg1 = %c0_i32, %arg2 = %c2147483647_i32, %arg3 = %true)
        -> (i32, i32, i32, i1) {
      %inner:4 = fir.do_loop %i = %c1 to %c10 step %c1 unordered
          iter_args(%arg4 = %arg0, %arg5 = %arg1, %arg6 = %arg2, %arg7 = %arg3)
          -> (i32, i32, i32, i1) {
        %coor = fir.array_coor %input(%shape) %i, %j
            : (!fir.ref<!fir.array<10x20xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
        %ld = fir.load %coor : !fir.ref<i32>
        %mask = arith.cmpi eq, %ld, %val : i32
        %if:4 = fir.if %mask -> (i32, i32, i32, i1) {
          %cmp = arith.cmpi slt, %ld, %arg6 : i32
          %upd = arith.ori %cmp, %arg7 : i1
          %loci = fir.convert %i : (index) -> i32
          %locj = fir.convert %j : (index) -> i32
          %nloci = arith.select %upd, %loci, %arg4 : i32
          %nlocj = arith.select %upd, %locj, %arg5 : i32
          %nmin = arith.select %upd, %ld, %arg6 : i32
          fir.result %nloci, %nlocj, %nmin, %false : i32, i32, i32, i1
        } else {
          fir.result %arg4, %arg5, %arg6, %arg7 : i32, i32, i32, i1
        }
        fir.result %if#0, %if#1, %if#2, %if#3 : i32, i32, i32, i1
      }
      fir.result %inner#0, %inner#1, %inner#2, %inner#3 : i32, i32, i32, i1
    }
    %nextval = arith.addi %val, %c0_i32 : i32
    fir.result %nextval, %res#0, %res#1 : i32, i32, i32
  }
  return %outer#1, %outer#2 : i32, i32
}
// CHECK-LABEL: func.func @test_transform_2d_outer_iterarg_target(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (i32, i32, i32)
// CHECK:           fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK:             fir.do_loop {{.*}} iter_args(%{{.*}}, %{{.*}}, %{{.*}}) -> (i32, i32, i1)
// CHECK:               arith.cmpi eq
// CHECK:               arith.andi
// CHECK:               fir.if {{.*}} -> (i32, i32, i1) {
// CHECK:                 %[[FALSE:.*]] = arith.constant false
// CHECK:                 fir.result %{{.*}}, %{{.*}}, %[[FALSE]] : i32, i32, i1
// CHECK:               }
// CHECK-NOT:       arith.cmpi slt
// CHECK-NOT:       arith.select {{.*}} : i32

// 2D nest with an op feeding the inner bound, unrelated sibling loops before and
// after the inner loop, and two locs sharing a then-region op -- must transform.
func.func @test_minloc_2d_intermediate_ops(%inputA: !fir.ref<!fir.array<10x20xi32>>,
                                            %val: i32, %offset: index,
                                            %out: !fir.ref<i32>) -> (i32, i32) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
  %res:4 = fir.do_loop %j = %c1 to %c20 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c0_i32, %arg2 = %c2147483647_i32, %arg3 = %true)
      -> (i32, i32, i32, i1) {
    %j_adj = arith.addi %j, %offset : index
    fir.do_loop %k0 = %c1 to %c10 step %c1 {
      fir.store %c0_i32 to %out : !fir.ref<i32>
    }
    %inner:4 = fir.do_loop %i = %c1 to %j_adj step %c1 unordered
        iter_args(%arg4 = %arg0, %arg5 = %arg1, %arg6 = %arg2, %arg7 = %arg3)
        -> (i32, i32, i32, i1) {
      %coor = fir.array_coor %inputA(%shape) %i, %j_adj : (!fir.ref<!fir.array<10x20xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
      %ld = fir.load %coor : !fir.ref<i32>
      %mask = arith.cmpi eq, %ld, %val : i32
      %if:4 = fir.if %mask -> (i32, i32, i32, i1) {
        %cmp = arith.cmpi slt, %ld, %arg6 : i32
        %upd = arith.ori %cmp, %arg7 : i1
        %loci = fir.convert %i : (index) -> i32
        %locj = arith.addi %loci, %loci : i32
        %nloci = arith.select %upd, %loci, %arg4 : i32
        %nlocj = arith.select %upd, %locj, %arg5 : i32
        %nmin = arith.select %upd, %ld, %arg6 : i32
        fir.result %nloci, %nlocj, %nmin, %false : i32, i32, i32, i1
      } else {
        fir.result %arg4, %arg5, %arg6, %arg7 : i32, i32, i32, i1
      }
      fir.result %if#0, %if#1, %if#2, %if#3 : i32, i32, i32, i1
    }
    fir.do_loop %k = %c1 to %c10 step %c1 {
      fir.store %c0_i32 to %out : !fir.ref<i32>
    }
    fir.result %inner#0, %inner#1, %inner#2, %inner#3 : i32, i32, i32, i1
  }
  return %res#0, %res#1 : i32, i32
}
// CHECK-LABEL: func.func @test_minloc_2d_intermediate_ops(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK:           %[[ADJ:.*]] = arith.addi
// CHECK:           fir.store
// CHECK:           fir.do_loop {{.*}} to %[[ADJ]] step {{.*}} iter_args(%{{.*}}, %{{.*}}, %{{.*}}) -> (i32, i32, i1)
// CHECK:             arith.cmpi eq
// CHECK:             arith.andi
// CHECK:             fir.if {{.*}} -> (i32, i32, i1) {
// CHECK:               %[[C:.*]] = fir.convert
// CHECK:               %[[S:.*]] = arith.addi %[[C]], %[[C]]
// CHECK:               %[[FALSE:.*]] = arith.constant false
// CHECK:               fir.result %[[C]], %[[S]], %[[FALSE]] : i32, i32, i1
// CHECK:             }
// CHECK:           fir.store
// CHECK-NOT:     arith.cmpi slt
// CHECK-NOT:     arith.select {{.*}} : i32

// Impure call inside then-block has side effects -- no transform.
func.func private @side_effect_fn(i32)
func.func @test_no_transform_call_in_then(%input: !fir.ref<!fir.array<100xi32>>,
                                          %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c2147483647_i32, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      fir.call @side_effect_fn(%ld) : (i32) -> ()
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_call_in_then(
// CHECK:         arith.constant 2147483647
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// Unrelated op in then uses the minMax iter arg -- no transform.
func.func @test_no_transform_other_user_of_minmax_arg(%input: !fir.ref<!fir.array<100xi32>>,
                                                       %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c2147483647_i32, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %unused = arith.addi %arg1, %c0_i32 : i32
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_other_user_of_minmax_arg(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// Outer loop intermediate op uses outer minMaxArg -- no transform.
func.func @test_no_transform_outer_loop_uses_minmax_arg(
    %input: !fir.ref<!fir.array<10x20xi32>>, %val: i32) -> (i32, i32) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
  %res:4 = fir.do_loop %j = %c1 to %c20 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c0_i32, %arg2 = %c2147483647_i32, %arg3 = %true)
      -> (i32, i32, i32, i1) {
    fir.call @side_effect_fn(%arg2) : (i32) -> ()
    %inner:4 = fir.do_loop %i = %c1 to %c10 step %c1 unordered
        iter_args(%arg4 = %arg0, %arg5 = %arg1, %arg6 = %arg2, %arg7 = %arg3)
        -> (i32, i32, i32, i1) {
      %coor = fir.array_coor %input(%shape) %i, %j : (!fir.ref<!fir.array<10x20xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
      %ld = fir.load %coor : !fir.ref<i32>
      %mask = arith.cmpi eq, %ld, %val : i32
      %if:4 = fir.if %mask -> (i32, i32, i32, i1) {
        %cmp = arith.cmpi slt, %ld, %arg6 : i32
        %upd = arith.ori %cmp, %arg7 : i1
        %loci = fir.convert %i : (index) -> i32
        %locj = fir.convert %j : (index) -> i32
        %nloci = arith.select %upd, %loci, %arg4 : i32
        %nlocj = arith.select %upd, %locj, %arg5 : i32
        %nmin = arith.select %upd, %ld, %arg6 : i32
        fir.result %nloci, %nlocj, %nmin, %false : i32, i32, i32, i1
      } else {
        fir.result %arg4, %arg5, %arg6, %arg7 : i32, i32, i32, i1
      }
      fir.result %if#0, %if#1, %if#2, %if#3 : i32, i32, i32, i1
    }
    fir.result %inner#0, %inner#1, %inner#2, %inner#3 : i32, i32, i32, i1
  }
  return %res#0, %res#1 : i32, i32
}
// CHECK-LABEL: func.func @test_no_transform_outer_loop_uses_minmax_arg(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i32, i1)
// CHECK-NOT:     arith.andi

// Outer loop reads the inner minMax result -- must not transform.
func.func @test_no_transform_outer_uses_inner_minmax_result(
    %input: !fir.ref<!fir.array<10x20xi32>>, %val: i32) -> (i32, i32) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
  %res:4 = fir.do_loop %j = %c1 to %c20 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c0_i32, %arg2 = %c2147483647_i32, %arg3 = %true)
      -> (i32, i32, i32, i1) {
    %inner:4 = fir.do_loop %i = %c1 to %c10 step %c1 unordered
        iter_args(%arg4 = %arg0, %arg5 = %arg1, %arg6 = %arg2, %arg7 = %arg3)
        -> (i32, i32, i32, i1) {
      %coor = fir.array_coor %input(%shape) %i, %j : (!fir.ref<!fir.array<10x20xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
      %ld = fir.load %coor : !fir.ref<i32>
      %mask = arith.cmpi eq, %ld, %val : i32
      %if:4 = fir.if %mask -> (i32, i32, i32, i1) {
        %cmp = arith.cmpi slt, %ld, %arg6 : i32
        %upd = arith.ori %cmp, %arg7 : i1
        %loci = fir.convert %i : (index) -> i32
        %locj = fir.convert %j : (index) -> i32
        %nloci = arith.select %upd, %loci, %arg4 : i32
        %nlocj = arith.select %upd, %locj, %arg5 : i32
        %nmin = arith.select %upd, %ld, %arg6 : i32
        fir.result %nloci, %nlocj, %nmin, %false : i32, i32, i32, i1
      } else {
        fir.result %arg4, %arg5, %arg6, %arg7 : i32, i32, i32, i1
      }
      fir.result %if#0, %if#1, %if#2, %if#3 : i32, i32, i32, i1
    }
    fir.call @side_effect_fn(%inner#2) : (i32) -> ()
    fir.result %inner#0, %inner#1, %inner#2, %inner#3 : i32, i32, i32, i1
  }
  return %res#0, %res#1 : i32, i32
}
// CHECK-LABEL: func.func @test_no_transform_outer_uses_inner_minmax_result(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i32, i1)
// CHECK-NOT:     arith.andi

// Outer loop has an extra iter arg and a different slot order than the inner
// loop; its terminator must be rebuilt, not forwarded from the inner results.
func.func @test_minloc_2d_outer_extra_iterarg_permuted(
    %input: !fir.ref<!fir.array<10x20xi32>>, %val: i32) -> (i32, i1, i32) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
  %res:4 = fir.do_loop %j = %c1 to %c20 step %c1
      iter_args(%acc = %c0_i32, %mm = %cmax, %first = %true, %loc = %c0_i32)
      -> (i32, i32, i1, i32) {
    %inner:3 = fir.do_loop %i = %c1 to %c10 step %c1 unordered
        iter_args(%a0 = %mm, %a1 = %first, %a2 = %loc) -> (i32, i1, i32) {
      %coor = fir.array_coor %input(%shape) %i, %j : (!fir.ref<!fir.array<10x20xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
      %ld = fir.load %coor : !fir.ref<i32>
      %mask = arith.cmpi eq, %ld, %val : i32
      %if:3 = fir.if %mask -> (i32, i1, i32) {
        %cmp = arith.cmpi slt, %ld, %a0 : i32
        %upd = arith.ori %cmp, %a1 : i1
        %loci = fir.convert %i : (index) -> i32
        %nloc = arith.select %upd, %loci, %a2 : i32
        %nmin = arith.select %upd, %ld, %a0 : i32
        fir.result %nmin, %false, %nloc : i32, i1, i32
      } else {
        fir.result %a0, %a1, %a2 : i32, i1, i32
      }
      fir.result %if#0, %if#1, %if#2 : i32, i1, i32
    }
    %acc2 = arith.addi %acc, %c1_i32 : i32
    fir.result %acc2, %inner#0, %inner#1, %inner#2 : i32, i32, i1, i32
  }
  return %res#0, %res#2, %res#3 : i32, i1, i32
}
// CHECK-LABEL: func.func @test_minloc_2d_outer_extra_iterarg_permuted(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true, %{{.*}} = %{{.*}}) -> (i32, i1, i32)
// CHECK:           fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (i1, i32)
// CHECK:             arith.andi
// CHECK:             fir.if {{.*}} -> (i1, i32) {
// CHECK:           arith.addi
// CHECK:           fir.result %{{.*}}, %{{.*}}#0, %{{.*}}#1 : i32, i1, i32
// CHECK:         return %{{.*}}#0, %{{.*}}#1, %{{.*}}#2 : i32, i1, i32
// CHECK-NOT:     arith.cmpi slt

// Non-strict compare: equal elements update loc, so isFirst gating is not
// equivalent -- no transform.
func.func @test_no_transform_non_strict_cmp(%input: !fir.ref<!fir.array<100xi32>>,
                                             %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c2147483647_i32, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi sle, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_non_strict_cmp(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// loc is yielded without a select on the update condition, so it changes on
// every match -- no transform.
func.func @test_no_transform_loc_not_guarded_by_upd(
    %input: !fir.ref<!fir.array<100xi32>>, %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c2147483647_i32, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %loc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_loc_not_guarded_by_upd(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// minMax select false-branch is not the iter arg -- no transform.
func.func @test_no_transform_select_wrong_false_branch(
    %input: !fir.ref<!fir.array<100xi32>>, %val: i32, %other: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c2147483647_i32, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %other : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_select_wrong_false_branch(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// Location computation depends on an iter_arg recurrence -- no transform.
func.func @test_no_transform_loc_uses_iterarg(%input: !fir.ref<!fir.array<100xi32>>,
                                              %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c2147483647_i32, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %next_accum = arith.addi %arg0, %c1_i32 : i32
      %nloc = arith.select %upd, %next_accum, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_loc_uses_iterarg(
// CHECK:         arith.constant 2147483647
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// Two sibling minloc loops in one function -- both must transform independently.
func.func @test_two_sibling_loops(%input: !fir.ref<!fir.array<100xi32>>,
                                   %val: i32) -> (i32, i32) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %r1:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  %r2:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %r1#0, %r2#0 : i32, i32
}
// CHECK-LABEL: func.func @test_two_sibling_loops(
// CHECK-COUNT-2: fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK-NOT:     arith.cmpi slt
// CHECK-NOT:     arith.select {{.*}} : i32

// Two i1 args both init=true; only one pairs with the minMax select chain.
// The extra i1 at position 1 yields an arith.select in the then-branch, not a
// constant false -- the real isFirst is at position 3.
func.func @test_two_isFirst_candidates_one_valid(%input: !fir.ref<!fir.array<100xi32>>,
                                                  %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:4 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %true, %arg2 = %cmax, %arg3 = %true)
      -> (i32, i1, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:4 = fir.if %mask -> (i32, i1, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg2 : i32
      %upd = arith.ori %cmp, %arg3 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg2 : i32
      %extra = arith.andi %false, %false : i1
      %nextra = arith.select %upd, %extra, %arg1 : i1
      fir.result %nloc, %nextra, %nmin, %false : i32, i1, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2, %arg3 : i32, i1, i32, i1
    }
    fir.result %if#0, %if#1, %if#2, %if#3 : i32, i1, i32, i1
  }
  return %res#0 : i32
}
// %arg1 is an isFirst candidate but its then-yield is a select, not a constant
// false, so it is rejected. %arg3 is the real isFirst; minMax is %arg2.
// CHECK-LABEL: func.func @test_two_isFirst_candidates_one_valid(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true, %{{.*}} = %true) -> (i32, i1, i1)
// CHECK:           arith.cmpi eq
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i32, i1, i1) {
// CHECK:             %[[FALSE:.*]] = arith.constant false
// CHECK:             fir.result %{{.*}}, %{{.*}}, %[[FALSE]] : i32, i1, i1
// CHECK:           }
// CHECK-NOT:     arith.cmpi slt
// CHECK-NOT:     arith.select {{.*}} : i32

// Two unused i32 results; %arg1 has correct minMax select chain matching the mask
// condition; %arg2 select's trueValue is a different load not in the condition --
// matchEqualityMaskSearchIf rejects %arg2 via getFIREqualityMaskTarget. Only %arg1 pairs.
func.func @test_two_minmax_candidates_one_valid(%input: !fir.ref<!fir.array<100xi32>>,
                                                 %other: !fir.ref<!fir.array<100xi32>>,
                                                 %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:4 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg2 = %cmax, %arg1 = %cmax, %arg3 = %true)
      -> (i32, i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %ocoor = fir.array_coor %other(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %old = fir.load %ocoor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:4 = fir.if %mask -> (i32, i32, i32, i1) {
      %cmp1 = arith.cmpi slt, %ld, %arg1 : i32
      %upd1 = arith.ori %cmp1, %arg3 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd1, %loc, %arg0 : i32
      %nmin1 = arith.select %upd1, %ld, %arg1 : i32
      %nmin2 = arith.select %upd1, %old, %arg2 : i32
      fir.result %nloc, %nmin2, %nmin1, %false : i32, i32, i32, i1
    } else {
      fir.result %arg0, %arg2, %arg1, %arg3 : i32, i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2, %if#3 : i32, i32, i32, i1
  }
  return %res#0 : i32
}
// %res#1 and %res#2 are both unused minMax candidates. %arg1 select uses %ld which
// appears in the mask condition (cmpi eq %ld, %val) -- valid. %arg2 select uses %old
// which is NOT in the condition -- getFIREqualityMaskTarget rejects it. Only %arg1 pairs.
// CHECK-LABEL: func.func @test_two_minmax_candidates_one_valid(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK:           arith.cmpi eq
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i32, i32, i1) {
// CHECK:             %[[FALSE:.*]] = arith.constant false
// CHECK:             fir.result %{{.*}}, %{{.*}}, %[[FALSE]] : i32, i32, i1
// CHECK:           }
// CHECK-NOT:     arith.cmpi slt
// CHECK-NOT:     arith.select {{.*}} : i32

// Mask hoisted out of the loop (invariant element) -- must transform.
func.func @test_transform_hoisted_mask(%val: i32, %x: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %mask = arith.cmpi eq, %x, %val : i32
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %x, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %x, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_transform_hoisted_mask(
// CHECK:         %[[MASK:.*]] = arith.cmpi eq
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:           arith.andi %[[MASK]], %{{.*}}
// CHECK-NOT:     arith.cmpi slt

// Mask compares against a root loop iter arg, which is not invariant -- no transform.
func.func @test_no_transform_iterarg_target(%input: !fir.ref<!fir.array<100xi32>>) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %arg0 : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_iterarg_target(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// The reduced value is itself a loop-carried value -- no transform.
func.func @test_no_transform_minmax_val_is_iterarg(%val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %mask = arith.cmpi eq, %arg0, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %arg0, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %arg0, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_minmax_val_is_iterarg(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// Volatile load of the reduced value -- no transform.
func.func @test_no_transform_volatile_load(%input: !fir.ref<!fir.array<100xi32>>,
                                            %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %vcoor = fir.volatile_cast %coor : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
    %ld = fir.load %vcoor : !fir.ref<i32, volatile>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_volatile_load(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// The fir.if results have a use besides the loop yield -- no transform.
func.func @test_no_transform_if_result_extra_use(%input: !fir.ref<!fir.array<100xi32>>,
                                                 %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.call @side_effect_fn(%if#0) : (i32) -> ()
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_if_result_extra_use(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// loc select is not guarded by the same update condition as minMax -- no transform.
func.func @test_no_transform_loc_wrong_condition(%input: !fir.ref<!fir.array<100xi32>>,
                                                  %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %cmp, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_loc_wrong_condition(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// loc select keeps a value other than its own iter arg -- no transform.
func.func @test_no_transform_loc_wrong_false_value(%input: !fir.ref<!fir.array<100xi32>>,
                                                    %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %c0_i32 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_loc_wrong_false_value(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// Root loop with a final value -- no transform.
func.func @test_no_transform_final_value(%input: !fir.ref<!fir.array<100xi32>>,
                                          %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:4 = fir.do_loop %i = %c1 to %c100 step %c1
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (index, i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %i, %if#0, %if#1, %if#2 : index, i32, i32, i1
  }
  return %res#1 : i32
}
// CHECK-LABEL: func.func @test_no_transform_final_value(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (index, i32, i32, i1)
// CHECK-NOT:     arith.andi

// Inner loop of the nest has a reduce operand -- no transform.
func.func @test_no_transform_inner_reduce(%input: !fir.ref<!fir.array<10x20xi32>>,
                                           %val: i32, %sum: !fir.ref<i32>) -> i32 {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
  %res:3 = fir.do_loop %j = %c1 to %c20 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %inner:3 = fir.do_loop %i = %c1 to %c10 step %c1 unordered
        reduce(#fir.reduce_attr<add> -> %sum : !fir.ref<i32>)
        iter_args(%a0 = %arg0, %a1 = %arg1, %a2 = %arg2) -> (i32, i32, i1) {
      %coor = fir.array_coor %input(%shape) %i, %j : (!fir.ref<!fir.array<10x20xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
      %ld = fir.load %coor : !fir.ref<i32>
      %mask = arith.cmpi eq, %ld, %val : i32
      %if:3 = fir.if %mask -> (i32, i32, i1) {
        %cmp = arith.cmpi slt, %ld, %a1 : i32
        %upd = arith.ori %cmp, %a2 : i1
        %loc = fir.convert %i : (index) -> i32
        %nloc = arith.select %upd, %loc, %a0 : i32
        %nmin = arith.select %upd, %ld, %a1 : i32
        fir.result %nloc, %nmin, %false : i32, i32, i1
      } else {
        fir.result %a0, %a1, %a2 : i32, i32, i1
      }
      fir.result %if#0, %if#1, %if#2 : i32, i32, i1
    }
    fir.result %inner#0, %inner#1, %inner#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_inner_reduce(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// Inner loop seeds the minMax slot with something other than the outer iter
// arg -- no transform.
func.func @test_no_transform_inner_init_mismatch(%input: !fir.ref<!fir.array<10x20xi32>>,
                                                  %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c10, %c20 : (index, index) -> !fir.shape<2>
  %res:3 = fir.do_loop %j = %c1 to %c20 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %inner:3 = fir.do_loop %i = %c1 to %c10 step %c1 unordered
        iter_args(%a0 = %arg0, %a1 = %cmax, %a2 = %arg2) -> (i32, i32, i1) {
      %coor = fir.array_coor %input(%shape) %i, %j : (!fir.ref<!fir.array<10x20xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
      %ld = fir.load %coor : !fir.ref<i32>
      %mask = arith.cmpi eq, %ld, %val : i32
      %if:3 = fir.if %mask -> (i32, i32, i1) {
        %cmp = arith.cmpi slt, %ld, %a1 : i32
        %upd = arith.ori %cmp, %a2 : i1
        %loc = fir.convert %i : (index) -> i32
        %nloc = arith.select %upd, %loc, %a0 : i32
        %nmin = arith.select %upd, %ld, %a1 : i32
        fir.result %nloc, %nmin, %false : i32, i32, i1
      } else {
        fir.result %a0, %a1, %a2 : i32, i32, i1
      }
      fir.result %if#0, %if#1, %if#2 : i32, i32, i1
    }
    fir.result %inner#0, %inner#1, %inner#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_inner_init_mismatch(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// isFirst and minMax flow through different sibling loops -- no transform.
func.func @test_no_transform_chains_through_different_loops(
    %input: !fir.ref<!fir.array<100xi32>>, %val: i32, %n: index) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %outer:3 = fir.do_loop %k = %c1 to %n step %c1
      iter_args(%loc0 = %c0_i32, %mm0 = %cmax, %first0 = %true)
      -> (i32, i32, i1) {
    %a:3 = fir.do_loop %i = %c1 to %c100 step %c1
        iter_args(%a0 = %c0_i32, %a1 = %c0_i32, %a2 = %first0)
        -> (i32, i32, i1) {
      %cond = arith.cmpi eq, %val, %val : i32
      %aif:3 = fir.if %cond -> (i32, i32, i1) {
        fir.result %a0, %a1, %false : i32, i32, i1
      } else {
        fir.result %a0, %a1, %a2 : i32, i32, i1
      }
      fir.result %aif#0, %aif#1, %aif#2 : i32, i32, i1
    }
    %b:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
        iter_args(%b0 = %loc0, %b1 = %mm0, %b2 = %true)
        -> (i32, i32, i1) {
      %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
      %ld = fir.load %coor : !fir.ref<i32>
      %mask = arith.cmpi eq, %ld, %val : i32
      %bif:3 = fir.if %mask -> (i32, i32, i1) {
        %cmp = arith.cmpi slt, %ld, %b1 : i32
        %upd = arith.ori %cmp, %b2 : i1
        %loc = fir.convert %i : (index) -> i32
        %nloc = arith.select %upd, %loc, %b0 : i32
        %nmin = arith.select %upd, %ld, %b1 : i32
        fir.result %nloc, %nmin, %false : i32, i32, i1
      } else {
        fir.result %b0, %b1, %b2 : i32, i32, i1
      }
      fir.result %bif#0, %bif#1, %bif#2 : i32, i32, i1
    }
    fir.result %b#0, %b#1, %a#2 : i32, i32, i1
  }
  return %outer#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_chains_through_different_loops(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// Then-branch sets isFirst to true instead of false -- no transform.
func.func @test_no_transform_isFirst_set_true(%input: !fir.ref<!fir.array<100xi32>>,
                                              %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %true : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_isFirst_set_true(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// Update condition does not include the isFirst iter arg -- no transform.
func.func @test_no_transform_upd_without_isFirst(%input: !fir.ref<!fir.array<100xi32>>,
                                                 %val: i32, %flag: i1) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %flag : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_upd_without_isFirst(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// Compare is not against the minMax iter arg -- no transform.
func.func @test_no_transform_cmp_not_against_minmax(%input: !fir.ref<!fir.array<100xi32>>,
                                                    %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %val : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_cmp_not_against_minmax(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// isFirst init is true but not a constant -- no transform.
func.func @test_no_transform_isFirst_init_not_constant(%input: !fir.ref<!fir.array<100xi32>>,
                                                       %val: i32, %first: i1) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %first)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_isFirst_init_not_constant(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// Loop yield does not forward the fir.if results in order -- no transform.
func.func @test_no_transform_yield_permuted(%input: !fir.ref<!fir.array<100xi32>>,
                                            %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#1, %if#0, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_yield_permuted(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// Loc values defined outside the then-region (a loop-body op and a function
// argument) -- must transform.
func.func @test_loc_defined_outside_then(%input: !fir.ref<!fir.array<100xi32>>,
                                         %val: i32, %k: i32) -> (i32, i32) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:4 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c0_i32, %arg2 = %cmax, %arg3 = %true)
      -> (i32, i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %loc = fir.convert %i : (index) -> i32
    %if:4 = fir.if %mask -> (i32, i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg2 : i32
      %upd = arith.ori %cmp, %arg3 : i1
      %nloc0 = arith.select %upd, %loc, %arg0 : i32
      %nloc1 = arith.select %upd, %k, %arg1 : i32
      %nmin = arith.select %upd, %ld, %arg2 : i32
      fir.result %nloc0, %nloc1, %nmin, %false : i32, i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2, %arg3 : i32, i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2, %if#3 : i32, i32, i32, i1
  }
  return %res#0, %res#1 : i32, i32
}
// CHECK-LABEL: func.func @test_loc_defined_outside_then(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK:           %[[LOC:.*]] = fir.convert
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i32, i32, i1) {
// CHECK:             %[[FALSE:.*]] = arith.constant false
// CHECK:             fir.result %[[LOC]], %arg2, %[[FALSE]] : i32, i32, i1
// CHECK:           }
// CHECK-NOT:     arith.cmpi slt
// CHECK-NOT:     arith.select {{.*}} : i32

// Unsigned compare with swapped operands and isFirst on the left of the ori --
// must transform.
func.func @test_unsigned_swapped_cmp(%input: !fir.ref<!fir.array<100xi32>>,
                                     %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant -1 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi ugt, %arg1, %ld : i32
      %upd = arith.ori %arg2, %cmp : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_unsigned_swapped_cmp(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i32, i1) {
// CHECK:           }
// CHECK-NOT:     arith.cmpi ugt
// CHECK-NOT:     arith.select {{.*}} : i32

// minMax iter arg is also used in the loop body outside the fir.if -- no transform.
func.func @test_no_transform_minmax_arg_used_outside_if(%input: !fir.ref<!fir.array<100xi32>>,
                                                        %val: i32, %out: !fir.ref<i32>) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    fir.store %arg1 to %out : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_minmax_arg_used_outside_if(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// Update condition is a bare compare with no ori -- no transform.
func.func @test_no_transform_upd_is_bare_cmp(%input: !fir.ref<!fir.array<100xi32>>,
                                             %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %upd = arith.cmpi slt, %ld, %arg1 : i32
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_upd_is_bare_cmp(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// The ori operand next to isFirst is not a compare -- no transform.
func.func @test_no_transform_upd_operand_not_cmp(%input: !fir.ref<!fir.array<100xi32>>,
                                                 %val: i32, %flag: i1) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %upd = arith.ori %flag, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_upd_operand_not_cmp(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// Loc computed with an effect-free but non-speculatable op (divsi by an unknown
// divisor) -- no transform.
func.func @test_no_transform_loc_not_speculatable(%input: !fir.ref<!fir.array<100xi32>>,
                                                  %val: i32, %d: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %idx = fir.convert %i : (index) -> i32
      %loc = arith.divsi %idx, %d : i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_loc_not_speculatable(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi
