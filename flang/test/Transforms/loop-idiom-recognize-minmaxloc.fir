// RUN: fir-opt %s --flang-loop-idiom | FileCheck %s

// Tests for elimination of redundant min/max tracking in equality-mask loops.

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
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:           arith.cmpi eq
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i32, i1) {
// CHECK:             fir.result %{{.*}}, %false : i32, i1
// CHECK:           }
// CHECK-NOT:     arith.cmpi slt
// CHECK-NOT:     arith.select {{.*}} : i32

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
// CHECK:             fir.result %{{.*}}, %false : i32, i1
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
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %icoor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
      %ild = fir.load %icoor : !fir.ref<i32>
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

// Both cmpi eq operands are variant array loads -- no transform.
func.func @test_eq_mask_fir_both_variant_arrays(%input: !fir.ref<!fir.array<100xi32>>,
                                                 %other: !fir.ref<!fir.array<100xi32>>) -> i32 {
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
    %acoor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ald = fir.load %acoor : !fir.ref<i32>
    %bcoor = fir.array_coor %other(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %bld = fir.load %bcoor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ald, %bld : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %ald, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ald, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_eq_mask_fir_both_variant_arrays(
// CHECK:         arith.constant 2147483647
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// 2D total reduction equality mask.
func.func @test_minloc_2d_eq_mask(%inputA: !fir.ref<!fir.array<10x20xi32>>,
                                   %val: i32) -> (i32, i32) {
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
      %coor = fir.array_coor %inputA(%shape) %i, %j : (!fir.ref<!fir.array<10x20xi32>>, !fir.shape<2>, index, index) -> !fir.ref<i32>
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
// CHECK-LABEL: func.func @test_minloc_2d_eq_mask(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK:           fir.do_loop {{.*}} iter_args(%{{.*}}, %{{.*}}, %{{.*}}) -> (i32, i32, i1)
// CHECK:             arith.cmpi eq
// CHECK:             arith.andi
// CHECK:             fir.if {{.*}} -> (i32, i32, i1) {
// CHECK:               fir.result %{{.*}}, %{{.*}}, %false : i32, i32, i1
// CHECK:             }
// CHECK-NOT:     arith.cmpi slt
// CHECK-NOT:     arith.select {{.*}} : i32

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

// Then-yields iter arg instead of false at isFirstIdx -- no transform.
func.func @test_no_transform_isFirst_not_false(%input: !fir.ref<!fir.array<100xi32>>,
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
      fir.result %nloc, %nmin, %arg2 : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_isFirst_not_false(
// CHECK:         arith.constant 2147483647
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
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

// 1D minloc inside unrelated 2-arg outer loop -- must transform.
func.func @test_transform_inside_unrelated_outer_loop(
    %input: !fir.ref<!fir.array<100xi32>>, %val: i32, %n: index) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %outer:2 = fir.do_loop %k = %c1 to %n step %c1
      iter_args(%acc = %c0_i32, %last = %c0_i32) -> (i32, i32) {
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
    %sum = arith.addi %acc, %res#0 : i32
    fir.result %sum, %res#0 : i32, i32
  }
  return %outer#1 : i32
}
// CHECK-LABEL: func.func @test_transform_inside_unrelated_outer_loop(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (i32, i32)
// CHECK:           fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:             arith.cmpi eq
// CHECK:             arith.andi
// CHECK:             fir.if {{.*}} -> (i32, i1) {
// CHECK:               fir.result %{{.*}}, %false : i32, i1
// CHECK:             }
// CHECK-NOT:       arith.cmpi slt
// CHECK-NOT:       arith.select {{.*}} : i32

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
// CHECK:                 fir.result %{{.*}}, %{{.*}}, %{{.*}}, %false : i32, i32, i32, i1
// CHECK:               }
// CHECK-NOT:     arith.cmpi slt
// CHECK-NOT:     arith.select {{.*}} : i32

// hlfir.designate element access.
func.func @test_minloc_hlfir_designate(%input: !fir.box<!fir.array<?xi32>>,
                                        %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %dims:3 = fir.box_dims %input, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %res:3 = fir.do_loop %i = %c1 to %dims#1 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %elem = hlfir.designate %input (%i) : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %ld = fir.load %elem : !fir.ref<i32>
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
// CHECK-LABEL: func.func @test_minloc_hlfir_designate(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:           arith.cmpi eq
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i32, i1) {
// CHECK:             fir.result %{{.*}}, %false : i32, i1
// CHECK:           }
// CHECK-NOT:     arith.cmpi slt
// CHECK-NOT:     arith.select {{.*}} : i32

// hlfir.apply element access.
func.func @test_minloc_hlfir_apply(%input: !hlfir.expr<?xi32>, %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %cmax = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %cmax, %arg2 = %true)
      -> (i32, i32, i1) {
    %elem = hlfir.apply %input, %i : (!hlfir.expr<?xi32>, index) -> i32
    %mask = arith.cmpi eq, %elem, %val : i32
    %if:3 = fir.if %mask -> (i32, i32, i1) {
      %cmp = arith.cmpi slt, %elem, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %elem, %arg1 : i32
      fir.result %nloc, %nmin, %false : i32, i32, i1
    } else {
      fir.result %arg0, %arg1, %arg2 : i32, i32, i1
    }
    fir.result %if#0, %if#1, %if#2 : i32, i32, i1
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_minloc_hlfir_apply(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:           arith.cmpi eq
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i32, i1) {
// CHECK:             fir.result %{{.*}}, %false : i32, i1
// CHECK:           }
// CHECK-NOT:     arith.cmpi slt
// CHECK-NOT:     arith.select {{.*}} : i32

// Volatile load on search array element — transform fires.
func.func @test_transform_volatile_minval(%input: !fir.ref<!fir.array<100xi32>>,
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
    %coor = fir.array_coor %input(%shape) %i
        : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor { volatile } : !fir.ref<i32>
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
// CHECK-LABEL: func.func @test_transform_volatile_minval(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:           arith.cmpi eq
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i32, i1) {
// CHECK:             fir.result %{{.*}}, %false : i32, i1
// CHECK:           }
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
// CHECK:               fir.result %{{.*}}, %{{.*}}, %false : i32, i32, i1
// CHECK:             }
// CHECK-NOT:     arith.cmpi sgt
// CHECK-NOT:     arith.select {{.*}} : i32

// Both cmpi eq operands load from the same array -- no transform.
func.func @test_no_transform_eq_same_array(%input: !fir.ref<!fir.array<100xi32>>) -> i32 {
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
    %mask = arith.cmpi eq, %ld, %ld : i32
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
// CHECK-LABEL: func.func @test_no_transform_eq_same_array(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

// isFirst arg at a non-last position -- no transform.
func.func @test_no_transform_isFirst_not_last(%input: !fir.ref<!fir.array<100xi32>>,
                                               %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:4 = fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c2147483647_i32, %arg2 = %true, %arg3 = %c0_i32)
      -> (i32, i32, i1, i32) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    %if:4 = fir.if %mask -> (i32, i32, i1, i32) {
      %cmp = arith.cmpi slt, %ld, %arg1 : i32
      %upd = arith.ori %cmp, %arg2 : i1
      %loc = fir.convert %i : (index) -> i32
      %nloc = arith.select %upd, %loc, %arg0 : i32
      %nmin = arith.select %upd, %ld, %arg1 : i32
      fir.result %nloc, %nmin, %false, %arg3 : i32, i32, i1, i32
    } else {
      fir.result %arg0, %arg1, %arg2, %arg3 : i32, i32, i1, i32
    }
    fir.result %if#0, %if#1, %if#2, %if#3 : i32, i32, i1, i32
  }
  return %res#0 : i32
}
// CHECK-LABEL: func.func @test_no_transform_isFirst_not_last(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true, %{{.*}} = %{{.*}}) -> (i32, i32, i1, i32)
// CHECK-NOT:     arith.andi

// fir.if missing else region -- no transform.
func.func @test_no_transform_no_else(%input: !fir.ref<!fir.array<100xi32>>,
                                      %val: i32) {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  fir.do_loop %i = %c1 to %c100 step %c1 unordered
      iter_args(%arg0 = %c0_i32, %arg1 = %c2147483647_i32, %arg2 = %true)
      -> (i32, i32, i1) {
    %coor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %coor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld, %val : i32
    fir.if %mask {
      fir.store %ld to %coor : !fir.ref<i32>
    }
    fir.result %arg0, %arg1, %arg2 : i32, i32, i1
  }
  return
}
// CHECK-LABEL: func.func @test_no_transform_no_else(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

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
// CHECK:             fir.result %{{.*}}, %false : i32, i1
// CHECK:           }
// CHECK-NOT:     arith.cmpi slt
// CHECK-NOT:     arith.select {{.*}} : i32

// Volatile mask-array load — no transform.
func.func @test_no_transform_volatile_mask_array(%input: !fir.ref<!fir.array<100xi32>>,
                                                  %mask_arr: !fir.ref<!fir.array<100xi32>>) -> i32 {
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
    %mcoor = fir.array_coor %mask_arr(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld_vol = fir.load %mcoor { volatile } : !fir.ref<i32>
    %icoor = fir.array_coor %input(%shape) %i : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %ld = fir.load %icoor : !fir.ref<i32>
    %mask = arith.cmpi eq, %ld_vol, %ld : i32
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
// CHECK-LABEL: func.func @test_no_transform_volatile_mask_array(
// CHECK:         iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i32, i1)
// CHECK-NOT:     arith.andi

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
// CHECK:                 fir.result %{{.*}}, %{{.*}}, %false : i32, i32, i1
// CHECK:               }
// CHECK-NOT:       arith.cmpi slt
// CHECK-NOT:       arith.select {{.*}} : i32

// Outer loop has same numArgs with unrelated i1 last arg — must transform inner nest.
func.func @test_transform_outer_same_numargs_unrelated_i1(
    %input: !fir.ref<!fir.array<100xi32>>, %val: i32, %n: index) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %outer:3 = fir.do_loop %k = %c1 to %n step %c1
      iter_args(%acc = %c0_i32, %last = %c0_i32, %flag = %false)
      -> (i32, i32, i1) {
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
    %sum = arith.addi %acc, %res#0 : i32
    %newflag = arith.cmpi eq, %res#0, %c0_i32 : i32
    fir.result %sum, %res#0, %newflag : i32, i32, i1
  }
  return %outer#1 : i32
}
// CHECK-LABEL: func.func @test_transform_outer_same_numargs_unrelated_i1(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}, %{{.*}} = %false) -> (i32, i32, i1)
// CHECK:           fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:             arith.cmpi eq
// CHECK:             arith.andi
// CHECK:             fir.if {{.*}} -> (i32, i1) {
// CHECK:               fir.result %{{.*}}, %false : i32, i1
// CHECK:             }
// CHECK-NOT:       arith.cmpi slt
// CHECK-NOT:       arith.select {{.*}} : i32

// User-written ordered loop (no unordered) -- transform fires without LICM.
func.func @test_user_written_ordered_loop(%input: !fir.ref<!fir.array<100xi32>>,
                                          %val: i32) -> i32 {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %c0_i32 = arith.constant 0 : i32
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %true = arith.constant true
  %false = arith.constant false
  %shape = fir.shape %c100 : (index) -> !fir.shape<1>
  %res:3 = fir.do_loop %i = %c1 to %c100 step %c1
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
// CHECK-LABEL: func.func @test_user_written_ordered_loop(
// CHECK:         fir.do_loop {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %true) -> (i32, i1)
// CHECK:           arith.cmpi eq
// CHECK:           arith.andi
// CHECK:           fir.if {{.*}} -> (i32, i1) {
// CHECK:             fir.result %{{.*}}, %false : i32, i1
// CHECK:           }
// CHECK-NOT:     arith.cmpi slt
// CHECK-NOT:     arith.select {{.*}} : i32
