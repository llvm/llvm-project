// RUN: fir-opt --loop-versioning %s | FileCheck %s


//  subroutine sum1d(a, n)
//    real*8 :: a(:)
//    integer :: n
//    real*8 :: sum
//    integer :: i
//    sum = 0
//    do i=1,n
//       sum = sum + a(i)
//    end do
//  end subroutine sum1d
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {
  func.func @sum1d(%arg0: !fir.box<!fir.array<?xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
    %decl = fir.declare %arg0 {uniq_name = "a"} : (!fir.box<!fir.array<?xf64>>) -> !fir.box<!fir.array<?xf64>>
    %rebox = fir.rebox %decl : (!fir.box<!fir.array<?xf64>>) -> !fir.box<!fir.array<?xf64>>
    %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QMmoduleFsum1dEi"}
    %1 = fir.alloca f64 {bindc_name = "sum", uniq_name = "_QMmoduleFsum1dEsum"}
    %cst = arith.constant 0.000000e+00 : f64
    fir.store %cst to %1 : !fir.ref<f64>
    %c1_i32 = arith.constant 1 : i32
    %2 = fir.convert %c1_i32 : (i32) -> index
    %3 = fir.load %arg1 : !fir.ref<i32>
    %4 = fir.convert %3 : (i32) -> index
    %c1 = arith.constant 1 : index
    %5 = fir.convert %2 : (index) -> i32
    %6:2 = fir.do_loop %arg2 = %2 to %4 step %c1 iter_args(%arg3 = %5) -> (index, i32) {
      fir.store %arg3 to %0 : !fir.ref<i32>
      %7 = fir.load %1 : !fir.ref<f64>
      %8 = fir.load %0 : !fir.ref<i32>
      %9 = fir.convert %8 : (i32) -> i64
      %c1_i64 = arith.constant 1 : i64
      %10 = arith.subi %9, %c1_i64 : i64
      %11 = fir.coordinate_of %rebox, %10 : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %12 = fir.load %11 : !fir.ref<f64>
      %13 = arith.addf %7, %12 fastmath<contract> : f64
      fir.store %13 to %1 : !fir.ref<f64>
      %14 = arith.addi %arg2, %c1 : index
      %15 = fir.convert %c1 : (index) -> i32
      %16 = fir.load %0 : !fir.ref<i32>
      %17 = arith.addi %16, %15 : i32
      fir.result %14, %17 : index, i32
    }
    fir.store %6#1 to %0 : !fir.ref<i32>
    return
  }

// Note this only checks the expected transformation, not the entire generated code:
// CHECK-LABEL: func.func @sum1d(
// CHECK-SAME:                  %[[ARG0:.*]]: !fir.box<!fir.array<?xf64>> {{.*}})
// CHECK: %[[DECL:.*]] = fir.declare %arg0 {uniq_name = "a"} : (!fir.box<!fir.array<?xf64>>) -> !fir.box<!fir.array<?xf64>>
// CHECK: %[[REBOX:.*]] = fir.rebox %[[DECL]]
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[DIMS:.*]]:3 = fir.box_dims %[[REBOX]], %[[ZERO]] : {{.*}}
// CHECK: %[[SIZE:.*]] = arith.constant 8 : index
// CHECK: %[[CMP:.*]] = arith.cmpi eq, %[[DIMS]]#2, %[[SIZE]]
// CHECK: %[[IF_RES:.*]]:2 = fir.if %[[CMP]] -> {{.*}}
// CHECK: %[[NEWARR:.*]] = fir.convert %[[REBOX]]
// CHECK: %[[BOXADDR:.*]] = fir.box_addr %[[NEWARR]] : {{.*}} -> !fir.ref<!fir.array<?xf64>>
// CHECK: %[[LOOP_RES:.*]]:2 = fir.do_loop {{.*}}
// CHECK: %[[COORD:.*]] = fir.coordinate_of %[[BOXADDR]], %{{.*}} : (!fir.ref<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// CHECK: %{{.*}} = fir.load %[[COORD]] : !fir.ref<f64>
// CHECK: fir.result %{{.*}}, %{{.*}}
// CHECK: }
// CHECK: fir.result %[[LOOP_RES]]#0, %[[LOOP_RES]]#1
// CHECK: } else {
// CHECK: %[[LOOP_RES2:.*]]:2 = fir.do_loop {{.*}}
// CHECK: %[[COORD2:.*]] = fir.coordinate_of %[[REBOX]], %{{.*}} : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
// CHECK: %{{.*}}= fir.load %[[COORD2]] : !fir.ref<f64>
// CHECK: fir.result %{{.*}}, %{{.*}}
// CHECK: }
// CHECK: fir.result %[[LOOP_RES2]]#0, %[[LOOP_RES2]]#1
// CHECK: }
// CHECK: fir.store %[[IF_RES]]#1 to %{{.*}}
// CHECK: return

// -----

// Test that loop-versioning pass doesn't expand known size arrays.
func.func @sum1dfixed(%arg0: !fir.ref<!fir.array<?xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
    %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsum1dfixedEi"}
    %1 = fir.alloca f64 {bindc_name = "sum", uniq_name = "_QFsum1dfixedEsum"}
    %cst = arith.constant 0.000000e+00 : f64
    fir.store %cst to %1 : !fir.ref<f64>
    %c1_i32 = arith.constant 1 : i32
    %2 = fir.convert %c1_i32 : (i32) -> index
    %3 = fir.load %arg1 : !fir.ref<i32>
    %4 = fir.convert %3 : (i32) -> index
    %c1 = arith.constant 1 : index
    %5 = fir.convert %2 : (index) -> i32
    %6:2 = fir.do_loop %arg2 = %2 to %4 step %c1 iter_args(%arg3 = %5) -> (index, i32) {
      fir.store %arg3 to %0 : !fir.ref<i32>
      %7 = fir.load %1 : !fir.ref<f64>
      %8 = fir.load %0 : !fir.ref<i32>
      %9 = fir.convert %8 : (i32) -> i64
      %c1_i64 = arith.constant 1 : i64
      %10 = arith.subi %9, %c1_i64 : i64
      %11 = fir.coordinate_of %arg0, %10 : (!fir.ref<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %12 = fir.load %11 : !fir.ref<f64>
      %13 = arith.addf %7, %12 fastmath<contract> : f64
      fir.store %13 to %1 : !fir.ref<f64>
      %14 = arith.addi %arg2, %c1 : index
      %15 = fir.convert %c1 : (index) -> i32
      %16 = fir.load %0 : !fir.ref<i32>
      %17 = arith.addi %16, %15 : i32
      fir.result %14, %17 : index, i32
    }
    fir.store %6#1 to %0 : !fir.ref<i32>
    return
  }

// CHECK-LABEL: func.func @sum1dfixed(
// CHECK-SAME:                        %[[ARG0:.*]]: !fir.ref<!fir.array<?xf64>> {{.*}})
// CHECK: fir.do_loop {{.*}}
// CHECK-NOT: fir.do_loop
// CHECK: %[[COORD:.*]] = fir.coordinate_of %[[ARG0]], {{.*}}
// CHECK: %{{.*}} = fir.load %[[COORD]]
// CHECK-NOT: fir.do_loop

// -----

// Check that "no result" from a versioned loop works correctly
// This code was the basis for this, but `read` is replaced with a function called Func
// subroutine test3(x, y)
//  integer :: y(:)
//  integer :: x(:)
//  read(*,*) x(y)
// end subroutine

  func.func @test3(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "x"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "y"}) {
    %c0 = arith.constant 0 : index
    %3:3 = fir.box_dims %arg1, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
    %c1 = arith.constant 1 : index
    %4 = fir.slice %c1, %3#1, %c1 : (index, index, index) -> !fir.slice<1>
    %c1_0 = arith.constant 1 : index
    %c0_1 = arith.constant 0 : index
    %5 = arith.subi %3#1, %c1_0 : index
    fir.do_loop %arg2 = %c0_1 to %5 step %c1_0 {
      %7 = fir.coordinate_of %arg1, %arg2 : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
      %8 = fir.load %7 : !fir.ref<i32>
      %9 = fir.convert %8 : (i32) -> index
      %10 = fir.array_coor %arg0 [%4] %9 : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index) -> !fir.ref<i32>
      %12 = fir.call @Func(%10) fastmath<contract> : (!fir.ref<i32>) -> i1
    }
    return
  }
  func.func private @Func(!fir.ref<i8>, !fir.ref<i32>) -> i1

// CHECK-LABEL: func.func @test3(
// CHECK-SAME:                    %[[X:.*]]: !fir.box<!fir.array<?xi32>> {{.*}},
// CHECK-SAME:                    %[[Y:.*]]: !fir.box<!fir.array<?xi32>> {{.*}}) {
// Look for arith.subi to locate the correct part of code.
// CHECK:    {{.*}} arith.subi {{.*}}
// CHECK:    %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:     %[[DIMS:.*]]:3 = fir.box_dims %[[Y]], %[[ZERO]]
// CHECK:    %[[FOUR:.*]] = arith.constant 4 : index
// CHECK:    %[[COMP:.*]] = arith.cmpi eq, %[[DIMS]]#2, %[[FOUR]] : index
// CHECK:    fir.if %[[COMP]] {
// CHECK:    %[[CONV:.*]]  = fir.convert %[[Y]] : {{.*}}
// CHECK:    %[[BOX_ADDR:.*]] = fir.box_addr %[[CONV]] : {{.*}}
// CHECK:    fir.do_loop %[[INDEX:.*]] = {{.*}}
// CHECK:    %[[YADDR:.*]] = fir.coordinate_of %[[BOX_ADDR]], %[[INDEX]]
// CHECK:    %[[YINT:.*]] = fir.load %[[YADDR]] : {{.*}}
// CHECK:    %[[YINDEX:.*]] = fir.convert %[[YINT]]
// CHECK:    %[[XADDR:.*]] = fir.array_coor %[[X]] [%{{.*}}] %[[YINDEX]]
// CHECK:    fir.call @Func(%[[XADDR]])
// CHECK-NEXT: }
// CHECK-NEXT: } else {
// CHECK:    fir.do_loop %[[INDEX2:.*]] = {{.*}}
// CHECK:    %[[YADDR2:.*]] = fir.coordinate_of %[[Y]], %[[INDEX2]]
// CHECK:    %[[YINT2:.*]] = fir.load %[[YADDR2]] : {{.*}}
// CHECK:    %[[YINDEX2:.*]] = fir.convert %[[YINT2]]
// CHECK:    %[[XADDR2:.*]] = fir.array_coor %[[X]] [%{{.*}}] %[[YINDEX2]]
// CHECK:    fir.call @Func(%[[XADDR2]])
// CHECK-NEXT: }


// ----

// Test array initialization.
//
// This code has been modified to simplify it - removing the realloc generated to grow
// the constructed 
//subroutine test4(a, b, n1, m1)
//  real :: a(:)  
//  real :: b(:,:)
//          
//  a = [ ((b(i,j), j=1,n1,m1), i=1,n1,m1) ]
//end subroutine test4

  func.func @test4(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "b"}, %arg2: !fir.ref<i32> {fir.bindc_name = "n1"}, %arg3: !fir.ref<i32> {fir.bindc_name = "m1"}) {
    %0 = fir.alloca index {bindc_name = ".buff.pos"}
    %1 = fir.alloca index {bindc_name = ".buff.size"}
    %c0 = arith.constant 0 : index
    %2:3 = fir.box_dims %arg0, %c0 : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
    %3 = fir.array_load %arg0 : (!fir.box<!fir.array<?xf32>>) -> !fir.array<?xf32>
    %c0_0 = arith.constant 0 : index
    fir.store %c0_0 to %0 : !fir.ref<index>
    %c32 = arith.constant 32 : index
    %4 = fir.allocmem f32, %c32
    fir.store %c32 to %1 : !fir.ref<index>
    %c1_i64 = arith.constant 1 : i64
    %5 = fir.convert %c1_i64 : (i64) -> index
    %6 = fir.load %arg2 : !fir.ref<i32>
    %7 = fir.convert %6 : (i32) -> i64
    %8 = fir.convert %7 : (i64) -> index
    %9 = fir.load %arg3 : !fir.ref<i32>
    %10 = fir.convert %9 : (i32) -> i64
    %11 = fir.convert %10 : (i64) -> index
    %12 = fir.do_loop %arg4 = %5 to %8 step %11 iter_args(%arg5 = %4) -> (!fir.heap<f32>) {
      %c1_i64_2 = arith.constant 1 : i64
      %19 = fir.convert %c1_i64_2 : (i64) -> index
      %20 = fir.load %arg2 : !fir.ref<i32>
      %21 = fir.convert %20 : (i32) -> i64
      %22 = fir.convert %21 : (i64) -> index
      %23 = fir.load %arg3 : !fir.ref<i32>
      %24 = fir.convert %23 : (i32) -> i64
      %25 = fir.convert %24 : (i64) -> index
      %26 = fir.do_loop %arg6 = %19 to %22 step %25 iter_args(%arg7 = %arg5) -> (!fir.heap<f32>) {
        %27 = fir.convert %arg4 : (index) -> i32
        %28 = fir.convert %27 : (i32) -> i64
        %c1_i64_3 = arith.constant 1 : i64
        %29 = arith.subi %28, %c1_i64_3 : i64
        %30 = fir.convert %arg6 : (index) -> i32
        %31 = fir.convert %30 : (i32) -> i64
        %c1_i64_4 = arith.constant 1 : i64
        %32 = arith.subi %31, %c1_i64_4 : i64
        %33 = fir.coordinate_of %arg1, %29, %32 : (!fir.box<!fir.array<?x?xf32>>, i64, i64) -> !fir.ref<f32>
        %34 = fir.load %33 : !fir.ref<f32>
        %c1_5 = arith.constant 1 : index
        %35 = fir.zero_bits !fir.ref<!fir.array<?xf32>>
        %36 = fir.coordinate_of %35, %c1_5 : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
        %37 = fir.convert %36 : (!fir.ref<f32>) -> index
        %38 = fir.load %0 : !fir.ref<index>
        %39 = fir.load %1 : !fir.ref<index>
        %c1_6 = arith.constant 1 : index
        %40 = arith.addi %38, %c1_6 : index

        fir.store %40 to %0 : !fir.ref<index>
        fir.result %arg7 : !fir.heap<f32>
      }
      fir.result %26 : !fir.heap<f32>
    }
    %13 = fir.convert %12 : (!fir.heap<f32>) -> !fir.heap<!fir.array<?xf32>>
    %14 = fir.load %0 : !fir.ref<index>
    %15 = fir.shape %14 : (index) -> !fir.shape<1>
    %16 = fir.array_load %13(%15) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.array<?xf32>
    %c1 = arith.constant 1 : index
    %c0_1 = arith.constant 0 : index
    %17 = arith.subi %2#1, %c1 : index
    %18 = fir.do_loop %arg4 = %c0_1 to %17 step %c1 unordered iter_args(%arg5 = %3) -> (!fir.array<?xf32>) {
      %19 = fir.array_fetch %16, %arg4 : (!fir.array<?xf32>, index) -> f32
      %20 = fir.array_update %arg5, %19, %arg4 : (!fir.array<?xf32>, f32, index) -> !fir.array<?xf32>
      fir.result %20 : !fir.array<?xf32>
    }
    fir.array_merge_store %3, %18 to %arg0 : !fir.array<?xf32>, !fir.array<?xf32>, !fir.box<!fir.array<?xf32>>
    fir.freemem %13 : !fir.heap<!fir.array<?xf32>>
    return
  }

// CHECK: func.func @test4(
// CHECK-SAME:            %[[A:.*]]: !fir.box<!fir.array<?xf32>>
// CHECK-SAME:            %[[B:.*]]: !fir.box<!fir.array<?x?xf32>>
// CHECK-SAME:            %[[N1:.*]]: !fir.ref<i32> {{.*}},
// CHECK-SAME:            %[[M1:.*]]: !fir.ref<i32> {{.*}}) {
// CHECK: fir.do_loop
// CHECK:   %[[FOUR:.*]] = arith.constant 4 : index
// CHECK:   %[[COMP:.*]] = arith.cmpi {{.*}}, %[[FOUR]]
// CHECK:   fir.if %[[COMP]] -> {{.*}} {
// CHECK:     %[[CONV:.*]] = fir.convert %[[B]] :
// CHECK:     %[[BOX_ADDR:.*]] = fir.box_addr %[[CONV]]
// CHECK:     %[[RES:.*]] = fir.do_loop {{.*}} {
// CHECK:     %[[ADDR:.*]] = fir.coordinate_of %[[BOX_ADDR]], %{{.*}}
// CHECK:     %{{.*}} = fir.load %[[ADDR]] : !fir.ref<f32>
// CHECK:   }
// CHECK:   fir.result %[[RES]] : {{.*}}
// CHECK: } else {
// CHECK:   %[[RES2:.*]] = fir.do_loop
// CHECK:     %{{.*}} = fir.coordinate_of %[[B]], %{{.*}}
// CHECK:   }
// CHECK:   fir.result %[[RES2]]
// CHECK: }

// -----


// Check that 2D arrays are identified and converted.
// Source code:
//   subroutine sum2d(a, nx, ny)
//    real*8 :: a(:,:)
//    integer :: nx, ny
//    real*8 :: sum
//    integer :: i, j
//    sum = 0
//    do i=1,nx
//       do j=1,ny
//          sum = sum + a(j,i)
//       end do
//    end do
//  end subroutine sum2d

  func.func @sum2d(%arg0: !fir.box<!fir.array<?x?xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "nx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "ny"}) {
    %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QMmoduleFsum2dEi"}
    %1 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QMmoduleFsum2dEj"}
    %2 = fir.alloca f64 {bindc_name = "sum", uniq_name = "_QMmoduleFsum2dEsum"}
    %cst = arith.constant 0.000000e+00 : f64
    fir.store %cst to %2 : !fir.ref<f64>
    %c1_i32 = arith.constant 1 : i32
    %3 = fir.convert %c1_i32 : (i32) -> index
    %4 = fir.load %arg1 : !fir.ref<i32>
    %5 = fir.convert %4 : (i32) -> index
    %c1 = arith.constant 1 : index
    %6 = fir.convert %3 : (index) -> i32
    %7:2 = fir.do_loop %arg3 = %3 to %5 step %c1 iter_args(%arg4 = %6) -> (index, i32) {
      fir.store %arg4 to %0 : !fir.ref<i32>
      %c1_i32_0 = arith.constant 1 : i32
      %8 = fir.convert %c1_i32_0 : (i32) -> index
      %9 = fir.load %arg2 : !fir.ref<i32>
      %10 = fir.convert %9 : (i32) -> index
      %c1_1 = arith.constant 1 : index
      %11 = fir.convert %8 : (index) -> i32
      %12:2 = fir.do_loop %arg5 = %8 to %10 step %c1_1 iter_args(%arg6 = %11) -> (index, i32) {
        fir.store %arg6 to %1 : !fir.ref<i32>
        %17 = fir.load %2 : !fir.ref<f64>
        %18 = fir.load %1 : !fir.ref<i32>
        %19 = fir.convert %18 : (i32) -> i64
        %c1_i64 = arith.constant 1 : i64
        %20 = arith.subi %19, %c1_i64 : i64
        %21 = fir.load %0 : !fir.ref<i32>
        %22 = fir.convert %21 : (i32) -> i64
        %c1_i64_2 = arith.constant 1 : i64
        %23 = arith.subi %22, %c1_i64_2 : i64
        %24 = fir.coordinate_of %arg0, %20, %23 : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %25 = fir.load %24 : !fir.ref<f64>
        %26 = arith.addf %17, %25 fastmath<contract> : f64
        fir.store %26 to %2 : !fir.ref<f64>
        %27 = arith.addi %arg5, %c1_1 : index
        %28 = fir.convert %c1_1 : (index) -> i32
        %29 = fir.load %1 : !fir.ref<i32>
        %30 = arith.addi %29, %28 : i32
        fir.result %27, %30 : index, i32
      }
      fir.store %12#1 to %1 : !fir.ref<i32>
      %13 = arith.addi %arg3, %c1 : index
      %14 = fir.convert %c1 : (index) -> i32
      %15 = fir.load %0 : !fir.ref<i32>
      %16 = arith.addi %15, %14 : i32
      fir.result %13, %16 : index, i32
    }
    fir.store %7#1 to %0 : !fir.ref<i32>
    return
  }

// Note this only checks the expected transformation, not the entire generated code:
// CHECK-LABEL: func.func @sum2d(
// CHECK-SAME:                  %[[ARG0:.*]]: !fir.box<!fir.array<?x?xf64>> {{.*}})
// Only inner loop should be verisoned.
// CHECK: fir.do_loop
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[DIMS0:.*]]:3 = fir.box_dims %[[ARG0]], %[[ZERO]] : {{.*}}
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[DIMS1:.*]]:3 = fir.box_dims %[[ARG0]], %[[ONE]] : {{.*}}
// CHECK: %[[SIZE:.*]] = arith.constant 8 : index
// CHECK: %[[CMP:.*]] = arith.cmpi eq, %[[DIMS0]]#2, %[[SIZE]]
// CHECK: %[[IF_RES:.*]]:2 = fir.if %[[CMP]] -> {{.*}}
// CHECK: %[[NEWARR:.*]] = fir.convert %[[ARG0]]
// CHECK: %[[BOXADDR:.*]] = fir.box_addr %[[NEWARR]] : {{.*}} -> !fir.ref<!fir.array<?xf64>>
// CHECK: %[[LOOP_RES:.*]]:2 = fir.do_loop {{.*}}
// Check the 2D -> 1D coordinate conversion, should have a multiply and a final add.
// Some other operations are checked to synch the different parts.
// CHECK: %[[OUTER_IDX:.*]] = arith.muli %[[DIMS1]]#2, {{.*}}
// CHECK: %[[INNER_IDX:.*]] = fir.convert {{.*}}
// CHECK: %[[ITEMSHIFT:.*]] = arith.constant 3 : index
// CHECK: %[[OUTER_DIV:.*]] = arith.shrsi %[[OUTER_IDX]], %[[ITEMSHIFT]]
// CHECK: %[[C2D:.*]] = arith.addi %[[OUTER_DIV]], %[[INNER_IDX]]
// CHECK: %[[COORD:.*]] = fir.coordinate_of %[[BOXADDR]], %[[C2D]] : (!fir.ref<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// CHECK: %{{.*}} = fir.load %[[COORD]] : !fir.ref<f64>
// CHECK: fir.result %{{.*}}, %{{.*}}
// CHECK: }
// CHECK: fir.result %[[LOOP_RES]]#0, %[[LOOP_RES]]#1
// CHECK: } else {
// CHECK: %[[LOOP_RES2:.*]]:2 = fir.do_loop {{.*}}
// CHECK: %[[COORD2:.*]] = fir.coordinate_of %[[ARG0]], %{{.*}} : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
// CHECK: %{{.*}}= fir.load %[[COORD2]] : !fir.ref<f64>
// CHECK: fir.result %{{.*}}, %{{.*}}
// CHECK: }
// CHECK: fir.result %[[LOOP_RES2]]#0, %[[LOOP_RES2]]#1
// CHECK: }
// CHECK: fir.store %[[IF_RES]]#1 to %{{.*}}
// CHECK: return

// -----

//   subroutine sum3d(a, nx, ny, nz)
//    real*8 :: a(:, :, :)
//    integer :: nx, ny, nz
//    real*8 :: sum
//    integer :: i, j, k
//    sum = 0
//    do k=1,nz
//       do j=1,ny
//          do i=0,nx
//             sum = sum + a(i, j, k)
//          end do
//       end do
//    end do
//  end subroutine sum3d


  func.func @sum3d(%arg0: !fir.box<!fir.array<?x?x?xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "nx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "ny"}, %arg3: !fir.ref<i32> {fir.bindc_name = "nz"}) {
    %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QMmoduleFsum3dEi"}
    %1 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QMmoduleFsum3dEj"}
    %2 = fir.alloca i32 {bindc_name = "k", uniq_name = "_QMmoduleFsum3dEk"}
    %3 = fir.alloca f64 {bindc_name = "sum", uniq_name = "_QMmoduleFsum3dEsum"}
    %cst = arith.constant 0.000000e+00 : f64
    fir.store %cst to %3 : !fir.ref<f64>
    %c1_i32 = arith.constant 1 : i32
    %4 = fir.convert %c1_i32 : (i32) -> index
    %5 = fir.load %arg3 : !fir.ref<i32>
    %6 = fir.convert %5 : (i32) -> index
    %c1 = arith.constant 1 : index
    %7 = fir.convert %4 : (index) -> i32
    %8:2 = fir.do_loop %arg4 = %4 to %6 step %c1 iter_args(%arg5 = %7) -> (index, i32) {
      fir.store %arg5 to %2 : !fir.ref<i32>
      %c1_i32_0 = arith.constant 1 : i32
      %9 = fir.convert %c1_i32_0 : (i32) -> index
      %10 = fir.load %arg2 : !fir.ref<i32>
      %11 = fir.convert %10 : (i32) -> index
      %c1_1 = arith.constant 1 : index
      %12 = fir.convert %9 : (index) -> i32
      %13:2 = fir.do_loop %arg6 = %9 to %11 step %c1_1 iter_args(%arg7 = %12) -> (index, i32) {
        fir.store %arg7 to %1 : !fir.ref<i32>
        %c0_i32 = arith.constant 0 : i32
        %18 = fir.convert %c0_i32 : (i32) -> index
        %19 = fir.load %arg1 : !fir.ref<i32>
        %20 = fir.convert %19 : (i32) -> index
        %c1_2 = arith.constant 1 : index
        %21 = fir.convert %18 : (index) -> i32
        %22:2 = fir.do_loop %arg8 = %18 to %20 step %c1_2 iter_args(%arg9 = %21) -> (index, i32) {
          fir.store %arg9 to %0 : !fir.ref<i32>
          %27 = fir.load %3 : !fir.ref<f64>
          %28 = fir.load %0 : !fir.ref<i32>
          %29 = fir.convert %28 : (i32) -> i64
          %c1_i64 = arith.constant 1 : i64
          %30 = arith.subi %29, %c1_i64 : i64
          %31 = fir.load %1 : !fir.ref<i32>
          %32 = fir.convert %31 : (i32) -> i64
          %c1_i64_3 = arith.constant 1 : i64
          %33 = arith.subi %32, %c1_i64_3 : i64
          %34 = fir.load %2 : !fir.ref<i32>
          %35 = fir.convert %34 : (i32) -> i64
          %c1_i64_4 = arith.constant 1 : i64
          %36 = arith.subi %35, %c1_i64_4 : i64
          %37 = fir.coordinate_of %arg0, %30, %33, %36 : (!fir.box<!fir.array<?x?x?xf64>>, i64, i64, i64) -> !fir.ref<f64>
          %38 = fir.load %37 : !fir.ref<f64>
          %39 = arith.addf %27, %38 fastmath<contract> : f64
          fir.store %39 to %3 : !fir.ref<f64>
          %40 = arith.addi %arg8, %c1_2 : index
          %41 = fir.convert %c1_2 : (index) -> i32
          %42 = fir.load %0 : !fir.ref<i32>
          %43 = arith.addi %42, %41 : i32
          fir.result %40, %43 : index, i32
        }
        fir.store %22#1 to %0 : !fir.ref<i32>
        %23 = arith.addi %arg6, %c1_1 : index
        %24 = fir.convert %c1_1 : (index) -> i32
        %25 = fir.load %1 : !fir.ref<i32>
        %26 = arith.addi %25, %24 : i32
        fir.result %23, %26 : index, i32
      }
      fir.store %13#1 to %1 : !fir.ref<i32>
      %14 = arith.addi %arg4, %c1 : index
      %15 = fir.convert %c1 : (index) -> i32
      %16 = fir.load %2 : !fir.ref<i32>
      %17 = arith.addi %16, %15 : i32
      fir.result %14, %17 : index, i32
    }
    fir.store %8#1 to %2 : !fir.ref<i32>
    return
  }
  
// Note this only checks the expected transformation, not the entire generated code:
// CHECK-LABEL: func.func @sum3d(
// CHECK-SAME:                  %[[ARG0:.*]]: !fir.box<!fir.array<?x?x?xf64>> {{.*}})
// Only inner loop should be verisoned.
// CHECK: fir.do_loop
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[DIMS0:.*]]:3 = fir.box_dims %[[ARG0]], %[[ZERO]] : {{.*}}
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[DIMS1:.*]]:3 = fir.box_dims %[[ARG0]], %[[ONE]] : {{.*}}
// CHECK: %[[TWO:.*]] = arith.constant 2 : index
// CHECK: %[[DIMS2:.*]]:3 = fir.box_dims %[[ARG0]], %[[TWO]] : {{.*}}
// CHECK: %[[SIZE:.*]] = arith.constant 8 : index
// CHECK: %[[CMP:.*]] = arith.cmpi eq, %[[DIMS0]]#2, %[[SIZE]]
// CHECK: %[[IF_RES:.*]]:2 = fir.if %[[CMP]] -> {{.*}}
// CHECK: %[[NEWARR:.*]] = fir.convert %[[ARG0]]
// CHECK: %[[BOXADDR:.*]] = fir.box_addr %[[NEWARR]] : {{.*}} -> !fir.ref<!fir.array<?xf64>>
// CHECK: %[[LOOP_RES:.*]]:2 = fir.do_loop {{.*}}
// Check the 3D -> 1D coordinate conversion, should have a multiply and a final add.
// Some other operations are checked to synch the different parts.
// CHECK: %[[OUTER_IDX:.*]] = arith.muli %[[DIMS2]]#2, {{.*}}
// CHECK: %[[MIDDLE_IDX:.*]] = arith.muli %[[DIMS1]]#2, {{.*}}
// CHECK: %[[MIDDLE_SUM:.*]] = arith.addi %[[MIDDLE_IDX]], %[[OUTER_IDX]]
// CHECK: %[[INNER_IDX:.*]] = fir.convert {{.*}}
// CHECK: %[[ITEMSHIFT:.*]] = arith.constant 3 : index
// CHECK: %[[MIDDLE_DIV:.*]] = arith.shrsi %[[MIDDLE_SUM]], %[[ITEMSHIFT]]
// CHECK: %[[C3D:.*]] = arith.addi %[[MIDDLE_DIV]], %[[INNER_IDX]]
// CHECK: %[[COORD:.*]] = fir.coordinate_of %[[BOXADDR]], %[[C3D]] : (!fir.ref<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// CHECK: %{{.*}} = fir.load %[[COORD]] : !fir.ref<f64>
// CHECK: fir.result %{{.*}}, %{{.*}}
// CHECK: }
// CHECK: fir.result %[[LOOP_RES]]#0, %[[LOOP_RES]]#1
// CHECK: } else {
// CHECK: %[[LOOP_RES2:.*]]:2 = fir.do_loop {{.*}}
// CHECK: %[[COORD2:.*]] = fir.coordinate_of %[[ARG0]], %{{.*}} : (!fir.box<!fir.array<?x?x?xf64>>, i64, i64, i64) -> !fir.ref<f64>
// CHECK: %{{.*}}= fir.load %[[COORD2]] : !fir.ref<f64>
// CHECK: fir.result %{{.*}}, %{{.*}}
// CHECK: }
// CHECK: fir.result %[[LOOP_RES2]]#0, %[[LOOP_RES2]]#1
// CHECK: }
// CHECK: fir.store %[[IF_RES]]#1 to %{{.*}}
// CHECK: return

// Check that OPTIONAL argument's box is not accessed illegally
// before the loop.
func.func @test_optional_arg(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.optional}) {
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %c0_i64 = arith.constant 0 : i64
  %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QMcheck_modFtestEi"}
  %1 = fir.convert %c1 : (index) -> i32
  %2:2 = fir.do_loop %arg1 = %c1 to %c20 step %c1 iter_args(%arg2 = %1) -> (index, i32) {
    fir.store %arg2 to %0 : !fir.ref<i32>
    %3 = fir.is_present %arg0 : (!fir.box<!fir.array<?xf32>>) -> i1
    fir.if %3 {
      %8 = fir.coordinate_of %arg0, %c0_i64 : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
    } else {
    }
    %4 = arith.addi %arg1, %c1 : index
    %5 = fir.convert %c1 : (index) -> i32
    %6 = fir.load %0 : !fir.ref<i32>
    %7 = arith.addi %6, %5 : i32
    fir.result %4, %7 : index, i32
  }
  fir.store %2#1 to %0 : !fir.ref<i32>
  return
}
// CHECK-LABEL:   func.func @test_optional_arg(
// CHECK-SAME:                                 %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x", fir.optional}) {
// CHECK-NEXT:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:           %[[VAL_2:.*]] = arith.constant 20 : index
// CHECK-NEXT:           %[[VAL_3:.*]] = arith.constant 0 : i64
// CHECK-NEXT:           %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QMcheck_modFtestEi"}
// CHECK-NEXT:           %[[VAL_5:.*]] = fir.convert %[[VAL_1]] : (index) -> i32
// CHECK-NEXT:           %[[VAL_6:.*]]:2 = fir.do_loop %[[VAL_7:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_1]] iter_args(%[[VAL_8:.*]] = %[[VAL_5]]) -> (index, i32) {
// CHECK-NEXT:             fir.store %[[VAL_8]] to %[[VAL_4]] : !fir.ref<i32>
// CHECK-NEXT:             %[[VAL_9:.*]] = fir.is_present %[[VAL_0]] : (!fir.box<!fir.array<?xf32>>) -> i1
// CHECK-NEXT:             fir.if %[[VAL_9]] {
// CHECK-NEXT:               %[[VAL_10:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_3]] : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
// CHECK-NEXT:             } else {
// CHECK-NEXT:             }
// CHECK-NEXT:             %[[VAL_11:.*]] = arith.addi %[[VAL_7]], %[[VAL_1]] : index
// CHECK-NEXT:             %[[VAL_12:.*]] = fir.convert %[[VAL_1]] : (index) -> i32
// CHECK-NEXT:             %[[VAL_13:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
// CHECK-NEXT:             %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_12]] : i32
// CHECK-NEXT:             fir.result %[[VAL_11]], %[[VAL_14]] : index, i32
// CHECK-NEXT:           }
// CHECK-NEXT:           fir.store %[[VAL_15:.*]]#1 to %[[VAL_4]] : !fir.ref<i32>
// CHECK-NEXT:           return
// CHECK-NEXT:         }

  // test sum1d with hlfir lowering
  func.func @_QPsum1d(%arg0: !fir.box<!fir.array<?xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %0 = fir.declare %arg0 {uniq_name = "_QFsum1dEa"} : (!fir.box<!fir.array<?xf64>>) -> !fir.box<!fir.array<?xf64>>
    %1 = fir.rebox %0 : (!fir.box<!fir.array<?xf64>>) -> !fir.box<!fir.array<?xf64>>
    %2 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsum1dEi"}
    %3 = fir.declare %2 {uniq_name = "_QFsum1dEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %4 = fir.declare %arg1 {uniq_name = "_QFsum1dEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %5 = fir.alloca f64 {bindc_name = "sum", uniq_name = "_QFsum1dEsum"}
    %6 = fir.declare %5 {uniq_name = "_QFsum1dEsum"} : (!fir.ref<f64>) -> !fir.ref<f64>
    fir.store %cst to %6 : !fir.ref<f64>
    %7 = fir.load %4 : !fir.ref<i32>
    %8 = fir.convert %7 : (i32) -> index
    %9 = fir.convert %c1 : (index) -> i32
    %10:2 = fir.do_loop %arg2 = %c1 to %8 step %c1 iter_args(%arg3 = %9) -> (index, i32) {
      fir.store %arg3 to %3 : !fir.ref<i32>
      %11 = fir.load %6 : !fir.ref<f64>
      %12 = fir.load %3 : !fir.ref<i32>
      %13 = fir.convert %12 : (i32) -> i64
      %14 = fir.array_coor %1 %13 : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
      %15 = fir.load %14 : !fir.ref<f64>
      %16 = arith.addf %11, %15 fastmath<contract> : f64
      fir.store %16 to %6 : !fir.ref<f64>
      %17 = arith.addi %arg2, %c1 : index
      %18 = fir.load %3 : !fir.ref<i32>
      %19 = arith.addi %18, %9 : i32
      fir.result %17, %19 : index, i32
    }
    fir.store %10#1 to %3 : !fir.ref<i32>
    return
  }
// CHECK-LABEL:   func.func @_QPsum1d(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.box<!fir.array<?xf64>> {fir.bindc_name = "a"},
// CHECK-SAME:                        %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[VAL_4:.*]] = fir.declare %[[VAL_0]] {uniq_name = "_QFsum1dEa"} : (!fir.box<!fir.array<?xf64>>) -> !fir.box<!fir.array<?xf64>>
// CHECK:           %[[VAL_5:.*]] = fir.rebox %[[VAL_4]] : (!fir.box<!fir.array<?xf64>>) -> !fir.box<!fir.array<?xf64>>
// CHECK:           %[[VAL_6:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsum1dEi"}
// CHECK:           %[[VAL_7:.*]] = fir.declare %[[VAL_6]] {uniq_name = "_QFsum1dEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_8:.*]] = fir.declare %[[VAL_1]] {uniq_name = "_QFsum1dEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_9:.*]] = fir.alloca f64 {bindc_name = "sum", uniq_name = "_QFsum1dEsum"}
// CHECK:           %[[VAL_10:.*]] = fir.declare %[[VAL_9]] {uniq_name = "_QFsum1dEsum"} : (!fir.ref<f64>) -> !fir.ref<f64>
// CHECK:           fir.store %[[VAL_3]] to %[[VAL_10]] : !fir.ref<f64>
// CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
// CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i32) -> index
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_2]] : (index) -> i32
// CHECK:           %[[VAL_14:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_15:.*]]:3 = fir.box_dims %[[VAL_5]], %[[VAL_14]] : (!fir.box<!fir.array<?xf64>>, index) -> (index, index, index)
// CHECK:           %[[VAL_16:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_17:.*]] = arith.cmpi eq, %[[VAL_15]]#2, %[[VAL_16]] : index
// CHECK:           %[[VAL_18:.*]]:2 = fir.if %[[VAL_17]] -> (index, i32) {
// CHECK:             %[[VAL_19:.*]] = fir.convert %[[VAL_5]] : (!fir.box<!fir.array<?xf64>>) -> !fir.box<!fir.array<?xf64>>
// CHECK:             %[[VAL_20:.*]] = fir.box_addr %[[VAL_19]] : (!fir.box<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>>
// CHECK:             %[[VAL_21:.*]]:2 = fir.do_loop %[[VAL_22:.*]] = %[[VAL_2]] to %[[VAL_12]] step %[[VAL_2]] iter_args(%[[VAL_23:.*]] = %[[VAL_13]]) -> (index, i32) {
// CHECK:               fir.store %[[VAL_23]] to %[[VAL_7]] : !fir.ref<i32>
// CHECK:               %[[VAL_24:.*]] = fir.load %[[VAL_10]] : !fir.ref<f64>
// CHECK:               %[[VAL_25:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
// CHECK:               %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i32) -> i64
// CHECK:               %[[VAL_27:.*]] = arith.constant 1 : i64
// CHECK:               %[[VAL_28:.*]] = arith.subi %[[VAL_26]], %[[VAL_27]] : i64
// CHECK:               %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i64) -> index
// CHECK:               %[[VAL_30:.*]] = fir.coordinate_of %[[VAL_20]], %[[VAL_29]] : (!fir.ref<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// CHECK:               %[[VAL_31:.*]] = fir.load %[[VAL_30]] : !fir.ref<f64>
// CHECK:               %[[VAL_32:.*]] = arith.addf %[[VAL_24]], %[[VAL_31]] fastmath<contract> : f64
// CHECK:               fir.store %[[VAL_32]] to %[[VAL_10]] : !fir.ref<f64>
// CHECK:               %[[VAL_33:.*]] = arith.addi %[[VAL_22]], %[[VAL_2]] : index
// CHECK:               %[[VAL_34:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
// CHECK:               %[[VAL_35:.*]] = arith.addi %[[VAL_34]], %[[VAL_13]] : i32
// CHECK:               fir.result %[[VAL_33]], %[[VAL_35]] : index, i32
// CHECK:             }
// CHECK:             fir.result %[[VAL_36:.*]]#0, %[[VAL_36]]#1 : index, i32
// CHECK:           } else {
// CHECK:             %[[VAL_37:.*]]:2 = fir.do_loop %[[VAL_38:.*]] = %[[VAL_2]] to %[[VAL_12]] step %[[VAL_2]] iter_args(%[[VAL_39:.*]] = %[[VAL_13]]) -> (index, i32) {
// CHECK:               fir.store %[[VAL_39]] to %[[VAL_7]] : !fir.ref<i32>
// CHECK:               %[[VAL_40:.*]] = fir.load %[[VAL_10]] : !fir.ref<f64>
// CHECK:               %[[VAL_41:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
// CHECK:               %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (i32) -> i64
// CHECK:               %[[VAL_43:.*]] = fir.array_coor %[[VAL_5]] %[[VAL_42]] : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
// CHECK:               %[[VAL_44:.*]] = fir.load %[[VAL_43]] : !fir.ref<f64>
// CHECK:               %[[VAL_45:.*]] = arith.addf %[[VAL_40]], %[[VAL_44]] fastmath<contract> : f64
// CHECK:               fir.store %[[VAL_45]] to %[[VAL_10]] : !fir.ref<f64>
// CHECK:               %[[VAL_46:.*]] = arith.addi %[[VAL_38]], %[[VAL_2]] : index
// CHECK:               %[[VAL_47:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
// CHECK:               %[[VAL_48:.*]] = arith.addi %[[VAL_47]], %[[VAL_13]] : i32
// CHECK:               fir.result %[[VAL_46]], %[[VAL_48]] : index, i32
// CHECK:             }
// CHECK:             fir.result %[[VAL_49:.*]]#0, %[[VAL_49]]#1 : index, i32
// CHECK:           }
// CHECK:           fir.store %[[VAL_50:.*]]#1 to %[[VAL_7]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }

// test sum2d with hlfir lowering
  func.func @_QPsum2d(%arg0: !fir.box<!fir.array<?x?xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "nx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "ny"}) {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %0 = fir.declare %arg0 {uniq_name = "_QFsum2dEa"} : (!fir.box<!fir.array<?x?xf64>>) -> !fir.box<!fir.array<?x?xf64>>
    %1 = fir.rebox %0 : (!fir.box<!fir.array<?x?xf64>>) -> !fir.box<!fir.array<?x?xf64>>
    %2 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsum2dEi"}
    %3 = fir.declare %2 {uniq_name = "_QFsum2dEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %4 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFsum2dEj"}
    %5 = fir.declare %4 {uniq_name = "_QFsum2dEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %6 = fir.declare %arg1 {uniq_name = "_QFsum2dEnx"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %7 = fir.declare %arg2 {uniq_name = "_QFsum2dEny"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %8 = fir.alloca f64 {bindc_name = "sum", uniq_name = "_QFsum2dEsum"}
    %9 = fir.declare %8 {uniq_name = "_QFsum2dEsum"} : (!fir.ref<f64>) -> !fir.ref<f64>
    fir.store %cst to %9 : !fir.ref<f64>
    %10 = fir.load %6 : !fir.ref<i32>
    %11 = fir.convert %10 : (i32) -> index
    %12 = fir.convert %c1 : (index) -> i32
    %13:2 = fir.do_loop %arg3 = %c1 to %11 step %c1 iter_args(%arg4 = %12) -> (index, i32) {
      fir.store %arg4 to %3 : !fir.ref<i32>
      %14 = fir.load %7 : !fir.ref<i32>
      %15 = fir.convert %14 : (i32) -> index
      %16:2 = fir.do_loop %arg5 = %c1 to %15 step %c1 iter_args(%arg6 = %12) -> (index, i32) {
        fir.store %arg6 to %5 : !fir.ref<i32>
        %20 = fir.load %9 : !fir.ref<f64>
        %21 = fir.load %5 : !fir.ref<i32>
        %22 = fir.convert %21 : (i32) -> i64
        %23 = fir.load %3 : !fir.ref<i32>
        %24 = fir.convert %23 : (i32) -> i64
        %25 = fir.array_coor %1 %22, %24 : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
        %26 = fir.load %25 : !fir.ref<f64>
        %27 = arith.addf %20, %26 fastmath<contract> : f64
        fir.store %27 to %9 : !fir.ref<f64>
        %28 = arith.addi %arg5, %c1 : index
        %29 = fir.load %5 : !fir.ref<i32>
        %30 = arith.addi %29, %12 : i32
        fir.result %28, %30 : index, i32
      }
      fir.store %16#1 to %5 : !fir.ref<i32>
      %17 = arith.addi %arg3, %c1 : index
      %18 = fir.load %3 : !fir.ref<i32>
      %19 = arith.addi %18, %12 : i32
      fir.result %17, %19 : index, i32
    }
    fir.store %13#1 to %3 : !fir.ref<i32>
    return
  }
// CHECK-LABEL:   func.func @_QPsum2d(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf64>> {fir.bindc_name = "a"},
// CHECK-SAME:                        %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "nx"},
// CHECK-SAME:                        %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "ny"}) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[VAL_5:.*]] = fir.declare %[[VAL_0]] {uniq_name = "_QFsum2dEa"} : (!fir.box<!fir.array<?x?xf64>>) -> !fir.box<!fir.array<?x?xf64>>
// CHECK:           %[[VAL_6:.*]] = fir.rebox %[[VAL_5]] : (!fir.box<!fir.array<?x?xf64>>) -> !fir.box<!fir.array<?x?xf64>>
// CHECK:           %[[VAL_7:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsum2dEi"}
// CHECK:           %[[VAL_8:.*]] = fir.declare %[[VAL_7]] {uniq_name = "_QFsum2dEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_9:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFsum2dEj"}
// CHECK:           %[[VAL_10:.*]] = fir.declare %[[VAL_9]] {uniq_name = "_QFsum2dEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_11:.*]] = fir.declare %[[VAL_1]] {uniq_name = "_QFsum2dEnx"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_12:.*]] = fir.declare %[[VAL_2]] {uniq_name = "_QFsum2dEny"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_13:.*]] = fir.alloca f64 {bindc_name = "sum", uniq_name = "_QFsum2dEsum"}
// CHECK:           %[[VAL_14:.*]] = fir.declare %[[VAL_13]] {uniq_name = "_QFsum2dEsum"} : (!fir.ref<f64>) -> !fir.ref<f64>
// CHECK:           fir.store %[[VAL_4]] to %[[VAL_14]] : !fir.ref<f64>
// CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_11]] : !fir.ref<i32>
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> index
// CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_3]] : (index) -> i32
// CHECK:           %[[VAL_18:.*]]:2 = fir.do_loop %[[VAL_19:.*]] = %[[VAL_3]] to %[[VAL_16]] step %[[VAL_3]] iter_args(%[[VAL_20:.*]] = %[[VAL_17]]) -> (index, i32) {
// CHECK:             fir.store %[[VAL_20]] to %[[VAL_8]] : !fir.ref<i32>
// CHECK:             %[[VAL_21:.*]] = fir.load %[[VAL_12]] : !fir.ref<i32>
// CHECK:             %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i32) -> index
// CHECK:             %[[VAL_23:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_24:.*]]:3 = fir.box_dims %[[VAL_6]], %[[VAL_23]] : (!fir.box<!fir.array<?x?xf64>>, index) -> (index, index, index)
// CHECK:             %[[VAL_25:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_26:.*]]:3 = fir.box_dims %[[VAL_6]], %[[VAL_25]] : (!fir.box<!fir.array<?x?xf64>>, index) -> (index, index, index)
// CHECK:             %[[VAL_27:.*]] = arith.constant 8 : index
// CHECK:             %[[VAL_28:.*]] = arith.cmpi eq, %[[VAL_24]]#2, %[[VAL_27]] : index
// CHECK:             %[[VAL_29:.*]]:2 = fir.if %[[VAL_28]] -> (index, i32) {
// CHECK:               %[[VAL_30:.*]] = fir.convert %[[VAL_6]] : (!fir.box<!fir.array<?x?xf64>>) -> !fir.box<!fir.array<?xf64>>
// CHECK:               %[[VAL_31:.*]] = fir.box_addr %[[VAL_30]] : (!fir.box<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>>
// CHECK:               %[[VAL_32:.*]]:2 = fir.do_loop %[[VAL_33:.*]] = %[[VAL_3]] to %[[VAL_22]] step %[[VAL_3]] iter_args(%[[VAL_34:.*]] = %[[VAL_17]]) -> (index, i32) {
// CHECK:                 fir.store %[[VAL_34]] to %[[VAL_10]] : !fir.ref<i32>
// CHECK:                 %[[VAL_35:.*]] = fir.load %[[VAL_14]] : !fir.ref<f64>
// CHECK:                 %[[VAL_36:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
// CHECK:                 %[[VAL_37:.*]] = fir.convert %[[VAL_36]] : (i32) -> i64
// CHECK:                 %[[VAL_38:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
// CHECK:                 %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (i32) -> i64
// CHECK:                 %[[VAL_40:.*]] = arith.constant 1 : i64
// CHECK:                 %[[VAL_41:.*]] = arith.subi %[[VAL_39]], %[[VAL_40]] : i64
// CHECK:                 %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (i64) -> index
// CHECK:                 %[[VAL_43:.*]] = arith.muli %[[VAL_26]]#2, %[[VAL_42]] : index
// CHECK:                 %[[VAL_44:.*]] = arith.constant 1 : i64
// CHECK:                 %[[VAL_45:.*]] = arith.subi %[[VAL_37]], %[[VAL_44]] : i64
// CHECK:                 %[[VAL_46:.*]] = fir.convert %[[VAL_45]] : (i64) -> index
// CHECK:                 %[[VAL_47:.*]] = arith.constant 3 : index
// CHECK:                 %[[VAL_48:.*]] = arith.shrsi %[[VAL_43]], %[[VAL_47]] : index
// CHECK:                 %[[VAL_49:.*]] = arith.addi %[[VAL_48]], %[[VAL_46]] : index
// CHECK:                 %[[VAL_50:.*]] = fir.coordinate_of %[[VAL_31]], %[[VAL_49]] : (!fir.ref<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// CHECK:                 %[[VAL_51:.*]] = fir.load %[[VAL_50]] : !fir.ref<f64>
// CHECK:                 %[[VAL_52:.*]] = arith.addf %[[VAL_35]], %[[VAL_51]] fastmath<contract> : f64
// CHECK:                 fir.store %[[VAL_52]] to %[[VAL_14]] : !fir.ref<f64>
// CHECK:                 %[[VAL_53:.*]] = arith.addi %[[VAL_33]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_54:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
// CHECK:                 %[[VAL_55:.*]] = arith.addi %[[VAL_54]], %[[VAL_17]] : i32
// CHECK:                 fir.result %[[VAL_53]], %[[VAL_55]] : index, i32
// CHECK:               }
// CHECK:               fir.result %[[VAL_56:.*]]#0, %[[VAL_56]]#1 : index, i32
// CHECK:             } else {
// CHECK:               %[[VAL_57:.*]]:2 = fir.do_loop %[[VAL_58:.*]] = %[[VAL_3]] to %[[VAL_22]] step %[[VAL_3]] iter_args(%[[VAL_59:.*]] = %[[VAL_17]]) -> (index, i32) {
// CHECK:                 fir.store %[[VAL_59]] to %[[VAL_10]] : !fir.ref<i32>
// CHECK:                 %[[VAL_60:.*]] = fir.load %[[VAL_14]] : !fir.ref<f64>
// CHECK:                 %[[VAL_61:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
// CHECK:                 %[[VAL_62:.*]] = fir.convert %[[VAL_61]] : (i32) -> i64
// CHECK:                 %[[VAL_63:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
// CHECK:                 %[[VAL_64:.*]] = fir.convert %[[VAL_63]] : (i32) -> i64
// CHECK:                 %[[VAL_65:.*]] = fir.array_coor %[[VAL_6]] %[[VAL_62]], %[[VAL_64]] : (!fir.box<!fir.array<?x?xf64>>, i64, i64) -> !fir.ref<f64>
// CHECK:                 %[[VAL_66:.*]] = fir.load %[[VAL_65]] : !fir.ref<f64>
// CHECK:                 %[[VAL_67:.*]] = arith.addf %[[VAL_60]], %[[VAL_66]] fastmath<contract> : f64
// CHECK:                 fir.store %[[VAL_67]] to %[[VAL_14]] : !fir.ref<f64>
// CHECK:                 %[[VAL_68:.*]] = arith.addi %[[VAL_58]], %[[VAL_3]] : index
// CHECK:                 %[[VAL_69:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
// CHECK:                 %[[VAL_70:.*]] = arith.addi %[[VAL_69]], %[[VAL_17]] : i32
// CHECK:                 fir.result %[[VAL_68]], %[[VAL_70]] : index, i32
// CHECK:               }
// CHECK:               fir.result %[[VAL_71:.*]]#0, %[[VAL_71]]#1 : index, i32
// CHECK:             }
// CHECK:             fir.store %[[VAL_72:.*]]#1 to %[[VAL_10]] : !fir.ref<i32>
// CHECK:             %[[VAL_73:.*]] = arith.addi %[[VAL_19]], %[[VAL_3]] : index
// CHECK:             %[[VAL_74:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
// CHECK:             %[[VAL_75:.*]] = arith.addi %[[VAL_74]], %[[VAL_17]] : i32
// CHECK:             fir.result %[[VAL_73]], %[[VAL_75]] : index, i32
// CHECK:           }
// CHECK:           fir.store %[[VAL_76:.*]]#1 to %[[VAL_8]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }

// test sum3d with hlfir lowering
  func.func @_QPsum3d(%arg0: !fir.box<!fir.array<?x?x?xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "nx"}, %arg2: !fir.ref<i32> {fir.bindc_name = "ny"}, %arg3: !fir.ref<i32> {fir.bindc_name = "nz"}) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f64
    %0 = fir.declare %arg0 {uniq_name = "_QFsum3dEa"} : (!fir.box<!fir.array<?x?x?xf64>>) -> !fir.box<!fir.array<?x?x?xf64>>
    %1 = fir.rebox %0 : (!fir.box<!fir.array<?x?x?xf64>>) -> !fir.box<!fir.array<?x?x?xf64>>
    %2 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsum3dEi"}
    %3 = fir.declare %2 {uniq_name = "_QFsum3dEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %4 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFsum3dEj"}
    %5 = fir.declare %4 {uniq_name = "_QFsum3dEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %6 = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFsum3dEk"}
    %7 = fir.declare %6 {uniq_name = "_QFsum3dEk"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %8 = fir.declare %arg1 {uniq_name = "_QFsum3dEnx"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %9 = fir.declare %arg2 {uniq_name = "_QFsum3dEny"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %10 = fir.declare %arg3 {uniq_name = "_QFsum3dEnz"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %11 = fir.alloca f64 {bindc_name = "sum", uniq_name = "_QFsum3dEsum"}
    %12 = fir.declare %11 {uniq_name = "_QFsum3dEsum"} : (!fir.ref<f64>) -> !fir.ref<f64>
    fir.store %cst to %12 : !fir.ref<f64>
    %13 = fir.load %10 : !fir.ref<i32>
    %14 = fir.convert %13 : (i32) -> index
    %15 = fir.convert %c1 : (index) -> i32
    %16:2 = fir.do_loop %arg4 = %c1 to %14 step %c1 iter_args(%arg5 = %15) -> (index, i32) {
      fir.store %arg5 to %7 : !fir.ref<i32>
      %17 = fir.load %9 : !fir.ref<i32>
      %18 = fir.convert %17 : (i32) -> index
      %19:2 = fir.do_loop %arg6 = %c1 to %18 step %c1 iter_args(%arg7 = %15) -> (index, i32) {
        fir.store %arg7 to %5 : !fir.ref<i32>
        %23 = fir.load %8 : !fir.ref<i32>
        %24 = fir.convert %23 : (i32) -> index
        %25 = fir.convert %c0 : (index) -> i32
        %26:2 = fir.do_loop %arg8 = %c0 to %24 step %c1 iter_args(%arg9 = %25) -> (index, i32) {
          fir.store %arg9 to %3 : !fir.ref<i32>
          %30 = fir.load %12 : !fir.ref<f64>
          %31 = fir.load %3 : !fir.ref<i32>
          %32 = fir.convert %31 : (i32) -> i64
          %33 = fir.load %5 : !fir.ref<i32>
          %34 = fir.convert %33 : (i32) -> i64
          %35 = fir.load %7 : !fir.ref<i32>
          %36 = fir.convert %35 : (i32) -> i64
          %37 = fir.array_coor %1 %32, %34, %36 : (!fir.box<!fir.array<?x?x?xf64>>, i64, i64, i64) -> !fir.ref<f64>
          %38 = fir.load %37 : !fir.ref<f64>
          %39 = arith.addf %30, %38 fastmath<contract> : f64
          fir.store %39 to %12 : !fir.ref<f64>
          %40 = arith.addi %arg8, %c1 : index
          %41 = fir.load %3 : !fir.ref<i32>
          %42 = arith.addi %41, %15 : i32
          fir.result %40, %42 : index, i32
        }
        fir.store %26#1 to %3 : !fir.ref<i32>
        %27 = arith.addi %arg6, %c1 : index
        %28 = fir.load %5 : !fir.ref<i32>
        %29 = arith.addi %28, %15 : i32
        fir.result %27, %29 : index, i32
      }
      fir.store %19#1 to %5 : !fir.ref<i32>
      %20 = arith.addi %arg4, %c1 : index
      %21 = fir.load %7 : !fir.ref<i32>
      %22 = arith.addi %21, %15 : i32
      fir.result %20, %22 : index, i32
    }
    fir.store %16#1 to %7 : !fir.ref<i32>
    return
  }
// CHECK-LABEL:   func.func @_QPsum3d(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.box<!fir.array<?x?x?xf64>> {fir.bindc_name = "a"},
// CHECK-SAME:                        %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "nx"},
// CHECK-SAME:                        %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "ny"},
// CHECK-SAME:                        %[[VAL_3:.*]]: !fir.ref<i32> {fir.bindc_name = "nz"}) {
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[VAL_7:.*]] = fir.declare %[[VAL_0]] {uniq_name = "_QFsum3dEa"} : (!fir.box<!fir.array<?x?x?xf64>>) -> !fir.box<!fir.array<?x?x?xf64>>
// CHECK:           %[[VAL_8:.*]] = fir.rebox %[[VAL_7]] : (!fir.box<!fir.array<?x?x?xf64>>) -> !fir.box<!fir.array<?x?x?xf64>>
// CHECK:           %[[VAL_9:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFsum3dEi"}
// CHECK:           %[[VAL_10:.*]] = fir.declare %[[VAL_9]] {uniq_name = "_QFsum3dEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_11:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFsum3dEj"}
// CHECK:           %[[VAL_12:.*]] = fir.declare %[[VAL_11]] {uniq_name = "_QFsum3dEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_13:.*]] = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFsum3dEk"}
// CHECK:           %[[VAL_14:.*]] = fir.declare %[[VAL_13]] {uniq_name = "_QFsum3dEk"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_15:.*]] = fir.declare %[[VAL_1]] {uniq_name = "_QFsum3dEnx"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_16:.*]] = fir.declare %[[VAL_2]] {uniq_name = "_QFsum3dEny"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_17:.*]] = fir.declare %[[VAL_3]] {uniq_name = "_QFsum3dEnz"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_18:.*]] = fir.alloca f64 {bindc_name = "sum", uniq_name = "_QFsum3dEsum"}
// CHECK:           %[[VAL_19:.*]] = fir.declare %[[VAL_18]] {uniq_name = "_QFsum3dEsum"} : (!fir.ref<f64>) -> !fir.ref<f64>
// CHECK:           fir.store %[[VAL_6]] to %[[VAL_19]] : !fir.ref<f64>
// CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_17]] : !fir.ref<i32>
// CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> index
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_5]] : (index) -> i32
// CHECK:           %[[VAL_23:.*]]:2 = fir.do_loop %[[VAL_24:.*]] = %[[VAL_5]] to %[[VAL_21]] step %[[VAL_5]] iter_args(%[[VAL_25:.*]] = %[[VAL_22]]) -> (index, i32) {
// CHECK:             fir.store %[[VAL_25]] to %[[VAL_14]] : !fir.ref<i32>
// CHECK:             %[[VAL_26:.*]] = fir.load %[[VAL_16]] : !fir.ref<i32>
// CHECK:             %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i32) -> index
// CHECK:             %[[VAL_28:.*]]:2 = fir.do_loop %[[VAL_29:.*]] = %[[VAL_5]] to %[[VAL_27]] step %[[VAL_5]] iter_args(%[[VAL_30:.*]] = %[[VAL_22]]) -> (index, i32) {
// CHECK:               fir.store %[[VAL_30]] to %[[VAL_12]] : !fir.ref<i32>
// CHECK:               %[[VAL_31:.*]] = fir.load %[[VAL_15]] : !fir.ref<i32>
// CHECK:               %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i32) -> index
// CHECK:               %[[VAL_33:.*]] = fir.convert %[[VAL_4]] : (index) -> i32
// CHECK:               %[[VAL_34:.*]] = arith.constant 0 : index
// CHECK:               %[[VAL_35:.*]]:3 = fir.box_dims %[[VAL_8]], %[[VAL_34]] : (!fir.box<!fir.array<?x?x?xf64>>, index) -> (index, index, index)
// CHECK:               %[[VAL_36:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_37:.*]]:3 = fir.box_dims %[[VAL_8]], %[[VAL_36]] : (!fir.box<!fir.array<?x?x?xf64>>, index) -> (index, index, index)
// CHECK:               %[[VAL_38:.*]] = arith.constant 2 : index
// CHECK:               %[[VAL_39:.*]]:3 = fir.box_dims %[[VAL_8]], %[[VAL_38]] : (!fir.box<!fir.array<?x?x?xf64>>, index) -> (index, index, index)
// CHECK:               %[[VAL_40:.*]] = arith.constant 8 : index
// CHECK:               %[[VAL_41:.*]] = arith.cmpi eq, %[[VAL_35]]#2, %[[VAL_40]] : index
// CHECK:               %[[VAL_42:.*]]:2 = fir.if %[[VAL_41]] -> (index, i32) {
// CHECK:                 %[[VAL_43:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.array<?x?x?xf64>>) -> !fir.box<!fir.array<?xf64>>
// CHECK:                 %[[VAL_44:.*]] = fir.box_addr %[[VAL_43]] : (!fir.box<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>>
// CHECK:                 %[[VAL_45:.*]]:2 = fir.do_loop %[[VAL_46:.*]] = %[[VAL_4]] to %[[VAL_32]] step %[[VAL_5]] iter_args(%[[VAL_47:.*]] = %[[VAL_33]]) -> (index, i32) {
// CHECK:                   fir.store %[[VAL_47]] to %[[VAL_10]] : !fir.ref<i32>
// CHECK:                   %[[VAL_48:.*]] = fir.load %[[VAL_19]] : !fir.ref<f64>
// CHECK:                   %[[VAL_49:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
// CHECK:                   %[[VAL_50:.*]] = fir.convert %[[VAL_49]] : (i32) -> i64
// CHECK:                   %[[VAL_51:.*]] = fir.load %[[VAL_12]] : !fir.ref<i32>
// CHECK:                   %[[VAL_52:.*]] = fir.convert %[[VAL_51]] : (i32) -> i64
// CHECK:                   %[[VAL_53:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
// CHECK:                   %[[VAL_54:.*]] = fir.convert %[[VAL_53]] : (i32) -> i64
// CHECK:                   %[[VAL_55:.*]] = arith.constant 1 : i64
// CHECK:                   %[[VAL_56:.*]] = arith.subi %[[VAL_54]], %[[VAL_55]] : i64
// CHECK:                   %[[VAL_57:.*]] = fir.convert %[[VAL_56]] : (i64) -> index
// CHECK:                   %[[VAL_58:.*]] = arith.muli %[[VAL_39]]#2, %[[VAL_57]] : index
// CHECK:                   %[[VAL_59:.*]] = arith.constant 1 : i64
// CHECK:                   %[[VAL_60:.*]] = arith.subi %[[VAL_52]], %[[VAL_59]] : i64
// CHECK:                   %[[VAL_61:.*]] = fir.convert %[[VAL_60]] : (i64) -> index
// CHECK:                   %[[VAL_62:.*]] = arith.muli %[[VAL_37]]#2, %[[VAL_61]] : index
// CHECK:                   %[[VAL_63:.*]] = arith.addi %[[VAL_62]], %[[VAL_58]] : index
// CHECK:                   %[[VAL_64:.*]] = arith.constant 1 : i64
// CHECK:                   %[[VAL_65:.*]] = arith.subi %[[VAL_50]], %[[VAL_64]] : i64
// CHECK:                   %[[VAL_66:.*]] = fir.convert %[[VAL_65]] : (i64) -> index
// CHECK:                   %[[VAL_67:.*]] = arith.constant 3 : index
// CHECK:                   %[[VAL_68:.*]] = arith.shrsi %[[VAL_63]], %[[VAL_67]] : index
// CHECK:                   %[[VAL_69:.*]] = arith.addi %[[VAL_68]], %[[VAL_66]] : index
// CHECK:                   %[[VAL_70:.*]] = fir.coordinate_of %[[VAL_44]], %[[VAL_69]] : (!fir.ref<!fir.array<?xf64>>, index) -> !fir.ref<f64>
// CHECK:                   %[[VAL_71:.*]] = fir.load %[[VAL_70]] : !fir.ref<f64>
// CHECK:                   %[[VAL_72:.*]] = arith.addf %[[VAL_48]], %[[VAL_71]] fastmath<contract> : f64
// CHECK:                   fir.store %[[VAL_72]] to %[[VAL_19]] : !fir.ref<f64>
// CHECK:                   %[[VAL_73:.*]] = arith.addi %[[VAL_46]], %[[VAL_5]] : index
// CHECK:                   %[[VAL_74:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
// CHECK:                   %[[VAL_75:.*]] = arith.addi %[[VAL_74]], %[[VAL_22]] : i32
// CHECK:                   fir.result %[[VAL_73]], %[[VAL_75]] : index, i32
// CHECK:                 }
// CHECK:                 fir.result %[[VAL_76:.*]]#0, %[[VAL_76]]#1 : index, i32
// CHECK:               } else {
// CHECK:                 %[[VAL_77:.*]]:2 = fir.do_loop %[[VAL_78:.*]] = %[[VAL_4]] to %[[VAL_32]] step %[[VAL_5]] iter_args(%[[VAL_79:.*]] = %[[VAL_33]]) -> (index, i32) {
// CHECK:                   fir.store %[[VAL_79]] to %[[VAL_10]] : !fir.ref<i32>
// CHECK:                   %[[VAL_80:.*]] = fir.load %[[VAL_19]] : !fir.ref<f64>
// CHECK:                   %[[VAL_81:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
// CHECK:                   %[[VAL_82:.*]] = fir.convert %[[VAL_81]] : (i32) -> i64
// CHECK:                   %[[VAL_83:.*]] = fir.load %[[VAL_12]] : !fir.ref<i32>
// CHECK:                   %[[VAL_84:.*]] = fir.convert %[[VAL_83]] : (i32) -> i64
// CHECK:                   %[[VAL_85:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
// CHECK:                   %[[VAL_86:.*]] = fir.convert %[[VAL_85]] : (i32) -> i64
// CHECK:                   %[[VAL_87:.*]] = fir.array_coor %[[VAL_8]] %[[VAL_82]], %[[VAL_84]], %[[VAL_86]] : (!fir.box<!fir.array<?x?x?xf64>>, i64, i64, i64) -> !fir.ref<f64>
// CHECK:                   %[[VAL_88:.*]] = fir.load %[[VAL_87]] : !fir.ref<f64>
// CHECK:                   %[[VAL_89:.*]] = arith.addf %[[VAL_80]], %[[VAL_88]] fastmath<contract> : f64
// CHECK:                   fir.store %[[VAL_89]] to %[[VAL_19]] : !fir.ref<f64>
// CHECK:                   %[[VAL_90:.*]] = arith.addi %[[VAL_78]], %[[VAL_5]] : index
// CHECK:                   %[[VAL_91:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
// CHECK:                   %[[VAL_92:.*]] = arith.addi %[[VAL_91]], %[[VAL_22]] : i32
// CHECK:                   fir.result %[[VAL_90]], %[[VAL_92]] : index, i32
// CHECK:                 }
// CHECK:                 fir.result %[[VAL_93:.*]]#0, %[[VAL_93]]#1 : index, i32
// CHECK:               }
// CHECK:               fir.store %[[VAL_94:.*]]#1 to %[[VAL_10]] : !fir.ref<i32>
// CHECK:               %[[VAL_95:.*]] = arith.addi %[[VAL_29]], %[[VAL_5]] : index
// CHECK:               %[[VAL_96:.*]] = fir.load %[[VAL_12]] : !fir.ref<i32>
// CHECK:               %[[VAL_97:.*]] = arith.addi %[[VAL_96]], %[[VAL_22]] : i32
// CHECK:               fir.result %[[VAL_95]], %[[VAL_97]] : index, i32
// CHECK:             }
// CHECK:             fir.store %[[VAL_98:.*]]#1 to %[[VAL_12]] : !fir.ref<i32>
// CHECK:             %[[VAL_99:.*]] = arith.addi %[[VAL_24]], %[[VAL_5]] : index
// CHECK:             %[[VAL_100:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
// CHECK:             %[[VAL_101:.*]] = arith.addi %[[VAL_100]], %[[VAL_22]] : i32
// CHECK:             fir.result %[[VAL_99]], %[[VAL_101]] : index, i32
// CHECK:           }
// CHECK:           fir.store %[[VAL_102:.*]]#1 to %[[VAL_14]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }

// test non-default lower bounds are handled correctly
  func.func @_QPlbounds_repro(%arg0: !fir.box<!fir.array<?x?x?xf32>> {fir.bindc_name = "u"}, %arg1: !fir.ref<i32> {fir.bindc_name = "ims"}, %arg2: !fir.ref<i32> {fir.bindc_name = "jms"}, %arg3: !fir.ref<i32> {fir.bindc_name = "kms"}, %arg4: !fir.ref<i32> {fir.bindc_name = "ips"}, %arg5: !fir.ref<i32> {fir.bindc_name = "ipe"}, %arg6: !fir.ref<i32> {fir.bindc_name = "jps"}, %arg7: !fir.ref<i32> {fir.bindc_name = "jpe"}, %arg8: !fir.ref<i32> {fir.bindc_name = "kps"}, %arg9: !fir.ref<i32> {fir.bindc_name = "kpe"}) {
    %c1_i32 = arith.constant 1 : i32
    %c1 = arith.constant 1 : index
    %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFlbounds_reproEi"}
    %1 = fir.declare %0 {uniq_name = "_QFlbounds_reproEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.declare %arg1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFlbounds_reproEims"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.declare %arg5 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFlbounds_reproEipe"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %4 = fir.declare %arg4 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFlbounds_reproEips"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %5 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFlbounds_reproEj"}
    %6 = fir.declare %5 {uniq_name = "_QFlbounds_reproEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %7 = fir.declare %arg2 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFlbounds_reproEjms"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %8 = fir.declare %arg7 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFlbounds_reproEjpe"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %9 = fir.declare %arg6 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFlbounds_reproEjps"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %10 = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFlbounds_reproEk"}
    %11 = fir.declare %10 {uniq_name = "_QFlbounds_reproEk"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %12 = fir.declare %arg3 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFlbounds_reproEkms"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %13 = fir.declare %arg9 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFlbounds_reproEkpe"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %14 = fir.declare %arg8 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFlbounds_reproEkps"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %15 = fir.alloca f32 {bindc_name = "vmax", uniq_name = "_QFlbounds_reproEvmax"}
    %16 = fir.declare %15 {uniq_name = "_QFlbounds_reproEvmax"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %17 = fir.load %12 : !fir.ref<i32>
    %18 = fir.convert %17 : (i32) -> index
    %19 = fir.load %2 : !fir.ref<i32>
    %20 = fir.convert %19 : (i32) -> index
    %21 = fir.load %7 : !fir.ref<i32>
    %22 = fir.convert %21 : (i32) -> index
    %23 = fir.shift %18, %20, %22 : (index, index, index) -> !fir.shift<3>
    %24 = fir.declare %arg0(%23) {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFlbounds_reproEu"} : (!fir.box<!fir.array<?x?x?xf32>>, !fir.shift<3>) -> !fir.box<!fir.array<?x?x?xf32>>
    %25 = fir.rebox %24(%23) : (!fir.box<!fir.array<?x?x?xf32>>, !fir.shift<3>) -> !fir.box<!fir.array<?x?x?xf32>>
    %26 = fir.array_coor %25(%23) %c1, %c1, %c1 : (!fir.box<!fir.array<?x?x?xf32>>, !fir.shift<3>, index, index, index) -> !fir.ref<f32>
    %27 = fir.load %26 : !fir.ref<f32>
    fir.store %27 to %16 : !fir.ref<f32>
    %28 = fir.load %9 : !fir.ref<i32>
    %29 = fir.convert %28 : (i32) -> index
    %30 = fir.load %8 : !fir.ref<i32>
    %31 = arith.subi %30, %c1_i32 : i32
    %32 = fir.convert %31 : (i32) -> index
    %33 = fir.convert %29 : (index) -> i32
    %34:2 = fir.do_loop %arg10 = %29 to %32 step %c1 iter_args(%arg11 = %33) -> (index, i32) {
      fir.store %arg11 to %6 : !fir.ref<i32>
      %35 = fir.load %4 : !fir.ref<i32>
      %36 = fir.convert %35 : (i32) -> index
      %37 = fir.load %3 : !fir.ref<i32>
      %38 = fir.convert %37 : (i32) -> index
      %39 = fir.convert %36 : (index) -> i32
      %40:2 = fir.do_loop %arg12 = %36 to %38 step %c1 iter_args(%arg13 = %39) -> (index, i32) {
        fir.store %arg13 to %1 : !fir.ref<i32>
        %45 = fir.load %14 : !fir.ref<i32>
        %46 = fir.convert %45 : (i32) -> index
        %47 = fir.load %13 : !fir.ref<i32>
        %48 = arith.subi %47, %c1_i32 : i32
        %49 = fir.convert %48 : (i32) -> index
        %50 = fir.convert %46 : (index) -> i32
        %51:2 = fir.do_loop %arg14 = %46 to %49 step %c1 iter_args(%arg15 = %50) -> (index, i32) {
          fir.store %arg15 to %11 : !fir.ref<i32>
          %56 = fir.load %11 : !fir.ref<i32>
          %57 = fir.convert %56 : (i32) -> i64
          %58 = fir.load %1 : !fir.ref<i32>
          %59 = fir.convert %58 : (i32) -> i64
          %60 = fir.load %6 : !fir.ref<i32>
          %61 = fir.convert %60 : (i32) -> i64
          %62 = fir.array_coor %25(%23) %57, %59, %61 : (!fir.box<!fir.array<?x?x?xf32>>, !fir.shift<3>, i64, i64, i64) -> !fir.ref<f32>
          %63 = fir.load %62 : !fir.ref<f32>
          %64 = fir.load %16 : !fir.ref<f32>
          %65 = arith.cmpf ogt, %63, %64 : f32
          fir.if %65 {
            %70 = fir.load %11 : !fir.ref<i32>
            %71 = fir.convert %70 : (i32) -> i64
            %72 = fir.load %1 : !fir.ref<i32>
            %73 = fir.convert %72 : (i32) -> i64
            %74 = fir.load %6 : !fir.ref<i32>
            %75 = fir.convert %74 : (i32) -> i64
            %76 = fir.array_coor %25(%23) %71, %73, %75 : (!fir.box<!fir.array<?x?x?xf32>>, !fir.shift<3>, i64, i64, i64) -> !fir.ref<f32>
            %77 = fir.load %76 : !fir.ref<f32>
            fir.store %77 to %16 : !fir.ref<f32>
          } else {
          }
          %66 = arith.addi %arg14, %c1 : index
          %67 = fir.convert %c1 : (index) -> i32
          %68 = fir.load %11 : !fir.ref<i32>
          %69 = arith.addi %68, %67 : i32
          fir.result %66, %69 : index, i32
        }
        fir.store %51#1 to %11 : !fir.ref<i32>
        %52 = arith.addi %arg12, %c1 : index
        %53 = fir.convert %c1 : (index) -> i32
        %54 = fir.load %1 : !fir.ref<i32>
        %55 = arith.addi %54, %53 : i32
        fir.result %52, %55 : index, i32
      }
      fir.store %40#1 to %1 : !fir.ref<i32>
      %41 = arith.addi %arg10, %c1 : index
      %42 = fir.convert %c1 : (index) -> i32
      %43 = fir.load %6 : !fir.ref<i32>
      %44 = arith.addi %43, %42 : i32
      fir.result %41, %44 : index, i32
    }
    fir.store %34#1 to %6 : !fir.ref<i32>
    return
  }
// CHECK-LABEL:   func.func @_QPlbounds_repro(
// CHECK-SAME:                                %[[VAL_0:.*]]: !fir.box<!fir.array<?x?x?xf32>> {fir.bindc_name = "u"},
// CHECK-SAME:                                %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "ims"},
// CHECK-SAME:                                %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "jms"},
// CHECK-SAME:                                %[[VAL_3:.*]]: !fir.ref<i32> {fir.bindc_name = "kms"},
// CHECK-SAME:                                %[[VAL_4:.*]]: !fir.ref<i32> {fir.bindc_name = "ips"},
// CHECK-SAME:                                %[[VAL_5:.*]]: !fir.ref<i32> {fir.bindc_name = "ipe"},
// CHECK-SAME:                                %[[VAL_6:.*]]: !fir.ref<i32> {fir.bindc_name = "jps"},
// CHECK-SAME:                                %[[VAL_7:.*]]: !fir.ref<i32> {fir.bindc_name = "jpe"},
// CHECK-SAME:                                %[[VAL_8:.*]]: !fir.ref<i32> {fir.bindc_name = "kps"},
// CHECK-SAME:                                %[[VAL_9:.*]]: !fir.ref<i32> {fir.bindc_name = "kpe"}) {
// CHECK:           %[[VAL_10:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_11:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_12:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFlbounds_reproEi"}
// CHECK:           %[[VAL_13:.*]] = fir.declare %[[VAL_12]] {uniq_name = "_QFlbounds_reproEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_14:.*]] = fir.declare %[[VAL_1]] {fortran_attrs = {{.*}}, uniq_name = "_QFlbounds_reproEims"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_15:.*]] = fir.declare %[[VAL_5]] {fortran_attrs = {{.*}}, uniq_name = "_QFlbounds_reproEipe"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_16:.*]] = fir.declare %[[VAL_4]] {fortran_attrs = {{.*}}, uniq_name = "_QFlbounds_reproEips"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_17:.*]] = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFlbounds_reproEj"}
// CHECK:           %[[VAL_18:.*]] = fir.declare %[[VAL_17]] {uniq_name = "_QFlbounds_reproEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_19:.*]] = fir.declare %[[VAL_2]] {fortran_attrs = {{.*}}, uniq_name = "_QFlbounds_reproEjms"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_20:.*]] = fir.declare %[[VAL_7]] {fortran_attrs = {{.*}}, uniq_name = "_QFlbounds_reproEjpe"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_21:.*]] = fir.declare %[[VAL_6]] {fortran_attrs = {{.*}}, uniq_name = "_QFlbounds_reproEjps"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_22:.*]] = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFlbounds_reproEk"}
// CHECK:           %[[VAL_23:.*]] = fir.declare %[[VAL_22]] {uniq_name = "_QFlbounds_reproEk"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_24:.*]] = fir.declare %[[VAL_3]] {fortran_attrs = {{.*}}, uniq_name = "_QFlbounds_reproEkms"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_25:.*]] = fir.declare %[[VAL_9]] {fortran_attrs = {{.*}}, uniq_name = "_QFlbounds_reproEkpe"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_26:.*]] = fir.declare %[[VAL_8]] {fortran_attrs = {{.*}}, uniq_name = "_QFlbounds_reproEkps"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_27:.*]] = fir.alloca f32 {bindc_name = "vmax", uniq_name = "_QFlbounds_reproEvmax"}
// CHECK:           %[[VAL_28:.*]] = fir.declare %[[VAL_27]] {uniq_name = "_QFlbounds_reproEvmax"} : (!fir.ref<f32>) -> !fir.ref<f32>
// CHECK:           %[[VAL_29:.*]] = fir.load %[[VAL_24]] : !fir.ref<i32>
// CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> index
// CHECK:           %[[VAL_31:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
// CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i32) -> index
// CHECK:           %[[VAL_33:.*]] = fir.load %[[VAL_19]] : !fir.ref<i32>
// CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i32) -> index
// CHECK:           %[[VAL_35:.*]] = fir.shift %[[VAL_30]], %[[VAL_32]], %[[VAL_34]] : (index, index, index) -> !fir.shift<3>
// CHECK:           %[[VAL_36:.*]] = fir.declare %[[VAL_0]](%[[VAL_35]]) {fortran_attrs = {{.*}}, uniq_name = "_QFlbounds_reproEu"} : (!fir.box<!fir.array<?x?x?xf32>>, !fir.shift<3>) -> !fir.box<!fir.array<?x?x?xf32>>
// CHECK:           %[[VAL_37:.*]] = fir.rebox %[[VAL_36]](%[[VAL_35]]) : (!fir.box<!fir.array<?x?x?xf32>>, !fir.shift<3>) -> !fir.box<!fir.array<?x?x?xf32>>
// CHECK:           %[[VAL_38:.*]] = fir.array_coor %[[VAL_37]](%[[VAL_35]]) %[[VAL_11]], %[[VAL_11]], %[[VAL_11]] : (!fir.box<!fir.array<?x?x?xf32>>, !fir.shift<3>, index, index, index) -> !fir.ref<f32>
// CHECK:           %[[VAL_39:.*]] = fir.load %[[VAL_38]] : !fir.ref<f32>
// CHECK:           fir.store %[[VAL_39]] to %[[VAL_28]] : !fir.ref<f32>
// CHECK:           %[[VAL_40:.*]] = fir.load %[[VAL_21]] : !fir.ref<i32>
// CHECK:           %[[VAL_41:.*]] = fir.convert %[[VAL_40]] : (i32) -> index
// CHECK:           %[[VAL_42:.*]] = fir.load %[[VAL_20]] : !fir.ref<i32>
// CHECK:           %[[VAL_43:.*]] = arith.subi %[[VAL_42]], %[[VAL_10]] : i32
// CHECK:           %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (i32) -> index
// CHECK:           %[[VAL_45:.*]] = fir.convert %[[VAL_41]] : (index) -> i32
// CHECK:           %[[VAL_46:.*]]:2 = fir.do_loop %[[VAL_47:.*]] = %[[VAL_41]] to %[[VAL_44]] step %[[VAL_11]] iter_args(%[[VAL_48:.*]] = %[[VAL_45]]) -> (index, i32) {
// CHECK:             fir.store %[[VAL_48]] to %[[VAL_18]] : !fir.ref<i32>
// CHECK:             %[[VAL_49:.*]] = fir.load %[[VAL_16]] : !fir.ref<i32>
// CHECK:             %[[VAL_50:.*]] = fir.convert %[[VAL_49]] : (i32) -> index
// CHECK:             %[[VAL_51:.*]] = fir.load %[[VAL_15]] : !fir.ref<i32>
// CHECK:             %[[VAL_52:.*]] = fir.convert %[[VAL_51]] : (i32) -> index
// CHECK:             %[[VAL_53:.*]] = fir.convert %[[VAL_50]] : (index) -> i32
// CHECK:             %[[VAL_54:.*]]:2 = fir.do_loop %[[VAL_55:.*]] = %[[VAL_50]] to %[[VAL_52]] step %[[VAL_11]] iter_args(%[[VAL_56:.*]] = %[[VAL_53]]) -> (index, i32) {
// CHECK:               fir.store %[[VAL_56]] to %[[VAL_13]] : !fir.ref<i32>
// CHECK:               %[[VAL_57:.*]] = fir.load %[[VAL_26]] : !fir.ref<i32>
// CHECK:               %[[VAL_58:.*]] = fir.convert %[[VAL_57]] : (i32) -> index
// CHECK:               %[[VAL_59:.*]] = fir.load %[[VAL_25]] : !fir.ref<i32>
// CHECK:               %[[VAL_60:.*]] = arith.subi %[[VAL_59]], %[[VAL_10]] : i32
// CHECK:               %[[VAL_61:.*]] = fir.convert %[[VAL_60]] : (i32) -> index
// CHECK:               %[[VAL_62:.*]] = fir.convert %[[VAL_58]] : (index) -> i32
// CHECK:               %[[VAL_63:.*]] = arith.constant 0 : index
// CHECK:               %[[VAL_64:.*]]:3 = fir.box_dims %[[VAL_37]], %[[VAL_63]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_65:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_66:.*]]:3 = fir.box_dims %[[VAL_37]], %[[VAL_65]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_67:.*]] = arith.constant 2 : index
// CHECK:               %[[VAL_68:.*]]:3 = fir.box_dims %[[VAL_37]], %[[VAL_67]] : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
// CHECK:               %[[VAL_69:.*]] = arith.constant 4 : index
// CHECK:               %[[VAL_70:.*]] = arith.cmpi eq, %[[VAL_64]]#2, %[[VAL_69]] : index
// CHECK:               %[[VAL_71:.*]]:2 = fir.if %[[VAL_70]] -> (index, i32) {
// CHECK:                 %[[VAL_72:.*]] = fir.convert %[[VAL_37]] : (!fir.box<!fir.array<?x?x?xf32>>) -> !fir.box<!fir.array<?xf32>>
// CHECK:                 %[[VAL_73:.*]] = fir.box_addr %[[VAL_72]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
// CHECK:                 %[[VAL_74:.*]]:2 = fir.do_loop %[[VAL_75:.*]] = %[[VAL_58]] to %[[VAL_61]] step %[[VAL_11]] iter_args(%[[VAL_76:.*]] = %[[VAL_62]]) -> (index, i32) {
// CHECK:                   fir.store %[[VAL_76]] to %[[VAL_23]] : !fir.ref<i32>
// CHECK:                   %[[VAL_77:.*]] = fir.load %[[VAL_23]] : !fir.ref<i32>
// CHECK:                   %[[VAL_78:.*]] = fir.convert %[[VAL_77]] : (i32) -> i64
// CHECK:                   %[[VAL_79:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
// CHECK:                   %[[VAL_80:.*]] = fir.convert %[[VAL_79]] : (i32) -> i64
// CHECK:                   %[[VAL_81:.*]] = fir.load %[[VAL_18]] : !fir.ref<i32>
// CHECK:                   %[[VAL_82:.*]] = fir.convert %[[VAL_81]] : (i32) -> i64
// CHECK:                   %[[VAL_83:.*]] = fir.convert %[[VAL_34]] : (index) -> i64
// CHECK:                   %[[VAL_84:.*]] = arith.subi %[[VAL_82]], %[[VAL_83]] : i64
// CHECK:                   %[[VAL_85:.*]] = fir.convert %[[VAL_84]] : (i64) -> index
// CHECK:                   %[[VAL_86:.*]] = arith.muli %[[VAL_68]]#2, %[[VAL_85]] : index
// CHECK:                   %[[VAL_87:.*]] = fir.convert %[[VAL_32]] : (index) -> i64
// CHECK:                   %[[VAL_88:.*]] = arith.subi %[[VAL_80]], %[[VAL_87]] : i64
// CHECK:                   %[[VAL_89:.*]] = fir.convert %[[VAL_88]] : (i64) -> index
// CHECK:                   %[[VAL_90:.*]] = arith.muli %[[VAL_66]]#2, %[[VAL_89]] : index
// CHECK:                   %[[VAL_91:.*]] = arith.addi %[[VAL_90]], %[[VAL_86]] : index
// CHECK:                   %[[VAL_92:.*]] = fir.convert %[[VAL_30]] : (index) -> i64
// CHECK:                   %[[VAL_93:.*]] = arith.subi %[[VAL_78]], %[[VAL_92]] : i64
// CHECK:                   %[[VAL_94:.*]] = fir.convert %[[VAL_93]] : (i64) -> index
// CHECK:                   %[[VAL_95:.*]] = arith.constant 2 : index
// CHECK:                   %[[VAL_96:.*]] = arith.shrsi %[[VAL_91]], %[[VAL_95]] : index
// CHECK:                   %[[VAL_97:.*]] = arith.addi %[[VAL_96]], %[[VAL_94]] : index
// CHECK:                   %[[VAL_98:.*]] = fir.coordinate_of %[[VAL_73]], %[[VAL_97]] : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:                   %[[VAL_99:.*]] = fir.load %[[VAL_98]] : !fir.ref<f32>
// CHECK:                   %[[VAL_100:.*]] = fir.load %[[VAL_28]] : !fir.ref<f32>
// CHECK:                   %[[VAL_101:.*]] = arith.cmpf ogt, %[[VAL_99]], %[[VAL_100]] : f32
// CHECK:                   fir.if %[[VAL_101]] {
// CHECK:                     %[[VAL_102:.*]] = fir.load %[[VAL_23]] : !fir.ref<i32>
// CHECK:                     %[[VAL_103:.*]] = fir.convert %[[VAL_102]] : (i32) -> i64
// CHECK:                     %[[VAL_104:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
// CHECK:                     %[[VAL_105:.*]] = fir.convert %[[VAL_104]] : (i32) -> i64
// CHECK:                     %[[VAL_106:.*]] = fir.load %[[VAL_18]] : !fir.ref<i32>
// CHECK:                     %[[VAL_107:.*]] = fir.convert %[[VAL_106]] : (i32) -> i64
// CHECK:                     %[[VAL_108:.*]] = fir.convert %[[VAL_34]] : (index) -> i64
// CHECK:                     %[[VAL_109:.*]] = arith.subi %[[VAL_107]], %[[VAL_108]] : i64
// CHECK:                     %[[VAL_110:.*]] = fir.convert %[[VAL_109]] : (i64) -> index
// CHECK:                     %[[VAL_111:.*]] = arith.muli %[[VAL_68]]#2, %[[VAL_110]] : index
// CHECK:                     %[[VAL_112:.*]] = fir.convert %[[VAL_32]] : (index) -> i64
// CHECK:                     %[[VAL_113:.*]] = arith.subi %[[VAL_105]], %[[VAL_112]] : i64
// CHECK:                     %[[VAL_114:.*]] = fir.convert %[[VAL_113]] : (i64) -> index
// CHECK:                     %[[VAL_115:.*]] = arith.muli %[[VAL_66]]#2, %[[VAL_114]] : index
// CHECK:                     %[[VAL_116:.*]] = arith.addi %[[VAL_115]], %[[VAL_111]] : index
// CHECK:                     %[[VAL_117:.*]] = fir.convert %[[VAL_30]] : (index) -> i64
// CHECK:                     %[[VAL_118:.*]] = arith.subi %[[VAL_103]], %[[VAL_117]] : i64
// CHECK:                     %[[VAL_119:.*]] = fir.convert %[[VAL_118]] : (i64) -> index
// CHECK:                     %[[VAL_120:.*]] = arith.constant 2 : index
// CHECK:                     %[[VAL_121:.*]] = arith.shrsi %[[VAL_116]], %[[VAL_120]] : index
// CHECK:                     %[[VAL_122:.*]] = arith.addi %[[VAL_121]], %[[VAL_119]] : index
// CHECK:                     %[[VAL_123:.*]] = fir.coordinate_of %[[VAL_73]], %[[VAL_122]] : (!fir.ref<!fir.array<?xf32>>, index) -> !fir.ref<f32>
// CHECK:                     %[[VAL_124:.*]] = fir.load %[[VAL_123]] : !fir.ref<f32>
// CHECK:                     fir.store %[[VAL_124]] to %[[VAL_28]] : !fir.ref<f32>
// CHECK:                   } else {
// CHECK:                   }
// CHECK:                   %[[VAL_125:.*]] = arith.addi %[[VAL_75]], %[[VAL_11]] : index
// CHECK:                   %[[VAL_126:.*]] = fir.convert %[[VAL_11]] : (index) -> i32
// CHECK:                   %[[VAL_127:.*]] = fir.load %[[VAL_23]] : !fir.ref<i32>
// CHECK:                   %[[VAL_128:.*]] = arith.addi %[[VAL_127]], %[[VAL_126]] : i32
// CHECK:                   fir.result %[[VAL_125]], %[[VAL_128]] : index, i32
// CHECK:                 }
// CHECK:                 fir.result %[[VAL_129:.*]]#0, %[[VAL_129]]#1 : index, i32
// CHECK:               } else {
// CHECK:                 %[[VAL_130:.*]]:2 = fir.do_loop %[[VAL_131:.*]] = %[[VAL_58]] to %[[VAL_61]] step %[[VAL_11]] iter_args(%[[VAL_132:.*]] = %[[VAL_62]]) -> (index, i32) {
// CHECK:                   fir.store %[[VAL_132]] to %[[VAL_23]] : !fir.ref<i32>
// CHECK:                   %[[VAL_133:.*]] = fir.load %[[VAL_23]] : !fir.ref<i32>
// CHECK:                   %[[VAL_134:.*]] = fir.convert %[[VAL_133]] : (i32) -> i64
// CHECK:                   %[[VAL_135:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
// CHECK:                   %[[VAL_136:.*]] = fir.convert %[[VAL_135]] : (i32) -> i64
// CHECK:                   %[[VAL_137:.*]] = fir.load %[[VAL_18]] : !fir.ref<i32>
// CHECK:                   %[[VAL_138:.*]] = fir.convert %[[VAL_137]] : (i32) -> i64
// CHECK:                   %[[VAL_139:.*]] = fir.array_coor %[[VAL_37]](%[[VAL_35]]) %[[VAL_134]], %[[VAL_136]], %[[VAL_138]] : (!fir.box<!fir.array<?x?x?xf32>>, !fir.shift<3>, i64, i64, i64) -> !fir.ref<f32>
// CHECK:                   %[[VAL_140:.*]] = fir.load %[[VAL_139]] : !fir.ref<f32>
// CHECK:                   %[[VAL_141:.*]] = fir.load %[[VAL_28]] : !fir.ref<f32>
// CHECK:                   %[[VAL_142:.*]] = arith.cmpf ogt, %[[VAL_140]], %[[VAL_141]] : f32
// CHECK:                   fir.if %[[VAL_142]] {
// CHECK:                     %[[VAL_143:.*]] = fir.load %[[VAL_23]] : !fir.ref<i32>
// CHECK:                     %[[VAL_144:.*]] = fir.convert %[[VAL_143]] : (i32) -> i64
// CHECK:                     %[[VAL_145:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
// CHECK:                     %[[VAL_146:.*]] = fir.convert %[[VAL_145]] : (i32) -> i64
// CHECK:                     %[[VAL_147:.*]] = fir.load %[[VAL_18]] : !fir.ref<i32>
// CHECK:                     %[[VAL_148:.*]] = fir.convert %[[VAL_147]] : (i32) -> i64
// CHECK:                     %[[VAL_149:.*]] = fir.array_coor %[[VAL_37]](%[[VAL_35]]) %[[VAL_144]], %[[VAL_146]], %[[VAL_148]] : (!fir.box<!fir.array<?x?x?xf32>>, !fir.shift<3>, i64, i64, i64) -> !fir.ref<f32>
// CHECK:                     %[[VAL_150:.*]] = fir.load %[[VAL_149]] : !fir.ref<f32>
// CHECK:                     fir.store %[[VAL_150]] to %[[VAL_28]] : !fir.ref<f32>
// CHECK:                   } else {
// CHECK:                   }
// CHECK:                   %[[VAL_151:.*]] = arith.addi %[[VAL_131]], %[[VAL_11]] : index
// CHECK:                   %[[VAL_152:.*]] = fir.convert %[[VAL_11]] : (index) -> i32
// CHECK:                   %[[VAL_153:.*]] = fir.load %[[VAL_23]] : !fir.ref<i32>
// CHECK:                   %[[VAL_154:.*]] = arith.addi %[[VAL_153]], %[[VAL_152]] : i32
// CHECK:                   fir.result %[[VAL_151]], %[[VAL_154]] : index, i32
// CHECK:                 }
// CHECK:                 fir.result %[[VAL_155:.*]]#0, %[[VAL_155]]#1 : index, i32
// CHECK:               }
// CHECK:               fir.store %[[VAL_156:.*]]#1 to %[[VAL_23]] : !fir.ref<i32>
// CHECK:               %[[VAL_157:.*]] = arith.addi %[[VAL_55]], %[[VAL_11]] : index
// CHECK:               %[[VAL_158:.*]] = fir.convert %[[VAL_11]] : (index) -> i32
// CHECK:               %[[VAL_159:.*]] = fir.load %[[VAL_13]] : !fir.ref<i32>
// CHECK:               %[[VAL_160:.*]] = arith.addi %[[VAL_159]], %[[VAL_158]] : i32
// CHECK:               fir.result %[[VAL_157]], %[[VAL_160]] : index, i32
// CHECK:             }
// CHECK:             fir.store %[[VAL_161:.*]]#1 to %[[VAL_13]] : !fir.ref<i32>
// CHECK:             %[[VAL_162:.*]] = arith.addi %[[VAL_47]], %[[VAL_11]] : index
// CHECK:             %[[VAL_163:.*]] = fir.convert %[[VAL_11]] : (index) -> i32
// CHECK:             %[[VAL_164:.*]] = fir.load %[[VAL_18]] : !fir.ref<i32>
// CHECK:             %[[VAL_165:.*]] = arith.addi %[[VAL_164]], %[[VAL_163]] : i32
// CHECK:             fir.result %[[VAL_162]], %[[VAL_165]] : index, i32
// CHECK:           }
// CHECK:           fir.store %[[VAL_166:.*]]#1 to %[[VAL_18]] : !fir.ref<i32>
// CHECK:           return
// CHECK:         }

// ! Verify that neither of the loops is versioned
// ! due to the array section in the inner loop:
// subroutine test_slice(x)
//   real :: x(:,:)
//   do i=10,100
//      x(i,7) = 1.0
//      x(i,3:5) = 2.0
//   end do
// end subroutine test_slice
func.func @_QPtest_slice(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}) {
  %c10 = arith.constant 10 : index
  %c100 = arith.constant 100 : index
  %c6_i64 = arith.constant 6 : i64
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  %cst = arith.constant 2.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1_i64 = arith.constant 1 : i64
  %cst_0 = arith.constant 1.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_sliceEi"}
  %1 = fir.convert %c10 : (index) -> i32
  %2:2 = fir.do_loop %arg1 = %c10 to %c100 step %c1 iter_args(%arg2 = %1) -> (index, i32) {
    fir.store %arg2 to %0 : !fir.ref<i32>
    %3 = fir.load %0 : !fir.ref<i32>
    %4 = fir.convert %3 : (i32) -> i64
    %5 = arith.subi %4, %c1_i64 : i64
    %6 = fir.coordinate_of %arg0, %5, %c6_i64 : (!fir.box<!fir.array<?x?xf32>>, i64, i64) -> !fir.ref<f32>
    fir.store %cst_0 to %6 : !fir.ref<f32>
    %7 = fir.load %0 : !fir.ref<i32>
    %8 = fir.convert %7 : (i32) -> i64
    %9 = fir.undefined index
    %10 = fir.convert %7 : (i32) -> index
    %11 = fir.slice %8, %9, %9, %c3, %c5, %c1 : (i64, index, index, index, index, index) -> !fir.slice<2>
    %12 = fir.undefined !fir.array<?x?xf32>
    %13 = fir.do_loop %arg3 = %c0 to %c2 step %c1 unordered iter_args(%arg4 = %12) -> (!fir.array<?x?xf32>) {
      %18 = arith.addi %arg3, %c1 : index
      %19 = fir.array_coor %arg0 [%11] %10, %18 : (!fir.box<!fir.array<?x?xf32>>, !fir.slice<2>, index, index) -> !fir.ref<f32>
      fir.store %cst to %19 : !fir.ref<f32>
      fir.result %12 : !fir.array<?x?xf32>
    }
    %14 = arith.addi %arg1, %c1 : index
    %15 = fir.convert %c1 : (index) -> i32
    %16 = fir.load %0 : !fir.ref<i32>
    %17 = arith.addi %16, %15 : i32
    fir.result %14, %17 : index, i32
  }
  fir.store %2#1 to %0 : !fir.ref<i32>
  return
}
// CHECK-LABEL:   func.func @_QPtest_slice(
// CHECK-NOT: fir.if

// ! Verify versioning for argument 'x' but not for 'y':
// subroutine test_independent_args(x, y)
//   real :: x(:,:), y(:,:)
//   do i=10,100
//      x(i,7) = 1.0
//      y(i,3:5) = 2.0
//   end do
// end subroutine test_independent_args
func.func @_QPtest_independent_args(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "y"}) {
  %c10 = arith.constant 10 : index
  %c100 = arith.constant 100 : index
  %c6_i64 = arith.constant 6 : i64
  %c3 = arith.constant 3 : index
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  %cst = arith.constant 2.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1_i64 = arith.constant 1 : i64
  %cst_0 = arith.constant 1.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_independent_argsEi"}
  %1 = fir.convert %c10 : (index) -> i32
  %2:2 = fir.do_loop %arg2 = %c10 to %c100 step %c1 iter_args(%arg3 = %1) -> (index, i32) {
    fir.store %arg3 to %0 : !fir.ref<i32>
    %3 = fir.load %0 : !fir.ref<i32>
    %4 = fir.convert %3 : (i32) -> i64
    %5 = arith.subi %4, %c1_i64 : i64
    %6 = fir.coordinate_of %arg0, %5, %c6_i64 : (!fir.box<!fir.array<?x?xf32>>, i64, i64) -> !fir.ref<f32>
    fir.store %cst_0 to %6 : !fir.ref<f32>
    %7 = fir.load %0 : !fir.ref<i32>
    %8 = fir.convert %7 : (i32) -> i64
    %9 = fir.undefined index
    %10 = fir.convert %7 : (i32) -> index
    %11 = fir.slice %8, %9, %9, %c3, %c5, %c1 : (i64, index, index, index, index, index) -> !fir.slice<2>
    %12 = fir.undefined !fir.array<?x?xf32>
    %13 = fir.do_loop %arg4 = %c0 to %c2 step %c1 unordered iter_args(%arg5 = %12) -> (!fir.array<?x?xf32>) {
      %18 = arith.addi %arg4, %c1 : index
      %19 = fir.array_coor %arg1 [%11] %10, %18 : (!fir.box<!fir.array<?x?xf32>>, !fir.slice<2>, index, index) -> !fir.ref<f32>
      fir.store %cst to %19 : !fir.ref<f32>
      fir.result %12 : !fir.array<?x?xf32>
    }
    %14 = arith.addi %arg2, %c1 : index
    %15 = fir.convert %c1 : (index) -> i32
    %16 = fir.load %0 : !fir.ref<i32>
    %17 = arith.addi %16, %15 : i32
    fir.result %14, %17 : index, i32
  }
  fir.store %2#1 to %0 : !fir.ref<i32>
  return
}
// CHECK-LABEL:   func.func @_QPtest_independent_args(
// CHECK-SAME:        %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"},
// CHECK-SAME:        %[[VAL_1:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "y"}) {
// CHECK:           %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_0]], %{{.*}} : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_19:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_20:.*]] = arith.cmpi eq, %[[VAL_16]]#2, %[[VAL_19]] : index
// CHECK:           %[[VAL_21:.*]]:2 = fir.if %[[VAL_20]] -> (index, i32) {
// CHECK-NOT: fir.if


// ! Verify that the whole loop nest is versioned
// ! without additional contiguity check for the inner loop:
// subroutine test_loop_nest(x)
//   real :: x(:)
//   do i=10,100
//      x(i) = 1.0
//      do j=10,100
//         x(j) = 2.0
//      end do
//   end do
// end subroutine test_loop_nest
func.func @_QPtest_loop_nest(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}) {
  %c10 = arith.constant 10 : index
  %c100 = arith.constant 100 : index
  %cst = arith.constant 2.000000e+00 : f32
  %c1_i64 = arith.constant 1 : i64
  %cst_0 = arith.constant 1.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_loop_nestEi"}
  %1 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFtest_loop_nestEj"}
  %2 = fir.convert %c10 : (index) -> i32
  %3:2 = fir.do_loop %arg1 = %c10 to %c100 step %c1 iter_args(%arg2 = %2) -> (index, i32) {
    fir.store %arg2 to %0 : !fir.ref<i32>
    %4 = fir.load %0 : !fir.ref<i32>
    %5 = fir.convert %4 : (i32) -> i64
    %6 = arith.subi %5, %c1_i64 : i64
    %7 = fir.coordinate_of %arg0, %6 : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
    fir.store %cst_0 to %7 : !fir.ref<f32>
    %8:2 = fir.do_loop %arg3 = %c10 to %c100 step %c1 iter_args(%arg4 = %2) -> (index, i32) {
      fir.store %arg4 to %1 : !fir.ref<i32>
      %13 = fir.load %1 : !fir.ref<i32>
      %14 = fir.convert %13 : (i32) -> i64
      %15 = arith.subi %14, %c1_i64 : i64
      %16 = fir.coordinate_of %arg0, %15 : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
      fir.store %cst to %16 : !fir.ref<f32>
      %17 = arith.addi %arg3, %c1 : index
      %18 = fir.convert %c1 : (index) -> i32
      %19 = fir.load %1 : !fir.ref<i32>
      %20 = arith.addi %19, %18 : i32
      fir.result %17, %20 : index, i32
    }
    fir.store %8#1 to %1 : !fir.ref<i32>
    %9 = arith.addi %arg1, %c1 : index
    %10 = fir.convert %c1 : (index) -> i32
    %11 = fir.load %0 : !fir.ref<i32>
    %12 = arith.addi %11, %10 : i32
    fir.result %9, %12 : index, i32
  }
  fir.store %3#1 to %0 : !fir.ref<i32>
  return
}
// CHECK-LABEL:   func.func @_QPtest_loop_nest(
// CHECK: fir.if
// CHECK-NOT: fir.if


//-----

// Check that a non-dominating operand isn't causing a problem.
// Just check it compiles, and doesn't version this loop.
func.func @sum1drebox(%arg0: !fir.box<!fir.array<?xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %decl = fir.declare %arg0 {uniq_name = "a"} : (!fir.box<!fir.array<?xf64>>) -> !fir.box<!fir.array<?xf64>>
  %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QMmoduleFsum1dEi"}
  %1 = fir.alloca f64 {bindc_name = "sum", uniq_name = "_QMmoduleFsum1dEsum"}
  %cst = arith.constant 0.000000e+00 : f64
  fir.store %cst to %1 : !fir.ref<f64>
  %c1_i32 = arith.constant 1 : i32
  %2 = fir.convert %c1_i32 : (i32) -> index
  %3 = fir.load %arg1 : !fir.ref<i32>
  %4 = fir.convert %3 : (i32) -> index
  %c1 = arith.constant 1 : index
  %5 = fir.convert %2 : (index) -> i32
  %6:2 = fir.do_loop %arg2 = %2 to %4 step %c1 iter_args(%arg3 = %5) -> (index, i32) {
    // rebox is not dominating the loop.
    %rebox = fir.rebox %decl : (!fir.box<!fir.array<?xf64>>) -> !fir.box<!fir.array<?xf64>>
    fir.store %arg3 to %0 : !fir.ref<i32>
    %7 = fir.load %1 : !fir.ref<f64>
    %8 = fir.load %0 : !fir.ref<i32>
    %9 = fir.convert %8 : (i32) -> i64
    %c1_i64 = arith.constant 1 : i64
    %10 = arith.subi %9, %c1_i64 : i64
    %11 = fir.coordinate_of %rebox, %10 : (!fir.box<!fir.array<?xf64>>, i64) -> !fir.ref<f64>
    %12 = fir.load %11 : !fir.ref<f64>
    %13 = arith.addf %7, %12 fastmath<contract> : f64
    fir.store %13 to %1 : !fir.ref<f64>
    %14 = arith.addi %arg2, %c1 : index
    %15 = fir.convert %c1 : (index) -> i32
    %16 = fir.load %0 : !fir.ref<i32>
    %17 = arith.addi %16, %15 : i32
    fir.result %14, %17 : index, i32
  }
  fir.store %6#1 to %0 : !fir.ref<i32>
  return
}
// CHECK-LABEL: func @sum1drebox
// No versioning -> no if-operation.
// CHECK-NOT: fir.if


// Check for a use in a different block (%12 = do_loop is used inside the if %14 block)
func.func @minloc(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "x"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "mask"}) -> f32 {
  %c2147483647_i32 = arith.constant 2147483647 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c5_i32 = arith.constant 5 : i32
  %c5 = arith.constant 5 : index
  %c1 = arith.constant 1 : index
  %0 = fir.alloca i32
  %1 = fir.alloca !fir.array<1xi32>
  %2 = fir.declare %arg1 {uniq_name = "_QFtestEmask"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
  %3 = fir.rebox %2 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
  %4 = fir.alloca f32 {bindc_name = "test", uniq_name = "_QFtestEtest"}
  %5 = fir.declare %4 {uniq_name = "_QFtestEtest"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %6 = fir.declare %arg0 {uniq_name = "_QFtestEx"} : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
  %7 = fir.rebox %6 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
  %8 = fir.shape %c1 : (index) -> !fir.shape<1>
  %9 = fir.array_coor %1(%8) %c1 : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  fir.store %c0_i32 to %9 : !fir.ref<i32>
  fir.store %c0_i32 to %0 : !fir.ref<i32>
  %10:3 = fir.box_dims %7, %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %11 = arith.subi %10#1, %c1 : index
  %12 = fir.do_loop %arg2 = %c0 to %11 step %c1 iter_args(%arg3 = %c2147483647_i32) -> (i32) {
    %18 = arith.addi %arg2, %c1 : index
    %19 = fir.array_coor %3 %18 : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
    %20 = fir.load %19 : !fir.ref<i32>
    %21 = arith.cmpi sge, %20, %c5_i32 : i32
    %22 = fir.if %21 -> (i32) {
      fir.store %c1_i32 to %0 : !fir.ref<i32>
      %23 = arith.subi %10#0, %c1 : index
      %24 = arith.addi %18, %23 : index
      %25 = fir.array_coor %7 %24 : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
      %26 = fir.load %25 : !fir.ref<i32>
      %27 = arith.cmpi slt, %26, %arg3 : i32
      %28 = fir.if %27 -> (i32) {
        %29 = fir.convert %18 : (index) -> i32
        fir.store %29 to %9 : !fir.ref<i32>
        fir.result %26 : i32
      } else {
        fir.result %arg3 : i32
      }
      fir.result %28 : i32
    } else {
      fir.result %arg3 : i32
    }
    fir.result %22 : i32
  }
  %13 = fir.load %0 : !fir.ref<i32>
  %14 = arith.cmpi eq, %13, %c1_i32 : i32
  fir.if %14 {
    %18 = arith.cmpi eq, %12, %c2147483647_i32 : i32
    fir.if %18 {
      %19 = fir.array_coor %1(%8) %c0 : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
      fir.store %c1_i32 to %19 : !fir.ref<i32>
    }
  }
  %15 = fir.slice %c5, %c5, %c1 : (index, index, index) -> !fir.slice<1>
  %16 = fir.rebox %7 [%15] : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>) -> !fir.box<!fir.array<1xi32>>
  fir.do_loop %arg2 = %c1 to %c1 step %c1 unordered {
    %18 = fir.array_coor %1(%8) %arg2 : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
    %19 = fir.load %18 : !fir.ref<i32>
    %20 = fir.array_coor %16 %arg2 : (!fir.box<!fir.array<1xi32>>, index) -> !fir.ref<i32>
    fir.store %19 to %20 : !fir.ref<i32>
  }
  %17 = fir.load %5 : !fir.ref<f32>
  return %17 : f32
}
// CHECK-LABEL: func @minloc
// CHECK: %[[V17:.*]] = fir.if %{{.*}} -> (i32) {
// CHECK:   %[[V27:.*]] = fir.do_loop
// CHECK:   fir.result %[[V27]] : i32
// CHECK: } else {
// CHECK:   %[[V23:.*]] = fir.do_loop
// CHECK:   fir.result %[[V23]] : i32
// CHECK: fir.if %{{.*}} {
// CHECK:   {{.*}} = arith.cmpi eq, %[[V17]], %c2147483647_i32

func.func @_QPtest_real10(%arg0: !fir.box<!fir.array<?x?xf80>> {fir.bindc_name = "a"}) -> f80 {
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f80
  %0 = fir.declare %arg0 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest_real10Ea"} : (!fir.box<!fir.array<?x?xf80>>) -> !fir.box<!fir.array<?x?xf80>>
  %1 = fir.rebox %0 : (!fir.box<!fir.array<?x?xf80>>) -> !fir.box<!fir.array<?x?xf80>>
  %2 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_real10Ei"}
  %3 = fir.declare %2 {uniq_name = "_QFtest_real10Ei"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %4 = fir.alloca f80 {bindc_name = "res", uniq_name = "_QFtest_real10Eres"}
  %5 = fir.declare %4 {uniq_name = "_QFtest_real10Eres"} : (!fir.ref<f80>) -> !fir.ref<f80>
  %6 = fir.address_of(@_QFtest_real10ECxdp) : !fir.ref<i32>
  %7 = fir.declare %6 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFtest_real10ECxdp"} : (!fir.ref<i32>) -> !fir.ref<i32>
  fir.store %cst to %5 : !fir.ref<f80>
  %8 = fir.convert %c1 : (index) -> i32
  %9:2 = fir.do_loop %arg1 = %c1 to %c10 step %c1 iter_args(%arg2 = %8) -> (index, i32) {
    fir.store %arg2 to %3 : !fir.ref<i32>
    %11 = fir.load %5 : !fir.ref<f80>
    %12 = fir.load %3 : !fir.ref<i32>
    %13 = fir.convert %12 : (i32) -> i64
    %14 = fir.array_coor %1 %13, %13 : (!fir.box<!fir.array<?x?xf80>>, i64, i64) -> !fir.ref<f80>
    %15 = fir.load %14 : !fir.ref<f80>
    %16 = arith.addf %11, %15 fastmath<contract> : f80
    fir.store %16 to %5 : !fir.ref<f80>
    %17 = arith.addi %arg1, %c1 : index
    %18 = fir.load %3 : !fir.ref<i32>
    %19 = arith.addi %18, %8 : i32
    fir.result %17, %19 : index, i32
  }
  fir.store %9#1 to %3 : !fir.ref<i32>
  %10 = fir.load %5 : !fir.ref<f80>
  return %10 : f80
}
// CHECK-LABEL:   func.func @_QPtest_real10(
// CHECK:           fir.if
// CHECK:             fir.do_loop
// CHECK-DAG:           arith.shrsi %{{[^,]*}}, %[[SHIFT:.*]] : index
// CHECK-DAG:           %[[SHIFT]] = arith.constant 4 : index
// CHECK:             fir.result
// CHECK:           } else {
// CHECK:             fir.do_loop

func.func @_QPtest_complex10(%arg0: !fir.box<!fir.array<?x?xcomplex<f80>>> {fir.bindc_name = "a"}) -> complex<f80> {
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f80
  %0 = fir.declare %arg0 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest_complex10Ea"} : (!fir.box<!fir.array<?x?xcomplex<f80>>>) -> !fir.box<!fir.array<?x?xcomplex<f80>>>
  %1 = fir.rebox %0 : (!fir.box<!fir.array<?x?xcomplex<f80>>>) -> !fir.box<!fir.array<?x?xcomplex<f80>>>
  %2 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_complex10Ei"}
  %3 = fir.declare %2 {uniq_name = "_QFtest_complex10Ei"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %4 = fir.alloca complex<f80> {bindc_name = "res", uniq_name = "_QFtest_complex10Eres"}
  %5 = fir.declare %4 {uniq_name = "_QFtest_complex10Eres"} : (!fir.ref<complex<f80>>) -> !fir.ref<complex<f80>>
  %6 = fir.address_of(@_QFtest_complex10ECxdp) : !fir.ref<i32>
  %7 = fir.declare %6 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFtest_complex10ECxdp"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %8 = fir.undefined complex<f80>
  %9 = fir.insert_value %8, %cst, [0 : index] : (complex<f80>, f80) -> complex<f80>
  %10 = fir.insert_value %9, %cst, [1 : index] : (complex<f80>, f80) -> complex<f80>
  fir.store %10 to %5 : !fir.ref<complex<f80>>
  %11 = fir.convert %c1 : (index) -> i32
  %12:2 = fir.do_loop %arg1 = %c1 to %c10 step %c1 iter_args(%arg2 = %11) -> (index, i32) {
    fir.store %arg2 to %3 : !fir.ref<i32>
    %14 = fir.load %5 : !fir.ref<complex<f80>>
    %15 = fir.load %3 : !fir.ref<i32>
    %16 = fir.convert %15 : (i32) -> i64
    %17 = fir.array_coor %1 %16, %16 : (!fir.box<!fir.array<?x?xcomplex<f80>>>, i64, i64) -> !fir.ref<complex<f80>>
    %18 = fir.load %17 : !fir.ref<complex<f80>>
    %19 = fir.addc %14, %18 {fastmath = #arith.fastmath<contract>} : complex<f80>
    fir.store %19 to %5 : !fir.ref<complex<f80>>
    %20 = arith.addi %arg1, %c1 : index
    %21 = fir.load %3 : !fir.ref<i32>
    %22 = arith.addi %21, %11 : i32
    fir.result %20, %22 : index, i32
  }
  fir.store %12#1 to %3 : !fir.ref<i32>
  %13 = fir.load %5 : !fir.ref<complex<f80>>
  return %13 : complex<f80>
}
// CHECK-LABEL:   func.func @_QPtest_complex10(
// CHECK:           fir.if
// CHECK:             fir.do_loop
// CHECK-DAG:           arith.shrsi %{{[^,]*}}, %[[SHIFT:.*]] : index
// CHECK-DAG:           %[[SHIFT]] = arith.constant 5 : index
// CHECK:             fir.result
// CHECK:           } else {
// CHECK:             fir.do_loop

// Test that the loop is not versioned with non-contiguous slices:
//subroutine test_step2_slice(x, y)
//  real :: x(:,:), y(:,:)
//  do i=1,10
//     x(::2,i) = y(::2,i) + 1.0
//  end do
//end subroutine
func.func @_QPtest_step2_slice(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "y"}) {
  %c10 = arith.constant 10 : index
  %cst = arith.constant 1.000000e+00 : f32
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_step2_sliceEi"}
  %2 = fir.declare %1 {uniq_name = "_QFtest_step2_sliceEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg0 dummy_scope %0 {uniq_name = "_QFtest_step2_sliceEx"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
  %4 = fir.rebox %3 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
  %5 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFtest_step2_sliceEy"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
  %6 = fir.rebox %5 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
  %7 = fir.convert %c1 : (index) -> i32
  %8:2 = fir.do_loop %arg2 = %c1 to %c10 step %c1 iter_args(%arg3 = %7) -> (index, i32) {
    fir.store %arg3 to %2 : !fir.ref<i32>
    %9:3 = fir.box_dims %6, %c0 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
    %10 = arith.addi %9#1, %c1 : index
    %11 = arith.divsi %10, %c2 : index
    %12 = arith.cmpi sgt, %11, %c0 : index
    %13 = arith.select %12, %11, %c0 : index
    %14 = fir.load %2 : !fir.ref<i32>
    %15 = fir.convert %14 : (i32) -> i64
    %16 = fir.undefined index
    %17 = fir.slice %c1, %9#1, %c2, %15, %16, %16 : (index, index, index, i64, index, index) -> !fir.slice<2>
    %18 = fir.rebox %6 [%17] : (!fir.box<!fir.array<?x?xf32>>, !fir.slice<2>) -> !fir.box<!fir.array<?xf32>>
    %19:3 = fir.box_dims %4, %c0 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
    %20 = fir.slice %c1, %19#1, %c2, %15, %16, %16 : (index, index, index, i64, index, index) -> !fir.slice<2>
    %21 = fir.rebox %4 [%20] : (!fir.box<!fir.array<?x?xf32>>, !fir.slice<2>) -> !fir.box<!fir.array<?xf32>>
    fir.do_loop %arg4 = %c1 to %13 step %c1 unordered {
      %25 = fir.array_coor %18 %arg4 : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
      %26 = fir.load %25 : !fir.ref<f32>
      %27 = arith.addf %26, %cst fastmath<fast> : f32
      %28 = fir.array_coor %21 %arg4 : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
      fir.store %27 to %28 : !fir.ref<f32>
    }
    %22 = arith.addi %arg2, %c1 overflow<nsw> : index
    %23 = fir.load %2 : !fir.ref<i32>
    %24 = arith.addi %23, %7 overflow<nsw> : i32
    fir.result %22, %24 : index, i32
  }
  fir.store %8#1 to %2 : !fir.ref<i32>
  return
}
// CHECK-LABEL:   func.func @_QPtest_step2_slice(
// CHECK-NOT: fir.if

// Test that the loop is versioned with most probably
// contiguous slices:
//subroutine test_step1_slice(x, y)
//  real :: x(:,:), y(:,:)
//  do i=1,10
//     x(:,i) = y(:,i) + 1.0
//  end do
//end subroutine
func.func @_QPtest_step1_slice(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "y"}) {
  %c10 = arith.constant 10 : index
  %cst = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_step1_sliceEi"}
  %2 = fir.declare %1 {uniq_name = "_QFtest_step1_sliceEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg0 dummy_scope %0 {uniq_name = "_QFtest_step1_sliceEx"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
  %4 = fir.rebox %3 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
  %5 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFtest_step1_sliceEy"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
  %6 = fir.rebox %5 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
  %7 = fir.convert %c1 : (index) -> i32
  %8:2 = fir.do_loop %arg2 = %c1 to %c10 step %c1 iter_args(%arg3 = %7) -> (index, i32) {
    fir.store %arg3 to %2 : !fir.ref<i32>
    %9:3 = fir.box_dims %6, %c0 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
    %10 = arith.cmpi sgt, %9#1, %c0 : index
    %11 = arith.select %10, %9#1, %c0 : index
    %12 = fir.load %2 : !fir.ref<i32>
    %13 = fir.convert %12 : (i32) -> i64
    %14 = fir.undefined index
    %15 = fir.slice %c1, %9#1, %c1, %13, %14, %14 : (index, index, index, i64, index, index) -> !fir.slice<2>
    %16 = fir.rebox %6 [%15] : (!fir.box<!fir.array<?x?xf32>>, !fir.slice<2>) -> !fir.box<!fir.array<?xf32>>
    %17:3 = fir.box_dims %4, %c0 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
    %18 = fir.slice %c1, %17#1, %c1, %13, %14, %14 : (index, index, index, i64, index, index) -> !fir.slice<2>
    %19 = fir.rebox %4 [%18] : (!fir.box<!fir.array<?x?xf32>>, !fir.slice<2>) -> !fir.box<!fir.array<?xf32>>
    fir.do_loop %arg4 = %c1 to %11 step %c1 unordered {
      %23 = fir.array_coor %16 %arg4 : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
      %24 = fir.load %23 : !fir.ref<f32>
      %25 = arith.addf %24, %cst fastmath<fast> : f32
      %26 = fir.array_coor %19 %arg4 : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
      fir.store %25 to %26 : !fir.ref<f32>
    }
    %20 = arith.addi %arg2, %c1 overflow<nsw> : index
    %21 = fir.load %2 : !fir.ref<i32>
    %22 = arith.addi %21, %7 overflow<nsw> : i32
    fir.result %20, %22 : index, i32
  }
  fir.store %8#1 to %2 : !fir.ref<i32>
  return
}
// CHECK-LABEL:   func.func @_QPtest_step1_slice(
// CHECK:           fir.do_loop
// CHECK:             fir.if
// CHECK:               fir.do_loop
// CHECK:             } else {
// CHECK:               fir.do_loop

// Test that the loop is versioned with logical arrays:
//subroutine test_logical_slice(x, y)
//  logical :: x(:,:), y(:,:)
//  do i=1,10
//     x(:,i) = y(:,i) .or. y(i,:)
//  end do
//end subroutine
func.func @_QPtest_logical_slice(%arg0: !fir.box<!fir.array<?x?x!fir.logical<4>>> {fir.bindc_name = "x"}, %arg1: !fir.box<!fir.array<?x?x!fir.logical<4>>> {fir.bindc_name = "y"}) {
  %c10 = arith.constant 10 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_logical_sliceEi"}
  %2 = fir.declare %1 {uniq_name = "_QFtest_logical_sliceEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg0 dummy_scope %0 {uniq_name = "_QFtest_logical_sliceEx"} : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.logical<4>>>
  %4 = fir.rebox %3 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.box<!fir.array<?x?x!fir.logical<4>>>
  %5 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFtest_logical_sliceEy"} : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, !fir.dscope) -> !fir.box<!fir.array<?x?x!fir.logical<4>>>
  %6 = fir.rebox %5 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> !fir.box<!fir.array<?x?x!fir.logical<4>>>
  %7 = fir.convert %c1 : (index) -> i32
  %8:2 = fir.do_loop %arg2 = %c1 to %c10 step %c1 iter_args(%arg3 = %7) -> (index, i32) {
    fir.store %arg3 to %2 : !fir.ref<i32>
    %9:3 = fir.box_dims %6, %c0 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index) -> (index, index, index)
    %10 = arith.cmpi sgt, %9#1, %c0 : index
    %11 = arith.select %10, %9#1, %c0 : index
    %12 = fir.load %2 : !fir.ref<i32>
    %13 = fir.convert %12 : (i32) -> i64
    %14 = fir.undefined index
    %15 = fir.slice %c1, %9#1, %c1, %13, %14, %14 : (index, index, index, i64, index, index) -> !fir.slice<2>
    %16 = fir.rebox %6 [%15] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, !fir.slice<2>) -> !fir.box<!fir.array<?x!fir.logical<4>>>
    %17:3 = fir.box_dims %6, %c1 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index) -> (index, index, index)
    %18 = fir.slice %13, %14, %14, %c1, %17#1, %c1 : (i64, index, index, index, index, index) -> !fir.slice<2>
    %19 = fir.rebox %6 [%18] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, !fir.slice<2>) -> !fir.box<!fir.array<?x!fir.logical<4>>>
    %20:3 = fir.box_dims %4, %c0 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, index) -> (index, index, index)
    %21 = fir.slice %c1, %20#1, %c1, %13, %14, %14 : (index, index, index, i64, index, index) -> !fir.slice<2>
    %22 = fir.rebox %4 [%21] : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, !fir.slice<2>) -> !fir.box<!fir.array<?x!fir.logical<4>>>
    fir.do_loop %arg4 = %c1 to %11 step %c1 unordered {
      %26 = fir.array_coor %16 %arg4 : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
      %27 = fir.array_coor %19 %arg4 : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
      %28 = fir.load %26 : !fir.ref<!fir.logical<4>>
      %29 = fir.load %27 : !fir.ref<!fir.logical<4>>
      %30 = fir.convert %28 : (!fir.logical<4>) -> i1
      %31 = fir.convert %29 : (!fir.logical<4>) -> i1
      %32 = arith.ori %30, %31 : i1
      %33 = fir.convert %32 : (i1) -> !fir.logical<4>
      %34 = fir.array_coor %22 %arg4 : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
      fir.store %33 to %34 : !fir.ref<!fir.logical<4>>
    }
    %23 = arith.addi %arg2, %c1 overflow<nsw> : index
    %24 = fir.load %2 : !fir.ref<i32>
    %25 = arith.addi %24, %7 overflow<nsw> : i32
    fir.result %23, %25 : index, i32
  }
  fir.store %8#1 to %2 : !fir.ref<i32>
  return
}
// CHECK-LABEL:   func.func @_QPtest_logical_slice(
// CHECK:           fir.do_loop
// CHECK:             fir.if
// CHECK:               fir.do_loop
// CHECK:             } else {
// CHECK:               fir.do_loop

// Test that the loop is versioned when a most probably
// contiguous slices have known shape:
//subroutine test_known_shape_slice(x, y)
//  integer :: x(:,:), y(:,:)
//  do i=1,10
//     x(1:10,i) = y(1:10,i) + 1
//  end do
//end subroutine
func.func @_QPtest_known_shape_slice(%arg0: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "x"}, %arg1: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "y"}) {
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_known_shape_sliceEi"}
  %2 = fir.declare %1 {uniq_name = "_QFtest_known_shape_sliceEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg0 dummy_scope %0 {uniq_name = "_QFtest_known_shape_sliceEx"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xi32>>
  %4 = fir.rebox %3 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %5 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFtest_known_shape_sliceEy"} : (!fir.box<!fir.array<?x?xi32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xi32>>
  %6 = fir.rebox %5 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.array<?x?xi32>>
  %7 = fir.convert %c1 : (index) -> i32
  %8:2 = fir.do_loop %arg2 = %c1 to %c10 step %c1 iter_args(%arg3 = %7) -> (index, i32) {
    fir.store %arg3 to %2 : !fir.ref<i32>
    %9 = fir.load %2 : !fir.ref<i32>
    %10 = fir.convert %9 : (i32) -> i64
    %11 = fir.undefined index
    %12 = fir.slice %c1, %c10, %c1, %10, %11, %11 : (index, index, index, i64, index, index) -> !fir.slice<2>
    %13 = fir.rebox %6 [%12] : (!fir.box<!fir.array<?x?xi32>>, !fir.slice<2>) -> !fir.box<!fir.array<10xi32>>
    %14 = fir.rebox %4 [%12] : (!fir.box<!fir.array<?x?xi32>>, !fir.slice<2>) -> !fir.box<!fir.array<10xi32>>
    fir.do_loop %arg4 = %c1 to %c10 step %c1 unordered {
      %18 = fir.array_coor %13 %arg4 : (!fir.box<!fir.array<10xi32>>, index) -> !fir.ref<i32>
      %19 = fir.load %18 : !fir.ref<i32>
      %20 = arith.addi %19, %c1_i32 : i32
      %21 = fir.array_coor %14 %arg4 : (!fir.box<!fir.array<10xi32>>, index) -> !fir.ref<i32>
      fir.store %20 to %21 : !fir.ref<i32>
    }
    %15 = arith.addi %arg2, %c1 overflow<nsw> : index
    %16 = fir.load %2 : !fir.ref<i32>
    %17 = arith.addi %16, %7 overflow<nsw> : i32
    fir.result %15, %17 : index, i32
  }
  fir.store %8#1 to %2 : !fir.ref<i32>
  return
}
// CHECK-LABEL:   func.func @_QPtest_known_shape_slice(
// CHECK:           fir.do_loop
// CHECK:             fir.if
// CHECK:               fir.do_loop
// CHECK:             } else {
// CHECK:               fir.do_loop

// Test that the loop is not versioned for most probably
// not-contiguous slices:
//subroutine test_maybe_noncontig_slice(x, y)
//  real :: x(:,:), y(:,:)
//  do i=1,10
//     x(i,:) = y(i,:) + 1.0
//  end do
//end subroutine
func.func @_QPtest_maybe_noncontig_slice(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "y"}) {
  %c10 = arith.constant 10 : index
  %cst = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_maybe_noncontig_sliceEi"}
  %2 = fir.declare %1 {uniq_name = "_QFtest_maybe_noncontig_sliceEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.declare %arg0 dummy_scope %0 {uniq_name = "_QFtest_maybe_noncontig_sliceEx"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
  %4 = fir.rebox %3 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
  %5 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFtest_maybe_noncontig_sliceEy"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
  %6 = fir.rebox %5 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>>
  %7 = fir.convert %c1 : (index) -> i32
  %8:2 = fir.do_loop %arg2 = %c1 to %c10 step %c1 iter_args(%arg3 = %7) -> (index, i32) {
    fir.store %arg3 to %2 : !fir.ref<i32>
    %9 = fir.load %2 : !fir.ref<i32>
    %10 = fir.convert %9 : (i32) -> i64
    %11:3 = fir.box_dims %6, %c1 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
    %12 = arith.cmpi sgt, %11#1, %c0 : index
    %13 = arith.select %12, %11#1, %c0 : index
    %14 = fir.undefined index
    %15 = fir.slice %10, %14, %14, %c1, %11#1, %c1 : (i64, index, index, index, index, index) -> !fir.slice<2>
    %16 = fir.rebox %6 [%15] : (!fir.box<!fir.array<?x?xf32>>, !fir.slice<2>) -> !fir.box<!fir.array<?xf32>>
    %17:3 = fir.box_dims %4, %c1 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
    %18 = fir.slice %10, %14, %14, %c1, %17#1, %c1 : (i64, index, index, index, index, index) -> !fir.slice<2>
    %19 = fir.rebox %4 [%18] : (!fir.box<!fir.array<?x?xf32>>, !fir.slice<2>) -> !fir.box<!fir.array<?xf32>>
    fir.do_loop %arg4 = %c1 to %13 step %c1 unordered {
      %23 = fir.array_coor %16 %arg4 : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
      %24 = fir.load %23 : !fir.ref<f32>
      %25 = arith.addf %24, %cst fastmath<fast> : f32
      %26 = fir.array_coor %19 %arg4 : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
      fir.store %25 to %26 : !fir.ref<f32>
    }
    %20 = arith.addi %arg2, %c1 overflow<nsw> : index
    %21 = fir.load %2 : !fir.ref<i32>
    %22 = arith.addi %21, %7 overflow<nsw> : i32
    fir.result %20, %22 : index, i32
  }
  fir.store %8#1 to %2 : !fir.ref<i32>
  return
}
// CHECK-LABEL:   func.func @_QPtest_maybe_noncontig_slice(
// CHECK-NOT: fir.if

// Regression test for facerec's GraphSimFct:
//real function test_graphsimfct(a1, a2)
//  integer :: i
//  real, intent(in) :: a1(:,:,:)
//  real, intent(in) :: a2(:,:,:,:)
//  graphsimfct = 0.0
//  do i=1,10
//     test_graphsimfct = test_graphsimfct + SUM(a1(:,:,i) * a2(:,:,i,i))
//  end do
//end function
func.func @_QPtest_graphsimfct(%arg0: !fir.box<!fir.array<?x?x?xf32>> {fir.bindc_name = "a1"}, %arg1: !fir.box<!fir.array<?x?x?x?xf32>> {fir.bindc_name = "a2"}) -> f32 {
  %c10 = arith.constant 10 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest_graphsimfctEa1"} : (!fir.box<!fir.array<?x?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?x?xf32>>
  %2 = fir.rebox %1 : (!fir.box<!fir.array<?x?x?xf32>>) -> !fir.box<!fir.array<?x?x?xf32>>
  %3 = fir.declare %arg1 dummy_scope %0 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFtest_graphsimfctEa2"} : (!fir.box<!fir.array<?x?x?x?xf32>>, !fir.dscope) -> !fir.box<!fir.array<?x?x?x?xf32>>
  %4 = fir.rebox %3 : (!fir.box<!fir.array<?x?x?x?xf32>>) -> !fir.box<!fir.array<?x?x?x?xf32>>
  %5 = fir.alloca f32 {bindc_name = "graphsimfct", uniq_name = "_QFtest_graphsimfctEgraphsimfct"}
  %6 = fir.declare %5 {uniq_name = "_QFtest_graphsimfctEgraphsimfct"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %7 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_graphsimfctEi"}
  %8 = fir.declare %7 {uniq_name = "_QFtest_graphsimfctEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %9 = fir.alloca f32 {bindc_name = "test_graphsimfct", uniq_name = "_QFtest_graphsimfctEtest_graphsimfct"}
  %10 = fir.declare %9 {uniq_name = "_QFtest_graphsimfctEtest_graphsimfct"} : (!fir.ref<f32>) -> !fir.ref<f32>
  fir.store %cst to %6 : !fir.ref<f32>
  %11 = fir.convert %c1 : (index) -> i32
  %12:2 = fir.do_loop %arg2 = %c1 to %c10 step %c1 iter_args(%arg3 = %11) -> (index, i32) {
    fir.store %arg3 to %8 : !fir.ref<i32>
    %14 = fir.load %10 : !fir.ref<f32>
    %15:3 = fir.box_dims %2, %c0 : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
    %16:3 = fir.box_dims %2, %c1 : (!fir.box<!fir.array<?x?x?xf32>>, index) -> (index, index, index)
    %17 = arith.cmpi sgt, %15#1, %c0 : index
    %18 = arith.select %17, %15#1, %c0 : index
    %19 = arith.cmpi sgt, %16#1, %c0 : index
    %20 = arith.select %19, %16#1, %c0 : index
    %21 = fir.load %8 : !fir.ref<i32>
    %22 = fir.convert %21 : (i32) -> i64
    %23 = fir.undefined index
    %24 = fir.slice %c1, %15#1, %c1, %c1, %16#1, %c1, %22, %23, %23 : (index, index, index, index, index, index, i64, index, index) -> !fir.slice<3>
    %25 = fir.rebox %2 [%24] : (!fir.box<!fir.array<?x?x?xf32>>, !fir.slice<3>) -> !fir.box<!fir.array<?x?xf32>>
    %26:3 = fir.box_dims %4, %c0 : (!fir.box<!fir.array<?x?x?x?xf32>>, index) -> (index, index, index)
    %27:3 = fir.box_dims %4, %c1 : (!fir.box<!fir.array<?x?x?x?xf32>>, index) -> (index, index, index)
    %28 = fir.slice %c1, %26#1, %c1, %c1, %27#1, %c1, %22, %23, %23, %22, %23, %23 : (index, index, index, index, index, index, i64, index, index, i64, index, index) -> !fir.slice<4>
    %29 = fir.rebox %4 [%28] : (!fir.box<!fir.array<?x?x?x?xf32>>, !fir.slice<4>) -> !fir.box<!fir.array<?x?xf32>>
    %30 = fir.do_loop %arg4 = %c1 to %20 step %c1 unordered iter_args(%arg5 = %cst) -> (f32) {
      %35 = fir.do_loop %arg6 = %c1 to %18 step %c1 unordered iter_args(%arg7 = %arg5) -> (f32) {
        %36 = fir.array_coor %25 %arg6, %arg4 : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
        %37 = fir.array_coor %29 %arg6, %arg4 : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
        %38 = fir.load %36 : !fir.ref<f32>
        %39 = fir.load %37 : !fir.ref<f32>
        %40 = arith.mulf %38, %39 fastmath<fast> : f32
        %41 = arith.addf %arg7, %40 fastmath<fast> : f32
        fir.result %41 : f32
      }
      fir.result %35 : f32
    }
    %31 = arith.addf %14, %30 fastmath<fast> : f32
    fir.store %31 to %10 : !fir.ref<f32>
    %32 = arith.addi %arg2, %c1 overflow<nsw> : index
    %33 = fir.load %8 : !fir.ref<i32>
    %34 = arith.addi %33, %11 overflow<nsw> : i32
    fir.result %32, %34 : index, i32
  }
  fir.store %12#1 to %8 : !fir.ref<i32>
  %13 = fir.load %10 : !fir.ref<f32>
  return %13 : f32
}
// CHECK-LABEL:   func.func @_QPtest_graphsimfct(
// CHECK:           fir.do_loop
// CHECK:             fir.do_loop
// CHECK:               fir.if
// CHECK:                 fir.do_loop
// CHECK:               } else {
// CHECK:                 fir.do_loop

} // End module
