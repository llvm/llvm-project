// RUN: fir-opt --split-input-file --cuf-predefined-var-to-gpu --canonicalize %s | FileCheck %s
// RUN: fir-opt --split-input-file --cuf-predefined-var-to-gpu --canonicalize %s | fir-opt --cuf-predefined-var-to-gpu --canonicalize | FileCheck %s

// attributes(device) subroutine sub1(i)
//   integer :: i
//   i = threadidx%x
//   i = blockdim%x
//   i = blockidx%x
//   i = griddim%x
//   i = warpsize
// end subroutine

// The following FIR output is coming from the small CUDA Fortran code above.
// To reproduce the output or update it:
//   bbc -emit-hlfir -fcuda %s -o - | fir-opt --convert-hlfir-to-fir
func.func @_QPsub1(%arg0: !fir.ref<i32> {fir.bindc_name = "i", cuf.data_attr = #cuf.cuda<device>}) attributes {cuf.proc_attr = #cuf.cuda_proc<device>} {
  %0 = fir.address_of(@_QM__fortran_builtinsE__builtin_blockdim) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %1 = fir.declare %0 {uniq_name = "_QM__fortran_builtinsE__builtin_blockdim"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %2 = fir.address_of(@_QM__fortran_builtinsE__builtin_blockidx) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %3 = fir.declare %2 {uniq_name = "_QM__fortran_builtinsE__builtin_blockidx"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %4 = fir.address_of(@_QM__fortran_builtinsE__builtin_griddim) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %5 = fir.declare %4 {uniq_name = "_QM__fortran_builtinsE__builtin_griddim"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %6 = fir.declare %arg0 {uniq_name = "_QFsub1Ei"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %7 = fir.address_of(@_QM__fortran_builtinsE__builtin_threadidx) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %8 = fir.declare %7 {uniq_name = "_QM__fortran_builtinsE__builtin_threadidx"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %9 = fir.alloca i32 {bindc_name = "__builtin_warpsize", uniq_name = "_QM__fortran_builtinsEC__builtin_warpsize"}
  %10 = fir.declare %9 {uniq_name = "_QM__fortran_builtinsEC__builtin_warpsize"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %12 = fir.coordinate_of %8, x : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
  %13 = fir.load %12 : !fir.ref<i32>
  fir.store %13 to %6 : !fir.ref<i32>
  %15 = fir.coordinate_of %1, x : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
  %16 = fir.load %15 : !fir.ref<i32>
  fir.store %16 to %6 : !fir.ref<i32>
  %18 = fir.coordinate_of %3, x : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
  %19 = fir.load %18 : !fir.ref<i32>
  fir.store %19 to %6 : !fir.ref<i32>
  %21 = fir.coordinate_of %5, y : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
  %22 = fir.load %21 : !fir.ref<i32>
  fir.store %22 to %6 : !fir.ref<i32>
  %c32_i32 = arith.constant 32 : i32
  fir.store %c32_i32 to %6 : !fir.ref<i32>
  %24 = fir.coordinate_of %8, x : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
  %25 = fir.load %24 : !fir.ref<i32>
  %c0_i32 = arith.constant 0 : i32
  %26 = arith.cmpi eq, %25, %c0_i32 : i32
  fir.if %26 {
    %c0_i32_0 = arith.constant 0 : i32
    fir.store %c0_i32_0 to %6 : !fir.ref<i32>
  }
  return
}

// CHECK-LABEL: func.func @_QPsub1

// CHECK: %[[WARPSIZE:.*]] = arith.constant 32 : i32

// CHECK: %[[BASE_THREAD_ID_X:.*]] = nvvm.read.ptx.sreg.tid.x : i32
// CHECK: %[[THREAD_ID_X:.*]] = arith.addi %[[BASE_THREAD_ID_X]], %c1{{.*}} : i32
// CHECK: %[[BASE_BLOCK_ID_X:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
// CHECK: %[[BLOCK_ID_X:.*]] = arith.addi %[[BASE_BLOCK_ID_X]], %c1{{.*}} : i32
// CHECK: %[[GRID_DIM_Y:.*]] = nvvm.read.ptx.sreg.nctaid.y : i32
// CHECK: %[[BLOCK_DIM_X:.*]] = nvvm.read.ptx.sreg.ntid.x : i32

// CHECK: %[[I:.*]] = fir.declare %{{.*}} {uniq_name = "_QFsub1Ei"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK: fir.store %[[THREAD_ID_X]] to %[[I]] : !fir.ref<i32>
// CHECK: fir.store %[[BLOCK_DIM_X]] to %[[I]] : !fir.ref<i32>
// CHECK: fir.store %[[BLOCK_ID_X]] to %[[I]] : !fir.ref<i32>
// CHECK: fir.store %[[GRID_DIM_Y]] to %[[I]] : !fir.ref<i32>

// CHECK: fir.store %[[WARPSIZE]] to %[[I]] : !fir.ref<i32>

// CHECK: %[[CMP:.*]] = arith.cmpi eq, %[[THREAD_ID_X]], %c0{{.*}} : i32
// CHECK: fir.if %[[CMP]] {
// CHECK:   fir.store %c0{{.*}} to %[[I]] : !fir.ref<i32>
// CHECK: }


// These function should not be transformed. Just here to make sure the pass
// does not crash on them.

func.func private @_QPsub2(%arg0: !fir.ref<i32> {fir.bindc_name = "i"})

func.func @_QPsub3(%arg0: !fir.ref<i32> {fir.bindc_name = "i"}) {
  return
}

// -----

func.func @_QPsub1(%arg0: !fir.ref<i32> {fir.bindc_name = "i", cuf.data_attr = #cuf.cuda<device>}) attributes {cuf.proc_attr = #cuf.cuda_proc<grid_global>} {
  %0 = fir.address_of(@_QM__fortran_builtinsE__builtin_blockdim) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %1 = fir.declare %0 {uniq_name = "_QM__fortran_builtinsE__builtin_blockdim"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %2 = fir.address_of(@_QM__fortran_builtinsE__builtin_blockidx) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %3 = fir.declare %2 {uniq_name = "_QM__fortran_builtinsE__builtin_blockidx"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %4 = fir.address_of(@_QM__fortran_builtinsE__builtin_griddim) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %5 = fir.declare %4 {uniq_name = "_QM__fortran_builtinsE__builtin_griddim"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %6 = fir.declare %arg0 {uniq_name = "_QFsub1Ei"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %7 = fir.address_of(@_QM__fortran_builtinsE__builtin_threadidx) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %8 = fir.declare %7 {uniq_name = "_QM__fortran_builtinsE__builtin_threadidx"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %9 = fir.alloca i32 {bindc_name = "__builtin_warpsize", uniq_name = "_QM__fortran_builtinsEC__builtin_warpsize"}
  %10 = fir.declare %9 {uniq_name = "_QM__fortran_builtinsEC__builtin_warpsize"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %12 = fir.coordinate_of %8, x : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
  %13 = fir.load %12 : !fir.ref<i32>
  fir.store %13 to %6 : !fir.ref<i32>
  %15 = fir.coordinate_of %1, x : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
  %16 = fir.load %15 : !fir.ref<i32>
  fir.store %16 to %6 : !fir.ref<i32>
  %18 = fir.coordinate_of %3, x : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
  %19 = fir.load %18 : !fir.ref<i32>
  fir.store %19 to %6 : !fir.ref<i32>
  %21 = fir.coordinate_of %5, y : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
  %22 = fir.load %21 : !fir.ref<i32>
  fir.store %22 to %6 : !fir.ref<i32>
  %c32_i32 = arith.constant 32 : i32
  fir.store %c32_i32 to %6 : !fir.ref<i32>
  %24 = fir.coordinate_of %8, x : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
  %25 = fir.load %24 : !fir.ref<i32>
  %c0_i32 = arith.constant 0 : i32
  %26 = arith.cmpi eq, %25, %c0_i32 : i32
  fir.if %26 {
    %c0_i32_0 = arith.constant 0 : i32
    fir.store %c0_i32_0 to %6 : !fir.ref<i32>
  }
  return
}

// CHECK-LABEL: func.func @_QPsub1

// CHECK: %{{.*}} = arith.constant 32 : i32

// CHECK: %[[BASE_THREAD_ID_X:.*]] = nvvm.read.ptx.sreg.tid.x : i32
// CHECK: %{{.*}} = arith.addi %[[BASE_THREAD_ID_X]], %c1{{.*}} : i32
// CHECK: %[[BASE_BLOCK_ID_X:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
// CHECK: %{{.*}} = arith.addi %[[BASE_BLOCK_ID_X]], %c1{{.*}} : i32
// CHECK: %{{.*}} = nvvm.read.ptx.sreg.nctaid.y : i32
// CHECK: %{{.*}} = nvvm.read.ptx.sreg.ntid.x : i32


// -----

func.func @_QPsub1(%arg0: !fir.ref<i32> {fir.bindc_name = "i", cuf.data_attr = #cuf.cuda<device>}) -> i32 attributes {cuf.proc_attr = #cuf.cuda_proc<grid_global>} {
  %0 = fir.address_of(@_QM__fortran_builtinsE__builtin_blockdim) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %1 = fir.declare %0 {uniq_name = "_QM__fortran_builtinsE__builtin_blockdim"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %2 = fir.address_of(@_QM__fortran_builtinsE__builtin_blockidx) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %3 = fir.declare %2 {uniq_name = "_QM__fortran_builtinsE__builtin_blockidx"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %4 = fir.address_of(@_QM__fortran_builtinsE__builtin_griddim) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %5 = fir.declare %4 {uniq_name = "_QM__fortran_builtinsE__builtin_griddim"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %6 = fir.declare %arg0 {uniq_name = "_QFsub1Ei"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %7 = fir.address_of(@_QM__fortran_builtinsE__builtin_threadidx) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %8 = fir.declare %7 {uniq_name = "_QM__fortran_builtinsE__builtin_threadidx"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %9 = fir.alloca i32 {bindc_name = "__builtin_warpsize", uniq_name = "_QM__fortran_builtinsEC__builtin_warpsize"}
  %10 = fir.declare %9 {uniq_name = "_QM__fortran_builtinsEC__builtin_warpsize"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %tid = nvvm.read.ptx.sreg.tid.x : i32
  %12 = fir.coordinate_of %8, x : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
  %13 = fir.load %12 : !fir.ref<i32>
  fir.store %13 to %6 : !fir.ref<i32>
  %15 = fir.coordinate_of %1, x : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
  %16 = fir.load %15 : !fir.ref<i32>
  fir.store %16 to %6 : !fir.ref<i32>
  %18 = fir.coordinate_of %3, x : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
  %19 = fir.load %18 : !fir.ref<i32>
  fir.store %19 to %6 : !fir.ref<i32>
  %21 = fir.coordinate_of %5, y : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
  %22 = fir.load %21 : !fir.ref<i32>
  fir.store %22 to %6 : !fir.ref<i32>
  %c32_i32 = arith.constant 32 : i32
  fir.store %c32_i32 to %6 : !fir.ref<i32>
  %24 = fir.coordinate_of %8, x : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
  %25 = fir.load %24 : !fir.ref<i32>
  %c0_i32 = arith.constant 0 : i32
  %26 = arith.cmpi eq, %25, %c0_i32 : i32
  fir.if %26 {
    %c0_i32_0 = arith.constant 0 : i32
    fir.store %c0_i32_0 to %6 : !fir.ref<i32>
  }
  return %tid : i32
}


// CHECK-LABEL: func.func @_QPsub1

// CHECK: %{{.*}} = arith.constant 32 : i32

// CHECK: %[[BASE_THREAD_ID_X:.*]] = nvvm.read.ptx.sreg.tid.x : i32
// CHECK: %{{.*}} = arith.addi %[[BASE_THREAD_ID_X]], %c1{{.*}} : i32
// CHECK: %[[BASE_BLOCK_ID_X:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
// CHECK: %{{.*}} = arith.addi %[[BASE_BLOCK_ID_X]], %c1{{.*}} : i32
// CHECK: %{{.*}} = nvvm.read.ptx.sreg.nctaid.y : i32
// CHECK: %{{.*}} = nvvm.read.ptx.sreg.ntid.x : i32
