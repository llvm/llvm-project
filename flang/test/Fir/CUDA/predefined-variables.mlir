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

// -----

func.func @_QMbarPgfoo(%arg0: !fir.ref<i32> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "a"}) attributes {cuf.proc_attr = #cuf.cuda_proc<global>, no_inline} {
  %c100_i32 = arith.constant 100 : i32
  %cond = arith.cmpi sle, %c100_i32, %c100_i32 : i32
  fir.if %cond {
    %0 = fir.address_of(@_QM__fortran_builtinsE__builtin_threadidx) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
    %1 = fir.declare %0 {uniq_name = "_QM__fortran_builtinsE__builtin_threadidx"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
    %2 = fir.coordinate_of %1, x : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
    %3 = fir.load %2 : !fir.ref<i32>
    fir.store %3 to %arg0 : !fir.ref<i32>
  }
  return
}

// CHECK-LABEL: func.func @_QMbarPgfoo
// CHECK: %[[THREAD_ID_X:.*]] = nvvm.read.ptx.sreg.tid.x : i32
// CHECK: %[[ADD:.*]] = arith.addi %[[THREAD_ID_X]], %c1_i32 : i32
// CHECK: fir.if
// CHECK: fir.store %[[ADD]] to %{{.*}} : !fir.ref<i32>

// -----

func.func @_QMbarPgfoo2(%arg0: !fir.ref<i32> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {cuf.data_attr = #cuf.cuda<device>, fir.bindc_name = "b"}) attributes {cuf.proc_attr = #cuf.cuda_proc<global>, no_inline} {
  %c100_i32 = arith.constant 100 : i32
  %0 = fir.address_of(@_QM__fortran_builtinsE__builtin_threadidx) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %1 = fir.declare %0 {uniq_name = "_QM__fortran_builtinsE__builtin_threadidx"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %2 = fir.coordinate_of %1, x : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
  %3 = fir.load %2 : !fir.ref<i32>
  fir.store %3 to %arg0 : !fir.ref<i32>
  %cond = arith.cmpi sle, %c100_i32, %c100_i32 : i32
  fir.if %cond {
    %4 = fir.address_of(@_QM__fortran_builtinsE__builtin_threadidx) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
    %5 = fir.declare %4 {uniq_name = "_QM__fortran_builtinsE__builtin_threadidx"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
    %6 = fir.coordinate_of %5, x : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
    %7 = fir.load %6 : !fir.ref<i32>
    fir.store %7 to %arg1 : !fir.ref<i32>
  }
  return
}

// CHECK-LABEL: func.func @_QMbarPgfoo2
// CHECK: %[[THREAD_ID_X:.*]] = nvvm.read.ptx.sreg.tid.x : i32
// CHECK: %[[ADD:.*]] = arith.addi %[[THREAD_ID_X]], %c1_i32 : i32
// CHECK: fir.store %[[ADD]] to %{{.*}} : !fir.ref<i32>
// CHECK: fir.if
// CHECK: fir.store %[[ADD]] to %{{.*}} : !fir.ref<i32>

// -----

func.func @surviving_predefined_vars(%arg0: i32, %arg1: i32, %arg2: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.000000e+00 : f64
  %i0 = arith.constant 0 : i32
  %i1 = arith.constant 1 : i32

  %v0 = fir.alloca i32 {uniq_name = "_QFminiEi"}
  %v1 = fir.alloca i32 {uniq_name = "_QFminiEj"}
  %sum = fir.alloca f64 {uniq_name = "_QFminiEsum"}
  %sum_d = fir.declare %sum {uniq_name = "_QFminiEsum"} : (!fir.ref<f64>) -> !fir.ref<f64>

  cuf.kernel<<<*, *>>> (%iv0 : index, %iv1 : index, %iv2 : index) = (%c0, %c0, %c0 : index, index, index) to (%c1, %c1, %c1 : index, index, index) step (%c1, %c1, %c1 : index, index, index) {
    %dim = fir.address_of(@_QM__fortran_builtinsE__builtin_blockdim) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
    %bid = fir.address_of(@_QM__fortran_builtinsE__builtin_blockidx) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
    %gdim = fir.address_of(@_QM__fortran_builtinsE__builtin_griddim) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
    %tid = fir.address_of(@_QM__fortran_builtinsE__builtin_threadidx) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>

    %d_dim = fir.declare %dim {uniq_name = "_QM__fortran_builtinsE__builtin_blockdim"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
    %d_bid = fir.declare %bid {uniq_name = "_QM__fortran_builtinsE__builtin_blockidx"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
    %d_gdim = fir.declare %gdim {uniq_name = "_QM__fortran_builtinsE__builtin_griddim"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
    %d_tid = fir.declare %tid {uniq_name = "_QM__fortran_builtinsE__builtin_threadidx"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>

    fir.store %i0 to %v0 : !fir.ref<i32>
    fir.store %i1 to %v1 : !fir.ref<i32>
    fir.store %cst to %sum_d : !fir.ref<f64>
    "fir.end"() : () -> ()
  } {n = 3 : i64}

  return
}

// CHECK-LABEL: surviving_predefined_vars
// CHECK-NOT: _QM__fortran_builtinsE__builtin_blockdim
// CHECK-NOT: _QM__fortran_builtinsE__builtin_blockidx
// CHECK-NOT: _QM__fortran_builtinsE__builtin_griddim
// CHECK-NOT: _QM__fortran_builtinsE__builtin_threadidx


// -----

func.func @surviving_predefined_vars(%arg0: i32, %arg1: i32, %arg2: i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.000000e+00 : f64
  %i0 = arith.constant 0 : i32
  %i1 = arith.constant 1 : i32

  %v0 = fir.alloca i32 {uniq_name = "_QFminiEi"}
  %v1 = fir.alloca i32 {uniq_name = "_QFminiEj"}
  %sum = fir.alloca f64 {uniq_name = "_QFminiEsum"}
  %sum_d = fir.declare %sum {uniq_name = "_QFminiEsum"} : (!fir.ref<f64>) -> !fir.ref<f64>

  cuf.kernel<<<*, *>>> (%iv0 : index, %iv1 : index, %iv2 : index) = (%c0, %c0, %c0 : index, index, index) to (%c1, %c1, %c1 : index, index, index) step (%c1, %c1, %c1 : index, index, index) {
    %dim = fir.address_of(@_QM__fortran_builtinsE__builtin_blockdim) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
    %bid = fir.address_of(@_QM__fortran_builtinsE__builtin_blockidx) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
    %gdim = fir.address_of(@_QM__fortran_builtinsE__builtin_griddim) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
    %tid = fir.address_of(@_QM__fortran_builtinsE__builtin_threadidx) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>

    %d_dim = fir.declare %dim {uniq_name = "_QM__fortran_builtinsE__builtin_blockdim"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
    %d_bid = fir.declare %bid {uniq_name = "_QM__fortran_builtinsE__builtin_blockidx"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
    %d_gdim = fir.declare %gdim {uniq_name = "_QM__fortran_builtinsE__builtin_griddim"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
    %d_tid = fir.declare %tid {uniq_name = "_QM__fortran_builtinsE__builtin_threadidx"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>

    %0 = fir.coordinate_of %d_tid, x : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
    %1 = fir.load %0 : !fir.ref<i32>
    fir.store %1 to %v0 : !fir.ref<i32>
    "fir.end"() : () -> ()
  } {n = 3 : i64}

  return
}

// CHECK-LABEL: surviving_predefined_vars
// CHECK-NOT: _QM__fortran_builtinsE__builtin_blockdim
// CHECK-NOT: _QM__fortran_builtinsE__builtin_blockidx
// CHECK-NOT: _QM__fortran_builtinsE__builtin_griddim
// CHECK-NOT: _QM__fortran_builtinsE__builtin_threadidx
// CHECK: nvvm.read.ptx.sreg.tid.x

// -----

func.func @_QMoutermodPouter(%arg0: !fir.ref<!fir.array<?x?xf64>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<?x?xf64>> {fir.bindc_name = "b"}, %arg2: i32 {fir.bindc_name = "stride"}, %arg3: i32 {fir.bindc_name = "n"}) attributes {no_inline} {
  %c1 = arith.constant 1 : index
  %cst = arith.constant 2.000000e+00 : f64
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32
  fir.store %arg2 to %1 : !fir.ref<i32>
  %2 = fir.declare %1 dummy_scope %0 arg 3 {fortran_attrs = #fir.var_attrs<intent_in, value>, uniq_name = "_QMoutermodFouterEstride"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %3 = fir.alloca i32
  fir.store %arg3 to %3 : !fir.ref<i32>
  %4 = fir.declare %3 dummy_scope %0 arg 4 {fortran_attrs = #fir.var_attrs<intent_in, value>, uniq_name = "_QMoutermodFouterEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %5 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QMoutermodFouterEj"}
  %6 = fir.declare %5 {uniq_name = "_QMoutermodFouterEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %7 = fir.load %2 : !fir.ref<i32>
  %8 = fir.convert %7 : (i32) -> index
  %9 = arith.maxsi %8, %c0 : index
  %10 = fir.load %4 : !fir.ref<i32>
  %11 = fir.convert %10 : (i32) -> index
  %12 = arith.maxsi %11, %c0 : index
  %13 = fir.shape %9, %12 : (index, index) -> !fir.shape<2>
  %14 = fir.declare %arg0(%13) dummy_scope %0 arg 1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMoutermodFouterEa"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<?x?xf64>>
  %15 = fir.embox %14(%13) : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf64>>
  %16 = fir.declare %arg1(%13) dummy_scope %0 arg 2 {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "_QMoutermodFouterEb"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<?x?xf64>>
  %17 = fir.embox %16(%13) : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf64>>
  %18 = acc.present var(%15 : !fir.box<!fir.array<?x?xf64>>) -> !fir.box<!fir.array<?x?xf64>> {name = "a"}
  %19 = acc.present var(%17 : !fir.box<!fir.array<?x?xf64>>) -> !fir.box<!fir.array<?x?xf64>> {name = "b"}
  acc.parallel combined(loop) dataOperands(%18, %19 : !fir.box<!fir.array<?x?xf64>>, !fir.box<!fir.array<?x?xf64>>) {
    %20 = fir.box_addr %18 : (!fir.box<!fir.array<?x?xf64>>) -> !fir.ref<!fir.array<?x?xf64>>
    %21 = fir.dummy_scope : !fir.dscope
    %22 = fir.declare %20(%13) dummy_scope %21 arg 1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMoutermodFouterEa"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<?x?xf64>>
    %23 = fir.box_addr %19 : (!fir.box<!fir.array<?x?xf64>>) -> !fir.ref<!fir.array<?x?xf64>>
    %24 = fir.declare %23(%13) dummy_scope %21 arg 2 {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "_QMoutermodFouterEb"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<?x?xf64>>
    %25 = fir.load %4 : !fir.ref<i32>
    %26 = acc.private varPtr(%6 : !fir.ref<i32>) recipe(@privatization_ref_i32) -> !fir.ref<i32> {implicit = true, name = "j"}
    %27 = fir.load %2 : !fir.ref<i32>
    %28 = fir.address_of(@_QM__fortran_builtinsE__builtin_blockdim) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
    %29 = fir.address_of(@_QM__fortran_builtinsE__builtin_blockidx) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
    %30 = fir.address_of(@_QM__fortran_builtinsE__builtin_griddim) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
    %31 = fir.address_of(@_QM__fortran_builtinsE__builtin_threadidx) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
    %32 = fir.assumed_size_extent : index
    %33 = fir.convert %c1 : (index) -> i32
    acc.loop combined(parallel) gang vector private(%26 : !fir.ref<i32>) control(%arg4 : i32) = (%c1_i32 : i32) to (%25 : i32)  step (%c1_i32 : i32) {
      %34 = fir.alloca i32
      %35 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QMoutermodFinner_loopEi"}
      %36 = fir.alloca i32
      %37 = fir.alloca i32 {bindc_name = "warpsize", uniq_name = "_QMcudadeviceECwarpsize"}
      %38 = fir.declare %26 {uniq_name = "_QMoutermodFouterEj"} : (!fir.ref<i32>) -> !fir.ref<i32>
      fir.store %arg4 to %38 : !fir.ref<i32>
      %39 = fir.load %38 : !fir.ref<i32>
      %40 = fir.dummy_scope : !fir.dscope
      fir.store %27 to %34 : !fir.ref<i32>
      %41 = fir.declare %34 dummy_scope %40 arg 3 {fortran_attrs = #fir.var_attrs<intent_in, value>, uniq_name = "_QMoutermodFinner_loopEstride"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
      %42 = fir.declare %28 {uniq_name = "_QM__fortran_builtinsE__builtin_blockdim"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
      %43 = fir.declare %29 {uniq_name = "_QM__fortran_builtinsE__builtin_blockidx"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
      %44 = fir.declare %30 {uniq_name = "_QM__fortran_builtinsE__builtin_griddim"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
      %45 = fir.declare %35 {uniq_name = "_QMoutermodFinner_loopEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
      fir.store %39 to %36 : !fir.ref<i32>
      %46 = fir.declare %36 dummy_scope %40 arg 4 {fortran_attrs = #fir.var_attrs<intent_in, value>, uniq_name = "_QMoutermodFinner_loopEj"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
      %47 = fir.declare %31 {uniq_name = "_QM__fortran_builtinsE__builtin_threadidx"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
      %48 = fir.declare %37 {uniq_name = "_QMcudadeviceECwarpsize"} : (!fir.ref<i32>) -> !fir.ref<i32>
      %49 = fir.load %41 : !fir.ref<i32>
      %50 = fir.convert %49 : (i32) -> index
      %51 = arith.maxsi %50, %c0 : index
      %52 = fir.shape %51, %32 : (index, index) -> !fir.shape<2>
      %53 = fir.declare %22(%52) dummy_scope %40 arg 1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QMoutermodFinner_loopEa"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<?x?xf64>>
      %54 = fir.declare %24(%52) dummy_scope %40 arg 2 {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "_QMoutermodFinner_loopEb"} : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<?x?xf64>>
      %55 = fir.load %46 : !fir.ref<i32>
      %56 = fir.convert %55 : (i32) -> i64
      %57 = fir.do_loop %arg5 = %c1 to %50 step %c1 iter_args(%arg6 = %33) -> (i32) {
        fir.store %arg6 to %45 : !fir.ref<i32>
        %58 = fir.load %45 : !fir.ref<i32>
        %59 = fir.convert %58 : (i32) -> i64
        %60 = fir.array_coor %53(%52) %59, %56 : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>, i64, i64) -> !fir.ref<f64>
        %61 = fir.load %60 : !fir.ref<f64>
        %62 = arith.mulf %61, %cst fastmath<reassoc,contract> : f64
        %63 = fir.array_coor %54(%52) %59, %56 : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>, i64, i64) -> !fir.ref<f64>
        fir.store %62 to %63 : !fir.ref<f64>
        %64 = fir.load %45 : !fir.ref<i32>
        %65 = arith.addi %64, %33 overflow<nsw> : i32
        fir.result %65 : i32
      }
      fir.store %57 to %45 : !fir.ref<i32>
      acc.yield
    } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
    acc.yield
  }
  acc.delete accVar(%18 : !fir.box<!fir.array<?x?xf64>>) {dataClause = #acc<data_clause acc_present>, name = "a"}
  acc.delete accVar(%19 : !fir.box<!fir.array<?x?xf64>>) {dataClause = #acc<data_clause acc_present>, name = "b"}
  return
}

// CHECK-LABEL: _QMoutermodPouter
// CHECK-NOT: _QM__fortran_builtinsE__builtin_blockdim
// CHECK-NOT: _QM__fortran_builtinsE__builtin_blockidx
// CHECK-NOT: _QM__fortran_builtinsE__builtin_griddim
// CHECK-NOT: _QM__fortran_builtinsE__builtin_threadidx

// -----

// A single fir.address_of shared by two declares must not be erased while a
// declare still uses it.
func.func @_QMdevmodPkernel() attributes {cuf.proc_attr = #cuf.cuda_proc<global>, no_inline} {
  %0 = fir.address_of(@_QM__fortran_builtinsE__builtin_threadidx) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %1 = fir.declare %0 {uniq_name = "_QM__fortran_builtinsE__builtin_threadidx"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %2 = fir.declare %0 {uniq_name = "_QM__fortran_builtinsE__builtin_threadidx"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>
  %3 = fir.alloca i32 {bindc_name = "a"}
  %4 = fir.alloca i32 {bindc_name = "b"}
  %5 = fir.coordinate_of %1, x : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
  %6 = fir.load %5 : !fir.ref<i32>
  fir.store %6 to %3 : !fir.ref<i32>
  %7 = fir.coordinate_of %2, x : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_dim3{x:i32,y:i32,z:i32}>>) -> !fir.ref<i32>
  %8 = fir.load %7 : !fir.ref<i32>
  fir.store %8 to %4 : !fir.ref<i32>
  return
}

// CHECK-LABEL: func.func @_QMdevmodPkernel
// CHECK-NOT: _QM__fortran_builtinsE__builtin_threadidx
// CHECK: %[[TID:.*]] = nvvm.read.ptx.sreg.tid.x : i32
// CHECK: %[[ADD:.*]] = arith.addi %[[TID]], %c1{{.*}} : i32
// CHECK: fir.store %[[ADD]] to %{{.*}} : !fir.ref<i32>
// CHECK: fir.store %[[ADD]] to %{{.*}} : !fir.ref<i32>
