! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes="CHECK-FIR" %s
! RUN: %flang_fc1 -emit-fir %s -o - | fir-opt --fir-to-llvm-ir | FileCheck --check-prefixes="CHECK-LLVMIR" %s
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck --check-prefixes="CHECK" %s
! REQUIRES: target=powerpc{{.*}}

! vec_add

! CHECK-LABEL: vec_add_testf32
subroutine vec_add_testf32(x, y)
  vector(real(4)) :: vsum, x, y
  vsum = vec_add(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[vsum:.*]] = arith.addf %[[vx]], %[[vy]] fastmath<contract> : vector<4xf32>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vsum]] : (vector<4xf32>) -> !fir.vector<4:f32>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.fadd %[[x]], %[[y]]  {fastmathFlags = #llvm.fastmath<contract>} : vector<4xf32>

! CHECK: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = fadd contract <4 x float> %[[x]], %[[y]]
end subroutine vec_add_testf32

! CHECK-LABEL: vec_add_testf64
subroutine vec_add_testf64(x, y)
  vector(real(8)) :: vsum, x, y
  vsum = vec_add(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[vsum:.*]] = arith.addf %[[vx]], %[[vy]] fastmath<contract> : vector<2xf64>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vsum]] : (vector<2xf64>) -> !fir.vector<2:f64>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.fadd %[[x]], %[[y]]  {fastmathFlags = #llvm.fastmath<contract>} : vector<2xf64>

! CHECK: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = fadd contract <2 x double> %[[x]], %[[y]]
end subroutine vec_add_testf64

! CHECK-LABEL: vec_add_testi8
subroutine vec_add_testi8(x, y)
  vector(integer(1)) :: vsum, x, y
  vsum = vec_add(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[vsum:.*]] = arith.addi %[[vx]], %[[vy]] : vector<16xi8>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vsum]] : (vector<16xi8>) -> !fir.vector<16:i8>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.add %[[x]], %[[y]] : vector<16xi8>

! CHECK: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = add <16 x i8> %[[x]], %[[y]]
end subroutine vec_add_testi8

! CHECK-LABEL: vec_add_testi16
subroutine vec_add_testi16(x, y)
  vector(integer(2)) :: vsum, x, y
  vsum = vec_add(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[vsum:.*]] = arith.addi %[[vx]], %[[vy]] : vector<8xi16>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vsum]] : (vector<8xi16>) -> !fir.vector<8:i16>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.add %[[x]], %[[y]] : vector<8xi16>

! CHECK: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = add <8 x i16> %[[x]], %[[y]]
end subroutine vec_add_testi16

! CHECK-LABEL: vec_add_testi32
subroutine vec_add_testi32(x, y)
  vector(integer(4)) :: vsum, x, y
  vsum = vec_add(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[vsum:.*]] = arith.addi %[[vx]], %[[vy]] : vector<4xi32>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vsum]] : (vector<4xi32>) -> !fir.vector<4:i32>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.add %[[x]], %[[y]] : vector<4xi32>

! CHECK: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = add <4 x i32> %[[x]], %[[y]]
end subroutine vec_add_testi32

! CHECK-LABEL: vec_add_testi64
subroutine vec_add_testi64(x, y)
  vector(integer(8)) :: vsum, x, y
  vsum = vec_add(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[vsum:.*]] = arith.addi %[[vx]], %[[vy]] : vector<2xi64>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vsum]] : (vector<2xi64>) -> !fir.vector<2:i64>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.add %[[x]], %[[y]] : vector<2xi64>

! CHECK: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = add <2 x i64> %[[x]], %[[y]]
end subroutine vec_add_testi64

! CHECK-LABEL: vec_add_testui8
subroutine vec_add_testui8(x, y)
  vector(unsigned(1)) :: vsum, x, y
  vsum = vec_add(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[vsum:.*]] = arith.addi %[[vx]], %[[vy]] : vector<16xi8>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vsum]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.add %[[x]], %[[y]] : vector<16xi8>

! CHECK: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = add <16 x i8> %[[x]], %[[y]]
end subroutine vec_add_testui8

! CHECK-LABEL: vec_add_testui16
subroutine vec_add_testui16(x, y)
  vector(unsigned(2)) :: vsum, x, y
  vsum = vec_add(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[vsum:.*]] = arith.addi %[[vx]], %[[vy]] : vector<8xi16>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vsum]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.add %[[x]], %[[y]] : vector<8xi16>

! CHECK: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = add <8 x i16> %[[x]], %[[y]]
end subroutine vec_add_testui16

! CHECK-LABEL: vec_add_testui32
subroutine vec_add_testui32(x, y)
  vector(unsigned(4)) :: vsum, x, y
  vsum = vec_add(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[vsum:.*]] = arith.addi %[[vx]], %[[vy]] : vector<4xi32>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vsum]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.add %[[x]], %[[y]] : vector<4xi32>

! CHECK: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = add <4 x i32> %[[x]], %[[y]]
end subroutine vec_add_testui32

! CHECK-LABEL: vec_add_testui64
subroutine vec_add_testui64(x, y)
  vector(unsigned(8)) :: vsum, x, y
  vsum = vec_add(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[vsum:.*]] = arith.addi %[[vx]], %[[vy]] : vector<2xi64>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vsum]] : (vector<2xi64>) -> !fir.vector<2:ui64>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.add %[[x]], %[[y]] : vector<2xi64>

! CHECK: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = add <2 x i64> %[[x]], %[[y]]
end subroutine vec_add_testui64

! vec_mul

! CHECK-LABEL: vec_mul_testf32
subroutine vec_mul_testf32(x, y)
  vector(real(4)) :: vmul, x, y
  vmul = vec_mul(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[vmul:.*]] = arith.mulf %[[vx]], %[[vy]] fastmath<contract> : vector<4xf32>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vmul]] : (vector<4xf32>) -> !fir.vector<4:f32>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.fmul %[[x]], %[[y]]  {fastmathFlags = #llvm.fastmath<contract>} : vector<4xf32>

! CHECK: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = fmul contract <4 x float> %[[x]], %[[y]]
end subroutine vec_mul_testf32

! CHECK-LABEL: vec_mul_testf64
subroutine vec_mul_testf64(x, y)
  vector(real(8)) :: vmul, x, y
  vmul = vec_mul(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[vmul:.*]] = arith.mulf %[[vx]], %[[vy]] fastmath<contract> : vector<2xf64>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vmul]] : (vector<2xf64>) -> !fir.vector<2:f64>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.fmul %[[x]], %[[y]]  {fastmathFlags = #llvm.fastmath<contract>} : vector<2xf64>

! CHECK: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = fmul contract <2 x double> %[[x]], %[[y]]
end subroutine vec_mul_testf64

! CHECK-LABEL: vec_mul_testi8
subroutine vec_mul_testi8(x, y)
  vector(integer(1)) :: vmul, x, y
  vmul = vec_mul(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[vmul:.*]] = arith.muli %[[vx]], %[[vy]] : vector<16xi8>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vmul]] : (vector<16xi8>) -> !fir.vector<16:i8>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.mul %[[x]], %[[y]] : vector<16xi8>

! CHECK: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = mul <16 x i8> %[[x]], %[[y]]
end subroutine vec_mul_testi8

! CHECK-LABEL: vec_mul_testi16
subroutine vec_mul_testi16(x, y)
  vector(integer(2)) :: vmul, x, y
  vmul = vec_mul(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[vmul:.*]] = arith.muli %[[vx]], %[[vy]] : vector<8xi16>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vmul]] : (vector<8xi16>) -> !fir.vector<8:i16>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.mul %[[x]], %[[y]] : vector<8xi16>

! CHECK: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = mul <8 x i16> %[[x]], %[[y]]
end subroutine vec_mul_testi16

! CHECK-LABEL: vec_mul_testi32
subroutine vec_mul_testi32(x, y)
  vector(integer(4)) :: vmul, x, y
  vmul = vec_mul(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[vmul:.*]] = arith.muli %[[vx]], %[[vy]] : vector<4xi32>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vmul]] : (vector<4xi32>) -> !fir.vector<4:i32>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.mul %[[x]], %[[y]] : vector<4xi32>

! CHECK: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = mul <4 x i32> %[[x]], %[[y]]
end subroutine vec_mul_testi32

! CHECK-LABEL: vec_mul_testi64
subroutine vec_mul_testi64(x, y)
  vector(integer(8)) :: vmul, x, y
  vmul = vec_mul(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[vmul:.*]] = arith.muli %[[vx]], %[[vy]] : vector<2xi64>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vmul]] : (vector<2xi64>) -> !fir.vector<2:i64>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.mul %[[x]], %[[y]] : vector<2xi64>

! CHECK: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = mul <2 x i64> %[[x]], %[[y]]
end subroutine vec_mul_testi64

! CHECK-LABEL: vec_mul_testui8
subroutine vec_mul_testui8(x, y)
  vector(unsigned(1)) :: vmul, x, y
  vmul = vec_mul(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[vmul:.*]] = arith.muli %[[vx]], %[[vy]] : vector<16xi8>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vmul]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.mul %[[x]], %[[y]] : vector<16xi8>

! CHECK: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = mul <16 x i8> %[[x]], %[[y]]
end subroutine vec_mul_testui8

! CHECK-LABEL: vec_mul_testui16
subroutine vec_mul_testui16(x, y)
  vector(unsigned(2)) :: vmul, x, y
  vmul = vec_mul(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[vmul:.*]] = arith.muli %[[vx]], %[[vy]] : vector<8xi16>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vmul]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.mul %[[x]], %[[y]] : vector<8xi16>

! CHECK: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = mul <8 x i16> %[[x]], %[[y]]
end subroutine vec_mul_testui16

! CHECK-LABEL: vec_mul_testui32
subroutine vec_mul_testui32(x, y)
  vector(unsigned(4)) :: vmul, x, y
  vmul = vec_mul(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[vmul:.*]] = arith.muli %[[vx]], %[[vy]] : vector<4xi32>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vmul]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.mul %[[x]], %[[y]] : vector<4xi32>

! CHECK: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = mul <4 x i32> %[[x]], %[[y]]
end subroutine vec_mul_testui32

! CHECK-LABEL: vec_mul_testui64
subroutine vec_mul_testui64(x, y)
  vector(unsigned(8)) :: vmul, x, y
  vmul = vec_mul(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[vmul:.*]] = arith.muli %[[vx]], %[[vy]] : vector<2xi64>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vmul]] : (vector<2xi64>) -> !fir.vector<2:ui64>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.mul %[[x]], %[[y]] : vector<2xi64>

! CHECK: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = mul <2 x i64> %[[x]], %[[y]]
end subroutine vec_mul_testui64

! vec_sub

! CHECK-LABEL: vec_sub_testf32
subroutine vec_sub_testf32(x, y)
  vector(real(4)) :: vsub, x, y
  vsub = vec_sub(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[vsub:.*]] = arith.subf %[[vx]], %[[vy]] fastmath<contract> : vector<4xf32>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vsub]] : (vector<4xf32>) -> !fir.vector<4:f32>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.fsub %[[x]], %[[y]]  {fastmathFlags = #llvm.fastmath<contract>} : vector<4xf32>

! CHECK: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = fsub contract <4 x float> %[[x]], %[[y]]
end subroutine vec_sub_testf32

! CHECK-LABEL: vec_sub_testf64
subroutine vec_sub_testf64(x, y)
  vector(real(8)) :: vsub, x, y
  vsub = vec_sub(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[vsub:.*]] = arith.subf %[[vx]], %[[vy]] fastmath<contract> : vector<2xf64>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vsub]] : (vector<2xf64>) -> !fir.vector<2:f64>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.fsub %[[x]], %[[y]]  {fastmathFlags = #llvm.fastmath<contract>} : vector<2xf64>

! CHECK: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = fsub contract <2 x double> %[[x]], %[[y]]
end subroutine vec_sub_testf64

! CHECK-LABEL: vec_sub_testi8
subroutine vec_sub_testi8(x, y)
  vector(integer(1)) :: vsub, x, y
  vsub = vec_sub(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[vsub:.*]] = arith.subi %[[vx]], %[[vy]] : vector<16xi8>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vsub]] : (vector<16xi8>) -> !fir.vector<16:i8>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.sub %[[x]], %[[y]] : vector<16xi8>

! CHECK: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = sub <16 x i8> %[[x]], %[[y]]
end subroutine vec_sub_testi8

! CHECK-LABEL: vec_sub_testi16
subroutine vec_sub_testi16(x, y)
  vector(integer(2)) :: vsub, x, y
  vsub = vec_sub(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[vsub:.*]] = arith.subi %[[vx]], %[[vy]] : vector<8xi16>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vsub]] : (vector<8xi16>) -> !fir.vector<8:i16>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.sub %[[x]], %[[y]] : vector<8xi16>

! CHECK: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = sub <8 x i16> %[[x]], %[[y]]
end subroutine vec_sub_testi16

! CHECK-LABEL: vec_sub_testi32
subroutine vec_sub_testi32(x, y)
  vector(integer(4)) :: vsub, x, y
  vsub = vec_sub(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[vsub:.*]] = arith.subi %[[vx]], %[[vy]] : vector<4xi32>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vsub]] : (vector<4xi32>) -> !fir.vector<4:i32>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.sub %[[x]], %[[y]] : vector<4xi32>

! CHECK: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = sub <4 x i32> %[[x]], %[[y]]
end subroutine vec_sub_testi32

! CHECK-LABEL: vec_sub_testi64
subroutine vec_sub_testi64(x, y)
  vector(integer(8)) :: vsub, x, y
  vsub = vec_sub(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[vsub:.*]] = arith.subi %[[vx]], %[[vy]] : vector<2xi64>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vsub]] : (vector<2xi64>) -> !fir.vector<2:i64>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.sub %[[x]], %[[y]] : vector<2xi64>

! CHECK: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = sub <2 x i64> %[[x]], %[[y]]
end subroutine vec_sub_testi64

! CHECK-LABEL: vec_sub_testui8
subroutine vec_sub_testui8(x, y)
  vector(unsigned(1)) :: vsub, x, y
  vsub = vec_sub(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[vsub:.*]] = arith.subi %[[vx]], %[[vy]] : vector<16xi8>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vsub]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.sub %[[x]], %[[y]] : vector<16xi8>

! CHECK: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = sub <16 x i8> %[[x]], %[[y]]
end subroutine vec_sub_testui8

! CHECK-LABEL: vec_sub_testui16
subroutine vec_sub_testui16(x, y)
  vector(unsigned(2)) :: vsub, x, y
  vsub = vec_sub(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[vsub:.*]] = arith.subi %[[vx]], %[[vy]] : vector<8xi16>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vsub]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.sub %[[x]], %[[y]] : vector<8xi16>

! CHECK: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = sub <8 x i16> %[[x]], %[[y]]
end subroutine vec_sub_testui16

! CHECK-LABEL: vec_sub_testui32
subroutine vec_sub_testui32(x, y)
  vector(unsigned(4)) :: vsub, x, y
  vsub = vec_sub(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[vsub:.*]] = arith.subi %[[vx]], %[[vy]] : vector<4xi32>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vsub]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.sub %[[x]], %[[y]] : vector<4xi32>

! CHECK: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = sub <4 x i32> %[[x]], %[[y]]
end subroutine vec_sub_testui32

! CHECK-LABEL: vec_sub_testui64
subroutine vec_sub_testui64(x, y)
  vector(unsigned(8)) :: vsub, x, y
  vsub = vec_sub(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[vsub:.*]] = arith.subi %[[vx]], %[[vy]] : vector<2xi64>
! CHECK-FIR: %{{[0-9]}} = fir.convert %[[vsub]] : (vector<2xi64>) -> !fir.vector<2:ui64>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %{{[0-9]}} = llvm.sub %[[x]], %[[y]] : vector<2xi64>

! CHECK: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %{{[0-9]}} = sub <2 x i64> %[[x]], %[[y]]
end subroutine vec_sub_testui64

!----------------------
! vec_and
!----------------------

! CHECK-LABEL: vec_and_test_i8
subroutine vec_and_test_i8(arg1, arg2)
  vector(integer(1)) :: r, arg1, arg2
  r = vec_and(arg1, arg2)
! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = arith.andi %[[varg1]], %[[varg2]] : vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.and %[[arg1]], %[[arg2]] : vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = and <16 x i8> %[[arg1]], %[[arg2]]
end subroutine vec_and_test_i8

! CHECK-LABEL: vec_and_test_i16
subroutine vec_and_test_i16(arg1, arg2)
  vector(integer(2)) :: r, arg1, arg2
  r = vec_and(arg1, arg2)
! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[r:.*]] = arith.andi %[[varg1]], %[[varg2]] : vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[r]] : (vector<8xi16>) -> !fir.vector<8:i16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.and %[[arg1]], %[[arg2]] : vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = and <8 x i16> %[[arg1]], %[[arg2]]
end subroutine vec_and_test_i16

! CHECK-LABEL: vec_and_test_i32
subroutine vec_and_test_i32(arg1, arg2)
  vector(integer(4)) :: r, arg1, arg2
  r = vec_and(arg1, arg2)
! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[r:.*]] = arith.andi %[[varg1]], %[[varg2]] : vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[r]] : (vector<4xi32>) -> !fir.vector<4:i32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.and %[[arg1]], %[[arg2]] : vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = and <4 x i32> %[[arg1]], %[[arg2]]
end subroutine vec_and_test_i32

! CHECK-LABEL: vec_and_test_i64
subroutine vec_and_test_i64(arg1, arg2)
  vector(integer(8)) :: r, arg1, arg2
  r = vec_and(arg1, arg2)
! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[r:.*]] = arith.andi %[[varg1]], %[[varg2]] : vector<2xi64>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[r]] : (vector<2xi64>) -> !fir.vector<2:i64>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.and %[[arg1]], %[[arg2]] : vector<2xi64>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = and <2 x i64> %[[arg1]], %[[arg2]]
end subroutine vec_and_test_i64

! CHECK-LABEL: vec_and_test_u8
subroutine vec_and_test_u8(arg1, arg2)
  vector(unsigned(1)) :: r, arg1, arg2
  r = vec_and(arg1, arg2)
! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = arith.andi %[[varg1]], %[[varg2]] : vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.and %[[arg1]], %[[arg2]] : vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = and <16 x i8> %[[arg1]], %[[arg2]]
end subroutine vec_and_test_u8

! CHECK-LABEL: vec_and_test_u16
subroutine vec_and_test_u16(arg1, arg2)
  vector(unsigned(2)) :: r, arg1, arg2
  r = vec_and(arg1, arg2)
! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[r:.*]] = arith.andi %[[varg1]], %[[varg2]] : vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[r]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.and %[[arg1]], %[[arg2]] : vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = and <8 x i16> %[[arg1]], %[[arg2]]
end subroutine vec_and_test_u16

! CHECK-LABEL: vec_and_test_u32
subroutine vec_and_test_u32(arg1, arg2)
  vector(unsigned(4)) :: r, arg1, arg2
  r = vec_and(arg1, arg2)
! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[r:.*]] = arith.andi %[[varg1]], %[[varg2]] : vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[r]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.and %[[arg1]], %[[arg2]] : vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = and <4 x i32> %[[arg1]], %[[arg2]]
end subroutine vec_and_test_u32

! CHECK-LABEL: vec_and_test_u64
subroutine vec_and_test_u64(arg1, arg2)
  vector(unsigned(8)) :: r, arg1, arg2
  r = vec_and(arg1, arg2)
! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[r:.*]] = arith.andi %[[varg1]], %[[varg2]] : vector<2xi64>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[r]] : (vector<2xi64>) -> !fir.vector<2:ui64>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.and %[[arg1]], %[[arg2]] : vector<2xi64>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = and <2 x i64> %[[arg1]], %[[arg2]]
end subroutine vec_and_test_u64

! CHECK-LABEL: vec_and_testf32
subroutine vec_and_testf32(arg1, arg2)
  vector(real(4)) :: r, arg1, arg2
  r = vec_and(arg1, arg2)
! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<4xf32> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<4xf32> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = arith.andi %[[bc1]], %[[bc2]] : vector<4xi32>
! CHECK-FIR: %[[vr:.*]] = vector.bitcast %[[r]] : vector<4xi32> to vector<4xf32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[vr]] : (vector<4xf32>) -> !fir.vector<4:f32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[bc1:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<4xi32>
! CHECK-LLVMIR: %[[bc2:.*]] = llvm.bitcast %[[arg2]] : vector<4xf32> to vector<4xi32>
! CHECK-LLVMIR: %[[r:.*]] = llvm.and %[[bc1]], %[[bc2]]  : vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[r]] : vector<4xi32> to vector<4xf32>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[bc1:.*]] = bitcast <4 x float> %[[arg1]] to <4 x i32>
! CHECK: %[[bc2:.*]] = bitcast <4 x float> %[[arg2]] to <4 x i32>
! CHECK: %[[r:.*]] = and <4 x i32> %[[bc1]], %[[bc2]]
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[r]] to <4 x float>
end subroutine vec_and_testf32

! CHECK-LABEL: vec_and_testf64
subroutine vec_and_testf64(arg1, arg2)
  vector(real(8)) :: r, arg1, arg2
  r = vec_and(arg1, arg2)
! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<2xf64> to vector<2xi64>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<2xf64> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = arith.andi %[[bc1]], %[[bc2]] : vector<2xi64>
! CHECK-FIR: %[[vr:.*]] = vector.bitcast %[[r]] : vector<2xi64> to vector<2xf64>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[vr]] : (vector<2xf64>) -> !fir.vector<2:f64>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[bc1:.*]] = llvm.bitcast %[[arg1]] : vector<2xf64> to vector<2xi64>
! CHECK-LLVMIR: %[[bc2:.*]] = llvm.bitcast %[[arg2]] : vector<2xf64> to vector<2xi64>
! CHECK-LLVMIR: %[[r:.*]] = llvm.and %[[bc1]], %[[bc2]]  : vector<2xi64>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[r]] : vector<2xi64> to vector<2xf64>

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[bc1:.*]] = bitcast <2 x double> %[[arg1]] to <2 x i64>
! CHECK: %[[bc2:.*]] = bitcast <2 x double> %[[arg2]] to <2 x i64>
! CHECK: %[[r:.*]] = and <2 x i64> %[[bc1]], %[[bc2]]
! CHECK: %{{[0-9]+}} = bitcast <2 x i64> %[[r]] to <2 x double>
end subroutine vec_and_testf64

!----------------------
! vec_xor
!----------------------

! CHECK-LABEL: vec_xor_test_i8
subroutine vec_xor_test_i8(arg1, arg2)
  vector(integer(1)) :: r, arg1, arg2
  r = vec_xor(arg1, arg2)
! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:i8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = arith.xori %[[varg1]], %[[varg2]] : vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:i8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[arg1]], %[[arg2]] : vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = xor <16 x i8> %[[arg1]], %[[arg2]]
end subroutine vec_xor_test_i8

! CHECK-LABEL: vec_xor_test_i16
subroutine vec_xor_test_i16(arg1, arg2)
  vector(integer(2)) :: r, arg1, arg2
  r = vec_xor(arg1, arg2)
! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:i16>) -> vector<8xi16>
! CHECK-FIR: %[[r:.*]] = arith.xori %[[varg1]], %[[varg2]] : vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[r]] : (vector<8xi16>) -> !fir.vector<8:i16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[arg1]], %[[arg2]] : vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = xor <8 x i16> %[[arg1]], %[[arg2]]
end subroutine vec_xor_test_i16

! CHECK-LABEL: vec_xor_test_i32
subroutine vec_xor_test_i32(arg1, arg2)
  vector(integer(4)) :: r, arg1, arg2
  r = vec_xor(arg1, arg2)
! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:i32>) -> vector<4xi32>
! CHECK-FIR: %[[r:.*]] = arith.xori %[[varg1]], %[[varg2]] : vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[r]] : (vector<4xi32>) -> !fir.vector<4:i32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[arg1]], %[[arg2]] : vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = xor <4 x i32> %[[arg1]], %[[arg2]]
end subroutine vec_xor_test_i32

! CHECK-LABEL: vec_xor_test_i64
subroutine vec_xor_test_i64(arg1, arg2)
  vector(integer(8)) :: r, arg1, arg2
  r = vec_xor(arg1, arg2)
! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:i64>) -> vector<2xi64>
! CHECK-FIR: %[[r:.*]] = arith.xori %[[varg1]], %[[varg2]] : vector<2xi64>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[r]] : (vector<2xi64>) -> !fir.vector<2:i64>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[arg1]], %[[arg2]] : vector<2xi64>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = xor <2 x i64> %[[arg1]], %[[arg2]]
end subroutine vec_xor_test_i64

! CHECK-LABEL: vec_xor_test_u8
subroutine vec_xor_test_u8(arg1, arg2)
  vector(unsigned(1)) :: r, arg1, arg2
  r = vec_xor(arg1, arg2)
! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<16:ui8>) -> vector<16xi8>
! CHECK-FIR: %[[r:.*]] = arith.xori %[[varg1]], %[[varg2]] : vector<16xi8>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[r]] : (vector<16xi8>) -> !fir.vector<16:ui8>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[arg1]], %[[arg2]] : vector<16xi8>

! CHECK: %[[arg1:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <16 x i8>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = xor <16 x i8> %[[arg1]], %[[arg2]]
end subroutine vec_xor_test_u8

! CHECK-LABEL: vec_xor_test_u16
subroutine vec_xor_test_u16(arg1, arg2)
  vector(unsigned(2)) :: r, arg1, arg2
  r = vec_xor(arg1, arg2)
! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<8:ui16>) -> vector<8xi16>
! CHECK-FIR: %[[r:.*]] = arith.xori %[[varg1]], %[[varg2]] : vector<8xi16>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[r]] : (vector<8xi16>) -> !fir.vector<8:ui16>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[arg1]], %[[arg2]] : vector<8xi16>

! CHECK: %[[arg1:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <8 x i16>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = xor <8 x i16> %[[arg1]], %[[arg2]]
end subroutine vec_xor_test_u16

! CHECK-LABEL: vec_xor_test_u32
subroutine vec_xor_test_u32(arg1, arg2)
  vector(unsigned(4)) :: r, arg1, arg2
  r = vec_xor(arg1, arg2)
! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:ui32>) -> vector<4xi32>
! CHECK-FIR: %[[r:.*]] = arith.xori %[[varg1]], %[[varg2]] : vector<4xi32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[r]] : (vector<4xi32>) -> !fir.vector<4:ui32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[arg1]], %[[arg2]] : vector<4xi32>

! CHECK: %[[arg1:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x i32>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = xor <4 x i32> %[[arg1]], %[[arg2]]
end subroutine vec_xor_test_u32

! CHECK-LABEL: vec_xor_test_u64
subroutine vec_xor_test_u64(arg1, arg2)
  vector(unsigned(8)) :: r, arg1, arg2
  r = vec_xor(arg1, arg2)
! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:ui64>) -> vector<2xi64>
! CHECK-FIR: %[[r:.*]] = arith.xori %[[varg1]], %[[varg2]] : vector<2xi64>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[r]] : (vector<2xi64>) -> !fir.vector<2:ui64>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.xor %[[arg1]], %[[arg2]] : vector<2xi64>

! CHECK: %[[arg1:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x i64>, ptr %{{.*}}, align 16
! CHECK: %{{[0-9]+}} = xor <2 x i64> %[[arg1]], %[[arg2]]
end subroutine vec_xor_test_u64

! CHECK-LABEL: vec_xor_testf32
subroutine vec_xor_testf32(arg1, arg2)
  vector(real(4)) :: r, arg1, arg2
  r = vec_xor(arg1, arg2)
! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<4xf32> to vector<4xi32>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<4xf32> to vector<4xi32>
! CHECK-FIR: %[[r:.*]] = arith.xori %[[bc1]], %[[bc2]] : vector<4xi32>
! CHECK-FIR: %[[vr:.*]] = vector.bitcast %[[r]] : vector<4xi32> to vector<4xf32>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[vr]] : (vector<4xf32>) -> !fir.vector<4:f32>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[bc1:.*]] = llvm.bitcast %[[arg1]] : vector<4xf32> to vector<4xi32>
! CHECK-LLVMIR: %[[bc2:.*]] = llvm.bitcast %[[arg2]] : vector<4xf32> to vector<4xi32>
! CHECK-LLVMIR: %[[r:.*]] = llvm.xor %[[bc1]], %[[bc2]]  : vector<4xi32>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[r]] : vector<4xi32> to vector<4xf32>

! CHECK: %[[arg1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
! CHECK: %[[bc1:.*]] = bitcast <4 x float> %[[arg1]] to <4 x i32>
! CHECK: %[[bc2:.*]] = bitcast <4 x float> %[[arg2]] to <4 x i32>
! CHECK: %[[r:.*]] = xor <4 x i32> %[[bc1]], %[[bc2]]
! CHECK: %{{[0-9]+}} = bitcast <4 x i32> %[[r]] to <4 x float>
end subroutine vec_xor_testf32

! CHECK-LABEL: vec_xor_testf64
subroutine vec_xor_testf64(arg1, arg2)
  vector(real(8)) :: r, arg1, arg2
  r = vec_xor(arg1, arg2)
! CHECK-FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[arg2:.*]] = fir.load %{{.*}} : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[varg1:.*]] = fir.convert %[[arg1]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[varg2:.*]] = fir.convert %[[arg2]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[bc1:.*]] = vector.bitcast %[[varg1]] : vector<2xf64> to vector<2xi64>
! CHECK-FIR: %[[bc2:.*]] = vector.bitcast %[[varg2]] : vector<2xf64> to vector<2xi64>
! CHECK-FIR: %[[r:.*]] = arith.xori %[[bc1]], %[[bc2]] : vector<2xi64>
! CHECK-FIR: %[[vr:.*]] = vector.bitcast %[[r]] : vector<2xi64> to vector<2xf64>
! CHECK-FIR: %{{[0-9]+}} = fir.convert %[[vr]] : (vector<2xf64>) -> !fir.vector<2:f64>

! CHECK-LLVMIR: %[[arg1:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[arg2:.*]] = llvm.load %{{.*}} : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[bc1:.*]] = llvm.bitcast %[[arg1]] : vector<2xf64> to vector<2xi64>
! CHECK-LLVMIR: %[[bc2:.*]] = llvm.bitcast %[[arg2]] : vector<2xf64> to vector<2xi64>
! CHECK-LLVMIR: %[[r:.*]] = llvm.xor %[[bc1]], %[[bc2]]  : vector<2xi64>
! CHECK-LLVMIR: %{{[0-9]+}} = llvm.bitcast %[[r]] : vector<2xi64> to vector<2xf64>

! CHECK: %[[arg1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[arg2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
! CHECK: %[[bc1:.*]] = bitcast <2 x double> %[[arg1]] to <2 x i64>
! CHECK: %[[bc2:.*]] = bitcast <2 x double> %[[arg2]] to <2 x i64>
! CHECK: %[[r:.*]] = xor <2 x i64> %[[bc1]], %[[bc2]]
! CHECK: %{{[0-9]+}} = bitcast <2 x i64> %[[r]] to <2 x double>
end subroutine vec_xor_testf64

