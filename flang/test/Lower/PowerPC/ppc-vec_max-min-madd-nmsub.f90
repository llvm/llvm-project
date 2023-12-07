! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes="CHECK-FIR" %s
! RUN: %flang_fc1 -emit-fir %s -o - | fir-opt --fir-to-llvm-ir | FileCheck --check-prefixes="CHECK-LLVMIR" %s
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck --check-prefixes="CHECK" %s
! REQUIRES: target=powerpc{{.*}}

! vec_max

! CHECK-LABEL: vec_max_testf32
subroutine vec_max_testf32(x, y)
  vector(real(4)) :: vmax, x, y
  vmax = vec_max(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[vmax:.*]] = fir.call @llvm.ppc.vsx.xvmaxsp(%[[x]], %[[y]]) fastmath<contract> : (!fir.vector<4:f32>, !fir.vector<4:f32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[vmax]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[vmax:.*]] = llvm.call @llvm.ppc.vsx.xvmaxsp(%[[x]], %[[y]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[vmax]], %{{[0-9]}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmax:.*]] = call contract <4 x float> @llvm.ppc.vsx.xvmaxsp(<4 x float> %[[x]], <4 x float> %[[y]])
! CHECK: store <4 x float> %[[vmax]], ptr %{{[0-9]}}, align 16
end subroutine vec_max_testf32

! CHECK-LABEL: vec_max_testf64
subroutine vec_max_testf64(x, y)
  vector(real(8)) :: vmax, x, y
  vmax = vec_max(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[vmax:.*]] = fir.call @llvm.ppc.vsx.xvmaxdp(%[[x]], %[[y]]) fastmath<contract> : (!fir.vector<2:f64>, !fir.vector<2:f64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[vmax]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[vmax:.*]] = llvm.call @llvm.ppc.vsx.xvmaxdp(%[[x]], %[[y]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<2xf64>, vector<2xf64>) -> vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[vmax]], %{{[0-9]}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmax:.*]] = call contract <2 x double> @llvm.ppc.vsx.xvmaxdp(<2 x double> %[[x]], <2 x double> %[[y]])
! CHECK: store <2 x double> %[[vmax]], ptr %{{[0-9]}}, align 16
end subroutine vec_max_testf64

! CHECK-LABEL: vec_max_testi8
subroutine vec_max_testi8(x, y)
  vector(integer(1)) :: vmax, x, y
  vmax = vec_max(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[vmax:.*]] = fir.call @llvm.ppc.altivec.vmaxsb(%[[x]], %[[y]]) fastmath<contract> : (!fir.vector<16:i8>, !fir.vector<16:i8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[vmax]] to %{{[0-9]}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[vmax:.*]] = llvm.call @llvm.ppc.altivec.vmaxsb(%[[x]], %[[y]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<16xi8>, vector<16xi8>) -> vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[vmax]], %{{[0-9]}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmax:.*]] = call <16 x i8> @llvm.ppc.altivec.vmaxsb(<16 x i8> %[[x]], <16 x i8> %[[y]])
! CHECK: store <16 x i8> %[[vmax]], ptr %{{[0-9]}}, align 16
end subroutine vec_max_testi8

! CHECK-LABEL: vec_max_testi16
subroutine vec_max_testi16(x, y)
  vector(integer(2)) :: vmax, x, y
  vmax = vec_max(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[vmax:.*]] = fir.call @llvm.ppc.altivec.vmaxsh(%[[x]], %[[y]]) fastmath<contract> : (!fir.vector<8:i16>, !fir.vector<8:i16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[vmax]] to %{{[0-9]}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[vmax:.*]] = llvm.call @llvm.ppc.altivec.vmaxsh(%[[x]], %[[y]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<8xi16>, vector<8xi16>) -> vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[vmax]], %{{[0-9]}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmax:.*]] = call <8 x i16> @llvm.ppc.altivec.vmaxsh(<8 x i16> %[[x]], <8 x i16> %[[y]])
! CHECK: store <8 x i16> %[[vmax]], ptr %{{[0-9]}}, align 16
end subroutine vec_max_testi16

! CHECK-LABEL: vec_max_testi32
subroutine vec_max_testi32(x, y)
  vector(integer(4)) :: vmax, x, y
  vmax = vec_max(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[vmax:.*]] = fir.call @llvm.ppc.altivec.vmaxsw(%[[x]], %[[y]]) fastmath<contract> : (!fir.vector<4:i32>, !fir.vector<4:i32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[vmax]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[vmax:.*]] = llvm.call @llvm.ppc.altivec.vmaxsw(%[[x]], %[[y]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[vmax]], %{{[0-9]}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmax:.*]] = call <4 x i32> @llvm.ppc.altivec.vmaxsw(<4 x i32> %[[x]], <4 x i32> %[[y]])
! CHECK: store <4 x i32> %[[vmax]], ptr %{{[0-9]}}, align 16
end subroutine vec_max_testi32

! CHECK-LABEL: vec_max_testi64
subroutine vec_max_testi64(x, y)
  vector(integer(8)) :: vmax, x, y
  vmax = vec_max(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[vmax:.*]] = fir.call @llvm.ppc.altivec.vmaxsd(%[[x]], %[[y]]) fastmath<contract> : (!fir.vector<2:i64>, !fir.vector<2:i64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[vmax]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[vmax:.*]] = llvm.call @llvm.ppc.altivec.vmaxsd(%[[x]], %[[y]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<2xi64>, vector<2xi64>) -> vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[vmax]], %{{[0-9]}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmax:.*]] = call <2 x i64> @llvm.ppc.altivec.vmaxsd(<2 x i64> %[[x]], <2 x i64> %[[y]])
! CHECK: store <2 x i64> %[[vmax]], ptr %{{[0-9]}}, align 16
end subroutine vec_max_testi64

! CHECK-LABEL: vec_max_testui8
subroutine vec_max_testui8(x, y)
  vector(unsigned(1)) :: vmax, x, y
  vmax = vec_max(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[vmax:.*]] = fir.call @llvm.ppc.altivec.vmaxub(%[[x]], %[[y]]) fastmath<contract> : (!fir.vector<16:ui8>, !fir.vector<16:ui8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[vmax]] to %{{[0-9]}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[vmax:.*]] = llvm.call @llvm.ppc.altivec.vmaxub(%[[x]], %[[y]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<16xi8>, vector<16xi8>) -> vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[vmax]], %{{[0-9]}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmax:.*]] = call <16 x i8> @llvm.ppc.altivec.vmaxub(<16 x i8> %[[x]], <16 x i8> %[[y]])
! CHECK: store <16 x i8> %[[vmax]], ptr %{{[0-9]}}, align 16
end subroutine vec_max_testui8

! CHECK-LABEL: vec_max_testui16
subroutine vec_max_testui16(x, y)
  vector(unsigned(2)) :: vmax, x, y
  vmax = vec_max(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[vmax:.*]] = fir.call @llvm.ppc.altivec.vmaxuh(%[[x]], %[[y]]) fastmath<contract> : (!fir.vector<8:ui16>, !fir.vector<8:ui16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[vmax]] to %{{[0-9]}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[vmax:.*]] = llvm.call @llvm.ppc.altivec.vmaxuh(%[[x]], %[[y]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<8xi16>, vector<8xi16>) -> vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[vmax]], %{{[0-9]}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmax:.*]] = call <8 x i16> @llvm.ppc.altivec.vmaxuh(<8 x i16> %[[x]], <8 x i16> %[[y]])
! CHECK: store <8 x i16> %[[vmax]], ptr %{{[0-9]}}, align 16
end subroutine vec_max_testui16

! CHECK-LABEL: vec_max_testui32
subroutine vec_max_testui32(x, y)
  vector(unsigned(4)) :: vmax, x, y
  vmax = vec_max(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[vmax:.*]] = fir.call @llvm.ppc.altivec.vmaxuw(%[[x]], %[[y]]) fastmath<contract> : (!fir.vector<4:ui32>, !fir.vector<4:ui32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[vmax]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[vmax:.*]] = llvm.call @llvm.ppc.altivec.vmaxuw(%[[x]], %[[y]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[vmax]], %{{[0-9]}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmax:.*]] = call <4 x i32> @llvm.ppc.altivec.vmaxuw(<4 x i32> %[[x]], <4 x i32> %[[y]])
! CHECK: store <4 x i32> %[[vmax]], ptr %{{[0-9]}}, align 16
end subroutine vec_max_testui32

! CHECK-LABEL: vec_max_testui64
subroutine vec_max_testui64(x, y)
  vector(unsigned(8)) :: vmax, x, y
  vmax = vec_max(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[vmax:.*]] = fir.call @llvm.ppc.altivec.vmaxud(%[[x]], %[[y]]) fastmath<contract> : (!fir.vector<2:ui64>, !fir.vector<2:ui64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[vmax]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[vmax:.*]] = llvm.call @llvm.ppc.altivec.vmaxud(%[[x]], %[[y]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<2xi64>, vector<2xi64>) -> vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[vmax]], %{{[0-9]}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmax:.*]] = call <2 x i64> @llvm.ppc.altivec.vmaxud(<2 x i64> %[[x]], <2 x i64> %[[y]])
! CHECK: store <2 x i64> %[[vmax]], ptr %{{[0-9]}}, align 16
end subroutine vec_max_testui64

! vec_min

! CHECK-LABEL: vec_min_testf32
subroutine vec_min_testf32(x, y)
  vector(real(4)) :: vmin, x, y
  vmin = vec_min(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[vmin:.*]] = fir.call @llvm.ppc.vsx.xvminsp(%[[x]], %[[y]]) fastmath<contract> : (!fir.vector<4:f32>, !fir.vector<4:f32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[vmin]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[vmin:.*]] = llvm.call @llvm.ppc.vsx.xvminsp(%[[x]], %[[y]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[vmin]], %{{[0-9]}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmin:.*]] = call contract <4 x float> @llvm.ppc.vsx.xvminsp(<4 x float> %[[x]], <4 x float> %[[y]])
! CHECK: store <4 x float> %[[vmin]], ptr %{{[0-9]}}, align 16
end subroutine vec_min_testf32

! CHECK-LABEL: vec_min_testf64
subroutine vec_min_testf64(x, y)
  vector(real(8)) :: vmin, x, y
  vmin = vec_min(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[vmin:.*]] = fir.call @llvm.ppc.vsx.xvmindp(%[[x]], %[[y]]) fastmath<contract> : (!fir.vector<2:f64>, !fir.vector<2:f64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[vmin]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[vmin:.*]] = llvm.call @llvm.ppc.vsx.xvmindp(%[[x]], %[[y]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<2xf64>, vector<2xf64>) -> vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[vmin]], %{{[0-9]}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmin:.*]] = call contract <2 x double> @llvm.ppc.vsx.xvmindp(<2 x double> %[[x]], <2 x double> %[[y]])
! CHECK: store <2 x double> %[[vmin]], ptr %{{[0-9]}}, align 16
end subroutine vec_min_testf64

! CHECK-LABEL: vec_min_testi8
subroutine vec_min_testi8(x, y)
  vector(integer(1)) :: vmin, x, y
  vmin = vec_min(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<16:i8>>
! CHECK-FIR: %[[vmin:.*]] = fir.call @llvm.ppc.altivec.vminsb(%[[x]], %[[y]]) fastmath<contract> : (!fir.vector<16:i8>, !fir.vector<16:i8>) -> !fir.vector<16:i8>
! CHECK-FIR: fir.store %[[vmin]] to %{{[0-9]}} : !fir.ref<!fir.vector<16:i8>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[vmin:.*]] = llvm.call @llvm.ppc.altivec.vminsb(%[[x]], %[[y]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<16xi8>, vector<16xi8>) -> vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[vmin]], %{{[0-9]}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmin:.*]] = call <16 x i8> @llvm.ppc.altivec.vminsb(<16 x i8> %[[x]], <16 x i8> %[[y]])
! CHECK: store <16 x i8> %[[vmin]], ptr %{{[0-9]}}, align 16
end subroutine vec_min_testi8

! CHECK-LABEL: vec_min_testi16
subroutine vec_min_testi16(x, y)
  vector(integer(2)) :: vmin, x, y
  vmin = vec_min(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<8:i16>>
! CHECK-FIR: %[[vmin:.*]] = fir.call @llvm.ppc.altivec.vminsh(%[[x]], %[[y]]) fastmath<contract> : (!fir.vector<8:i16>, !fir.vector<8:i16>) -> !fir.vector<8:i16>
! CHECK-FIR: fir.store %[[vmin]] to %{{[0-9]}} : !fir.ref<!fir.vector<8:i16>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[vmin:.*]] = llvm.call @llvm.ppc.altivec.vminsh(%[[x]], %[[y]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<8xi16>, vector<8xi16>) -> vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[vmin]], %{{[0-9]}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmin:.*]] = call <8 x i16> @llvm.ppc.altivec.vminsh(<8 x i16> %[[x]], <8 x i16> %[[y]])
! CHECK: store <8 x i16> %[[vmin]], ptr %{{[0-9]}}, align 16
end subroutine vec_min_testi16

! CHECK-LABEL: vec_min_testi32
subroutine vec_min_testi32(x, y)
  vector(integer(4)) :: vmin, x, y
  vmin = vec_min(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<4:i32>>
! CHECK-FIR: %[[vmin:.*]] = fir.call @llvm.ppc.altivec.vminsw(%[[x]], %[[y]]) fastmath<contract> : (!fir.vector<4:i32>, !fir.vector<4:i32>) -> !fir.vector<4:i32>
! CHECK-FIR: fir.store %[[vmin]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:i32>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[vmin:.*]] = llvm.call @llvm.ppc.altivec.vminsw(%[[x]], %[[y]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[vmin]], %{{[0-9]}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmin:.*]] = call <4 x i32> @llvm.ppc.altivec.vminsw(<4 x i32> %[[x]], <4 x i32> %[[y]])
! CHECK: store <4 x i32> %[[vmin]], ptr %{{[0-9]}}, align 16
end subroutine vec_min_testi32

! CHECK-LABEL: vec_min_testi64
subroutine vec_min_testi64(x, y)
  vector(integer(8)) :: vmin, x, y
  vmin = vec_min(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<2:i64>>
! CHECK-FIR: %[[vmin:.*]] = fir.call @llvm.ppc.altivec.vminsd(%[[x]], %[[y]]) fastmath<contract> : (!fir.vector<2:i64>, !fir.vector<2:i64>) -> !fir.vector<2:i64>
! CHECK-FIR: fir.store %[[vmin]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:i64>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[vmin:.*]] = llvm.call @llvm.ppc.altivec.vminsd(%[[x]], %[[y]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<2xi64>, vector<2xi64>) -> vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[vmin]], %{{[0-9]}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmin:.*]] = call <2 x i64> @llvm.ppc.altivec.vminsd(<2 x i64> %[[x]], <2 x i64> %[[y]])
! CHECK: store <2 x i64> %[[vmin]], ptr %{{[0-9]}}, align 16
end subroutine vec_min_testi64

! CHECK-LABEL: vec_min_testui8
subroutine vec_min_testui8(x, y)
  vector(unsigned(1)) :: vmin, x, y
  vmin = vec_min(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<16:ui8>>
! CHECK-FIR: %[[vmin:.*]] = fir.call @llvm.ppc.altivec.vminub(%[[x]], %[[y]]) fastmath<contract> : (!fir.vector<16:ui8>, !fir.vector<16:ui8>) -> !fir.vector<16:ui8>
! CHECK-FIR: fir.store %[[vmin]] to %{{[0-9]}} : !fir.ref<!fir.vector<16:ui8>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<16xi8>>
! CHECK-LLVMIR: %[[vmin:.*]] = llvm.call @llvm.ppc.altivec.vminub(%[[x]], %[[y]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<16xi8>, vector<16xi8>) -> vector<16xi8>
! CHECK-LLVMIR: llvm.store %[[vmin]], %{{[0-9]}} : !llvm.ptr<vector<16xi8>>

! CHECK: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmin:.*]] = call <16 x i8> @llvm.ppc.altivec.vminub(<16 x i8> %[[x]], <16 x i8> %[[y]])
! CHECK: store <16 x i8> %[[vmin]], ptr %{{[0-9]}}, align 16
end subroutine vec_min_testui8

! CHECK-LABEL: vec_min_testui16
subroutine vec_min_testui16(x, y)
  vector(unsigned(2)) :: vmin, x, y
  vmin = vec_min(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<8:ui16>>
! CHECK-FIR: %[[vmin:.*]] = fir.call @llvm.ppc.altivec.vminuh(%[[x]], %[[y]]) fastmath<contract> : (!fir.vector<8:ui16>, !fir.vector<8:ui16>) -> !fir.vector<8:ui16>
! CHECK-FIR: fir.store %[[vmin]] to %{{[0-9]}} : !fir.ref<!fir.vector<8:ui16>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<8xi16>>
! CHECK-LLVMIR: %[[vmin:.*]] = llvm.call @llvm.ppc.altivec.vminuh(%[[x]], %[[y]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<8xi16>, vector<8xi16>) -> vector<8xi16>
! CHECK-LLVMIR: llvm.store %[[vmin]], %{{[0-9]}} : !llvm.ptr<vector<8xi16>>

! CHECK: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmin:.*]] = call <8 x i16> @llvm.ppc.altivec.vminuh(<8 x i16> %[[x]], <8 x i16> %[[y]])
! CHECK: store <8 x i16> %[[vmin]], ptr %{{[0-9]}}, align 16
end subroutine vec_min_testui16

! CHECK-LABEL: vec_min_testui32
subroutine vec_min_testui32(x, y)
  vector(unsigned(4)) :: vmin, x, y
  vmin = vec_min(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<4:ui32>>
! CHECK-FIR: %[[vmin:.*]] = fir.call @llvm.ppc.altivec.vminuw(%[[x]], %[[y]]) fastmath<contract> : (!fir.vector<4:ui32>, !fir.vector<4:ui32>) -> !fir.vector<4:ui32>
! CHECK-FIR: fir.store %[[vmin]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:ui32>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<4xi32>>
! CHECK-LLVMIR: %[[vmin:.*]] = llvm.call @llvm.ppc.altivec.vminuw(%[[x]], %[[y]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xi32>, vector<4xi32>) -> vector<4xi32>
! CHECK-LLVMIR: llvm.store %[[vmin]], %{{[0-9]}} : !llvm.ptr<vector<4xi32>>

! CHECK: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmin:.*]] = call <4 x i32> @llvm.ppc.altivec.vminuw(<4 x i32> %[[x]], <4 x i32> %[[y]])
! CHECK: store <4 x i32> %[[vmin]], ptr %{{[0-9]}}, align 16
end subroutine vec_min_testui32

! CHECK-LABEL: vec_min_testui64
subroutine vec_min_testui64(x, y)
  vector(unsigned(8)) :: vmin, x, y
  vmin = vec_min(x, y)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<2:ui64>>
! CHECK-FIR: %[[vmin:.*]] = fir.call @llvm.ppc.altivec.vminud(%[[x]], %[[y]]) fastmath<contract> : (!fir.vector<2:ui64>, !fir.vector<2:ui64>) -> !fir.vector<2:ui64>
! CHECK-FIR: fir.store %[[vmin]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:ui64>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xi64>>
! CHECK-LLVMIR: %[[vmin:.*]] = llvm.call @llvm.ppc.altivec.vminud(%[[x]], %[[y]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<2xi64>, vector<2xi64>) -> vector<2xi64>
! CHECK-LLVMIR: llvm.store %[[vmin]], %{{[0-9]}} : !llvm.ptr<vector<2xi64>>

! CHECK: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmin:.*]] = call <2 x i64> @llvm.ppc.altivec.vminud(<2 x i64> %[[x]], <2 x i64> %[[y]])
! CHECK: store <2 x i64> %[[vmin]], ptr %{{[0-9]}}, align 16
end subroutine vec_min_testui64

! vec_madd

! CHECK-LABEL: vec_madd_testf32
subroutine vec_madd_testf32(x, y, z)
  vector(real(4)) :: vmsum, x, y, z
  vmsum = vec_madd(x, y, z)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[z:.*]] = fir.load %arg2 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[vmsum:.*]] = fir.call @llvm.fma.v4f32(%[[x]], %[[y]], %[[z]]) fastmath<contract> : (!fir.vector<4:f32>, !fir.vector<4:f32>, !fir.vector<4:f32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[vmsum]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[z:.*]] = llvm.load %arg2 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[vmsum:.*]] = llvm.call @llvm.fma.v4f32(%[[x]], %[[y]], %[[z]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[vmsum]], %{{[0-9]}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[z:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmsum:.*]] = call contract <4 x float> @llvm.fma.v4f32(<4 x float> %[[x]], <4 x float> %[[y]], <4 x float> %[[z]])
! CHECK: store <4 x float> %[[vmsum]], ptr %{{[0-9]}}, align 16
end subroutine vec_madd_testf32

! CHECK-LABEL: vec_madd_testf64
subroutine vec_madd_testf64(x, y, z)
  vector(real(8)) :: vmsum, x, y, z
  vmsum = vec_madd(x, y, z)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[z:.*]] = fir.load %arg2 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[vmsum:.*]] = fir.call @llvm.fma.v2f64(%[[x]], %[[y]], %[[z]]) fastmath<contract> : (!fir.vector<2:f64>, !fir.vector<2:f64>, !fir.vector<2:f64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[vmsum]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[z:.*]] = llvm.load %arg2 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[vmsum:.*]] = llvm.call @llvm.fma.v2f64(%[[x]], %[[y]], %[[z]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<2xf64>, vector<2xf64>, vector<2xf64>) -> vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[vmsum]], %{{[0-9]}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[z:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vmsum:.*]] = call contract <2 x double> @llvm.fma.v2f64(<2 x double> %[[x]], <2 x double> %[[y]], <2 x double> %[[z]])
! CHECK: store <2 x double> %[[vmsum]], ptr %{{[0-9]}}, align 16
end subroutine vec_madd_testf64

! vec_nmsub

! CHECK-LABEL: vec_nmsub_testf32
subroutine vec_nmsub_testf32(x, y, z)
  vector(real(4)) :: vnmsub, x, y, z
  vnmsub = vec_nmsub(x, y, z)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[z:.*]] = fir.load %arg2 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[nmsub:.*]] = fir.call @llvm.ppc.fnmsub.v4f32(%[[x]], %[[y]], %[[z]]) fastmath<contract> : (!fir.vector<4:f32>, !fir.vector<4:f32>, !fir.vector<4:f32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[nmsub]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[z:.*]] = llvm.load %arg2 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[vnmsub:.*]] = llvm.call @llvm.ppc.fnmsub.v4f32(%[[x]], %[[y]], %[[z]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[vnmsub]], %{{[0-9]}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[z:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vnmsub:.*]] = call contract <4 x float> @llvm.ppc.fnmsub.v4f32(<4 x float> %[[x]], <4 x float> %[[y]], <4 x float> %[[z]])
! CHECK: store <4 x float> %[[vnmsub]], ptr %{{[0-9]}}, align 16
end subroutine vec_nmsub_testf32

! CHECK-LABEL: vec_nmsub_testf64
subroutine vec_nmsub_testf64(x, y, z)
  vector(real(8)) :: vnmsub, x, y, z
  vnmsub = vec_nmsub(x, y, z)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[z:.*]] = fir.load %arg2 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[nmsub:.*]] = fir.call @llvm.ppc.fnmsub.v2f64(%[[x]], %[[y]], %[[z]]) fastmath<contract> : (!fir.vector<2:f64>, !fir.vector<2:f64>, !fir.vector<2:f64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[nmsub]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[z:.*]] = llvm.load %arg2 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[vnmsub:.*]] = llvm.call @llvm.ppc.fnmsub.v2f64(%[[x]], %[[y]], %[[z]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<2xf64>, vector<2xf64>, vector<2xf64>) -> vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[vnmsub]], %{{[0-9]}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[z:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[vnmsub:.*]] = call contract <2 x double> @llvm.ppc.fnmsub.v2f64(<2 x double> %[[x]], <2 x double> %[[y]], <2 x double> %[[z]])
! CHECK: store <2 x double> %[[vnmsub]], ptr %{{[0-9]}}, align 16
end subroutine vec_nmsub_testf64

! vec_msub

! CHECK-LABEL: vec_msub_testf32
subroutine vec_msub_testf32(x, y, z)
  vector(real(4)) :: vmsub, x, y, z
  vmsub = vec_msub(x, y, z)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[z:.*]] = fir.load %arg2 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[vz:.*]] = fir.convert %[[z]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[nz:.*]] = arith.negf %[[vz]] fastmath<contract> : vector<4xf32>
! CHECK-FIR: %[[vmsub:.*]] = fir.call @llvm.fma.v4f32(%[[vx]], %[[vy]], %[[nz]]) fastmath<contract> : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[vmsub]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[z:.*]] = llvm.load %arg2 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[nz:.*]] = llvm.fneg %[[z]]  {fastmathFlags = #llvm.fastmath<contract>} : vector<4xf32>
! CHECK-LLVMIR: %[[vmsub:.*]] = llvm.call @llvm.fma.v4f32(%[[x]], %[[y]], %[[nz]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[vmsub]], %{{[0-9]}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[z:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[nz:.*]] = fneg contract <4 x float> %[[z]]
! CHECK: %[[vmsub:.*]] = call contract <4 x float> @llvm.fma.v4f32(<4 x float> %[[x]], <4 x float> %[[y]], <4 x float> %[[nz]])
! CHECK: store <4 x float> %[[vmsub]], ptr %{{[0-9]}}, align 16
end subroutine vec_msub_testf32

! CHECK-LABEL: vec_msub_testf64
subroutine vec_msub_testf64(x, y, z)
  vector(real(8)) :: vmsub, x, y, z
  vmsub = vec_msub(x, y, z)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[z:.*]] = fir.load %arg2 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[vz:.*]] = fir.convert %[[z]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[nz:.*]] = arith.negf %[[vz]] fastmath<contract> : vector<2xf64>
! CHECK-FIR: %[[vmsub:.*]] = fir.call @llvm.fma.v2f64(%[[vx]], %[[vy]], %[[nz]]) fastmath<contract> : (vector<2xf64>, vector<2xf64>, vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[vmsub]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[z:.*]] = llvm.load %arg2 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[nz:.*]] = llvm.fneg %[[z]]  {fastmathFlags = #llvm.fastmath<contract>} : vector<2xf64>
! CHECK-LLVMIR: %[[vmsub:.*]] = llvm.call @llvm.fma.v2f64(%[[x]], %[[y]], %[[nz]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<2xf64>, vector<2xf64>, vector<2xf64>) -> vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[vmsub]], %{{[0-9]}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[z:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[nz:.*]] = fneg contract <2 x double> %[[z]]
! CHECK: %[[vmsub:.*]] = call contract <2 x double> @llvm.fma.v2f64(<2 x double> %[[x]], <2 x double> %[[y]], <2 x double> %[[nz]])
! CHECK: store <2 x double> %[[vmsub]], ptr %{{[0-9]}}, align 16
end subroutine vec_msub_testf64

! vec_nmadd

! CHECK-LABEL: vec_nmadd_testf32
subroutine vec_nmadd_testf32(x, y, z)
  vector(real(4)) :: vnmsum, x, y, z
  vnmsum = vec_nmadd(x, y, z)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[z:.*]] = fir.load %arg2 : !fir.ref<!fir.vector<4:f32>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[vz:.*]] = fir.convert %[[z]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[msum:.*]] = fir.call @llvm.fma.v4f32(%[[vx]], %[[vy]], %[[vz]]) fastmath<contract> : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: %[[vmsum:.*]] = fir.convert %[[msum]] : (!fir.vector<4:f32>) -> vector<4xf32>
! CHECK-FIR: %[[nmsum:.*]] = arith.negf %[[vmsum]] fastmath<contract> : vector<4xf32>
! CHECK-FIR: %[[vnmsum:.*]] = fir.convert %[[nmsum]] : (vector<4xf32>) -> !fir.vector<4:f32>
! CHECK-FIR: fir.store %[[vnmsum]] to %{{[0-9]}} : !fir.ref<!fir.vector<4:f32>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[z:.*]] = llvm.load %arg2 : !llvm.ptr<vector<4xf32>>
! CHECK-LLVMIR: %[[msum:.*]] = llvm.call @llvm.fma.v4f32(%[[x]], %[[y]], %[[z]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
! CHECK-LLVMIR: %[[vnmsum:.*]] = llvm.fneg %[[msum]]  {fastmathFlags = #llvm.fastmath<contract>} : vector<4xf32>
! CHECK-LLVMIR: llvm.store %[[vnmsum]], %{{[0-9]}} : !llvm.ptr<vector<4xf32>>

! CHECK: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[z:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! CHECK: %[[msum:.*]] = call contract <4 x float> @llvm.fma.v4f32(<4 x float> %[[x]], <4 x float> %[[y]], <4 x float> %[[z]])
! CHECK: %[[vnmsum:.*]] = fneg contract <4 x float> %[[msum]]
! CHECK: store <4 x float> %[[vnmsum]], ptr %{{[0-9]}}, align 16
end subroutine vec_nmadd_testf32

! CHECK-LABEL: vec_nmadd_testf64
subroutine vec_nmadd_testf64(x, y, z)
  vector(real(8)) :: vnmsum, x, y, z
  vnmsum = vec_nmadd(x, y, z)
! CHECK-FIR: %[[x:.*]] = fir.load %arg0 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[y:.*]] = fir.load %arg1 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[z:.*]] = fir.load %arg2 : !fir.ref<!fir.vector<2:f64>>
! CHECK-FIR: %[[vx:.*]] = fir.convert %[[x]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[vy:.*]] = fir.convert %[[y]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[vz:.*]] = fir.convert %[[z]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[msum:.*]] = fir.call @llvm.fma.v2f64(%[[vx]], %[[vy]], %[[vz]]) fastmath<contract> : (vector<2xf64>, vector<2xf64>, vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: %[[vmsum:.*]] = fir.convert %[[msum]] : (!fir.vector<2:f64>) -> vector<2xf64>
! CHECK-FIR: %[[nmsum:.*]] = arith.negf %[[vmsum]] fastmath<contract> : vector<2xf64>
! CHECK-FIR: %[[vnmsum:.*]] = fir.convert %[[nmsum]] : (vector<2xf64>) -> !fir.vector<2:f64>
! CHECK-FIR: fir.store %[[vnmsum]] to %{{[0-9]}} : !fir.ref<!fir.vector<2:f64>>

! CHECK-LLVMIR: %[[x:.*]] = llvm.load %arg0 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[y:.*]] = llvm.load %arg1 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[z:.*]] = llvm.load %arg2 : !llvm.ptr<vector<2xf64>>
! CHECK-LLVMIR: %[[msum:.*]] = llvm.call @llvm.fma.v2f64(%[[x]], %[[y]], %[[z]]) {fastmathFlags = #llvm.fastmath<contract>} : (vector<2xf64>, vector<2xf64>, vector<2xf64>) -> vector<2xf64>
! CHECK-LLVMIR: %[[vnmsum:.*]] = llvm.fneg %[[msum]]  {fastmathFlags = #llvm.fastmath<contract>} : vector<2xf64>
! CHECK-LLVMIR: llvm.store %[[vnmsum]], %{{[0-9]}} : !llvm.ptr<vector<2xf64>>

! CHECK: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[y:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[z:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! CHECK: %[[msum:.*]] = call contract <2 x double> @llvm.fma.v2f64(<2 x double> %[[x]], <2 x double> %[[y]], <2 x double> %[[z]])
! CHECK: %[[vnmsum:.*]] = fneg contract <2 x double> %[[msum]]
! CHECK: store <2 x double> %[[vnmsum]], ptr %{{[0-9]}}, align 16
end subroutine vec_nmadd_testf64
