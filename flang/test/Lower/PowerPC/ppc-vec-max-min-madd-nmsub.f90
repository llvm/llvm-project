! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

! vec_max

! CHECK-LABEL: vec_max_testf32
subroutine vec_max_testf32(x, y)
  vector(real(4)) :: vmax, x, y
  vmax = vec_max(x, y)

! LLVMIR: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmax:.*]] = call contract <4 x float> @llvm.ppc.vsx.xvmaxsp(<4 x float> %[[x]], <4 x float> %[[y]])
! LLVMIR: store <4 x float> %[[vmax]], ptr %{{[0-9]}}, align 16
end subroutine vec_max_testf32

! CHECK-LABEL: vec_max_testf64
subroutine vec_max_testf64(x, y)
  vector(real(8)) :: vmax, x, y
  vmax = vec_max(x, y)

! LLVMIR: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmax:.*]] = call contract <2 x double> @llvm.ppc.vsx.xvmaxdp(<2 x double> %[[x]], <2 x double> %[[y]])
! LLVMIR: store <2 x double> %[[vmax]], ptr %{{[0-9]}}, align 16
end subroutine vec_max_testf64

! CHECK-LABEL: vec_max_testi8
subroutine vec_max_testi8(x, y)
  vector(integer(1)) :: vmax, x, y
  vmax = vec_max(x, y)

! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmax:.*]] = call <16 x i8> @llvm.ppc.altivec.vmaxsb(<16 x i8> %[[x]], <16 x i8> %[[y]])
! LLVMIR: store <16 x i8> %[[vmax]], ptr %{{[0-9]}}, align 16
end subroutine vec_max_testi8

! CHECK-LABEL: vec_max_testi16
subroutine vec_max_testi16(x, y)
  vector(integer(2)) :: vmax, x, y
  vmax = vec_max(x, y)

! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmax:.*]] = call <8 x i16> @llvm.ppc.altivec.vmaxsh(<8 x i16> %[[x]], <8 x i16> %[[y]])
! LLVMIR: store <8 x i16> %[[vmax]], ptr %{{[0-9]}}, align 16
end subroutine vec_max_testi16

! CHECK-LABEL: vec_max_testi32
subroutine vec_max_testi32(x, y)
  vector(integer(4)) :: vmax, x, y
  vmax = vec_max(x, y)

! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmax:.*]] = call <4 x i32> @llvm.ppc.altivec.vmaxsw(<4 x i32> %[[x]], <4 x i32> %[[y]])
! LLVMIR: store <4 x i32> %[[vmax]], ptr %{{[0-9]}}, align 16
end subroutine vec_max_testi32

! CHECK-LABEL: vec_max_testi64
subroutine vec_max_testi64(x, y)
  vector(integer(8)) :: vmax, x, y
  vmax = vec_max(x, y)

! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmax:.*]] = call <2 x i64> @llvm.ppc.altivec.vmaxsd(<2 x i64> %[[x]], <2 x i64> %[[y]])
! LLVMIR: store <2 x i64> %[[vmax]], ptr %{{[0-9]}}, align 16
end subroutine vec_max_testi64

! CHECK-LABEL: vec_max_testui8
subroutine vec_max_testui8(x, y)
  vector(unsigned(1)) :: vmax, x, y
  vmax = vec_max(x, y)

! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmax:.*]] = call <16 x i8> @llvm.ppc.altivec.vmaxub(<16 x i8> %[[x]], <16 x i8> %[[y]])
! LLVMIR: store <16 x i8> %[[vmax]], ptr %{{[0-9]}}, align 16
end subroutine vec_max_testui8

! CHECK-LABEL: vec_max_testui16
subroutine vec_max_testui16(x, y)
  vector(unsigned(2)) :: vmax, x, y
  vmax = vec_max(x, y)

! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmax:.*]] = call <8 x i16> @llvm.ppc.altivec.vmaxuh(<8 x i16> %[[x]], <8 x i16> %[[y]])
! LLVMIR: store <8 x i16> %[[vmax]], ptr %{{[0-9]}}, align 16
end subroutine vec_max_testui16

! CHECK-LABEL: vec_max_testui32
subroutine vec_max_testui32(x, y)
  vector(unsigned(4)) :: vmax, x, y
  vmax = vec_max(x, y)

! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmax:.*]] = call <4 x i32> @llvm.ppc.altivec.vmaxuw(<4 x i32> %[[x]], <4 x i32> %[[y]])
! LLVMIR: store <4 x i32> %[[vmax]], ptr %{{[0-9]}}, align 16
end subroutine vec_max_testui32

! CHECK-LABEL: vec_max_testui64
subroutine vec_max_testui64(x, y)
  vector(unsigned(8)) :: vmax, x, y
  vmax = vec_max(x, y)

! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmax:.*]] = call <2 x i64> @llvm.ppc.altivec.vmaxud(<2 x i64> %[[x]], <2 x i64> %[[y]])
! LLVMIR: store <2 x i64> %[[vmax]], ptr %{{[0-9]}}, align 16
end subroutine vec_max_testui64

! vec_min

! CHECK-LABEL: vec_min_testf32
subroutine vec_min_testf32(x, y)
  vector(real(4)) :: vmin, x, y
  vmin = vec_min(x, y)

! LLVMIR: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmin:.*]] = call contract <4 x float> @llvm.ppc.vsx.xvminsp(<4 x float> %[[x]], <4 x float> %[[y]])
! LLVMIR: store <4 x float> %[[vmin]], ptr %{{[0-9]}}, align 16
end subroutine vec_min_testf32

! CHECK-LABEL: vec_min_testf64
subroutine vec_min_testf64(x, y)
  vector(real(8)) :: vmin, x, y
  vmin = vec_min(x, y)

! LLVMIR: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmin:.*]] = call contract <2 x double> @llvm.ppc.vsx.xvmindp(<2 x double> %[[x]], <2 x double> %[[y]])
! LLVMIR: store <2 x double> %[[vmin]], ptr %{{[0-9]}}, align 16
end subroutine vec_min_testf64

! CHECK-LABEL: vec_min_testi8
subroutine vec_min_testi8(x, y)
  vector(integer(1)) :: vmin, x, y
  vmin = vec_min(x, y)

! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmin:.*]] = call <16 x i8> @llvm.ppc.altivec.vminsb(<16 x i8> %[[x]], <16 x i8> %[[y]])
! LLVMIR: store <16 x i8> %[[vmin]], ptr %{{[0-9]}}, align 16
end subroutine vec_min_testi8

! CHECK-LABEL: vec_min_testi16
subroutine vec_min_testi16(x, y)
  vector(integer(2)) :: vmin, x, y
  vmin = vec_min(x, y)

! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmin:.*]] = call <8 x i16> @llvm.ppc.altivec.vminsh(<8 x i16> %[[x]], <8 x i16> %[[y]])
! LLVMIR: store <8 x i16> %[[vmin]], ptr %{{[0-9]}}, align 16
end subroutine vec_min_testi16

! CHECK-LABEL: vec_min_testi32
subroutine vec_min_testi32(x, y)
  vector(integer(4)) :: vmin, x, y
  vmin = vec_min(x, y)

! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmin:.*]] = call <4 x i32> @llvm.ppc.altivec.vminsw(<4 x i32> %[[x]], <4 x i32> %[[y]])
! LLVMIR: store <4 x i32> %[[vmin]], ptr %{{[0-9]}}, align 16
end subroutine vec_min_testi32

! CHECK-LABEL: vec_min_testi64
subroutine vec_min_testi64(x, y)
  vector(integer(8)) :: vmin, x, y
  vmin = vec_min(x, y)

! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmin:.*]] = call <2 x i64> @llvm.ppc.altivec.vminsd(<2 x i64> %[[x]], <2 x i64> %[[y]])
! LLVMIR: store <2 x i64> %[[vmin]], ptr %{{[0-9]}}, align 16
end subroutine vec_min_testi64

! CHECK-LABEL: vec_min_testui8
subroutine vec_min_testui8(x, y)
  vector(unsigned(1)) :: vmin, x, y
  vmin = vec_min(x, y)

! LLVMIR: %[[x:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <16 x i8>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmin:.*]] = call <16 x i8> @llvm.ppc.altivec.vminub(<16 x i8> %[[x]], <16 x i8> %[[y]])
! LLVMIR: store <16 x i8> %[[vmin]], ptr %{{[0-9]}}, align 16
end subroutine vec_min_testui8

! CHECK-LABEL: vec_min_testui16
subroutine vec_min_testui16(x, y)
  vector(unsigned(2)) :: vmin, x, y
  vmin = vec_min(x, y)

! LLVMIR: %[[x:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <8 x i16>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmin:.*]] = call <8 x i16> @llvm.ppc.altivec.vminuh(<8 x i16> %[[x]], <8 x i16> %[[y]])
! LLVMIR: store <8 x i16> %[[vmin]], ptr %{{[0-9]}}, align 16
end subroutine vec_min_testui16

! CHECK-LABEL: vec_min_testui32
subroutine vec_min_testui32(x, y)
  vector(unsigned(4)) :: vmin, x, y
  vmin = vec_min(x, y)

! LLVMIR: %[[x:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <4 x i32>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmin:.*]] = call <4 x i32> @llvm.ppc.altivec.vminuw(<4 x i32> %[[x]], <4 x i32> %[[y]])
! LLVMIR: store <4 x i32> %[[vmin]], ptr %{{[0-9]}}, align 16
end subroutine vec_min_testui32

! CHECK-LABEL: vec_min_testui64
subroutine vec_min_testui64(x, y)
  vector(unsigned(8)) :: vmin, x, y
  vmin = vec_min(x, y)

! LLVMIR: %[[x:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <2 x i64>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmin:.*]] = call <2 x i64> @llvm.ppc.altivec.vminud(<2 x i64> %[[x]], <2 x i64> %[[y]])
! LLVMIR: store <2 x i64> %[[vmin]], ptr %{{[0-9]}}, align 16
end subroutine vec_min_testui64

! vec_madd

! CHECK-LABEL: vec_madd_testf32
subroutine vec_madd_testf32(x, y, z)
  vector(real(4)) :: vmsum, x, y, z
  vmsum = vec_madd(x, y, z)

! LLVMIR: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[z:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmsum:.*]] = call contract <4 x float> @llvm.fma.v4f32(<4 x float> %[[x]], <4 x float> %[[y]], <4 x float> %[[z]])
! LLVMIR: store <4 x float> %[[vmsum]], ptr %{{[0-9]}}, align 16
end subroutine vec_madd_testf32

! CHECK-LABEL: vec_madd_testf64
subroutine vec_madd_testf64(x, y, z)
  vector(real(8)) :: vmsum, x, y, z
  vmsum = vec_madd(x, y, z)

! LLVMIR: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[z:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vmsum:.*]] = call contract <2 x double> @llvm.fma.v2f64(<2 x double> %[[x]], <2 x double> %[[y]], <2 x double> %[[z]])
! LLVMIR: store <2 x double> %[[vmsum]], ptr %{{[0-9]}}, align 16
end subroutine vec_madd_testf64

! vec_nmsub

! CHECK-LABEL: vec_nmsub_testf32
subroutine vec_nmsub_testf32(x, y, z)
  vector(real(4)) :: vnmsub, x, y, z
  vnmsub = vec_nmsub(x, y, z)

! LLVMIR: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[z:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vnmsub:.*]] = call contract <4 x float> @llvm.ppc.fnmsub.v4f32(<4 x float> %[[x]], <4 x float> %[[y]], <4 x float> %[[z]])
! LLVMIR: store <4 x float> %[[vnmsub]], ptr %{{[0-9]}}, align 16
end subroutine vec_nmsub_testf32

! CHECK-LABEL: vec_nmsub_testf64
subroutine vec_nmsub_testf64(x, y, z)
  vector(real(8)) :: vnmsub, x, y, z
  vnmsub = vec_nmsub(x, y, z)

! LLVMIR: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[z:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[vnmsub:.*]] = call contract <2 x double> @llvm.ppc.fnmsub.v2f64(<2 x double> %[[x]], <2 x double> %[[y]], <2 x double> %[[z]])
! LLVMIR: store <2 x double> %[[vnmsub]], ptr %{{[0-9]}}, align 16
end subroutine vec_nmsub_testf64

! vec_msub

! CHECK-LABEL: vec_msub_testf32
subroutine vec_msub_testf32(x, y, z)
  vector(real(4)) :: vmsub, x, y, z
  vmsub = vec_msub(x, y, z)

! LLVMIR: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[z:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[nz:.*]] = fneg contract <4 x float> %[[z]]
! LLVMIR: %[[vmsub:.*]] = call contract <4 x float> @llvm.fma.v4f32(<4 x float> %[[x]], <4 x float> %[[y]], <4 x float> %[[nz]])
! LLVMIR: store <4 x float> %[[vmsub]], ptr %{{[0-9]}}, align 16
end subroutine vec_msub_testf32

! CHECK-LABEL: vec_msub_testf64
subroutine vec_msub_testf64(x, y, z)
  vector(real(8)) :: vmsub, x, y, z
  vmsub = vec_msub(x, y, z)

! LLVMIR: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[z:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[nz:.*]] = fneg contract <2 x double> %[[z]]
! LLVMIR: %[[vmsub:.*]] = call contract <2 x double> @llvm.fma.v2f64(<2 x double> %[[x]], <2 x double> %[[y]], <2 x double> %[[nz]])
! LLVMIR: store <2 x double> %[[vmsub]], ptr %{{[0-9]}}, align 16
end subroutine vec_msub_testf64

! vec_nmadd

! CHECK-LABEL: vec_nmadd_testf32
subroutine vec_nmadd_testf32(x, y, z)
  vector(real(4)) :: vnmsum, x, y, z
  vnmsum = vec_nmadd(x, y, z)

! LLVMIR: %[[x:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[z:.*]] = load <4 x float>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[msum:.*]] = call contract <4 x float> @llvm.fma.v4f32(<4 x float> %[[x]], <4 x float> %[[y]], <4 x float> %[[z]])
! LLVMIR: %[[vnmsum:.*]] = fneg contract <4 x float> %[[msum]]
! LLVMIR: store <4 x float> %[[vnmsum]], ptr %{{[0-9]}}, align 16
end subroutine vec_nmadd_testf32

! CHECK-LABEL: vec_nmadd_testf64
subroutine vec_nmadd_testf64(x, y, z)
  vector(real(8)) :: vnmsum, x, y, z
  vnmsum = vec_nmadd(x, y, z)

! LLVMIR: %[[x:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[y:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[z:.*]] = load <2 x double>, ptr %{{[0-9]}}, align 16
! LLVMIR: %[[msum:.*]] = call contract <2 x double> @llvm.fma.v2f64(<2 x double> %[[x]], <2 x double> %[[y]], <2 x double> %[[z]])
! LLVMIR: %[[vnmsum:.*]] = fneg contract <2 x double> %[[msum]]
! LLVMIR: store <2 x double> %[[vnmsum]], ptr %{{[0-9]}}, align 16
end subroutine vec_nmadd_testf64
