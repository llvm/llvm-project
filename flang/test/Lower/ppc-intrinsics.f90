! RUN: bbc -emit-fir %s -outline-intrinsics -o - | FileCheck --check-prefixes="CHECK-FIR" %s
! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck --check-prefixes="CHECK-LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

! CHECK-LABEL: fmadd_testr
subroutine fmadd_testr(a, x, y)
  real :: a, x, y, z
  z = fmadd(a, x, y)
! CHECK-FIR: fir.call @fir.__ppc_fmadd.f32.f32.f32.f32
! CHECK-LLVMIR: call contract float @llvm.fma.f32(float %{{[0-9]}}, float %{{[0-9]}}, float %{{[0-9]}})
end

! CHECK-LABEL: fmadd_testd
subroutine fmadd_testd(a, x, y)
  real(8) :: a, x, y, z
  z = fmadd(a, x, y)
! CHECK-FIR: fir.call @fir.__ppc_fmadd.f64.f64.f64.f64
! CHECK-LLVMIR: call contract double @llvm.fma.f64(double %{{[0-9]}}, double %{{[0-9]}}, double %{{[0-9]}})
end

! CHECK-LABEL: fnmadd_testr
subroutine fnmadd_testr(a, x, y)
  real :: a, x, y, z
  z = fnmadd(a, x, y)
! CHECK-FIR: fir.call @fir.__ppc_fnmadd.f32.f32.f32.f32
! CHECK-LLVMIR: call contract float @llvm.ppc.fnmadds(float %{{[0-9]}}, float %{{[0-9]}}, float %{{[0-9]}})
end

! CHECK-LABEL: fnmadd_testd
subroutine fnmadd_testd(a, x, y)
  real(8) :: a, x, y, z
  z = fnmadd(a, x, y)
! CHECK-FIR: fir.call @fir.__ppc_fnmadd.f64.f64.f64.f64
! CHECK-LLVMIR: call contract double @llvm.ppc.fnmadd(double %{{[0-9]}}, double %{{[0-9]}}, double %{{[0-9]}})
end

! CHECK-LABEL: fmsub_testr
subroutine fmsub_testr(a, x, y)
  real :: a, x, y, z
  z = fmsub(a, x, y)
! CHECK-FIR: fir.call @fir.__ppc_fmsub.f32.f32.f32.f32
! CHECK-LLVMIR: call contract float @llvm.ppc.fmsubs(float %{{[0-9]}}, float %{{[0-9]}}, float %{{[0-9]}})
end

! CHECK-LABEL: fmsub_testd
subroutine fmsub_testd(a, x, y)
  real(8) :: a, x, y, z
  z = fmsub(a, x, y)
! CHECK-FIR: fir.call @fir.__ppc_fmsub.f64.f64.f64.f64
! CHECK-LLVMIR: call contract double @llvm.ppc.fmsub(double %{{[0-9]}}, double %{{[0-9]}}, double %{{[0-9]}})
end

! CHECK-LABEL: fnmsub_testr
subroutine fnmsub_testr(a, x, y)
  real :: a, x, y, z
  z = fnmsub(a, x, y)
! CHECK-FIR: fir.call @fir.__ppc_fnmsub.f32.f32.f32.f32
! CHECK-LLVMIR: call contract float @llvm.ppc.fnmsub.f32(float %{{[0-9]}}, float %{{[0-9]}}, float %{{[0-9]}})
end

! CHECK-LABEL: fnmsub_testd
subroutine fnmsub_testd(a, x, y)
  real(8) :: a, x, y, z
  z = fnmsub(a, x, y)
! CHECK-FIR: fir.call @fir.__ppc_fnmsub.f64.f64.f64.f64
! CHECK-LLVMIR: call contract double @llvm.ppc.fnmsub.f64(double %{{[0-9]}}, double %{{[0-9]}}, double %{{[0-9]}})
end
