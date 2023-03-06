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

! CHECK-LABEL: fctid_test
subroutine fctid_test(x)
  real(8) :: x, r
  r = fctid(x)
! CHECK-FIR: fir.call @fir.__ppc_fctid.f64.f64
! CHECK-LLVMIR: call contract double @llvm.ppc.fctid(double %{{[0-9]}})
end

! CHECK-LABEL: fctidz_test
subroutine fctidz_test(x)
  real(8) :: x, r
  r = fctidz(x)
! CHECK-FIR: fir.call @fir.__ppc_fctidz.f64.f64
! CHECK-LLVMIR: call contract double @llvm.ppc.fctidz(double %{{[0-9]}})
end

! CHECK-LABEL: fctiw_test
subroutine fctiw_test(x)
  real(8) :: x, r
  r = fctiw(x)
! CHECK-FIR: fir.call @fir.__ppc_fctiw.f64.f64
! CHECK-LLVMIR: call contract double @llvm.ppc.fctiw(double %{{[0-9]}})
end

! CHECK-LABEL: fctiwz_test
subroutine fctiwz_test(x)
  real(8) :: x, r
  r = fctiwz(x)
! CHECK-FIR: fir.call @fir.__ppc_fctiwz.f64.f64
! CHECK-LLVMIR: call contract double @llvm.ppc.fctiwz(double %{{[0-9]}})
end

! CHECK-LABEL: fctudz_test
subroutine fctudz_test(x)
  real(8) :: x, r
  r = fctudz(x)
! CHECK-FIR: fir.call @fir.__ppc_fctudz.f64.f64
! CHECK-LLVMIR: call contract double @llvm.ppc.fctudz(double %{{[0-9]}})
end

! CHECK-LABEL: fctuwz_test
subroutine fctuwz_test(x)
  real(8) :: x, r
  r = fctuwz(x)
! CHECK-FIR: fir.call @fir.__ppc_fctuwz.f64.f64
! CHECK-LLVMIR: call contract double @llvm.ppc.fctuwz(double %{{[0-9]}})
end

! CHECK-LABEL: fcfi_test
subroutine fcfi_test(i)
  real(8) :: i, r
  r = fcfi(i)
! CHECK-FIR: fir.call @fir.__ppc_fcfi.f64.f64
! CHECK-LLVMIR: call contract double @llvm.ppc.fcfid(double %{{[0-9]}})
end

! CHECK-LABEL: fcfid_test
subroutine fcfid_test(i)
  real(8) :: i, r
  r = fcfid(i)
! CHECK-FIR: fir.call @fir.__ppc_fcfid.f64.f64
! CHECK-LLVMIR: call contract double @llvm.ppc.fcfid(double %{{[0-9]}})
end

! CHECK-LABEL: fcfud_test
subroutine fcfud_test(i)
  real(8) :: i, r
  r = fcfud(i)
! CHECK-FIR: fir.call @fir.__ppc_fcfud.f64.f64
! CHECK-LLVMIR: call contract double @llvm.ppc.fcfud(double %{{[0-9]}})
end
