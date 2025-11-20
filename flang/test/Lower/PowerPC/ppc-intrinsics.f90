! RUN: %flang_fc1 -flang-experimental-hlfir -emit-llvm %s -o - | FileCheck --check-prefixes="LLVMIR" %s
! REQUIRES: target=powerpc{{.*}}

! CHECK-LABEL: fmadd_testr
subroutine fmadd_testr(a, x, y)
  real :: a, x, y, z
  z = fmadd(a, x, y)
! LLVMIR: call contract float @llvm.fma.f32(float %{{[0-9]}}, float %{{[0-9]}}, float %{{[0-9]}})
end

! CHECK-LABEL: fmadd_testd
subroutine fmadd_testd(a, x, y)
  real(8) :: a, x, y, z
  z = fmadd(a, x, y)
! LLVMIR: call contract double @llvm.fma.f64(double %{{[0-9]}}, double %{{[0-9]}}, double %{{[0-9]}})
end

! CHECK-LABEL: fnmadd_testr
subroutine fnmadd_testr(a, x, y)
  real :: a, x, y, z
  z = fnmadd(a, x, y)
! LLVMIR: call contract float @llvm.ppc.fnmadds(float %{{[0-9]}}, float %{{[0-9]}}, float %{{[0-9]}})
end

! CHECK-LABEL: fnmadd_testd
subroutine fnmadd_testd(a, x, y)
  real(8) :: a, x, y, z
  z = fnmadd(a, x, y)
! LLVMIR: call contract double @llvm.ppc.fnmadd(double %{{[0-9]}}, double %{{[0-9]}}, double %{{[0-9]}})
end

! CHECK-LABEL: fmsub_testr
subroutine fmsub_testr(a, x, y)
  real :: a, x, y, z
  z = fmsub(a, x, y)
! LLVMIR: call contract float @llvm.ppc.fmsubs(float %{{[0-9]}}, float %{{[0-9]}}, float %{{[0-9]}})
end

! CHECK-LABEL: fmsub_testd
subroutine fmsub_testd(a, x, y)
  real(8) :: a, x, y, z
  z = fmsub(a, x, y)
! LLVMIR: call contract double @llvm.ppc.fmsub(double %{{[0-9]}}, double %{{[0-9]}}, double %{{[0-9]}})
end

! CHECK-LABEL: fnmsub_testr
subroutine fnmsub_testr(a, x, y)
  real :: a, x, y, z
  z = fnmsub(a, x, y)
! LLVMIR: call contract float @llvm.ppc.fnmsub.f32(float %{{[0-9]}}, float %{{[0-9]}}, float %{{[0-9]}})
end

! CHECK-LABEL: fnmsub_testd
subroutine fnmsub_testd(a, x, y)
  real(8) :: a, x, y, z
  z = fnmsub(a, x, y)
! LLVMIR: call contract double @llvm.ppc.fnmsub.f64(double %{{[0-9]}}, double %{{[0-9]}}, double %{{[0-9]}})
end

! CHECK-LABEL: fctid_test
subroutine fctid_test(x)
  real(8) :: x, r
  r = fctid(x)
! LLVMIR: call contract double @llvm.ppc.fctid(double %{{[0-9]}})
end

! CHECK-LABEL: fctidz_test
subroutine fctidz_test(x)
  real(8) :: x, r
  r = fctidz(x)
! LLVMIR: call contract double @llvm.ppc.fctidz(double %{{[0-9]}})
end

! CHECK-LABEL: fctiw_test
subroutine fctiw_test(x)
  real(8) :: x, r
  r = fctiw(x)
! LLVMIR: call contract double @llvm.ppc.fctiw(double %{{[0-9]}})
end

! CHECK-LABEL: fctiwz_test
subroutine fctiwz_test(x)
  real(8) :: x, r
  r = fctiwz(x)
! LLVMIR: call contract double @llvm.ppc.fctiwz(double %{{[0-9]}})
end

! CHECK-LABEL: fctudz_test
subroutine fctudz_test(x)
  real(8) :: x, r
  r = fctudz(x)
! LLVMIR: call contract double @llvm.ppc.fctudz(double %{{[0-9]}})
end

! CHECK-LABEL: fctuwz_test
subroutine fctuwz_test(x)
  real(8) :: x, r
  r = fctuwz(x)
! LLVMIR: call contract double @llvm.ppc.fctuwz(double %{{[0-9]}})
end

! CHECK-LABEL: fcfi_test
subroutine fcfi_test(i)
  real(8) :: i, r
  r = fcfi(i)
! LLVMIR: call contract double @llvm.ppc.fcfid(double %{{[0-9]}})
end

! CHECK-LABEL: fcfid_test
subroutine fcfid_test(i)
  real(8) :: i, r
  r = fcfid(i)
! LLVMIR: call contract double @llvm.ppc.fcfid(double %{{[0-9]}})
end

! CHECK-LABEL: fcfud_test
subroutine fcfud_test(i)
  real(8) :: i, r
  r = fcfud(i)
! LLVMIR: call contract double @llvm.ppc.fcfud(double %{{[0-9]}})
end

! CHECK-LABEL: fnabs_testr(x)
subroutine fnabs_testr(x)
  real :: x, y
  y = fnabs(x)
! LLVMIR: call contract float @llvm.ppc.fnabss(float %{{[0-9]}})
end

! CHECK-LABEL: fnabs_testd(x)
subroutine fnabs_testd(x)
  real(8) :: x, y
  y = fnabs(x)
! LLVMIR: call contract double @llvm.ppc.fnabs(double %{{[0-9]}})
end

!CHECK-LABEL: fre_test(x)
subroutine fre_test(x)
  real(8) :: x, y
  y = fre(x)
! LLVMIR: call contract double @llvm.ppc.fre(double %{{[0-9]}})
end

!CHECK-LABEL: fres_test(x)
subroutine fres_test(x)
  real :: x, y
  y = fres(x)
! LLVMIR: call contract float @llvm.ppc.fres(float %{{[0-9]}})
end

!CHECK-LABEL: frsqrte_test(x)
subroutine frsqrte_test(x)
  real(8) :: x, y
  y = frsqrte(x)
! LLVMIR: call contract double @llvm.ppc.frsqrte(double %{{[0-9]}})
end

!CHECK-LABEL: frsqrtes_test(x)
subroutine frsqrtes_test(x)
  real :: x, y
  y = frsqrtes(x)
! LLVMIR: call contract float @llvm.ppc.frsqrtes(float %{{[0-9]}})
end

! CHECK-LABEL: mtfsf_test
subroutine mtfsf_test(r)
  real(8) :: r
  call mtfsf(1, r)
! LLVMIR: call void @llvm.ppc.mtfsf(i32 {{[0-9]}}, double %{{[0-9]}})
end

! CHECK-LABEL: mtfsfi_test
subroutine mtfsfi_test()
  call mtfsfi(1, 2)
! LLVMIR: call void @llvm.ppc.mtfsfi(i32 {{[0-9]}}, i32 {{[0-9]}})
end
