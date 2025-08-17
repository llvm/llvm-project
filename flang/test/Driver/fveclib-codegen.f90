! test that -fveclib= is passed to the backend
! RUN: %if aarch64-registered-target %{ %flang -S -Ofast -target aarch64-unknown-linux-gnu -fveclib=SLEEF -o - %s | FileCheck %s --check-prefix=SLEEF %}
! RUN: %if x86-registered-target %{ %flang -S -Ofast -target x86_64-unknown-linux-gnu -fveclib=libmvec -o - %s | FileCheck %s %}
! RUN: %if aarch64-registered-target %{ %flang -S -Ofast -target aarch64-unknown-linux-gnu -fveclib=libmvec -march=armv8.2-a+nosve -o - %s | FileCheck %s --check-prefix=LIBMVEC-AARCH64-NEON %}
! RUN: %if aarch64-registered-target %{ %flang -S -Ofast -target aarch64-unknown-linux-gnu -fveclib=libmvec -march=armv8.2-a+sve -o - %s | FileCheck %s --check-prefix=LIBMVEC-AARCH64-SVE %}
! RUN: %if x86-registered-target %{ %flang -S -O3 -ffast-math -target x86_64-unknown-linux-gnu -fveclib=AMDLIBM -o - %s | FileCheck %s --check-prefix=AMDLIBM %}
! RUN: %flang -S -Ofast -fveclib=NoLibrary -o - %s | FileCheck %s --check-prefix=NOLIB

subroutine sb(a, b)
  real :: a(:), b(:)
  integer :: i
  do i=1,100
! check that we used a vectorized call to powf()
! CHECK: _ZGVbN4vv_powf
! SLEEF: _ZGVnN4vv_powf
! LIBMVEC-AARCH64-NEON: _ZGVnN4vv_powf
! LIBMVEC-AARCH64-SVE: _ZGVsMxvv_powf
! AMDLIBM: amd_vrs4_powf
! NOLIB: powf
    a(i) = a(i) ** b(i)
  end do
end subroutine
