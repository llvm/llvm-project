!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! REQUIRES: flang-supports-f128-math

! RUN: %if x86-registered-target %{ %flang_fc1 -triple x86_64-unknown-linux-gnu -fopenmp -emit-llvm %s -o - | FileCheck %s --check-prefix=HOST %}
! RUN: %if amdgpu-registered-target %{ %flang_fc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device -emit-llvm %s -o - | FileCheck %s --check-prefix=DEVICE --implicit-check-not=@_FortranACAbsF128 %}

! Host-only quad-complex operations must survive lowering until late function
! filtering without triggering unsupported AMDGPU ABI rewriting. The same
! applies to host code in a function containing a target region.
! HOST: call {{.*}}@_FortranACAbsF128(
subroutine host_only(x)
  complex(16) :: x
  print *, abs(x)
end subroutine

subroutine host_associated(x)
  complex(16) :: x
  call internal_proc(x)
contains
  subroutine internal_proc(y)
    complex(16), value :: y
    complex(16) :: z
    z = x + y
  end subroutine
end subroutine

subroutine mixed(x, y)
  complex(16) :: x
  complex(8) :: y
  print *, abs(x)
  !$omp target map(tofrom:y)
    y = device_fn(y)
  !$omp end target
contains
  function device_fn(z) result(r)
    complex(8), value :: z
    complex(8) :: r
    r = z
  end function
end subroutine

! DEVICE-DAG: define {{.*}}{ double, double } @_QFmixedPdevice_fn(double {{.*}}, double {{.*}})
! DEVICE-DAG: define {{.*}}@__omp_offloading_{{.*}}mixed_{{.*}}
! DEVICE-DAG: call {{.*}}{ double, double } @_QFmixedPdevice_fn(double {{.*}}, double {{.*}})
