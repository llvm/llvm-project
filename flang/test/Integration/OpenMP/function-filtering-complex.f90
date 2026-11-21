!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! REQUIRES: flang-supports-f128-math, amdgpu-registered-target

! RUN: %flang_fc1 -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-target-device -emit-llvm %s -o - | FileCheck %s --implicit-check-not=@_FortranACAbsF128

! LLVM IR is needed because this regression depends on the driver's OpenMP
! device pipeline: early declare-target marking and selective TargetRewrite
! run before FIR-to-LLVM lowering, and host-only functions are removed only by
! the later function filtering. No earlier -emit-* stage runs this sequence.
! Pass-level behavior is covered by flang/test/Fir/target-rewrite-openmp.fir.

! Host-only quad-complex operations must survive lowering until late function
! filtering without triggering unsupported AMDGPU ABI rewriting. The same
! applies to host code in a function containing a target region.
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

! CHECK-DAG: define {{.*}}{ double, double } @_QFmixedPdevice_fn(double {{.*}}, double {{.*}})
! CHECK-DAG: define {{.*}}@__omp_offloading_{{.*}}mixed_{{.*}}
! CHECK-DAG: call {{.*}}{ double, double } @_QFmixedPdevice_fn(double {{.*}}, double {{.*}})
