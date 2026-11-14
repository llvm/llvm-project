!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: %if amdgpu-registered-target %{ %flang_fc1 -triple amdgcn-amd-amdhsa -fopenmp-simd -fopenmp-is-target-device -emit-llvm %s -o - | FileCheck %s %}

! SIMD-only device compilation does not filter host functions, so complex
! arguments must still use the AMDGPU target ABI.
! CHECK: define void @callee_(double {{.*}}, double {{.*}})
! CHECK: define void @caller_(double {{.*}}, double {{.*}})
! CHECK: call void @callee_(double {{.*}}, double {{.*}})
subroutine callee(value)
  complex(8), value :: value
end subroutine

subroutine caller(value)
  complex(8), value :: value
  interface
    subroutine callee(value)
      complex(8), value :: value
    end subroutine
  end interface
  call callee(value)
end subroutine
