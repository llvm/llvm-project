! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s

! The base and its variant are both external procedures (same visibility), so
! the variant is accessible at every reference to the base.

! CHECK-LABEL: func.func @_QPtest_vendor_llvm
! CHECK: fir.call @_QPvllvm(){{.*}}: () -> ()
! CHECK-NOT: fir.call @_QPbase_llvm
subroutine test_vendor_llvm
  call base_llvm()
end subroutine test_vendor_llvm

subroutine base_llvm
  interface
    subroutine vllvm()
    end subroutine
  end interface
  !$omp declare variant (base_llvm:vllvm) match (implementation={vendor(llvm)})
end subroutine base_llvm

! An unknown vendor does not match: the base call is kept.

! CHECK-LABEL: func.func @_QPtest_vendor_unknown
! CHECK: fir.call @_QPbase_unknown(){{.*}}: () -> ()
! CHECK-NOT: fir.call @_QPvunknown
subroutine test_vendor_unknown
  call base_unknown()
end subroutine test_vendor_unknown

subroutine base_unknown
  interface
    subroutine vunknown()
    end subroutine
  end interface
  !$omp declare variant (base_unknown:vunknown) match (implementation={vendor("unknown")})
end subroutine base_unknown
