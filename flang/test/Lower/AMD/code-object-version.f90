!REQUIRES: amdgpu-registered-target
!RUN: %flang_fc1 -emit-hlfir -triple amdgcn-amd-amdhsa -target-cpu gfx908 %s -o - | FileCheck  --check-prefix=COV_DEFAULT %s
!RUN: %flang_fc1 -emit-hlfir -triple amdgcn-amd-amdhsa -target-cpu gfx908 -mcode-object-version=none %s -o - | FileCheck  --check-prefix=COV_NONE %s
!RUN: %flang_fc1 -emit-hlfir -triple amdgcn-amd-amdhsa -target-cpu gfx908 -mcode-object-version=4 %s -o - | FileCheck  --check-prefix=COV_4 %s
!RUN: %flang_fc1 -emit-hlfir -triple amdgcn-amd-amdhsa -target-cpu gfx908 -mcode-object-version=5 %s -o - | FileCheck  --check-prefix=COV_5 %s
!RUN: %flang_fc1 -emit-hlfir -triple amdgcn-amd-amdhsa -target-cpu gfx908 -mcode-object-version=6 %s -o - | FileCheck  --check-prefix=COV_6 %s

!COV_DEFAULT: llvm.mlir.global weak_odr hidden local_unnamed_addr constant @__oclc_ABI_version(500 : i32) {addr_space = 4 : i32} : i32
!COV_NONE-NOT: llvm.mlir.global weak_odr hidden local_unnamed_addr constant @__oclc_ABI_version(500 : i32) {addr_space = 4 : i32} : i32
!COV_4: llvm.mlir.global weak_odr hidden local_unnamed_addr constant @__oclc_ABI_version(400 : i32) {addr_space = 4 : i32} : i32
!COV_5: llvm.mlir.global weak_odr hidden local_unnamed_addr constant @__oclc_ABI_version(500 : i32) {addr_space = 4 : i32} : i32
!COV_6: llvm.mlir.global weak_odr hidden local_unnamed_addr constant @__oclc_ABI_version(600 : i32) {addr_space = 4 : i32} : i32
subroutine target_simple
end subroutine target_simple
