!REQUIRES: amdgpu-registered-target

!RUN: %flang_fc1 -emit-llvm -triple amdgcn-amd-amdhsa -target-cpu gfx908 %s -o - | FileCheck --check-prefix=COV-DEFAULT %s
!RUN: %flang_fc1 -emit-llvm -triple amdgcn-amd-amdhsa -target-cpu gfx908 -mcode-object-version=none %s -o - | FileCheck --check-prefix=COV-NONE %s
!RUN: %flang_fc1 -emit-llvm -triple amdgcn-amd-amdhsa -target-cpu gfx908 -mcode-object-version=4 %s -o - | FileCheck --check-prefix=COV-4 %s
!RUN: %flang_fc1 -emit-llvm -triple amdgcn-amd-amdhsa -target-cpu gfx908 -mcode-object-version=5 %s -o - | FileCheck --check-prefix=COV-5 %s
!RUN: %flang_fc1 -emit-llvm -triple amdgcn-amd-amdhsa -target-cpu gfx908 -mcode-object-version=6 %s -o - | FileCheck --check-prefix=COV-6 %s

!COV-DEFAULT-NOT: !{{.*}} = !{{{.*}}, !"amdhsa_code_object_version", {{.*}}}
!COV-NONE-NOT: !{{.*}} = !{{{.*}}, !"amdhsa_code_object_version", {{.*}}}

!COV-4: !llvm.module.flags = !{{{.*}}, ![[COV_FLAG:.*]]}
!COV-4: ![[COV_FLAG]] = !{i32 1, !"amdhsa_code_object_version", i32 400}

!COV-5: !llvm.module.flags = !{{{.*}}, ![[COV_FLAG:.*]]}
!COV-5: ![[COV_FLAG]] = !{i32 1, !"amdhsa_code_object_version", i32 500}

!COV-6: !llvm.module.flags = !{{{.*}}, ![[COV_FLAG:.*]]}
!COV-6: ![[COV_FLAG]] = !{i32 1, !"amdhsa_code_object_version", i32 600}

subroutine target_simple
end subroutine
