# REQUIRES: amdgpu

# RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx906 -mattr=-xnack --amdhsa-code-object-version=4 -filetype=obj %s -o %t-xnack-off0.o
# RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx906 -mattr=-xnack --amdhsa-code-object-version=4 -filetype=obj %s -o %t-xnack-off1.o
# RUN: ld.lld -shared %t-xnack-off0.o %t-xnack-off1.o -o %t-xnack-off2.so
# RUN: llvm-readobj --file-headers %t-xnack-off2.so | FileCheck --check-prefix=XNACK-OFF %s

# RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx906 -mattr=+xnack --amdhsa-code-object-version=4 -filetype=obj %s -o %t-xnack-on0.o
# RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx906 -mattr=+xnack --amdhsa-code-object-version=4 -filetype=obj %s -o %t-xnack-on1.o
# RUN: ld.lld -shared %t-xnack-on0.o %t-xnack-on1.o -o %t-xnack-on2.so
# RUN: llvm-readobj --file-headers %t-xnack-on2.so | FileCheck --check-prefix=XNACK-ON %s

# RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx906 --amdhsa-code-object-version=4 -filetype=obj %s -o %t-xnack-any.o
# RUN: ld.lld -shared %t-xnack-off0.o %t-xnack-any.o -o %t-xnack-off3.so
# RUN: llvm-readobj --file-headers %t-xnack-off3.so | FileCheck --check-prefix=XNACK-OFF %s
# RUN: ld.lld -shared %t-xnack-on0.o %t-xnack-any.o -o %t-xnack-on3.so
# RUN: llvm-readobj --file-headers %t-xnack-on3.so | FileCheck --check-prefix=XNACK-ON %s

# RUN: not ld.lld -shared %t-xnack-off0.o %t-xnack-on0.o -o /dev/null 2>&1 | FileCheck --check-prefix=XNACK-INCOMPATIBLE %s

# XNACK-OFF:          EF_AMDGPU_FEATURE_XNACK_OFF_V4 (0x200)
# XNACK-ON:           EF_AMDGPU_FEATURE_XNACK_ON_V4 (0x300)
# XNACK-INCOMPATIBLE: incompatible xnack:

# RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx906 -mattr=-sramecc --amdhsa-code-object-version=4 -filetype=obj %s -o %t-sramecc-off0.o
# RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx906 -mattr=-sramecc --amdhsa-code-object-version=4 -filetype=obj %s -o %t-sramecc-off1.o
# RUN: ld.lld -shared %t-sramecc-off0.o %t-sramecc-off1.o -o %t-sramecc-off2.so
# RUN: llvm-readobj --file-headers %t-sramecc-off2.so | FileCheck --check-prefix=SRAMECC-OFF %s

# RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx906 -mattr=+sramecc --amdhsa-code-object-version=4 -filetype=obj %s -o %t-sramecc-on0.o
# RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx906 -mattr=+sramecc --amdhsa-code-object-version=4 -filetype=obj %s -o %t-sramecc-on1.o
# RUN: ld.lld -shared %t-sramecc-on0.o %t-sramecc-on1.o -o %t-sramecc-on2.so
# RUN: llvm-readobj --file-headers %t-sramecc-on2.so | FileCheck --check-prefix=SRAMECC-ON %s

# RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx906 --amdhsa-code-object-version=4 -filetype=obj %s -o %t-sramecc-any.o
# RUN: ld.lld -shared %t-sramecc-off0.o %t-sramecc-any.o -o %t-sramecc-off3.so
# RUN: llvm-readobj --file-headers %t-sramecc-off3.so | FileCheck --check-prefix=SRAMECC-OFF %s
# RUN: ld.lld -shared %t-sramecc-on0.o %t-sramecc-any.o -o %t-sramecc-on3.so
# RUN: llvm-readobj --file-headers %t-sramecc-on3.so | FileCheck --check-prefix=SRAMECC-ON %s

# RUN: not ld.lld -shared %t-sramecc-off0.o %t-sramecc-on0.o -o /dev/null 2>&1 | FileCheck --check-prefix=SRAMECC-INCOMPATIBLE %s

# SRAMECC-OFF:          EF_AMDGPU_FEATURE_SRAMECC_OFF_V4 (0x800)
# SRAMECC-ON:           EF_AMDGPU_FEATURE_SRAMECC_ON_V4 (0xC00)
# SRAMECC-INCOMPATIBLE: incompatible sramecc:

# RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=6 --amdgpu-force-generic-version=1 -filetype=obj %s -o %t-genericv1_0.o
# RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=6 --amdgpu-force-generic-version=1 -filetype=obj %s -o %t-genericv1_1.o
# RUN: ld.lld -shared %t-genericv1_0.o %t-genericv1_1.o -o %t-genericv1_2.so
# RUN: llvm-readobj --file-headers %t-genericv1_2.so | FileCheck --check-prefix=GENERICV1 %s

# RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=6 --amdgpu-force-generic-version=2 -filetype=obj %s -o %t-genericv2_0.o
# RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=6 --amdgpu-force-generic-version=2 -filetype=obj %s -o %t-genericv2_1.o
# RUN: ld.lld -shared %t-genericv2_0.o %t-genericv2_1.o -o %t-genericv2_2.so
# RUN: llvm-readobj --file-headers %t-genericv2_2.so | FileCheck --check-prefix=GENERICV2 %s

# RUN: not ld.lld -shared %t-genericv1_0.o %t-genericv2_0.o -o /dev/null 2>&1 | FileCheck --check-prefix=GENERIC-INCOMPATIBLE %s

# GENERICV1:            EF_AMDGPU_GENERIC_VERSION_V1 (0x1000000)
# GENERICV2:            EF_AMDGPU_GENERIC_VERSION_V2 (0x2000000)
# GENERIC-INCOMPATIBLE: incompatible generic version
