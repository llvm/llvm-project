// Tests that "gfxABC" gets correctly converted to a triple and arch
// argument"compute_YY" when llvm-offload-binary is invoked

// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx600 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX600 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx601 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX601 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx602 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX602 %s
//
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx700 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX700 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx701 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX701 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx702 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX702 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx703 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX703 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx704 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX704 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx705 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX705 %s
//
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx801 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX801 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx802 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX802 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx803 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX803 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx805 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX805 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx810 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX810 %s
//
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx900 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX900 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx902 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX902 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx906 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX906 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx908 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX908 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx90a -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX90A %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx90c -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX90C %s
//
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx942 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX942 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx950 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX950 %s
//
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1010 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1010 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1011 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1011 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1012 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1012 %s
//
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1030 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1030 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1031 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1031 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1032 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1032 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1033 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1033 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1034 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1034 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1035 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1035 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1036 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1036 %s
//
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1100 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1100 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1101 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1101 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1102 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1102 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1103 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1103 %s
//
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1150 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1150 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1151 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1151 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1152 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1152 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1153 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1153 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1170 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1170 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1171 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1171 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1172 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1172 %s

// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1200 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1200 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1201 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1201 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1250 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1250 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1251 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1251 %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx1310 -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX1310 %s


// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx9-generic -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX9_GENERIC %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx9-4-generic -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX9_4_GENERIC %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx10-1-generic -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX10_1_GENERIC %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx10-3-generic -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX10_3_GENERIC %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx11-generic -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX11_GENERIC %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx12-generic -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX12_GENERIC %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx12-5-generic -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX12_5_GENERIC %s

//
// Feature modifier coverage
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx90a:xnack+ -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX90A_XNACK %s
// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=gfx908:sramecc- -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,GFX908_SRAM %s

// RUN: %clang -x hip -### --target=x86_64-linux-gnu -c --cuda-gpu-arch=amdgcnspirv -nogpuinc -nogpulib %s 2>&1 | FileCheck --check-prefixes=HIP,SPIRV %s


// HIP: llvm-offload-binary

// GFX600:  triple=amdgcn-amd-amdhsa,arch=gfx600
// GFX601:  triple=amdgcn-amd-amdhsa,arch=gfx601
// GFX602:  triple=amdgcn-amd-amdhsa,arch=gfx602
// GFX700:  triple=amdgcn-amd-amdhsa,arch=gfx700
// GFX701:  triple=amdgcn-amd-amdhsa,arch=gfx701
// GFX702:  triple=amdgcn-amd-amdhsa,arch=gfx702
// GFX703:  triple=amdgcn-amd-amdhsa,arch=gfx703
// GFX704:  triple=amdgcn-amd-amdhsa,arch=gfx704
// GFX705:  triple=amdgcn-amd-amdhsa,arch=gfx705
// GFX801:  triple=amdgcn-amd-amdhsa,arch=gfx801
// GFX802:  triple=amdgcn-amd-amdhsa,arch=gfx802
// GFX803:  triple=amdgcn-amd-amdhsa,arch=gfx803
// GFX805:  triple=amdgcn-amd-amdhsa,arch=gfx805
// GFX810:  triple=amdgcn-amd-amdhsa,arch=gfx810
// GFX900:  triple=amdgcn-amd-amdhsa,arch=gfx900
// GFX902:  triple=amdgcn-amd-amdhsa,arch=gfx902
// GFX906:  triple=amdgcn-amd-amdhsa,arch=gfx906
// GFX908:  triple=amdgcn-amd-amdhsa,arch=gfx908
// GFX90A:  triple=amdgcn-amd-amdhsa,arch=gfx90a
// GFX90C:  triple=amdgcn-amd-amdhsa,arch=gfx90c
// GFX942:  triple=amdgcn-amd-amdhsa,arch=gfx942
// GFX950:  triple=amdgcn-amd-amdhsa,arch=gfx950
// GFX1010: triple=amdgcn-amd-amdhsa,arch=gfx1010
// GFX1011: triple=amdgcn-amd-amdhsa,arch=gfx1011
// GFX1012: triple=amdgcn-amd-amdhsa,arch=gfx1012
// GFX1030: triple=amdgcn-amd-amdhsa,arch=gfx1030
// GFX1031: triple=amdgcn-amd-amdhsa,arch=gfx1031
// GFX1032: triple=amdgcn-amd-amdhsa,arch=gfx1032
// GFX1033: triple=amdgcn-amd-amdhsa,arch=gfx1033
// GFX1034: triple=amdgcn-amd-amdhsa,arch=gfx1034
// GFX1035: triple=amdgcn-amd-amdhsa,arch=gfx1035
// GFX1036: triple=amdgcn-amd-amdhsa,arch=gfx1036
// GFX1100: triple=amdgcn-amd-amdhsa,arch=gfx1100
// GFX1101: triple=amdgcn-amd-amdhsa,arch=gfx1101
// GFX1102: triple=amdgcn-amd-amdhsa,arch=gfx1102
// GFX1103: triple=amdgcn-amd-amdhsa,arch=gfx1103
// GFX1150: triple=amdgcn-amd-amdhsa,arch=gfx1150
// GFX1151: triple=amdgcn-amd-amdhsa,arch=gfx1151
// GFX1152: triple=amdgcn-amd-amdhsa,arch=gfx1152
// GFX1153: triple=amdgcn-amd-amdhsa,arch=gfx1153
// GFX1170: triple=amdgcn-amd-amdhsa,arch=gfx1170
// GFX1171: triple=amdgcn-amd-amdhsa,arch=gfx1171
// GFX1172: triple=amdgcn-amd-amdhsa,arch=gfx1172
// GFX1200: triple=amdgcn-amd-amdhsa,arch=gfx1200
// GFX1201: triple=amdgcn-amd-amdhsa,arch=gfx1201
// GFX1250: triple=amdgcn-amd-amdhsa,arch=gfx1250
// GFX1251: triple=amdgcn-amd-amdhsa,arch=gfx1251
// GFX1310: triple=amdgcn-amd-amdhsa,arch=gfx1310

// GFX90A_XNACK: triple=amdgcn-amd-amdhsa,arch=gfx90a:xnack+
// GFX908_SRAM:  triple=amdgcn-amd-amdhsa,arch=gfx908:sramecc-

// GFX9_GENERIC: triple=amdgcn-amd-amdhsa,arch=gfx9-generic
// GFX9_4_GENERIC: triple=amdgcn-amd-amdhsa,arch=gfx9-4-generic
// GFX10_1_GENERIC: triple=amdgcn-amd-amdhsa,arch=gfx10-1-generic
// GFX10_3_GENERIC: triple=amdgcn-amd-amdhsa,arch=gfx10-3-generic
// GFX11_GENERIC: triple=amdgcn-amd-amdhsa,arch=gfx11-generic
// GFX12_GENERIC: triple=amdgcn-amd-amdhsa,arch=gfx12-generic
// GFX12_5_GENERIC: triple=amdgcn-amd-amdhsa,arch=gfx12-5-generic

// SPIRV: triple=spirv64-amd-amdhsa,arch=amdgcnspirv
