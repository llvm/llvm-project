/// Check that --offload-arch accepts the AMDGPU subarch triple spelling
/// (e.g. amdgpu9.00, amdgpu12.50) as an alias for the corresponding gfx
/// offload architecture, and that the legacy gfx names still work

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu6.00 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX600 %s
// GFX600: "-cc1" "-triple" "amdgpu6.00-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu6.01 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX601 %s
// GFX601: "-cc1" "-triple" "amdgpu6.01-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu6.02 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX602 %s
// GFX602: "-cc1" "-triple" "amdgpu6.02-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu7.00 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX700 %s
// GFX700: "-cc1" "-triple" "amdgpu7.00-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu7.01 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX701 %s
// GFX701: "-cc1" "-triple" "amdgpu7.01-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu7.02 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX702 %s
// GFX702: "-cc1" "-triple" "amdgpu7.02-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu7.03 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX703 %s
// GFX703: "-cc1" "-triple" "amdgpu7.03-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu7.04 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX704 %s
// GFX704: "-cc1" "-triple" "amdgpu7.04-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu7.05 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX705 %s
// GFX705: "-cc1" "-triple" "amdgpu7.05-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu8.01 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX801 %s
// GFX801: "-cc1" "-triple" "amdgpu8.01-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu8.02 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX802 %s
// GFX802: "-cc1" "-triple" "amdgpu8.02-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu8.03 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX803 %s
// GFX803: "-cc1" "-triple" "amdgpu8.03-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu8.05 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX805 %s
// GFX805: "-cc1" "-triple" "amdgpu8.05-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu8.10 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX810 %s
// GFX810: "-cc1" "-triple" "amdgpu8.10-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu9.00 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX900 %s
// GFX900: "-cc1" "-triple" "amdgpu9.00-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu9.02 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX902 %s
// GFX902: "-cc1" "-triple" "amdgpu9.02-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu9.04 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX904 %s
// GFX904: "-cc1" "-triple" "amdgpu9.04-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu9.06 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX906 %s
// GFX906: "-cc1" "-triple" "amdgpu9.06-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu9.08 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX908 %s
// GFX908: "-cc1" "-triple" "amdgpu9.08-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu9.09 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX909 %s
// GFX909: "-cc1" "-triple" "amdgpu9.09-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu9.0a -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX90A %s
// GFX90A: "-cc1" "-triple" "amdgpu9.0a-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu9.0c -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX90C %s
// GFX90C: "-cc1" "-triple" "amdgpu9.0c-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu9.42 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX942 %s
// GFX942: "-cc1" "-triple" "amdgpu9.42-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu9.50 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX950 %s
// GFX950: "-cc1" "-triple" "amdgpu9.50-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu10.10 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1010 %s
// GFX1010: "-cc1" "-triple" "amdgpu10.10-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu10.11 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1011 %s
// GFX1011: "-cc1" "-triple" "amdgpu10.11-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu10.12 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1012 %s
// GFX1012: "-cc1" "-triple" "amdgpu10.12-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu10.13 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1013 %s
// GFX1013: "-cc1" "-triple" "amdgpu10.13-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu10.30 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1030 %s
// GFX1030: "-cc1" "-triple" "amdgpu10.30-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu10.31 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1031 %s
// GFX1031: "-cc1" "-triple" "amdgpu10.31-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu10.32 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1032 %s
// GFX1032: "-cc1" "-triple" "amdgpu10.32-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu10.33 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1033 %s
// GFX1033: "-cc1" "-triple" "amdgpu10.33-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu10.34 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1034 %s
// GFX1034: "-cc1" "-triple" "amdgpu10.34-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu10.35 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1035 %s
// GFX1035: "-cc1" "-triple" "amdgpu10.35-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu10.36 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1036 %s
// GFX1036: "-cc1" "-triple" "amdgpu10.36-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu11.00 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1100 %s
// GFX1100: "-cc1" "-triple" "amdgpu11.00-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu11.01 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1101 %s
// GFX1101: "-cc1" "-triple" "amdgpu11.01-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu11.02 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1102 %s
// GFX1102: "-cc1" "-triple" "amdgpu11.02-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu11.03 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1103 %s
// GFX1103: "-cc1" "-triple" "amdgpu11.03-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu11.50 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1150 %s
// GFX1150: "-cc1" "-triple" "amdgpu11.50-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu11.51 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1151 %s
// GFX1151: "-cc1" "-triple" "amdgpu11.51-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu11.52 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1152 %s
// GFX1152: "-cc1" "-triple" "amdgpu11.52-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu11.53 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1153 %s
// GFX1153: "-cc1" "-triple" "amdgpu11.53-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu11.54 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1154 %s
// GFX1154: "-cc1" "-triple" "amdgpu11.54-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu11.70 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1170 %s
// GFX1170: "-cc1" "-triple" "amdgpu11.70-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu11.71 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1171 %s
// GFX1171: "-cc1" "-triple" "amdgpu11.71-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu11.72 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1172 %s
// GFX1172: "-cc1" "-triple" "amdgpu11.72-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu12.00 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1200 %s
// GFX1200: "-cc1" "-triple" "amdgpu12.00-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu12.01 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1201 %s
// GFX1201: "-cc1" "-triple" "amdgpu12.01-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu12.50 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1250 %s
// GFX1250: "-cc1" "-triple" "amdgpu12.50-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu12.51 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1251 %s
// GFX1251: "-cc1" "-triple" "amdgpu12.51-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu13.10 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1310 %s
// GFX1310: "-cc1" "-triple" "amdgpu13.10-amd-amdhsa"

//
// The "major" subarch spellings select the corresponding generic target.
//

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu9 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX9_GENERIC %s
// GFX9_GENERIC: "-cc1" "-triple" "amdgpu9-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu9.4 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX9_4_GENERIC %s
// GFX9_4_GENERIC: "-cc1" "-triple" "amdgpu9.4-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu10.1 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX10_1_GENERIC %s
// GFX10_1_GENERIC: "-cc1" "-triple" "amdgpu10.1-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu10.3 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX10_3_GENERIC %s
// GFX10_3_GENERIC: "-cc1" "-triple" "amdgpu10.3-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu11 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX11_GENERIC %s
// GFX11_GENERIC: "-cc1" "-triple" "amdgpu11-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu11.7 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX11_7_GENERIC %s
// GFX11_7_GENERIC: "-cc1" "-triple" "amdgpu11.7-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu12 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX12_GENERIC %s
// GFX12_GENERIC: "-cc1" "-triple" "amdgpu12-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu12.5 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX12_5_GENERIC %s
// GFX12_5_GENERIC: "-cc1" "-triple" "amdgpu12.5-amd-amdhsa"

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu13 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX13_GENERIC %s
// GFX13_GENERIC: "-cc1" "-triple" "amdgpu13-amd-amdhsa"

//
// The legacy gfx names continue to be accepted as aliases.
//

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=gfx900 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX900 %s

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=gfx1154 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX1154 %s

// RUN: %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=gfx9-generic -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=GFX9_GENERIC %s

//
// A bare amdgpu (no subarch) and family-major names without a generic target
// are rejected.
//

// RUN: not %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=BARE %s
// BARE: failed to deduce triple for target architecture 'amdgpu'

// RUN: not %clang -### --target=x86_64-pc-linux-gnu -fopenmp --offload-arch=amdgpu8 -nogpulib -nogpuinc %s 2>&1 | FileCheck --check-prefix=NO-GENERIC %s
// NO-GENERIC: failed to deduce triple for target architecture 'amdgpu8'
