// COM: Prefixes follow pattern (AMD_COMGR_SAVETEMPS)-(AMD_COMGR_USE_VFS)-(DataAction API)

// COM: Default behavior right now is to use the real file system
// RUN: source-to-bc-with-dev-libs %s -o %t-with-dev-libs.bc | FileCheck --check-prefixes=STATUS,OUT-NA-NA-NA %s

// COM: AMD_COMGR_USE_VFS=1 should force the compiler to use VFS, irrespective of the option provided via the DataAction API
// RUN: env AMD_COMGR_USE_VFS=1 source-to-bc-with-dev-libs %s --novfs -o %t-with-dev-libs.bc | FileCheck --check-prefixes=STATUS,OUT-NA-VFS-NOVFS %s
// RUN: env AMD_COMGR_USE_VFS=1 source-to-bc-with-dev-libs %s -o %t-with-dev-libs.bc | FileCheck --check-prefixes=STATUS,OUT-NA-VFS-NA %s

// COM: AMD_COMGR_USE_VFS=0 should force the compiler to not use VFS, irrespective of the option provided via the DataAction API
// RUN: env AMD_COMGR_USE_VFS=0 source-to-bc-with-dev-libs %s --vfs -o %t-with-dev-libs.bc | FileCheck --check-prefixes=STATUS,OUT-NA-NOVFS-VFS %s
// RUN: env AMD_COMGR_USE_VFS=0 source-to-bc-with-dev-libs %s -o %t-with-dev-libs.bc | FileCheck --check-prefixes=STATUS,OUT-NA-NOVFS-NA %s

// COM: No value for AMD_COMGR_USE_VFS should respect option provided via the DataAction API
// RUN: source-to-bc-with-dev-libs %s --vfs -o %t-with-dev-libs.bc | FileCheck --check-prefixes=STATUS,OUT-NA-NA-VFS %s
// RUN: source-to-bc-with-dev-libs %s --novfs -o %t-with-dev-libs.bc | FileCheck --check-prefixes=STATUS,OUT-NA-NA-NOVFS %s

// COM: AMD_COMGR_SAVE_TEMPS=1 should override all options and always use the real file system 
// RUN: env AMD_COMGR_SAVE_TEMPS=1 source-to-bc-with-dev-libs %s --vfs -o %t-with-dev-libs.bc | FileCheck --check-prefixes=STATUS,OUT-SAVETEMPS-NA-VFS %s
// RUN: env AMD_COMGR_SAVE_TEMPS=1 AMD_COMGR_USE_VFS=1 source-to-bc-with-dev-libs %s -o %t-with-dev-libs.bc | FileCheck --check-prefixes=STATUS,OUT-SAVETEMPS-VFS-NA %s
// RUN: env AMD_COMGR_SAVE_TEMPS=1 AMD_COMGR_USE_VFS=1 source-to-bc-with-dev-libs %s --vfs -o %t-with-dev-libs.bc | FileCheck --check-prefixes=STATUS,OUT-SAVETEMPS-VFS-VFS %s

// OUT-NA-NA-NA: File System: VFS
// OUT-NA-VFS-NOVFS: File System: VFS
// OUT-NA-VFS-NA: File System: VFS
// OUT-NA-NOVFS-VFS: File System: Real
// OUT-NA-NOVFS-NA: File System: Real
// OUT-NA-NA-VFS: File System: VFS
// OUT-NA-NA-NOVFS: File System: Real
// OUT-SAVETEMPS-NA-VFS: File System: Real
// OUT-SAVETEMPS-VFS-VFS: File System: Real
// OUT-SAVETEMPS-VFS-NA: File System: Real

// COM: Verify success of compilation for all scenarios
// STATUS: ReturnStatus: AMD_COMGR_STATUS_SUCCESS

extern const __constant bool __oclc_finite_only_opt;
extern const __constant bool __oclc_unsafe_math_opt;
extern const __constant bool __oclc_correctly_rounded_sqrt32;
extern const __constant bool __oclc_wavefrontsize64;
extern const __constant int __oclc_ISA_version;
extern const __constant int __oclc_ABI_version;

void kernel device_libs(__global float *status) {

  if (__oclc_finite_only_opt)            status[0] = 1.0;
  if (__oclc_unsafe_math_opt)            status[1] = 1.0;
  if (__oclc_correctly_rounded_sqrt32)   status[3] = 1.0;
  if (__oclc_wavefrontsize64)            status[4] = 1.0;
  if (__oclc_ISA_version)                status[5] = 1.0;
  if (__oclc_ABI_version)                status[6] = 1.0;

  // Math functions to test AMDGPULibCalls Folding optimizations
  // fold_sincos()
  float x = 0.25;
  status[7] = sin(x) + cos(x);
  status[8] = cos(x) + sin(x);

  // fold_rootn()
  float y = 725.0;
  status[9] = rootn(y, 3);
  status[10] = rootn(y, -1);
  status[11] = rootn(y, -2);

  // fold_pow()
  float z = 12.16;
  status[12] = pow(z, (float) 0.5);
  status[13] = powr(y, (float) 7.23);

  // printf()
  printf("testy\n");
}
