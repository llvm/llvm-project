// Test overriding warnings about complex range.
// range.c tests the settings of -complex-range=, and this test covers
// all warnings related to complex range.
// [draft] The tests for Case A are currently insufficient, but in the
// formal pull request, tests for all combinations of options will be
// added.

// Clang options related to complex range are as follows:
// -f[no-]cx-limited-range
// -f[no-]cx-fortran-rules
// -fcomplex-arithmetic=[full|improved|promoted|basic]
// -ffp-model=[strict|precise|fast|aggressive]
// -f[no-]fast-math

// Case A: Overriding Warnings
// Emit warnings about overriding when options implying different
// complex ranges are specified. However, warnings are not emitted in
// the following cases:
// (a) When the positive/negative form or a different value of the same
//     option is specified. 
//       Example: -fcx-limited-range -fno-cx-limited-range
// (b) When -ffp-model= is negated by -fno-fast-math. 
//       Example: -ffp-model=fast -fno-fast-math

// Case B: GCC Incompatibility Warnings
// Emit warnings because the following cases result in behavior
// incompatible with GCC:
// (a) -fcx-limited-range    -fno-fast-math
// (b) -fcx-fortran-rules    -fcx-limited-range
// (c) -fcx-fortran-rules    -fno-cx-limited-range
// (d) -fcx-fortran-rules    -ffast-math
// (e) -fcx-fortran-rules    -fno-fast-math
// (f) -fno-cx-fortran-rules -fcx-limited-range
// (g) -fno-cx-fortran-rules -ffast-math

// Case C: Combined A and B Warnings
// Emit combined warnings when both A and B apply. This is the default
// warning when both apply. The warning message changes to Case A or B
// if the user specifies -Wno-gcc-compat or -Wno-overriding-complex-range.


// Test A
// RUN: %clang -### -fcx-fortran-rules -fcx-limited-range -c -Woverriding-complex-range -Wno-gcc-compat %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=OVR_WARN1 %s

// RUN: %clang -### -fcomplex-arithmetic=improved  -ffp-model=aggressive -c -Woverriding-complex-range -Wno-gcc-compat %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=OVR_WARN2 %s

// RUN: %clang -### -Werror -fcx-limited-range -fno-cx-limited-range -c -Woverriding-complex-range -Wno-gcc-compat %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=OVR_WARN %s

// RUN: %clang -### -ffp-model=fast -ffp-model=strict -c -Woverriding-complex-range -Wno-gcc-compat %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=OVR_WARN %s

// RUN: %clang -### -Werror -ffp-model=fast -fno-fast-math -c -Woverriding-complex-range -Wno-gcc-compat %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=OVR_WARN %s


// Test B
// RUN: %clang -### -fcx-limited-range -fno-fast-math -c -Wno-overriding-complex-range -Wgcc-compat %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CMPT_WARN1 %s

// RUN: %clang -### -fcx-fortran-rules -fcx-limited-range -c -Wno-overriding-complex-range -Wgcc-compat %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CMPT_WARN2 %s

// RUN: %clang -### -fcx-fortran-rules -fno-cx-limited-range -c -Wno-overriding-complex-range -Wgcc-compat %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CMPT_WARN3 %s

// RUN: %clang -### -fcx-fortran-rules -ffast-math -c -Wno-overriding-complex-range -Wgcc-compat %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CMPT_WARN4 %s

// RUN: %clang -### -fcx-fortran-rules -fno-fast-math -c -Wno-overriding-complex-range -Wgcc-compat %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CMPT_WARN5 %s

// RUN: %clang -### -fno-cx-fortran-rules -fcx-limited-range -c -Wno-overriding-complex-range -Wgcc-compat %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CMPT_WARN6 %s

// RUN: %clang -### -fno-cx-fortran-rules -ffast-math -c -Wno-overriding-complex-range -Wgcc-compat %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CMPT_WARN7 %s


// Test C
// RUN: %clang -### -fcx-limited-range -fno-fast-math -c -Woverriding-complex-range -Wgcc-compat %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=COMBO_WARN1 %s

// RUN: %clang -### -fcx-fortran-rules -fcx-limited-range -c -Woverriding-complex-range -Wgcc-compat %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=COMBO_WARN2 %s

// RUN: %clang -### -fcx-fortran-rules -fno-cx-limited-range -c -Woverriding-complex-range -Wgcc-compat %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=COMBO_WARN3 %s

// RUN: %clang -### -fcx-fortran-rules -ffast-math -c -Woverriding-complex-range -Wgcc-compat %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=COMBO_WARN4 %s

// RUN: %clang -### -fcx-fortran-rules -fno-fast-math -c -Woverriding-complex-range -Wgcc-compat %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=COMBO_WARN5 %s

// RUN: %clang -### -fno-cx-fortran-rules -fcx-limited-range -c -Woverriding-complex-range -Wgcc-compat %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=COMBO_WARN6 %s

// RUN: %clang -### -fno-cx-fortran-rules -ffast-math -c -Woverriding-complex-range -Wgcc-compat %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=COMBO_WARN7 %s


// OVR_WARN-NOT: [-Woverriding-complex-range]
// OVR_WARN1: warning: '-fcx-limited-range' sets complex range to "basic" overriding the setting of "improved" that was implied by '-fcx-fortran-rules' [-Woverriding-complex-range]
// OVR_WARN2: warning: '-ffp-model=aggressive' sets complex range to "basic" overriding the setting of "improved" that was implied by '-fcomplex-arithmetic=improved' [-Woverriding-complex-range]
// CMPT_WARN1: warning: complex number caluculation is incompatible with GCC; specify '-fcx-limited-range' after '-fno-fast-math' for compatibility [-Wgcc-compat]
// CMPT_WARN2: warning: complex number caluculation is incompatible with GCC; specify '-fcx-fortran-rules' after '-fcx-limited-range' for compatibility [-Wgcc-compat]
// CMPT_WARN3: warning: complex number caluculation is incompatible with GCC; specify '-fcx-fortran-rules' after '-fno-cx-limited-range' for compatibility [-Wgcc-compat]
// CMPT_WARN4: warning: complex number caluculation is incompatible with GCC; specify '-fcx-fortran-rules' after '-ffast-math' for compatibility [-Wgcc-compat]
// CMPT_WARN5: warning: complex number caluculation is incompatible with GCC; specify '-fcx-fortran-rules' after '-fno-fast-math' for compatibility [-Wgcc-compat]
// CMPT_WARN6: warning: complex number caluculation is incompatible with GCC; specify '-fno-cx-fortran-rules' after '-fcx-limited-range' for compatibility [-Wgcc-compat]
// CMPT_WARN7: warning: complex number caluculation is incompatible with GCC; specify '-fno-cx-fortran-rules' after '-ffast-math' for compatibility [-Wgcc-compat]
// COMBO_WARN1: warning: '-fno-fast-math' sets complex range to "none" overriding the setting of "basic" that was implied by '-fcx-limited-range' and this is incompatible with GCC; specify '-fcx-limited-range' after '-fno-fast-math' for compatibility [-Wgcc-compat]
// COMBO_WARN2: warning: '-fcx-limited-range' sets complex range to "basic" overriding the setting of "improved" that was implied by '-fcx-fortran-rules' and this is incompatible with GCC; specify '-fcx-fortran-rules' after '-fcx-limited-range' for compatibility [-Wgcc-compat]
// COMBO_WARN3: warning: '-fno-cx-limited-range' sets complex range to "full" overriding the setting of "improved" that was implied by '-fcx-fortran-rules' and this is incompatible with GCC; specify '-fcx-fortran-rules' after '-fno-cx-limited-range' for compatibility [-Wgcc-compat]
// COMBO_WARN4: warning: '-ffast-math' sets complex range to "basic" overriding the setting of "improved" that was implied by '-fcx-fortran-rules' and this is incompatible with GCC; specify '-fcx-fortran-rules' after '-ffast-math' for compatibility [-Wgcc-compat]
// COMBO_WARN5: warning: '-fno-fast-math' sets complex range to "none" overriding the setting of "improved" that was implied by '-fcx-fortran-rules' and this is incompatible with GCC; specify '-fcx-fortran-rules' after '-fno-fast-math' for compatibility [-Wgcc-compat]
// COMBO_WARN6: warning: '-fcx-limited-range' sets complex range to "basic" overriding the setting of "full" that was implied by '-fno-cx-fortran-rules' and this is incompatible with GCC; specify '-fno-cx-fortran-rules' after '-fcx-limited-range' for compatibility [-Wgcc-compat]
// COMBO_WARN7: warning: '-ffast-math' sets complex range to "basic" overriding the setting of "full" that was implied by '-fno-cx-fortran-rules' and this is incompatible with GCC; specify '-fno-cx-fortran-rules' after '-ffast-math' for compatibility [-Wgcc-compat]
