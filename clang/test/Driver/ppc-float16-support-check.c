// Test the -mfloat16 front-end source-admission gate for _Float16 on PowerPC.
//
// The gate is target-independent: _Float16 is admitted with -mfloat16 on Power8+
// with hard-float, rejected on pre-Power8 CPUs or with -msoft-float, and is absent
// without the flag. -mfloat16 is front-end-only: it is forwarded to -cc1 and is
// never translated into a target feature.

// Power8/Power9 + -mfloat16: _Float16 is admitted (no diagnostic).
// RUN: %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only -mcpu=pwr9 -mfloat16 %s 2>&1 | FileCheck %s --check-prefix=HASFLOAT16 --allow-empty
// RUN: %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only -mcpu=pwr8 -mfloat16 %s 2>&1 | FileCheck %s --check-prefix=HASFLOAT16 --allow-empty
// RUN: %clang -target powerpc64-unknown-linux-gnu   -fsyntax-only -mcpu=pwr8 -mfloat16 %s 2>&1 | FileCheck %s --check-prefix=HASFLOAT16 --allow-empty
// RUN: %clang -target powerpc64-ibm-aix      -fsyntax-only -mcpu=pwr8 -mfloat16 %s 2>&1 | FileCheck %s --check-prefix=HASFLOAT16 --allow-empty
// HASFLOAT16-NOT: option '-mfloat16' cannot be specified with
// HASFLOAT16-NOT: _Float16 is not supported on this target

// Pre-Power8 + -mfloat16: rejected.
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only -mcpu=pwr7 -mfloat16 %s 2>&1 | FileCheck %s --check-prefix=BADCPU
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only -mcpu=pwr6 -mfloat16 %s 2>&1 | FileCheck %s --check-prefix=BADCPU
// RUN: not %clang -target powerpc64-unknown-linux-gnu   -fsyntax-only -mcpu=pwr7 -mfloat16 %s 2>&1 | FileCheck %s --check-prefix=BADCPU
// RUN: not %clang -target powerpc64-ibm-aix      -fsyntax-only -mcpu=pwr7 -mfloat16 %s 2>&1 | FileCheck %s --check-prefix=BADCPU
// BADCPU: option '-mfloat16' cannot be specified with

// -mfloat16 with -msoft-float: rejected.
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only -mcpu=pwr9 -mfloat16 -msoft-float %s 2>&1 | FileCheck %s --check-prefix=SOFTFLOAT
// RUN: not %clang -target powerpc64-unknown-linux-gnu   -fsyntax-only -mcpu=pwr9 -mfloat16 -msoft-float %s 2>&1 | FileCheck %s --check-prefix=SOFTFLOAT
// SOFTFLOAT: option '-mfloat16' cannot be specified with '-msoft-float'

// Without -mfloat16: the _Float16 token is not admitted.
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only -mcpu=pwr9 %s 2>&1 | FileCheck %s --check-prefix=NOFLAG
// RUN: not %clang -target powerpc64-ibm-aix      -fsyntax-only -mcpu=pwr9 %s 2>&1 | FileCheck %s --check-prefix=NOFLAG
// NOFLAG: _Float16 is not supported on this target

// Front-end-only: -mfloat16 is forwarded to -cc1 as "-mfloat16" and is NOT turned
// into a "+float16" target feature.
// RUN: %clang -target powerpc64le-unknown-linux-gnu -mcpu=pwr9 -mfloat16 -### -c %s 2>&1 | FileCheck %s --check-prefix=FORWARD
// RUN: %clang -target powerpc64-ibm-aix      -mcpu=pwr9 -mfloat16 -### -c %s 2>&1 | FileCheck %s --check-prefix=FORWARD
// FORWARD: "-mfloat16"

_Float16 f;
