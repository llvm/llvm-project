// Verify that the -ffastlib=AMDLIBM driver flag, together with fast-math at -O3,
// rewrites math library calls into their fast library entry points for X86,
// and leaves them untouched without the flag.

// REQUIRES: x86-registered-target

// RUN: %clang --target=x86_64-unknown-linux-gnu -O3 -ffast-math \
// RUN:   -ffastlib=AMDLIBM -S %s -o - | FileCheck %s --check-prefix=AMD
// RUN: %clang --target=x86_64-unknown-linux-gnu -O3 -ffast-math \
// RUN:   -S %s -o - | FileCheck %s --check-prefix=STD

double tan(double);
double exp(double);
double atan2(double, double);
double expm1(double);
float tanf(float);
double cbrt(double);

double call_tan(double x) { return tan(x) + x; }
// AMD-LABEL: call_tan:
// AMD: callq{{.*}}amd_fasttan
// STD-LABEL: call_tan:
// STD: callq{{.*}}tan

double call_exp(double x) { return exp(x) + x; }
// AMD-LABEL: call_exp:
// AMD: callq{{.*}}amd_fastexp
// STD-LABEL: call_exp:
// STD: callq{{.*}}exp

double call_atan2(double y, double x) { return atan2(y, x) + x; }
// AMD-LABEL: call_atan2:
// AMD: callq{{.*}}amd_fastatan2
// STD-LABEL: call_atan2:
// STD: callq{{.*}}atan2

double call_expm1(double x) { return expm1(x) + x; }
// AMD-LABEL: call_expm1:
// AMD: callq{{.*}}amd_fastexpm1
// STD-LABEL: call_expm1:
// STD: callq{{.*}}expm1

// Single-precision variant is rewritten too.
float call_tanf(float x) { return tanf(x) + x; }
// AMD-LABEL: call_tanf:
// AMD: callq{{.*}}amd_fasttanf
// STD-LABEL: call_tanf:
// STD: callq{{.*}}tanf

// cbrt has no fast library mapping and must stay even with the option enabled.
double call_cbrt(double x) { return cbrt(x) + x; }
// AMD-LABEL: call_cbrt:
// AMD-NOT: amd_fast
// AMD: callq{{.*}}cbrt
