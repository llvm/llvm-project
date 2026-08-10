// Test that the GCC fast-math floating point flags get lowered to the correct
// permutation of Clang frontend flags. This is non-trivial for a few reasons.
// First, the GCC flags have many different and surprising effects. Second,
// LLVM only supports three switches which is more coarse grained than GCC's
// support.
//
// Both of them use gcc driver for as.
//
// RUN: %clang -### -fno-honor-infinities -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NINF,NO-NNAN,NO-FINITE-ONLY %s
// infinites [sic] is a supported alternative spelling of infinities.
// RUN: %clang -### -fno-honor-infinites -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NINF,NO-NNAN,NO-FINITE-ONLY %s
//
// RUN: %clang -### -fno-fast-math -fno-honor-infinities -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,NINF,NO-NNAN,NO-FINITE-ONLY %s
//
// RUN: %clang -### -fno-honor-infinities -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,NO-NINF,NO-NNAN,NO-FINITE-ONLY %s
//
// RUN: %clang -### -fno-signed-zeros -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NSZ,NOROUNDING %s
//
// RUN: %clang -### -fno-fast-math -fno-signed-zeros -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,NSZ %s
//
// RUN: %clang -### -fno-signed-zeros -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,NO-NSZ,NOROUNDING %s
//
// RUN: %clang -### -freciprocal-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,ARCP,NOROUNDING %s
//
// RUN: %clang -### -fno-fast-math -freciprocal-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,ARCP,NOROUNDING %s
//
// RUN: %clang -### -freciprocal-math -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-ARCP,NOROUNDING %s
//
// RUN: %clang -### -fno-honor-nans -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NNAN,NO-NINF,NO-FINITE-ONLY,NOROUNDING %s
//
// RUN: %clang -### -fno-fast-math -fno-honor-nans -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,NNAN,NO-NINF,NO-FINITE-ONLY,NOROUNDING %s
//
// RUN: %clang -### -fno-honor-nans -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,NO-NNAN,NO-NINF,NO-FINITE-ONLY,NOROUNDING %s
//
// RUN: %clang -### -ffast-math -fno-approx-func -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,NINF,NNAN,FINITE-ONLY,NSZ,ARCP,NO-AFN,NO-ERRNO,NOROUNDING %s
//
// RUN: %clang -### -fno-approx-func -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,FAST,NINF,NNAN,FINITE-ONLY,NSZ,ARCP,AFN,NO-ERRNO,NOROUNDING %s
//
// RUN: %clang -### -fapprox-func -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,AFN %s
//
// RUN: %clang -### -fno-fast-math -fapprox-func -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,AFN,NOROUNDING %s
//
// RUN: %clang -### -fapprox-func -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,NO-AFN %s
//
// RUN: %clang -### -fmath-errno -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,ERRNO %s
//
// RUN: %clang -### -fmath-errno -fno-math-errno -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-ERRNO %s
//
// Target defaults for -fmath-errno (reusing the above checks).
// RUN: %clang -### --target=i686-unknown-linux -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,ERRNO %s
// RUN: %clang -### -target i686-apple-darwin -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-ERRNO %s
// RUN: %clang -### --target=x86_64-unknown-freebsd -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-ERRNO %s
// RUN: %clang -### --target=x86_64-unknown-netbsd -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-ERRNO %s
// RUN: %clang -### --target=x86_64-unknown-openbsd -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-ERRNO %s
// RUN: %clang -### --target=x86_64-unknown-haiku -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-ERRNO %s
// RUN: %clang -### --target=x86_64-unknown-dragonfly -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-ERRNO %s
// RUN: %clang -### --target=x86_64-fuchsia -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-ERRNO %s
// RUN: %clang -### --target=x86_64-linux-android -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-ERRNO %s
// RUN: %clang -### --target=x86_64-linux-musl -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-ERRNO %s
// RUN: %clang -### --target=amdgcn-amd-amdhsa -nogpuinc -nogpulib -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-ERRNO %s
// RUN: %clang -### --target=amdgcn-amd-amdpal -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-ERRNO %s
// RUN: %clang -### --target=amdgcn-mesa-mesa3d -c %s 2>&1   \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-ERRNO %s
//
// Check that -ffast-math disables -fmath-errno, and -fno-fast-math merely
// preserves the target default. Also check various flag set operations between
// the two flags. (Resuses above checks.)
// RUN: %clang -### -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-ERRNO %s
// RUN: %clang -### -fmath-errno -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-ERRNO %s
// RUN: %clang -### -ffast-math -fmath-errno -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,ERRNO %s
// RUN: %clang -### --target=i686-unknown-linux -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,ERRNO %s
// RUN: %clang -### --target=i686-unknown-linux -fno-math-errno -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,ERRNO %s
// RUN: %clang -### -target i686-apple-darwin -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-ERRNO %s
// RUN: %clang -### -target i686-apple-darwin -fno-math-errno -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-ERRNO %s
// RUN: %clang -### -fno-fast-math -fno-math-errno -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-ERRNO %s
//
// RUN: %clang -### -fno-math-errno -fassociative-math -freciprocal-math \
// RUN:     -fno-signed-zeros -fno-trapping-math -fapprox-func -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-ERRNO,UNSAFE,ARCP,NSZ,NO-TRAPPING,REASSOC %s
//
// RUN: %clang -### -fno-fast-math -fno-math-errno -fassociative-math -freciprocal-math \
// RUN:     -fno-signed-zeros -fno-trapping-math -fapprox-func -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-ERRNO,UNSAFE,ARCP,NSZ,NO-TRAPPING,REASSOC %s

// The 2nd -fno-fast-math overrides -fassociative-math.

// RUN: %clang -### -fno-fast-math -fno-math-errno -fassociative-math -freciprocal-math \
// RUN:     -fno-fast-math -fno-signed-zeros -fno-trapping-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-UNSAFE,NO-REASSOC,NO-ARCP,NSZ,NO-TRAPPING %s
//
// Check that various umbrella flags also enable these frontend options.
// RUN: %clang -### -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,FAST,NINF,NNAN,FINITE-ONLY,REASSOC,NSZ,ARCP,AFN,CONTRACT-FAST,NO-ERRNO,NOROUNDING %s
// RUN: %clang -### -ffinite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NINF,NNAN,FINITE-ONLY %s
// RUN: %clang -### -funsafe-math-optimizations -fno-math-errno -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,NO-NNAN,NO-NINF,NO-FINITE-ONLY,REASSOC,NSZ,ARCP,AFN,CONTRACT-FAST,NO-ERRNO,NOROUNDING %s
//
// One umbrella flag is *really* weird and also changes the semantics of the
// program by adding a special preprocessor macro. Check that the frontend flag
// modeling this semantic change is provided. Also check that the flag is not
// present if any of the optimizations are disabled.
// RUN: %clang -### -fno-fast-math -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,FAST,NINF,NNAN,FINITE-ONLY,REASSOC,NSZ,ARCP,AFN,CONTRACT-FAST,NO-ERRNO,NOROUNDING %s
// RUN: %clang -### -funsafe-math-optimizations -ffinite-math-only \
// RUN:     -fno-math-errno -ffp-contract=fast -fno-rounding-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,FAST,NINF,NNAN,FINITE-ONLY,NSZ,ARCP,AFN,CONTRACT-FAST,NO-ERRNO,NOROUNDING %s
// RUN: %clang -### -fno-honor-infinities -fno-honor-nans -fno-math-errno \
// RUN:     -fassociative-math -freciprocal-math -fno-signed-zeros -fapprox-func \
// RUN:     -fno-trapping-math -ffp-contract=fast -fno-rounding-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,FAST,NINF,NNAN,FINITE-ONLY,NSZ,ARCP,AFN,CONTRACT-FAST,NO-ERRNO,NOROUNDING %s
//
// RUN: %clang -### -ffast-math -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,NO-NINF,NO-NNAN,NO-FINITE-ONLY,NO-REASSOC,NO-NSZ,NO-ARCP,NO-AFN,NOROUNDING %s
// RUN: %clang -### -ffast-math -fno-finite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,NO-NINF,NO-NNAN,NO-FINITE-ONLY,REASSOC,NSZ,ARCP,AFN,CONTRACT-FAST,NOROUNDING %s

// FIXME: This case leaves nnan and ninf. That seems wrong!
// RUN: %clang -### -ffast-math -fno-unsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,NINF,NNAN,FINITE-ONLY,NO-REASSOC,NO-NSZ,NO-ARCP,NO-AFN,NOROUNDING,NO-TRAPPING %s
// RUN: %clang -### -ffast-math -fmath-errno -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,NINF,NNAN,FINITE-ONLY,REASSOC,NSZ,ARCP,AFN,CONTRACT-FAST,ERRNO,NOROUNDING %s
// RUN: %clang -### -ffast-math -fno-associative-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,NINF,NNAN,FINITE-ONLY,NO-REASSOC,NSZ,ARCP,AFN,CONTRACT-FAST,NO-ERRNO,NOROUNDING %s
//
// Check various means of disabling these flags, including disabling them after
// they've been enabled via an umbrella flag.
// RUN: %clang -### -fno-honor-infinities -fhonor-infinities -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-NINF,NO-NNAN,NO-FINITE-ONLY %s
// RUN: %clang -### -ffinite-math-only -fhonor-infinities -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-NINF,NNAN,NO-FINITE-ONLY %s
// RUN: %clang -### -ffinite-math-only -fno-finite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-NINF,NO-NNAN,NO-FINITE-ONLY %s
// RUN: %clang -### -ffast-math -fhonor-infinities -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,NO-NINF,NNAN,NO-FINITE-ONLY,REASSOC,NSZ,ARCP,AFN,CONTRACT-FAST,NO-ERRNO,NOROUNDING %s
// RUN: %clang -### -ffast-math -fno-finite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,NO-NINF,NO-NNAN,NO-FINITE-ONLY,REASSOC,NSZ,ARCP,AFN,CONTRACT-FAST,NO-ERRNO,NOROUNDING %s
//
// RUN: %clang -### -fno-honor-nans -fhonor-nans -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-NINF,NO-NNAN,NO-FINITE-ONLY %s
// RUN: %clang -### -ffinite-math-only -fhonor-nans -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NINF,NO-NNAN,NO-FINITE-ONLY %s
// RUN: %clang -### -ffinite-math-only -fno-finite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-NINF,NO-NNAN,NO-FINITE-ONLY %s
// RUN: %clang -### -ffast-math -fhonor-nans -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NINF,NO-NNAN,NO-FINITE-ONLY %s
// RUN: %clang -### -ffast-math -fno-finite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-NINF,NO-NNAN,NO-FINITE-ONLY %s

// A later inverted option overrides an earlier option.

// RUN: %clang -### -fassociative-math -freciprocal-math -fno-signed-zeros \
// RUN:     -fno-trapping-math -fno-associative-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-REASSOC,ARCP,NSZ,NO-TRAPPING %s

// RUN: %clang -### -funsafe-math-optimizations -fno-associative-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-UNSAFE,NO-REASSOC,ARCP,NSZ,AFN %s

// RUN: %clang -### -funsafe-math-optimizations -fno-reciprocal-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-UNSAFE,REASSOC,NO-ARCP,NSZ,AFN %s

// reassoc requires nsz
// RUN: %clang -### -funsafe-math-optimizations -fsigned-zeros -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-UNSAFE,NO-REASSOC,ARCP,NO-NSZ,AFN %s

// FIXME: Shouldn't trapping math disable all unsafe math?
// RUN: %clang -### -funsafe-math-optimizations -ftrapping-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-UNSAFE,ARCP,NSZ,AFN,TRAPPING %s

// RUN: %clang -### -funsafe-math-optimizations -fno-unsafe-math-optimizations \
// RUN:     -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-UNSAFE,NO-REASSOC,NO-ARCP,NO-NSZ,NO-AFN,NO-TRAPPING %s
// RUN: %clang -### -ffast-math -fno-associative-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-UNSAFE,NO-REASSOC,ARCP,NSZ,AFN %s

// RUN: %clang -### -ffast-math -fno-reciprocal-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,NO-UNSAFE,NNAN,NINF,FINITE-ONLY,REASSOC,NO-ARCP,CONTRACT-FAST,NSZ,AFN %s

// reassoc requires nsz
// RUN: %clang -### -ffast-math -fsigned-zeros -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,NO-UNSAFE,NNAN,NINF,FINITE-ONLY,NO-REASSOC,ARCP,CONTRACT-FAST,NO-NSZ,AFN %s

// FIXME: Shouldn't trapping math disable unsafe math?
// RUN: %clang -### -ffast-math -ftrapping-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,NO-UNSAFE,ARCP,NSZ,AFN,TRAPPING %s

// RUN: %clang -### -ffast-math -fno-unsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-FAST,NO-UNSAFE,NO-ARCP,NO-NSZ,NO-AFN,NO-TRAPPING %s

// Reassociate is allowed because it does not require reciprocal-math.

// RUN: %clang -### -fassociative-math -freciprocal-math -fno-signed-zeros \
// RUN:     -fno-trapping-math -fno-reciprocal-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,REASSOC,NO-ARCP,NSZ,NO-TRAPPING %s


// In these runs, reassociate is not allowed because both no-signed-zeros and no-trapping-math are required.

// RUN: %clang -### -fassociative-math -freciprocal-math -fno-signed-zeros \
// RUN:     -fno-trapping-math -fsigned-zeros -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-REASSOC,ARCP,NO-NSZ,NO-TRAPPING %s

// RUN: %clang -### -fassociative-math -freciprocal-math -fno-signed-zeros \
// RUN:     -fno-trapping-math -ftrapping-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefixes=CHECK,NO-REASSOC,ARCP,NSZ,TRAPPING %s

// The checks below allow stringing together prefixes to select the expected
// set of cc1 options for any combination of floating-point options.
// This is based on the assumption that the order of the flags when rendered
// is stable, so the negative checks only need to appear where the option would
// appear if used.

// start marker
// CHECK: "-cc1"

// NO-NINF-NOT:         "-menable-no-infs"
// NINF-SAME:           "-menable-no-infs"

// NO-NNAN-NOT:         "-menable-no-nans"
// NNAN-SAME:           "-menable-no-nans"

// NO-AFN-NOT:          "-fapprox-func"
// AFN-SAME:            "-fapprox-func"

// NO-ERRNO-NOT:        "-fmath-errno"
// ERRNO-SAME:          "-fmath-errno"

// NO-UNSAFE-NOT:       "-funsafe-math-optimizations"
// UNSAFE-SAME:         "-funsafe-math-optimizations"

// NO-NSZ-NOT:          "-fno-signed-zeros"
// NSZ-SAME:            "-fno-signed-zeros"

// NO-REASSOC-NOT:      "-mreassociate"
// REASSOC-SAME:        "-mreassociate"

// NO-ARCP-NOT:         "-freciprocal-math"
// ARCP-SAME:           "-freciprocal-math"

// NO-DENORM-NOT:       "-fdenormal-fp-math"
// DENORM-IEEE-SAME:    "-fdenormal-fp-math=ieee,ieee"
// DENORM-PS-SAME:      "-fdenormal-fp-math=preserve-sign,preserve-sign"
// DENORM-PZ-SAME:      "-fdenormal-fp-math=positive-zero,positive-zero"

// NO-CONTRACT-NOT:     "-ffp-contract"
// CONTRACT-OFF-SAME:   "-ffp-contract=off"
// CONTRACT-ON-SAME:    "-ffp-contract=on"
// CONTRACT-FAST-SAME:  "-ffp-contract=fast"

// This one is odd because -frounding-math is the default
// NO-NOROUNDING-NOT:   "-fno-rounding-math"
// NOROUNDING-SAME:     "-fno-rounding-math"

// NO-TRAPPING-NOT:     "-ffp-exception-behavior=strict"
// NO-TRAPPING-NOT:     "-ffp-exception-behavior=maytrap"
// TRAPPING-SAME:       "-ffp-exception-behavior=strict"

// NO-FAST-NOT:         "-ffast-math"
// FAST-SAME:           "-ffast-math"

// NO-FINITE-ONLY-NOT:  "-ffinite-math-only"
// FINITE-ONLY-SAME:    "-ffinite-math-only"

// NO-CX-RANGE-NOT:     "-complex-range"
// CX-RANGE-FULL-SAME:  "-complex-range=full"
// CX-RANGE-PROMO-SAME: "-complex-range=promoted"
// CX-RANGE-IMPRO-SAME: "-complex-range=improved"
// CX-RANGE-BASIC-SAME: "-complex-range=basic"

// end marker
// CHECK-SAME: "-o"


// This isn't fast-math, but the option is handled in the same place as other FP params.
// Last option wins, and strict behavior is assumed by default. 

// RUN: %clang -### -fno-strict-float-cast-overflow -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPOV-WORKAROUND %s
// CHECK-FPOV-WORKAROUND: "-cc1"
// CHECK-FPOV-WORKAROUND: "-fno-strict-float-cast-overflow"

// RUN: %clang -### -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPOV-WORKAROUND-DEFAULT %s
// CHECK-FPOV-WORKAROUND-DEFAULT: "-cc1"
// CHECK-FPOV-WORKAROUND-DEFAULT-NOT: "strict-float-cast-overflow"

// RUN: %clang -### -fstrict-float-cast-overflow -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FPOV-WORKAROUND %s
// CHECK-NO-FPOV-WORKAROUND: "-cc1"
// CHECK-NO-FPOV-WORKAROUND-NOT: "strict-float-cast-overflow"

// RUN: %clang -### -fno-strict-float-cast-overflow -fstrict-float-cast-overflow -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FPOV-WORKAROUND-OVERRIDE %s
// CHECK-NO-FPOV-WORKAROUND-OVERRIDE: "-cc1"
// CHECK-NO-FPOV-WORKAROUND-OVERRIDE-NOT: "strict-float-cast-overflow"

