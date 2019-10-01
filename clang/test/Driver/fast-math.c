// Test that the GCC fast-math floating point flags get lowered to the correct
// permutation of Clang frontend flags. This is non-trivial for a few reasons.
// First, the GCC flags have many different and surprising effects. Second,
// LLVM only supports three switches which is more coarse grained than GCC's
// support.
//
// Both of them use gcc driver for as.
// REQUIRES: clang-driver
//
// RUN: %clang -### -fno-honor-infinities -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-INFS %s
// infinites [sic] is a supported alternative spelling of infinities.
// RUN: %clang -### -fno-honor-infinites -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-INFS %s
// CHECK-NO-INFS: "-cc1"
// CHECK-NO-INFS: "-menable-no-infs"
//
// RUN: %clang -### -fno-fast-math -fno-honor-infinities -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FAST-MATH-NO-INFS %s
// CHECK-NO-FAST-MATH-NO-INFS: "-cc1"
// CHECK-NO-FAST-MATH-NO-INFS: "-menable-no-infs"
//
// RUN: %clang -### -fno-honor-infinities -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-INFS-NO-FAST-MATH %s
// CHECK-NO-INFS-NO-FAST-MATH: "-cc1"
// CHECK-NO-INFS-NO-FAST-MATH-NOT: "-menable-no-infs"
//
// RUN: %clang -### -fno-signed-zeros -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-SIGNED-ZEROS %s
// CHECK-NO-SIGNED-ZEROS: "-cc1"
// CHECK-NO-SIGNED-ZEROS: "-fno-signed-zeros"
//
// RUN: %clang -### -fno-fast-math -fno-signed-zeros -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FAST-MATH-NO-SIGNED-ZEROS %s
// CHECK-NO-FAST-MATH-NO-SIGNED-ZEROS: "-cc1"
// CHECK-NO-FAST-MATH-NO-SIGNED-ZEROS: "-fno-signed-zeros"
//
// RUN: %clang -### -fno-signed-zeros -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-SIGNED-ZEROS-NO-FAST-MATH %s
// CHECK-NO-SIGNED-ZEROS-NO-FAST-MATH: "-cc1"
// CHECK-NO-SIGNED-ZEROS-NO-FAST-MATH-NOT: "-fno-signed-zeros"
//
// RUN: %clang -### -freciprocal-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-RECIPROCAL-MATH %s
// CHECK-RECIPROCAL-MATH: "-cc1"
// CHECK-RECIPROCAL-MATH: "-freciprocal-math"
//
// RUN: %clang -### -fno-fast-math -freciprocal-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FAST-MATH-RECIPROCAL-MATH %s
// CHECK-NO-FAST-MATH-RECIPROCAL-MATH: "-cc1"
// CHECK-NO-FAST-MATH-RECIPROCAL-MATH: "-freciprocal-math"
//
// RUN: %clang -### -freciprocal-math -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-RECIPROCAL-MATH-NO-FAST-MATH %s
// CHECK-RECIPROCAL-MATH-NO-FAST-MATH: "-cc1"
// CHECK-RECIPROCAL-MATH-NO-FAST-MATH-NOT: "-freciprocal-math"
//
// RUN: %clang -### -fno-honor-nans -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NANS %s
// CHECK-NO-NANS: "-cc1"
// CHECK-NO-NANS: "-menable-no-nans"
//
// RUN: %clang -### -fno-fast-math -fno-honor-nans -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FAST-MATH-NO-NANS %s
// CHECK-NO-FAST-MATH-NO-NANS: "-cc1"
// CHECK-NO-FAST-MATH-NO-NANS: "-menable-no-nans"
//
// RUN: %clang -### -fno-honor-nans -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NANS-NO-FAST-MATH %s
// CHECK-NO-NANS-NO-FAST-MATH: "-cc1"
// CHECK-NO-NANS-NO-FAST-MATH-NOT: "-menable-no-nans"
//
// RUN: %clang -### -fmath-errno -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MATH-ERRNO %s
// CHECK-MATH-ERRNO: "-cc1"
// CHECK-MATH-ERRNO: "-fmath-errno"
//
// RUN: %clang -### -fmath-errno -fno-math-errno -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-MATH-ERRNO %s
// CHECK-NO-MATH-ERRNO: "-cc1"
// CHECK-NO-MATH-ERRNO-NOT: "-fmath-errno"
//
// Target defaults for -fmath-errno (reusing the above checks).
// RUN: %clang -### -target i686-unknown-linux -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MATH-ERRNO %s
// RUN: %clang -### -target i686-apple-darwin -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-MATH-ERRNO %s
// RUN: %clang -### -target x86_64-unknown-freebsd -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-MATH-ERRNO %s
// RUN: %clang -### -target x86_64-unknown-netbsd -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-MATH-ERRNO %s
// RUN: %clang -### -target x86_64-unknown-openbsd -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-MATH-ERRNO %s
// RUN: %clang -### -target x86_64-unknown-dragonfly -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-MATH-ERRNO %s
// RUN: %clang -### -target x86_64-fuchsia -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-MATH-ERRNO %s
// RUN: %clang -### -target x86_64-linux-android -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-MATH-ERRNO %s
// RUN: %clang -### -target amdgcn-amd-amdhsa -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-MATH-ERRNO %s
// RUN: %clang -### -target amdgcn-amd-amdpal -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-MATH-ERRNO %s
// RUN: %clang -### -target amdgcn-mesa-mesa3d -c %s 2>&1   \
// RUN:   | FileCheck --check-prefix=CHECK-NO-MATH-ERRNO %s
//
// Check that -ffast-math disables -fmath-errno, and -fno-fast-math merely
// preserves the target default. Also check various flag set operations between
// the two flags. (Resuses above checks.)
// RUN: %clang -### -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-MATH-ERRNO %s
// RUN: %clang -### -fmath-errno -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-MATH-ERRNO %s
// RUN: %clang -### -ffast-math -fmath-errno -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MATH-ERRNO %s
// RUN: %clang -### -target i686-unknown-linux -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MATH-ERRNO %s
// RUN: %clang -### -target i686-unknown-linux -fno-math-errno -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MATH-ERRNO %s
// RUN: %clang -### -target i686-apple-darwin -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-MATH-ERRNO %s
// RUN: %clang -### -target i686-apple-darwin -fno-math-errno -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-MATH-ERRNO %s
// RUN: %clang -### -fno-fast-math -fno-math-errno -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-MATH-ERRNO %s
//
// RUN: %clang -### -fno-math-errno -fassociative-math -freciprocal-math \
// RUN:     -fno-signed-zeros -fno-trapping-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-UNSAFE-MATH %s
// CHECK-UNSAFE-MATH: "-cc1"
// CHECK-UNSAFE-MATH: "-menable-unsafe-fp-math"
// CHECK-UNSAFE-MATH: "-mreassociate"
//
// RUN: %clang -### -fno-fast-math -fno-math-errno -fassociative-math -freciprocal-math \
// RUN:     -fno-signed-zeros -fno-trapping-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FAST-MATH-UNSAFE-MATH %s
// CHECK-NO-FAST-MATH-UNSAFE-MATH: "-cc1"
// CHECK-NO-FAST-MATH-UNSAFE-MATH: "-menable-unsafe-fp-math"
// CHECK-NO-FAST-MATH-UNSAFE-MATH: "-mreassociate"

// The 2nd -fno-fast-math overrides -fassociative-math.

// RUN: %clang -### -fno-fast-math -fno-math-errno -fassociative-math -freciprocal-math \
// RUN:     -fno-fast-math -fno-signed-zeros -fno-trapping-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-UNSAFE-MATH-NO-FAST-MATH %s
// CHECK-UNSAFE-MATH-NO-FAST-MATH: "-cc1"
// CHECK-UNSAFE-MATH-NO-FAST-MATH-NOT: "-menable-unsafe-fp-math"
// CHECK-UNSAFE-MATH-NO-FAST-MATH-NOT: "-mreassociate"
//
// Check that various umbrella flags also enable these frontend options.
// RUN: %clang -### -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-INFS %s
// RUN: %clang -### -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NANS %s
// RUN: %clang -### -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-UNSAFE-MATH %s
// RUN: %clang -### -ffinite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-INFS %s
// RUN: %clang -### -ffinite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NANS %s
// RUN: %clang -### -funsafe-math-optimizations -fno-math-errno -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-UNSAFE-MATH %s
//
// One umbrella flag is *really* weird and also changes the semantics of the
// program by adding a special preprocessor macro. Check that the frontend flag
// modeling this semantic change is provided. Also check that the flag is not
// present if any of the optimizations are disabled.
// RUN: %clang -### -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FAST-MATH %s
// RUN: %clang -### -fno-fast-math -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FAST-MATH %s
// RUN: %clang -### -funsafe-math-optimizations -ffinite-math-only \
// RUN:     -fno-math-errno -ffp-contract=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FAST-MATH %s
// RUN: %clang -### -fno-honor-infinities -fno-honor-nans -fno-math-errno \
// RUN:     -fassociative-math -freciprocal-math -fno-signed-zeros \
// RUN:     -fno-trapping-math -ffp-contract=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FAST-MATH %s
// CHECK-FAST-MATH: "-cc1"
// CHECK-FAST-MATH: "-ffast-math"
// CHECK-FAST-MATH: "-ffinite-math-only"
//
// RUN: %clang -### -ffast-math -fno-fast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FAST-MATH %s
// RUN: %clang -### -ffast-math -fno-finite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FAST-MATH %s
// RUN: %clang -### -ffast-math -fno-unsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FAST-MATH %s
// RUN: %clang -### -ffast-math -fmath-errno -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FAST-MATH %s
// RUN: %clang -### -ffast-math -fno-associative-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-FAST-MATH --check-prefix=CHECK-ASSOC-MATH %s
// CHECK-NO-FAST-MATH: "-cc1"
// CHECK-NO-FAST-MATH-NOT: "-ffast-math"
// CHECK-ASSOC-MATH-NOT: "-mreassociate"
//
// Check various means of disabling these flags, including disabling them after
// they've been enabled via an umbrella flag.
// RUN: %clang -### -fno-honor-infinities -fhonor-infinities -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NO-INFS %s
// RUN: %clang -### -ffinite-math-only -fhonor-infinities -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NO-INFS %s
// RUN: %clang -### -ffinite-math-only -fno-finite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NO-INFS %s
// RUN: %clang -### -ffast-math -fhonor-infinities -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NO-INFS %s
// RUN: %clang -### -ffast-math -fno-finite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NO-INFS %s
// CHECK-NO-NO-INFS: "-cc1"
// CHECK-NO-NO-INFS-NOT: "-menable-no-infs"
// CHECK-NO-NO-INFS-NOT: "-ffinite-math-only"
// CHECK-NO-NO-INFS: "-o"
//
// RUN: %clang -### -fno-honor-nans -fhonor-nans -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NO-NANS %s
// RUN: %clang -### -ffinite-math-only -fhonor-nans -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NO-NANS %s
// RUN: %clang -### -ffinite-math-only -fno-finite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NO-NANS %s
// RUN: %clang -### -ffast-math -fhonor-nans -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NO-NANS %s
// RUN: %clang -### -ffast-math -fno-finite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NO-NANS %s
// CHECK-NO-NO-NANS: "-cc1"
// CHECK-NO-NO-NANS-NOT: "-menable-no-nans"
// CHECK-NO-NO-NANS-NOT: "-ffinite-math-only"
// CHECK-NO-NO-NANS: "-o"

// A later inverted option overrides an earlier option.

// RUN: %clang -### -fassociative-math -freciprocal-math -fno-signed-zeros \
// RUN:     -fno-trapping-math -fno-associative-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s

// RUN: %clang -### -funsafe-math-optimizations -fno-associative-math -c %s \
// RUN:   2>&1 | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s

// RUN: %clang -### -funsafe-math-optimizations -fno-reciprocal-math -c %s \
// RUN:   2>&1 | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s
// RUN: %clang -### -funsafe-math-optimizations -fsigned-zeros -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s
// RUN: %clang -### -funsafe-math-optimizations -ftrapping-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s
// RUN: %clang -### -funsafe-math-optimizations -fno-unsafe-math-optimizations \
// RUN:     -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s
// RUN: %clang -### -ffast-math -fno-associative-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s

// RUN: %clang -### -ffast-math -fno-reciprocal-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s
// RUN: %clang -### -ffast-math -fsigned-zeros -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s
// RUN: %clang -### -ffast-math -ftrapping-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s
// RUN: %clang -### -ffast-math -fno-unsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s

// CHECK-NO-UNSAFE-MATH: "-cc1"
// CHECK-NO-UNSAFE-MATH-NOT: "-menable-unsafe-fp-math"
// CHECK-NO_UNSAFE-MATH-NOT: "-mreassociate"
// CHECK-NO-UNSAFE-MATH: "-o"


// Reassociate is allowed because it does not require reciprocal-math.

// RUN: %clang -### -fassociative-math -freciprocal-math -fno-signed-zeros \
// RUN:     -fno-trapping-math -fno-reciprocal-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-REASSOC-NO-UNSAFE-MATH %s

// CHECK-REASSOC-NO-UNSAFE-MATH: "-cc1"
// CHECK-REASSOC-NO-UNSAFE-MATH-NOT: "-menable-unsafe-fp-math"
// CHECK-REASSOC-NO_UNSAFE-MATH: "-mreassociate"
// CHECK-REASSOC-NO-UNSAFE-MATH-NOT: "-menable-unsafe-fp-math"
// CHECK-REASSOC-NO-UNSAFE-MATH: "-o"


// In these runs, reassociate is not allowed because both no-signed-zeros and no-trapping-math are required.

// RUN: %clang -### -fassociative-math -freciprocal-math -fno-signed-zeros \
// RUN:     -fno-trapping-math -fsigned-zeros -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-REASSOC-NO-UNSAFE-MATH %s

// RUN: %clang -### -fassociative-math -freciprocal-math -fno-signed-zeros \
// RUN:     -fno-trapping-math -ftrapping-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-REASSOC-NO-UNSAFE-MATH %s

// CHECK-NO-REASSOC-NO-UNSAFE-MATH: "-cc1"
// CHECK-NO-REASSOC-NO-UNSAFE-MATH-NOT: "-menable-unsafe-fp-math"
// CHECK-NO-REASSOC-NO_UNSAFE-MATH-NOT: "-mreassociate"
// CHECK-NO-REASSOC-NO-UNSAFE-MATH-NOT: "-menable-unsafe-fp-math"
// CHECK-NO-REASSOC-NO-UNSAFE-MATH: "-o"


// RUN: %clang -### -ftrapping-math -fno-trapping-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-TRAPPING-MATH %s
// CHECK-NO-TRAPPING-MATH: "-fno-trapping-math"

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

