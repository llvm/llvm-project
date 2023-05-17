// Test that incompatible combinations of -ffp-model= options
// and other floating point options get a warning diagnostic.
//

// RUN: %clang -### -ffp-model=fast -ffp-contract=off -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN %s
// WARN: warning: overriding '-ffp-model=fast' option with '-ffp-contract=off' [-Woverriding-t-option]

// RUN: %clang -### -ffp-model=fast -ffp-contract=on -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN1 %s
// WARN1: warning: overriding '-ffp-model=fast' option with '-ffp-contract=on' [-Woverriding-t-option]

// RUN: %clang -### -ffp-model=strict -fassociative-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN2 %s
// WARN2: warning: overriding '-ffp-model=strict' option with '-fassociative-math' [-Woverriding-t-option]

// RUN: %clang -### -ffp-model=strict -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN3 %s
// WARN3: warning: overriding '-ffp-model=strict' option with '-ffast-math' [-Woverriding-t-option]

// RUN: %clang -### -ffp-model=strict -ffinite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN4 %s
// WARN4: warning: overriding '-ffp-model=strict' option with '-ffinite-math-only' [-Woverriding-t-option]

// RUN: %clang -### -ffp-model=strict -ffp-contract=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN5 %s
// WARN5: warning: overriding '-ffp-model=strict' option with '-ffp-contract=fast' [-Woverriding-t-option]

// RUN: %clang -### -ffp-model=strict -ffp-contract=on -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN7 %s
// WARN7: warning: overriding '-ffp-model=strict' option with '-ffp-contract=on' [-Woverriding-t-option]

// RUN: %clang -### -ffp-model=strict -fno-honor-infinities -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN8 %s
// WARN8: warning: overriding '-ffp-model=strict' option with '-fno-honor-infinities' [-Woverriding-t-option]

// RUN: %clang -### -ffp-model=strict -fno-honor-nans -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN9 %s
// WARN9: warning: overriding '-ffp-model=strict' option with '-fno-honor-nans' [-Woverriding-t-option]

// RUN: %clang -### -ffp-model=strict -fno-rounding-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARNa %s
// WARNa: warning: overriding '-ffp-model=strict' option with '-fno-rounding-math' [-Woverriding-t-option]

// RUN: %clang -### -ffp-model=strict -fno-signed-zeros -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARNb %s
// WARNb: warning: overriding '-ffp-model=strict' option with '-fno-signed-zeros' [-Woverriding-t-option]

// RUN: %clang -### -ffp-model=strict -fno-trapping-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARNc %s
// WARNc: warning: overriding '-ffp-model=strict' option with '-fno-trapping-math' [-Woverriding-t-option]

// RUN: %clang -### -ffp-model=strict -freciprocal-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARNd %s
// WARNd: warning: overriding '-ffp-model=strict' option with '-freciprocal-math' [-Woverriding-t-option]

// RUN: %clang -### -ffp-model=strict -funsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARNe %s
// WARNe: warning: overriding '-ffp-model=strict' option with '-funsafe-math-optimizations' [-Woverriding-t-option]

// RUN: %clang -### -ffp-model=strict -Ofast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARNf %s
// WARNf: warning: overriding '-ffp-model=strict' option with '-Ofast' [-Woverriding-t-option]

// RUN: %clang -### -ffp-model=strict -fdenormal-fp-math=preserve-sign,preserve-sign -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN10 %s
// WARN10: warning: overriding '-ffp-model=strict' option with '-fdenormal-fp-math=preserve-sign,preserve-sign' [-Woverriding-t-option]

// RUN: %clang -### -ffp-model=fast -ffp-model=strict -c %s 2>&1 | FileCheck \
// RUN:   --check-prefix=WARN11 %s
// WARN11: warning: overriding '-ffp-model=fast' option with '-ffp-model=strict' [-Woverriding-t-option]
// WARN11-NOT: warning: overriding '-ffp-model=strict' option with '-ffp-model=strict' [-Woverriding-t-option]

// RUN: %clang -### -Ofast -ffp-model=strict -c %s 2>&1 | FileCheck \
// RUN:   --check-prefix=WARN12 %s
// RUN: %clang -### -ffast-math -ffp-model=strict -c %s 2>&1 | FileCheck \
// RUN:   --check-prefix=WARN12 %s
// WARN12-NOT: warning: overriding '-ffp-model=strict' option with '-ffp-model=strict' [-Woverriding-t-option]

// RUN: %clang -### -ffp-model=strict -fapprox-func -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=WARN13 %s
// WARN13: warning: overriding '-ffp-model=strict' option with '-fapprox-func' [-Woverriding-t-option]

// RUN: %clang -### -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NOROUND %s
// CHECK-NOROUND: "-cc1"
// CHECK-NOROUND: "-fno-rounding-math"

// RUN: %clang -### -frounding-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ROUND --implicit-check-not ffp-exception-behavior=strict %s
// CHECK-ROUND: "-cc1"
// CHECK-ROUND: "-frounding-math"

// RUN: %clang -### -ftrapping-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-TRAP %s
// CHECK-TRAP: "-cc1"
// CHECK-TRAP: "-ffp-exception-behavior=strict"

// RUN: %clang -### -nostdinc -ffp-model=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPM-FAST %s
// CHECK-FPM-FAST: "-cc1"
// CHECK-FPM-FAST: "-menable-no-infs"
// CHECK-FPM-FAST: "-menable-no-nans"
// CHECK-FPM-FAST: "-fapprox-func"
// CHECK-FPM-FAST: "-funsafe-math-optimizations"
// CHECK-FPM-FAST: "-fno-signed-zeros"
// CHECK-FPM-FAST: "-mreassociate"
// CHECK-FPM-FAST: "-freciprocal-math"
// CHECK-FPM-FAST: "-ffp-contract=fast"
// CHECK-FPM-FAST: "-fno-rounding-math"
// CHECK-FPM-FAST: "-ffast-math"
// CHECK-FPM-FAST: "-ffinite-math-only"

// RUN: %clang -### -nostdinc -ffp-model=precise -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPM-PRECISE %s
// CHECK-FPM-PRECISE: "-cc1"
// CHECK-FPM-PRECISE: "-ffp-contract=on"
// CHECK-FPM-PRECISE: "-fno-rounding-math"

// RUN: %clang -### -nostdinc -ffp-model=strict -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FPM-STRICT %s
// CHECK-FPM-STRICT: "-cc1"
// CHECK-FPM-STRICT: "-frounding-math"
// CHECK-FPM-STRICT: "-ffp-exception-behavior=strict"

// RUN: %clang -### -nostdinc -ffp-model=strict -ffp-model=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-EXCEPT %s
// RUN: %clang -### -nostdinc -ffp-model=strict -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-EXCEPT %s
// RUN: %clang -### -nostdinc -ffp-model=strict -Ofast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-EXCEPT %s
// CHECK-NO-EXCEPT-NOT: "-ffp-exception-behavior=strict"

// RUN: %clang -### -nostdinc -ffp-exception-behavior=strict -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FEB-STRICT %s
// CHECK-FEB-STRICT: "-cc1"
// CHECK-FEB-STRICT: "-fno-rounding-math"
// CHECK-FEB-STRICT: "-ffp-exception-behavior=strict"

// RUN: %clang -### -nostdinc -ffp-exception-behavior=maytrap -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FEB-MAYTRAP %s
// CHECK-FEB-MAYTRAP: "-cc1"
// CHECK-FEB-MAYTRAP: "-fno-rounding-math"
// CHECK-FEB-MAYTRAP: "-ffp-exception-behavior=maytrap"

// RUN: %clang -### -nostdinc -ffp-exception-behavior=ignore -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FEB-IGNORE %s
// CHECK-FEB-IGNORE: "-cc1"
// CHECK-FEB-IGNORE: "-fno-rounding-math"
// CHECK-FEB-IGNORE: "-ffp-exception-behavior=ignore"

// RUN: %clang -### -nostdinc -ffast-math -ffp-model=fast -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FASTMATH-FPM-FAST %s
// CHECK-FASTMATH-FPM-FAST: "-cc1"
// CHECK-FASTMATH-FPM-FAST: "-menable-no-infs"
// CHECK-FASTMATH-FPM-FAST: "-menable-no-nans"
// CHECK-FASTMATH-FPM-FAST: "-fapprox-func"
// CHECK-FASTMATH-FPM-FAST: "-funsafe-math-optimizations"
// CHECK-FASTMATH-FPM-FAST: "-fno-signed-zeros"
// CHECK-FASTMATH-FPM-FAST: "-mreassociate"
// CHECK-FASTMATH-FPM-FAST: "-freciprocal-math"
// CHECK-FASTMATH-FPM-FAST: "-ffp-contract=fast"
// CHECK-FASTMATH-FPM-FAST: "-fno-rounding-math"
// CHECK-FASTMATH-FPM-FAST: "-ffast-math"
// CHECK-FASTMATH-FPM-FAST: "-ffinite-math-only"

// RUN: %clang -### -nostdinc -ffast-math -ffp-model=precise -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FASTMATH-FPM-PRECISE %s
// CHECK-FASTMATH-FPM-PRECISE:     "-cc1"
// CHECK-FASTMATH-FPM-PRECISE-NOT: "-menable-no-infs"
// CHECK-FASTMATH-FPM-PRECISE-NOT: "-menable-no-nans"
// CHECK-FASTMATH-FPM-PRECISE-NOT: "-fapprox-func"
// CHECK-FASTMATH-FPM-PRECISE-NOT: "-funsafe-math-optimizations"
// CHECK-FASTMATH-FPM-PRECISE-NOT: "-fno-signed-zeros"
// CHECK-FASTMATH-FPM-PRECISE-NOT: "-mreassociate"
// CHECK-FASTMATH-FPM-PRECISE-NOT: "-freciprocal-math"
// CHECK-FASTMATH-FPM-PRECISE:     "-ffp-contract=on"
// CHECK-FASTMATH-FPM-PRECISE:     "-fno-rounding-math"
// CHECK-FASTMATH-FPM-PRECISE-NOT: "-ffast-math"
// CHECK-FASTMATH-FPM-PRECISE-NOT: "-ffinite-math-only"

// RUN: %clang -### -nostdinc -ffast-math -ffp-model=strict -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FASTMATH-FPM-STRICT %s
// CHECK-FASTMATH-FPM-STRICT:     "-cc1"
// CHECK-FASTMATH-FPM-STRICT-NOT: "-menable-no-infs"
// CHECK-FASTMATH-FPM-STRICT-NOT: "-menable-no-nans"
// CHECK-FASTMATH-FPM-STRICT-NOT: "-fapprox-func"
// CHECK-FASTMATH-FPM-STRICT-NOT: "-funsafe-math-optimizations"
// CHECK-FASTMATH-FPM-STRICT-NOT: "-fno-signed-zeros"
// CHECK-FASTMATH-FPM-STRICT-NOT: "-mreassociate"
// CHECK-FASTMATH-FPM-STRICT-NOT: "-freciprocal-math"
// CHECK-FASTMATH-FPM-STRICT:     "-ffp-contract=off"
// CHECK-FASTMATH-FPM-STRICT-NOT: "-fno-rounding-math"
// CHECK-FASTMATH-FPM-STRICT-NOT: "-ffast-math"
// CHECK-FASTMATH-FPM-STRICT-NOT: "-ffinite-math-only"
