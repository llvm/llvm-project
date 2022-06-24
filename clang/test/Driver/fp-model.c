// Test that incompatible combinations of -ffp-model= options
// and other floating point options get a warning diagnostic.


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
// CHECK-FPM-FAST: "-menable-unsafe-fp-math"
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

