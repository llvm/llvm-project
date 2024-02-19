////
//// Verify that valid options for the -mprefer-vector-width flag are passed through and invalid options cause an error.
////

//// If there are no options, convert to 'all'.

// RUN: %clang -### -S %s -mprefer-vector-width=none  2>&1 | FileCheck --check-prefix=WIDTHNONE %s
// WIDTHNONE: "-mprefer-vector-width=none"

//// Check options that cover all types.

// RUN: %clang -### -S %s -mprefer-vector-width=128  2>&1 | FileCheck --check-prefix=WIDTH128 %s
// WIDTH128: "-mprefer-vector-width=128"

//// Check invalid parameters.

// RUN: not %clang -### -S %s -mprefer-vector-width=one  2>&1 | FileCheck --check-prefix=WIDTHONE %s
// WIDTHONE: invalid value 'one' in 'mprefer-vector-width='

// RUN: not %clang -### -S %s -mprefer-vector-width=128.5  2>&1 | FileCheck --check-prefix=WIDTH128p5 %s
// WIDTH128p5: invalid value '128.5' in 'mprefer-vector-width='

// RUN: not %clang -### -S %s -mprefer-vector-width=-128  2>&1 | FileCheck --check-prefix=WIDTHNEG128 %s
// WIDTHNEG128: invalid value '-128' in 'mprefer-vector-width='
