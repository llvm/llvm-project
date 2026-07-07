// Check that -fstack-limit-variable=<var> is forwarded to -cc1, that the last
// value wins, and that an illegal variable name is rejected.

// RUN: %clang --target=arm-none-eabi -fstack-limit-variable=__stack_boundary -### -c %s 2>&1 | FileCheck %s -check-prefix=FORWARD
// RUN: %clang --target=thumbv7em-none-eabi -fstack-limit-variable=__stack_boundary -### -c %s 2>&1 | FileCheck %s -check-prefix=FORWARD
// FORWARD: "-cc1"
// FORWARD-SAME: "-fstack-limit-variable=__stack_boundary"

// The default is off.
// RUN: %clang --target=thumbv7em-none-eabi -### -c %s 2>&1 | FileCheck %s -check-prefix=DEFAULT
// DEFAULT-NOT: "-fstack-limit-variable=

// Last value wins.
// RUN: %clang --target=thumbv7em-none-eabi -fstack-limit-variable=first -fstack-limit-variable=second -### -c %s 2>&1 | FileCheck %s -check-prefix=LAST
// LAST-NOT: "-fstack-limit-variable=first"
// LAST: "-fstack-limit-variable=second"

// Illegal symbol names are rejected.
// RUN: not %clang --target=thumbv7em-none-eabi -fstack-limit-variable= -c %s 2>&1 | FileCheck %s -check-prefix=INVALID
// RUN: not %clang --target=thumbv7em-none-eabi -fstack-limit-variable=1bad -c %s 2>&1 | FileCheck %s -check-prefix=INVALID
// INVALID: error: invalid argument 'fstack-limit-variable=' only allowed with 'legal symbol name'

// -fstack-limit-trap-number=<n> is forwarded alongside the variable.
// RUN: %clang --target=thumbv7em-none-eabi -fstack-limit-variable=__stack_boundary -fstack-limit-trap-number=42 -### -c %s 2>&1 | FileCheck %s -check-prefix=TRAP-FORWARD
// TRAP-FORWARD: "-cc1"
// TRAP-FORWARD-SAME: "-fstack-limit-variable=__stack_boundary"
// TRAP-FORWARD-SAME: "-fstack-limit-trap-number=42"

// -fstack-limit-trap-number without -fstack-limit-variable is an error.
// RUN: not %clang --target=thumbv7em-none-eabi -fstack-limit-trap-number=42 -c %s 2>&1 | FileCheck %s -check-prefix=TRAP-ALONE
// TRAP-ALONE: error: invalid argument 'fstack-limit-trap-number=' only allowed with '-fstack-limit-variable='

// Non-integer trap values are rejected when cc1 parses the option.
// RUN: not %clang --target=thumbv7em-none-eabi -fstack-limit-variable=__stack_boundary -fstack-limit-trap-number=foo -c %s 2>&1 | FileCheck %s -check-prefix=TRAP-INVALID
// TRAP-INVALID: error: invalid integral value 'foo' in '-fstack-limit-trap-number=foo'

int foo(void) { return 0; }
