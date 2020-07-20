// Make sure we don't match the -NOT lines with the linker invocation.
// Delimiters match the start of the cc1 and the start of the linker lines
// DELIMITERS: {{^ (\(in-process\)|")}}

// RUN: %clang -### -pedantic -no-pedantic %s 2>&1 | FileCheck -check-prefix=NO_PEDANTIC -check-prefix=DELIMITERS %s
// RUN: %clang -### -pedantic -Wno-pedantic %s 2>&1 | FileCheck -check-prefix=PEDANTIC -check-prefix=DELIMITERS %s
// NO_PEDANTIC-NOT: -pedantic
// RUN: %clang -### -pedantic -pedantic -no-pedantic -pedantic %s 2>&1 | FileCheck -check-prefix=PEDANTIC -check-prefix=DELIMITERS %s
// RUN: %clang -### -pedantic -pedantic -no-pedantic -Wpedantic %s 2>&1 | FileCheck -check-prefix=NO_PEDANTIC -check-prefix=DELIMITERS %s
// PEDANTIC: -pedantic
// REQUIRES: clang-driver

// DELIMITERS: {{^ *"}}
