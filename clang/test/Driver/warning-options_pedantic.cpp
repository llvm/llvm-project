// RUN: %clang -### -pedantic -no-pedantic %s 2>&1 | FileCheck -check-prefix=NO_PEDANTIC %s
// RUN: %clang -### -pedantic -Wno-pedantic %s 2>&1 | FileCheck -check-prefix=PEDANTIC %s
// NO_PEDANTIC-NOT: -pedantic
// RUN: %clang -### -pedantic -pedantic -no-pedantic -pedantic %s 2>&1 | FileCheck -check-prefix=PEDANTIC %s
// RUN: %clang -### -pedantic -pedantic -no-pedantic -Wpedantic %s 2>&1 | FileCheck -check-prefix=NO_PEDANTIC %s
// PEDANTIC: -pedantic
// XFAIL: cygwin,mingw32
