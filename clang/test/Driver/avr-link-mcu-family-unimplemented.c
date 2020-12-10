// RUN: %clang -### -target avr -no-canonical-prefixes -save-temps -mmcu=attiny102 %s 2>&1 | FileCheck --check-prefix=WARN0 %s
// WARN0: warning: support for linking stdlibs for microcontroller 'attiny102' is not implemented
// WARN0: warning: standard library not linked and so no interrupt vector table or compiler runtime routines will be linked

// RUN: %clang -### -target avr -no-canonical-prefixes -save-temps -mmcu=atxmega32x1 %s 2>&1 | FileCheck --check-prefix=WARN1 %s
// WARN1: warning: support for linking stdlibs for microcontroller 'atxmega32x1' is not implemented
// WARN1: warning: standard library not linked and so no interrupt vector table or compiler runtime routines will be linked

int main() { return 0; }

