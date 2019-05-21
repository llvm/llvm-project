// RUN: %clang -### -target avr -no-canonical-prefixes -save-temps -mmcu=attiny13a %s 2>&1 | FileCheck --check-prefix=WARN %s

// WARN: warning: support for linking stdlibs for microcontroller 'attiny13a' is not implemented
// WARN: warning: standard library not linked and so no interrupt vector table or compiler runtime routines will be linked

int main() { return 0; }

