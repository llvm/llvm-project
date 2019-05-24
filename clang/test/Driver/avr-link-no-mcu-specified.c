// RUN: %clang -### -target avr -no-canonical-prefixes -save-temps %s 2>&1 | FileCheck --check-prefix=WARN %s
// RUN: %clang -### -target avr -no-canonical-prefixes -save-temps -mmcu=atmega328 %s 2>&1 | FileCheck --check-prefix=NOWARN %s

// WARN: warning: no target microcontroller specified on command line, cannot link standard libraries, please pass -mmcu=<mcu name>
// WARN: warning: standard library not linked and so no interrupt vector table or compiler runtime routines will be linked

// NOWARN: main

int main() { return 0; }

