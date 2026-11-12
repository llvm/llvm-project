// RUN: %clang -### --target=avr -save-temps -mmcu=atmega328 -nostdlib %s 2>&1 | FileCheck %s
// RUN: %clang -### --target=avr -save-temps -mmcu=atmega328 -nodefaultlibs %s 2>&1 | FileCheck %s

// nostdlib and nodefaultlibs programs should compile fine.

// CHECK: main
int main() { return 0; }

