// RUN: %clang -### --target=avr -mmcu=atmega328 --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINK %s
// LINK: {{".*ld.*"}} {{.*}} {{"-L.*avr5"}} {{.*}} "-Tdata=0x800100" {{.*}} "-latmega328" "-mavr5"
