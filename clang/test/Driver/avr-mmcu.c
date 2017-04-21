// A test for the propagation of the -mmcu option to -cc1 and -cc1as

// RUN: %clang -### -target avr -no-canonical-prefixes -mmcu=atmega328p -save-temps %s 2>&1 | FileCheck %s
// CHECK: clang{{.*}} "-cc1" {{.*}} "-target-cpu" "atmega328p"
// CHECK: clang{{.*}} "-cc1as" {{.*}} "-target-cpu" "atmega328p"
