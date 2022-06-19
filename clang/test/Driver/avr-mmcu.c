// A test for the propagation of the -mmcu option to -cc1 and -cc1as

// RUN: %clang -### --target=avr -mmcu=attiny11 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECK0 %s
// CHECK0: "-cc1" {{.*}} "-target-cpu" "attiny11"
// CHECK0: "-cc1as" {{.*}} "-target-cpu" "attiny11"

// RUN: %clang -### --target=avr -mmcu=at90s2313 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECK1 %s
// CHECK1: "-cc1" {{.*}} "-target-cpu" "at90s2313"
// CHECK1: "-cc1as" {{.*}} "-target-cpu" "at90s2313"

// RUN: %clang -### --target=avr -mmcu=at90s8515 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECK2 %s
// CHECK2: "-cc1" {{.*}} "-target-cpu" "at90s8515"
// CHECK2: "-cc1as" {{.*}} "-target-cpu" "at90s8515"

// RUN: %clang -### --target=avr -mmcu=attiny13a -save-temps %s 2>&1 | FileCheck -check-prefix=CHECK3 %s
// CHECK3: "-cc1" {{.*}} "-target-cpu" "attiny13a"
// CHECK3: "-cc1as" {{.*}} "-target-cpu" "attiny13a"

// RUN: %clang -### --target=avr -mmcu=attiny88 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECK4 %s
// CHECK4: "-cc1" {{.*}} "-target-cpu" "attiny88"
// CHECK4: "-cc1as" {{.*}} "-target-cpu" "attiny88"

// RUN: %clang -### --target=avr -mmcu=attiny88 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECK5 %s
// CHECK5: "-cc1" {{.*}} "-target-cpu" "attiny88"
// CHECK5: "-cc1as" {{.*}} "-target-cpu" "attiny88"

// RUN: %clang -### --target=avr -mmcu=atmega8u2 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECK6 %s
// CHECK6: "-cc1" {{.*}} "-target-cpu" "atmega8u2"
// CHECK6: "-cc1as" {{.*}} "-target-cpu" "atmega8u2"

// RUN: %clang -### --target=avr -mmcu=atmega8u2 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECK7 %s
// CHECK7: "-cc1" {{.*}} "-target-cpu" "atmega8u2"
// CHECK7: "-cc1as" {{.*}} "-target-cpu" "atmega8u2"

// RUN: %clang -### --target=avr -mmcu=atmega8a -save-temps %s 2>&1 | FileCheck -check-prefix=CHECK8 %s
// CHECK8: "-cc1" {{.*}} "-target-cpu" "atmega8a"
// CHECK8: "-cc1as" {{.*}} "-target-cpu" "atmega8a"

// RUN: %clang -### --target=avr -mmcu=atmega8a -save-temps %s 2>&1 | FileCheck -check-prefix=CHECK9 %s
// CHECK9: "-cc1" {{.*}} "-target-cpu" "atmega8a"
// CHECK9: "-cc1as" {{.*}} "-target-cpu" "atmega8a"

// RUN: %clang -### --target=avr -mmcu=atmega16a -save-temps %s 2>&1 | FileCheck -check-prefix=CHECKa %s
// CHECKa: "-cc1" {{.*}} "-target-cpu" "atmega16a"
// CHECKa: "-cc1as" {{.*}} "-target-cpu" "atmega16a"

// RUN: %clang -### --target=avr -mmcu=atmega16a -save-temps %s 2>&1 | FileCheck -check-prefix=CHECKb %s
// CHECKb: "-cc1" {{.*}} "-target-cpu" "atmega16a"
// CHECKb: "-cc1as" {{.*}} "-target-cpu" "atmega16a"

// RUN: %clang -### --target=avr -mmcu=atmega128a -save-temps %s 2>&1 | FileCheck -check-prefix=CHECKc %s
// CHECKc: "-cc1" {{.*}} "-target-cpu" "atmega128a"
// CHECKc: "-cc1as" {{.*}} "-target-cpu" "atmega128a"

// RUN: %clang -### --target=avr -mmcu=atmega2560 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECKd %s
// CHECKd: "-cc1" {{.*}} "-target-cpu" "atmega2560"
// CHECKd: "-cc1as" {{.*}} "-target-cpu" "atmega2560"

// RUN: %clang -### --target=avr -mmcu=attiny10 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECKe %s
// CHECKe: "-cc1" {{.*}} "-target-cpu" "attiny10"
// CHECKe: "-cc1as" {{.*}} "-target-cpu" "attiny10"

// RUN: %clang -### --target=avr -mmcu=atxmega16a4 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECKf %s
// CHECKf: "-cc1" {{.*}} "-target-cpu" "atxmega16a4"
// CHECKf: "-cc1as" {{.*}} "-target-cpu" "atxmega16a4"

// RUN: %clang -### --target=avr -mmcu=atxmega64b1 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECKg %s
// CHECKg: "-cc1" {{.*}} "-target-cpu" "atxmega64b1"
// CHECKg: "-cc1as" {{.*}} "-target-cpu" "atxmega64b1"

// RUN: %clang -### --target=avr -mmcu=atxmega64a1u -save-temps %s 2>&1 | FileCheck -check-prefix=CHECKh %s
// CHECKh: "-cc1" {{.*}} "-target-cpu" "atxmega64a1u"
// CHECKh: "-cc1as" {{.*}} "-target-cpu" "atxmega64a1u"

// RUN: %clang -### --target=avr -mmcu=atxmega128a3u -save-temps %s 2>&1 | FileCheck -check-prefix=CHECKj %s
// CHECKj: "-cc1" {{.*}} "-target-cpu" "atxmega128a3u"
// CHECKj: "-cc1as" {{.*}} "-target-cpu" "atxmega128a3u"

// RUN: %clang -### --target=avr -mmcu=atxmega128a4u -save-temps %s 2>&1 | FileCheck -check-prefix=CHECKi %s
// CHECKi: "-cc1" {{.*}} "-target-cpu" "atxmega128a4u"
// CHECKi: "-cc1as" {{.*}} "-target-cpu" "atxmega128a4u"
