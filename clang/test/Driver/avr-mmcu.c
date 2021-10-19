// A test for the propagation of the -mmcu option to -cc1 and -cc1as

// RUN: %clang -### -target avr -no-canonical-prefixes -mmcu=attiny11 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECK0 %s
// CHECK0: clang{{.*}} "-cc1" {{.*}} "-target-cpu" "attiny11"
// CHECK0: clang{{.*}} "-cc1as" {{.*}} "-target-cpu" "attiny11"

// RUN: %clang -### -target avr -no-canonical-prefixes -mmcu=at90s2313 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECK1 %s
// CHECK1: clang{{.*}} "-cc1" {{.*}} "-target-cpu" "at90s2313"
// CHECK1: clang{{.*}} "-cc1as" {{.*}} "-target-cpu" "at90s2313"

// RUN: %clang -### -target avr -no-canonical-prefixes -mmcu=at90s8515 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECK2 %s
// CHECK2: clang{{.*}} "-cc1" {{.*}} "-target-cpu" "at90s8515"
// CHECK2: clang{{.*}} "-cc1as" {{.*}} "-target-cpu" "at90s8515"

// RUN: %clang -### -target avr -no-canonical-prefixes -mmcu=attiny13a -save-temps %s 2>&1 | FileCheck -check-prefix=CHECK3 %s
// CHECK3: clang{{.*}} "-cc1" {{.*}} "-target-cpu" "attiny13a"
// CHECK3: clang{{.*}} "-cc1as" {{.*}} "-target-cpu" "attiny13a"

// RUN: %clang -### -target avr -no-canonical-prefixes -mmcu=attiny88 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECK4 %s
// CHECK4: clang{{.*}} "-cc1" {{.*}} "-target-cpu" "attiny88"
// CHECK4: clang{{.*}} "-cc1as" {{.*}} "-target-cpu" "attiny88"

// RUN: %clang -### -target avr -no-canonical-prefixes -mmcu=attiny88 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECK5 %s
// CHECK5: clang{{.*}} "-cc1" {{.*}} "-target-cpu" "attiny88"
// CHECK5: clang{{.*}} "-cc1as" {{.*}} "-target-cpu" "attiny88"

// RUN: %clang -### -target avr -no-canonical-prefixes -mmcu=atmega8u2 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECK6 %s
// CHECK6: clang{{.*}} "-cc1" {{.*}} "-target-cpu" "atmega8u2"
// CHECK6: clang{{.*}} "-cc1as" {{.*}} "-target-cpu" "atmega8u2"

// RUN: %clang -### -target avr -no-canonical-prefixes -mmcu=atmega8u2 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECK7 %s
// CHECK7: clang{{.*}} "-cc1" {{.*}} "-target-cpu" "atmega8u2"
// CHECK7: clang{{.*}} "-cc1as" {{.*}} "-target-cpu" "atmega8u2"

// RUN: %clang -### -target avr -no-canonical-prefixes -mmcu=atmega8a -save-temps %s 2>&1 | FileCheck -check-prefix=CHECK8 %s
// CHECK8: clang{{.*}} "-cc1" {{.*}} "-target-cpu" "atmega8a"
// CHECK8: clang{{.*}} "-cc1as" {{.*}} "-target-cpu" "atmega8a"

// RUN: %clang -### -target avr -no-canonical-prefixes -mmcu=atmega8a -save-temps %s 2>&1 | FileCheck -check-prefix=CHECK9 %s
// CHECK9: clang{{.*}} "-cc1" {{.*}} "-target-cpu" "atmega8a"
// CHECK9: clang{{.*}} "-cc1as" {{.*}} "-target-cpu" "atmega8a"

// RUN: %clang -### -target avr -no-canonical-prefixes -mmcu=atmega16a -save-temps %s 2>&1 | FileCheck -check-prefix=CHECKa %s
// CHECKa: clang{{.*}} "-cc1" {{.*}} "-target-cpu" "atmega16a"
// CHECKa: clang{{.*}} "-cc1as" {{.*}} "-target-cpu" "atmega16a"

// RUN: %clang -### -target avr -no-canonical-prefixes -mmcu=atmega16a -save-temps %s 2>&1 | FileCheck -check-prefix=CHECKb %s
// CHECKb: clang{{.*}} "-cc1" {{.*}} "-target-cpu" "atmega16a"
// CHECKb: clang{{.*}} "-cc1as" {{.*}} "-target-cpu" "atmega16a"

// RUN: %clang -### -target avr -no-canonical-prefixes -mmcu=atmega128a -save-temps %s 2>&1 | FileCheck -check-prefix=CHECKc %s
// CHECKc: clang{{.*}} "-cc1" {{.*}} "-target-cpu" "atmega128a"
// CHECKc: clang{{.*}} "-cc1as" {{.*}} "-target-cpu" "atmega128a"

// RUN: %clang -### -target avr -no-canonical-prefixes -mmcu=atmega2560 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECKd %s
// CHECKd: clang{{.*}} "-cc1" {{.*}} "-target-cpu" "atmega2560"
// CHECKd: clang{{.*}} "-cc1as" {{.*}} "-target-cpu" "atmega2560"

// RUN: %clang -### -target avr -no-canonical-prefixes -mmcu=attiny10 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECKe %s
// CHECKe: clang{{.*}} "-cc1" {{.*}} "-target-cpu" "attiny10"
// CHECKe: clang{{.*}} "-cc1as" {{.*}} "-target-cpu" "attiny10"

// RUN: %clang -### -target avr -no-canonical-prefixes -mmcu=atxmega16a4 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECKf %s
// CHECKf: clang{{.*}} "-cc1" {{.*}} "-target-cpu" "atxmega16a4"
// CHECKf: clang{{.*}} "-cc1as" {{.*}} "-target-cpu" "atxmega16a4"

// RUN: %clang -### -target avr -no-canonical-prefixes -mmcu=atxmega64b1 -save-temps %s 2>&1 | FileCheck -check-prefix=CHECKg %s
// CHECKg: clang{{.*}} "-cc1" {{.*}} "-target-cpu" "atxmega64b1"
// CHECKg: clang{{.*}} "-cc1as" {{.*}} "-target-cpu" "atxmega64b1"

// RUN: %clang -### -target avr -no-canonical-prefixes -mmcu=atxmega64a1u -save-temps %s 2>&1 | FileCheck -check-prefix=CHECKh %s
// CHECKh: clang{{.*}} "-cc1" {{.*}} "-target-cpu" "atxmega64a1u"
// CHECKh: clang{{.*}} "-cc1as" {{.*}} "-target-cpu" "atxmega64a1u"

// RUN: %clang -### -target avr -no-canonical-prefixes -mmcu=atxmega128a3u -save-temps %s 2>&1 | FileCheck -check-prefix=CHECKj %s
// CHECKj: clang{{.*}} "-cc1" {{.*}} "-target-cpu" "atxmega128a3u"
// CHECKj: clang{{.*}} "-cc1as" {{.*}} "-target-cpu" "atxmega128a3u"

// RUN: %clang -### -target avr -no-canonical-prefixes -mmcu=atxmega128a4u -save-temps %s 2>&1 | FileCheck -check-prefix=CHECKi %s
// CHECKi: clang{{.*}} "-cc1" {{.*}} "-target-cpu" "atxmega128a4u"
// CHECKi: clang{{.*}} "-cc1as" {{.*}} "-target-cpu" "atxmega128a4u"
