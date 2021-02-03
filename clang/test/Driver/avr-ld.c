// RUN: %clang -### --target=avr -mmcu=at90s2313 --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKA %s
// LINKA: {{".*ld.*"}} {{.*}} {{"-L.*tiny-stack"}} {{.*}} "-Tdata=0x800060" {{.*}} "-lat90s2313" "-mavr2"

// RUN: %clang -### --target=avr -mmcu=at90s8515 --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKB %s
// LINKB: {{".*ld.*"}} {{.*}} "-Tdata=0x800060" {{.*}} "-lat90s8515" "-mavr2"

// RUN: %clang -### --target=avr -mmcu=attiny13 --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKC %s
// LINKC: {{".*ld.*"}} {{.*}} {{"-L.*avr25/tiny-stack"}} {{.*}} "-Tdata=0x800060" {{.*}} "-lattiny13" "-mavr25"

// RUN: %clang -### --target=avr -mmcu=attiny44 --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKD %s
// LINKD: {{".*ld.*"}} {{.*}} {{"-L.*avr25"}} {{.*}} "-Tdata=0x800060" {{.*}} "-lattiny44" "-mavr25"

// RUN: %clang -### --target=avr -mmcu=atmega103 --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKE %s
// LINKE: {{".*ld.*"}} {{.*}} {{"-L.*avr31"}} {{.*}} "-Tdata=0x800060" {{.*}} "-latmega103" "-mavr31"

// RUN: %clang -### --target=avr -mmcu=atmega8u2 --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKF %s
// LINKF: {{".*ld.*"}} {{.*}} {{"-L.*avr35"}} {{.*}} "-Tdata=0x800100" {{.*}} "-latmega8u2" "-mavr35"

// RUN: %clang -### --target=avr -mmcu=atmega48pa --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKG %s
// LINKG: {{".*ld.*"}} {{.*}} {{"-L.*avr4"}} {{.*}} "-Tdata=0x800100" {{.*}} "-latmega48pa" "-mavr4"

// RUN: %clang -### --target=avr -mmcu=atmega328 --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKH %s
// LINKH: {{".*ld.*"}} {{.*}} {{"-L.*avr5"}} {{.*}} "-Tdata=0x800100" {{.*}} "-latmega328" "-mavr5"

// RUN: %clang -### --target=avr -mmcu=atmega1281 --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKI %s
// LINKI: {{".*ld.*"}} {{.*}} {{"-L.*avr51"}} {{.*}} "-Tdata=0x800200" {{.*}} "-latmega1281" "-mavr51"

// RUN: %clang -### --target=avr -mmcu=atmega2560 --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKJ %s
// LINKJ: {{".*ld.*"}} {{.*}} {{"-L.*avr6"}} {{.*}} "-Tdata=0x800200" {{.*}} "-latmega2560" "-mavr6"

// RUN: %clang -### --target=avr -mmcu=attiny10 --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKK %s
// LINKK: {{".*ld.*"}} {{.*}} {{"-L.*avrtiny"}} {{.*}} "-Tdata=0x800040" {{.*}} "-lattiny10" "-mavrtiny"

// RUN: %clang -### --target=avr -mmcu=atxmega16a4 --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKL %s
// LINKL: {{".*ld.*"}} {{.*}} {{"-L.*avrxmega2"}} {{.*}} "-Tdata=0x802000" {{.*}} "-latxmega16a4" "-mavrxmega2"

// RUN: %clang -### --target=avr -mmcu=atxmega64b3 --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKM %s
// LINKM: {{".*ld.*"}} {{.*}} {{"-L.*avrxmega4"}} {{.*}} "-Tdata=0x802000" {{.*}} "-latxmega64b3" "-mavrxmega4"

// RUN: %clang -### --target=avr -mmcu=atxmega128a3u --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKN %s
// LINKN: {{".*ld.*"}} {{.*}} {{"-L.*avrxmega6"}} {{.*}} "-Tdata=0x802000" {{.*}} "-latxmega128a3u" "-mavrxmega6"

// RUN: %clang -### --target=avr -mmcu=atxmega128a1 --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKO %s
// LINKO: {{".*ld.*"}} {{.*}} {{"-L.*avrxmega7"}} {{.*}} "-Tdata=0x802000" {{.*}} "-latxmega128a1" "-mavrxmega7"
