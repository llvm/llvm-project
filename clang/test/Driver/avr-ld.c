// RUN: %clang -### --target=avr -mmcu=at90s2313 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKA %s
// LINKA: {{".*ld.*"}} {{.*}} {{"-L.*tiny-stack"}} {{.*}} "-Tdata=0x800060" "--start-group" {{.*}} "-lat90s2313" {{.*}} "--end-group" "-mavr2"

// RUN: %clang -### --target=avr -mmcu=at90s8515 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKB %s
// LINKB: {{".*ld.*"}} {{.*}} "-Tdata=0x800060" "--start-group" {{.*}} "-lat90s8515" {{.*}} "--end-group" "-mavr2"

// RUN: %clang -### --target=avr -mmcu=attiny13 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKC %s
// LINKC: {{".*ld.*"}} {{.*}} {{"-L.*avr25/tiny-stack"}} {{.*}} "-Tdata=0x800060" "--start-group" {{.*}} "-lattiny13" {{.*}} "--end-group" "-mavr25"

// RUN: %clang -### --target=avr -mmcu=attiny44 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKD %s
// LINKD: {{".*ld.*"}} {{.*}} {{"-L.*avr25"}} {{.*}} "-Tdata=0x800060" "--start-group" {{.*}} "-lattiny44" {{.*}} "--end-group" "-mavr25"

// RUN: %clang -### --target=avr -mmcu=atmega103 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKE %s
// LINKE: {{".*ld.*"}} {{.*}} {{"-L.*avr31"}} {{.*}} "-Tdata=0x800060" "--start-group" {{.*}} "-latmega103" {{.*}} "--end-group" "-mavr31"

// RUN: %clang -### --target=avr -mmcu=atmega8u2 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKF %s
// LINKF: {{".*ld.*"}} {{.*}} {{"-L.*avr35"}} {{.*}} "-Tdata=0x800100" "--start-group" {{.*}} "-latmega8u2" {{.*}} "--end-group" "-mavr35"

// RUN: %clang -### --target=avr -mmcu=atmega48pa --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKG %s
// LINKG: {{".*ld.*"}} {{.*}} {{"-L.*avr4"}} {{.*}} "-Tdata=0x800100" "--start-group" {{.*}} "-latmega48pa" {{.*}} "--end-group" "-mavr4"

// RUN: %clang -### --target=avr -mmcu=atmega328 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKH %s
// LINKH: {{".*ld.*"}} {{.*}} {{"-L.*avr5"}} {{.*}} "-Tdata=0x800100" "--start-group" {{.*}} "-latmega328" {{.*}} "--end-group" "-mavr5"

// RUN: %clang -### --target=avr -mmcu=atmega1281 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKI %s
// LINKI: {{".*ld.*"}} {{.*}} {{"-L.*avr51"}} {{.*}} "-Tdata=0x800200" "--start-group" {{.*}} "-latmega1281" {{.*}} "--end-group" "-mavr51"

// RUN: %clang -### --target=avr -mmcu=atmega2560 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKJ %s
// LINKJ: {{".*ld.*"}} {{.*}} {{"-L.*avr6"}} {{.*}} "-Tdata=0x800200" "--start-group" {{.*}} "-latmega2560" {{.*}} "--end-group" "-mavr6"

// RUN: %clang -### --target=avr -mmcu=attiny10 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKK %s
// LINKK: {{".*ld.*"}} {{.*}} {{"-L.*avrtiny"}} {{.*}} "-Tdata=0x800040" "--start-group" {{.*}} "-lattiny10" {{.*}} "--end-group" "-mavrtiny"

// RUN: %clang -### --target=avr -mmcu=atxmega16a4 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKL %s
// LINKL: {{".*ld.*"}} {{.*}} {{"-L.*avrxmega2"}} {{.*}} "-Tdata=0x802000" "--start-group" {{.*}} "-latxmega16a4" {{.*}} "--end-group" "-mavrxmega2"

// RUN: %clang -### --target=avr -mmcu=atxmega64b3 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKM %s
// LINKM: {{".*ld.*"}} {{.*}} {{"-L.*avrxmega4"}} {{.*}} "-Tdata=0x802000" "--start-group" {{.*}} "-latxmega64b3" {{.*}} "--end-group" "-mavrxmega4"

// RUN: %clang -### --target=avr -mmcu=atxmega128a3u --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKN %s
// LINKN: {{".*ld.*"}} {{.*}} {{"-L.*avrxmega6"}} {{.*}} "-Tdata=0x802000" "--start-group" {{.*}} "-latxmega128a3u" {{.*}} "--end-group" "-mavrxmega6"

// RUN: %clang -### --target=avr -mmcu=atxmega128a1 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKO %s
// LINKO: {{".*ld.*"}} {{.*}} {{"-L.*avrxmega7"}} {{.*}} "-Tdata=0x802000" "--start-group" {{.*}} "-latxmega128a1" {{.*}} "--end-group" "-mavrxmega7"
