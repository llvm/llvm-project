// RUN: %clang -### --target=avr -mmcu=at90s2313 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKA %s
// LINKA: {{".*ld.*"}} {{.*}} {{"-L.*tiny-stack"}} {{.*}} "--defsym=__DATA_REGION_ORIGIN__=0x800060" "--start-group" {{.*}} "-lat90s2313" {{.*}} "--end-group" "--relax" "-mavr2"

// RUN: %clang -### --target=avr -mmcu=at90s8515 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s -mrelax 2>&1 | FileCheck -check-prefix LINKB %s
// LINKB: {{".*ld.*"}} {{.*}} "--defsym=__DATA_REGION_ORIGIN__=0x800060" "--start-group" {{.*}} "-lat90s8515" {{.*}} "--end-group" "--relax" "-mavr2"

// RUN: %clang -### --target=avr -mmcu=attiny13 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s -mno-relax 2>&1 | FileCheck -check-prefix LINKC %s
// LINKC: {{".*ld.*"}} {{.*}} {{"-L.*avr25/tiny-stack"}} {{.*}} "--defsym=__DATA_REGION_ORIGIN__=0x800060" "--start-group" {{.*}} "-lattiny13" {{.*}} "--end-group" "-mavr25"
// LINLC-NOT: "--relax"

// RUN: %clang -### --target=avr -mmcu=attiny44 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKD %s
// LINKD: {{".*ld.*"}} {{.*}} {{"-L.*avr25"}} {{.*}} "--defsym=__DATA_REGION_ORIGIN__=0x800060" "--start-group" {{.*}} "-lattiny44" {{.*}} "--end-group" "--relax" "-mavr25"

// RUN: %clang -### --target=avr -mmcu=atmega103 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKE %s
// LINKE: {{".*ld.*"}} {{.*}} {{"-L.*avr31"}} {{.*}} "--defsym=__DATA_REGION_ORIGIN__=0x800060" "--start-group" {{.*}} "-latmega103" {{.*}} "--end-group" "--relax" "-mavr31"

// RUN: %clang -### --target=avr -mmcu=atmega8u2 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKF %s
// LINKF: {{".*ld.*"}} {{.*}} {{"-L.*avr35"}} {{.*}} "--defsym=__DATA_REGION_ORIGIN__=0x800100" "--start-group" {{.*}} "-latmega8u2" {{.*}} "--end-group" "--relax" "-mavr35"

// RUN: %clang -### --target=avr -mmcu=atmega48pa --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKG %s
// LINKG: {{".*ld.*"}} {{.*}} {{"-L.*avr4"}} {{.*}} "--defsym=__DATA_REGION_ORIGIN__=0x800100" "--start-group" {{.*}} "-latmega48pa" {{.*}} "--end-group" "--relax" "-mavr4"

// RUN: %clang -### --target=avr -mmcu=atmega328 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKH %s
// LINKH: {{".*ld.*"}} {{.*}} {{"-L.*avr5"}} {{.*}} "--defsym=__DATA_REGION_ORIGIN__=0x800100" "--start-group" {{.*}} "-latmega328" {{.*}} "--end-group" "--relax" "-mavr5"

// RUN: %clang -### --target=avr -mmcu=atmega1281 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKI %s
// LINKI: {{".*ld.*"}} {{.*}} {{"-L.*avr51"}} {{.*}} "--defsym=__DATA_REGION_ORIGIN__=0x800200" "--start-group" {{.*}} "-latmega1281" {{.*}} "--end-group" "--relax" "-mavr51"

// RUN: %clang -### --target=avr -mmcu=atmega2560 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKJ %s
// LINKJ: {{".*ld.*"}} {{.*}} {{"-L.*avr6"}} {{.*}} "--defsym=__DATA_REGION_ORIGIN__=0x800200" "--start-group" {{.*}} "-latmega2560" {{.*}} "--end-group" "--relax" "-mavr6"

// RUN: %clang -### --target=avr -mmcu=attiny10 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKK %s
// LINKK: {{".*ld.*"}} {{.*}} {{"-L.*avrtiny"}} {{.*}} "--defsym=__DATA_REGION_ORIGIN__=0x800040" "--start-group" {{.*}} "-lattiny10" {{.*}} "--end-group" "--relax" "-mavrtiny"

// RUN: %clang -### --target=avr -mmcu=atxmega16a4 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKL %s
// LINKL: {{".*ld.*"}} {{.*}} {{"-L.*avrxmega2"}} {{.*}} "--defsym=__DATA_REGION_ORIGIN__=0x802000" "--start-group" {{.*}} "-latxmega16a4" {{.*}} "--end-group" "--relax" "-mavrxmega2"

// RUN: %clang -### --target=avr -mmcu=atxmega64b3 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKM %s
// LINKM: {{".*ld.*"}} {{.*}} {{"-L.*avrxmega4"}} {{.*}} "--defsym=__DATA_REGION_ORIGIN__=0x802000" "--start-group" {{.*}} "-latxmega64b3" {{.*}} "--end-group" "--relax" "-mavrxmega4"

// RUN: %clang -### --target=avr -mmcu=atxmega128a3u --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKN %s
// LINKN: {{".*ld.*"}} {{.*}} {{"-L.*avrxmega6"}} {{.*}} "--defsym=__DATA_REGION_ORIGIN__=0x802000" "--start-group" {{.*}} "-latxmega128a3u" {{.*}} "--end-group" "--relax" "-mavrxmega6"

// RUN: %clang -### --target=avr -mmcu=atxmega128a1 --rtlib=libgcc --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKO %s
// LINKO: {{".*ld.*"}} {{.*}} {{"-L.*avrxmega7"}} {{.*}} "--defsym=__DATA_REGION_ORIGIN__=0x802000" "--start-group" {{.*}} "-latxmega128a1" {{.*}} "--end-group" "--relax" "-mavrxmega7"

// RUN: %clang -### --target=avr -mmcu=atmega328 -fuse-ld=ld -flto --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck --check-prefix=LINKP %s
// LINKP: {{".*ld.*"}} {{.*}} "--defsym=__DATA_REGION_ORIGIN__=0x800100" "-plugin" {{.*}}  "-plugin-opt=mcpu=atmega328"

// RUN: %clang -### --target=avr -fuse-ld=ld -flto --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck --check-prefix=LINKQ %s
// LINKQ: {{".*ld.*"}} {{.*}} "-plugin"
// LINKQ-NOT: "-plugin-opt=mcpu"

// RUN: %clang -### --target=avr -mmcu=atmega328 -fuse-ld=lld -flto=thin --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKR %s
// LINKR: {{".*ld.*"}} {{.*}} "--defsym=__DATA_REGION_ORIGIN__=0x800100" "-plugin-opt=mcpu=atmega328" "-plugin-opt=thinlto"

// RUN: %clang -### --target=avr -mmcu=atmega328 -fuse-ld=lld -flto --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKS %s
// LINKS: {{".*ld.*"}} {{.*}} "--defsym=__DATA_REGION_ORIGIN__=0x800100" "-plugin-opt=mcpu=atmega328"
// LINKS-NOT: "-plugin-opt=thinlto"

// RUN: %clang -### --target=avr -mmcu=attiny40 -fuse-ld=lld --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKT0 %s
// LINKT0: {{".*lld.*"}} {{.*}} {{"-T.*avrtiny.x"}}
// LINKT0-NOT: "-m

// RUN: %clang -### --target=avr -mmcu=atxmega384c3 -fuse-ld=lld --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKT1 %s
// LINKT1: {{".*lld.*"}} {{.*}} {{"-T.*avrxmega6.x"}}
// LINKT1-NOT: "-m

// RUN: %clang -### --target=avr -mmcu=atmega328 -fuse-ld=lld --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKT2 %s
// LINKT2: {{".*lld.*"}} {{.*}} "--start-group" {{.*}} "--end-group"
// LINKT2-NOT: "-T
// LINKT2-NOT: "-m

// RUN: %clang -### --target=avr -mmcu=attiny40 --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck -check-prefix LINKT3 %s
// LINKT3: {{".*ld.*"}} {{.*}} "-mavrtiny"
// LINKT3-NOT: "-T

// RUN: %clang -### --target=avr -mmcu=attiny40 --sysroot %S/Inputs/basic_avr_tree -fuse-ld=lld -T %S/Inputs/basic_avr_tree/usr/lib/avr/lib/ldscripts/avrxmega6.x %s 2>&1 | FileCheck -check-prefix LINKT4 %s
// LINKT4: {{".*lld.*"}} {{.*}} {{"-T.*avrxmega6.x"}}
// LINKT4-NOT: {{"-T.*avrtiny.x"}}
// LINKT4-NOT: "-m

// RUN: %clang -### -r --target=avr -mmcu=atmega328 --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck --check-prefix=LINKU %s
// LINKU: {{".*ld.*"}} {{.*}} "-r" {{.*}} "-mavr5"
// LINKU-NOT: "--gc-sections"
// LINKU-NOT: "--defsym
// LINKU-NOT: "-l

// RUN: %clang -### -r --target=avr -mmcu=atmega328 -fuse-ld=lld --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck --check-prefix=LINKV %s
// LINKV: {{".*ld.*"}} {{.*}} "-r"
// LINKV-NOT: "--gc-sections"
// LINKV-NOT: "--defsym
// LINKV-NOT: "-l
// LINKV-NOT: "-m

// RUN: %clang -### -r --target=avr -mmcu=atmega328 -lm --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck --check-prefix=LINKW %s
// LINKW: {{".*ld.*"}} {{.*}} "-r" "-lm" {{.*}} "-mavr5"
// LINKW-NOT: "--gc-sections"
// LINKW-NOT: "--defsym

// RUN: %clang -### -r --target=avr --sysroot %S/Inputs/basic_avr_tree %s 2>&1 | FileCheck --check-prefix=LINKX %s
// LINKX: warning: no target microcontroller specified
// LINKX: {{".*ld.*"}} {{.*}} "-r" {{.*}}
// LINKX-NOT: warning: {{.*}} standard library
// LINKX-NOT: warning: {{.*}} data section address
// LINKX-NOT: "--gc-sections"
// LINKX-NOT: "--defsym
// LINKX-NOT: "-l
// LINKX-NOT: "-m
