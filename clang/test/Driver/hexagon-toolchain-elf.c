// -----------------------------------------------------------------------------
// Test standard include paths
// -----------------------------------------------------------------------------

// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK000 %s
// CHECK000: "-cc1" {{.*}} "-internal-externc-isystem" "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/include"

// RUN: %clangxx -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK001 %s
// CHECK001: "-cc1" {{.*}} "-internal-isystem" "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/include/c++"
// CHECK001:   "-internal-externc-isystem" "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/include"

// -----------------------------------------------------------------------------
// Test -nostdinc, -nostdlibinc, -nostdinc++
// -----------------------------------------------------------------------------

// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -nostdinc \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK010 %s
// CHECK010: "-cc1"
// CHECK010-NOT: "-internal-externc-isystem"

// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -nostdlibinc \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK011 %s
// CHECK011: "-cc1"
// CHECK011-NOT: "-internal-externc-isystem"

// RUN: %clangxx -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -nostdinc++ \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK012 %s
// CHECK012: "-cc1"
// CHECK012-NOT: "-internal-isystem"
// CHECK012-DAG: "-internal-externc-isystem" "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/include"

// RUN: %clangxx -### -target hexagon-unknown-elf -fno-integrated-as    \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/qc/bin \
// RUN:   --gcc-toolchain="" \
// RUN:   -nostdlibinc \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK013 %s
// CHECK013: "-cc1"
// CHECK013-NOT: "-internal-isystem"
// CHECK013-NOT: "-internal-externc-isystem"

// -----------------------------------------------------------------------------
// Test -mcpu=<cpuname> -mv<number>
// -----------------------------------------------------------------------------
// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv5 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK021 %s
// CHECK021: "-cc1" {{.*}} "-target-cpu" "hexagonv5"
// CHECK021: {{hexagon-link|ld}}{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v5/crt0

// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv55 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK022 %s
// CHECK022: "-cc1" {{.*}} "-target-cpu" "hexagonv55"
// CHECK022: {{hexagon-link|ld}}{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v55/crt0

// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK023 %s
// CHECK023: "-cc1" {{.*}} "-target-cpu" "hexagonv60"
// CHECK023: {{hexagon-link|ld}}{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/crt0

// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv62 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK024 %s
// CHECK024: "-cc1" {{.*}} "-target-cpu" "hexagonv62"
// CHECK024: {{hexagon-link|ld}}{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v62/crt0

// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv65 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK025 %s
// CHECK025: "-cc1" {{.*}} "-target-cpu" "hexagonv65"
// CHECK025: {{hexagon-link|ld}}{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v65/crt0

// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv66 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK026 %s
// CHECK026: "-cc1" {{.*}} "-target-cpu" "hexagonv66"
// CHECK026: {{hexagon-link|ld}}{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v66/crt0

// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -O3 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK027 %s
// CHECK027-NOT: "-ffp-contract=fast"
// CHECK027: {{hexagon-link|ld}}

// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -O3 -ffp-contract=off \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK028 %s
// CHECK028-NOT: "-ffp-contract=fast"
// CHECK028: {{hexagon-link|ld}}

// -----------------------------------------------------------------------------
// Test Linker related args
// -----------------------------------------------------------------------------

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Defaults for C
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK030 %s
// CHECK030: "-cc1"
// CHECK030: {{hexagon-link|ld}}
// CHECK030-NOT: "-static"
// CHECK030-NOT: "-shared"
// CHECK030: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/crt0_standalone.o"
// CHECK030: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/crt0.o"
// CHECK030: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/init.o"
// CHECK030: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60"
// CHECK030: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib"
// CHECK030: "{{[^"]+}}.o"
// CHECK030: "--start-group" "-lstandalone" "-lc" "-lgcc" "--end-group"
// CHECK030: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/fini.o"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Defaults for C++
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// RUN: %clangxx -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK031 %s
// CHECK031: "-cc1"
// CHECK031: {{hexagon-link|ld}}
// CHECK031-NOT: "-static"
// CHECK031-NOT: "-shared"
// CHECK031: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/crt0_standalone.o"
// CHECK031: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/crt0.o"
// CHECK031: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/init.o"
// CHECK031: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60"
// CHECK031: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib"
// CHECK031: "{{[^"]+}}.o"
// CHECK031: "-lstdc++" "-lm"
// CHECK031: "--start-group" "-lstandalone" "-lc" "-lgcc" "--end-group"
// CHECK031: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/fini.o"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Additional Libraries (-L)
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -Lone -L two -L three \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK032 %s
// CHECK032: "-cc1"
// CHECK032: {{hexagon-link|ld}}
// CHECK032-NOT: "-static"
// CHECK032-NOT: "-shared"
// CHECK032: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/crt0_standalone.o"
// CHECK032: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/crt0.o"
// CHECK032: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/init.o"
// CHECK032: "-Lone" "-Ltwo" "-Lthree"
// CHECK032: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60"
// CHECK032: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib"
// CHECK032: "{{[^"]+}}.o"
// CHECK032: "--start-group" "-lstandalone" "-lc" "-lgcc" "--end-group"
// CHECK032: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/fini.o"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// -static, -shared
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -static \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK033 %s
// CHECK033: "-cc1"
// CHECK033: {{hexagon-link|ld}}
// CHECK033: "-static"
// CHECK033: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/crt0_standalone.o"
// CHECK033: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/crt0.o"
// CHECK033: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/init.o"
// CHECK033: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60"
// CHECK033: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib"
// CHECK033: "{{[^"]+}}.o"
// CHECK033: "--start-group" "-lstandalone" "-lc" "-lgcc" "--end-group"
// CHECK033: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/fini.o"

// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -shared \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK034 %s
// CHECK034: "-cc1"
// CHECK034: {{hexagon-link|ld}}
// CHECK034: "-shared" "-call_shared"
// CHECK034-NOT: crt0_standalone.o
// CHECK034-NOT: crt0.o
// CHECK034: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/G0/pic/initS.o"
// CHECK034: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/G0"
// CHECK034: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60"
// CHECK034: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib"
// CHECK034: "{{[^"]+}}.o"
// CHECK034: "--start-group"
// CHECK034-NOT: "-lstandalone"
// CHECK034-NOT: "-lc"
// CHECK034: "-lgcc"
// CHECK034: "--end-group"
// CHECK034: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/G0/pic/finiS.o"

// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -shared \
// RUN:   -static \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK035 %s
// CHECK035: "-cc1"
// CHECK035: {{hexagon-link|ld}}
// CHECK035: "-shared" "-call_shared" "-static"
// CHECK035-NOT: crt0_standalone.o
// CHECK035-NOT: crt0.o
// CHECK035: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/G0/init.o"
// CHECK035: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/G0"
// CHECK035: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60"
// CHECK035: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib"
// CHECK035: "{{[^"]+}}.o"
// CHECK035: "--start-group"
// CHECK035-NOT: "-lstandalone"
// CHECK035-NOT: "-lc"
// CHECK035: "-lgcc"
// CHECK035: "--end-group"
// CHECK035: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/G0/fini.o"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// -nostdlib, -nostartfiles, -nodefaultlibs
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// RUN: %clangxx -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -nostdlib \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK036 %s
// CHECK036: "-cc1"
// CHECK036: {{hexagon-link|ld}}
// CHECK036-NOT: crt0_standalone.o
// CHECK036-NOT: crt0.o
// CHECK036-NOT: init.o
// CHECK036: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60"
// CHECK036: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib"
// CHECK036: "{{[^"]+}}.o"
// CHECK036-NOT: "-lstdc++"
// CHECK036-NOT: "-lm"
// CHECK036-NOT: "--start-group"
// CHECK036-NOT: "-lstandalone"
// CHECK036-NOT: "-lc"
// CHECK036-NOT: "-lgcc"
// CHECK036-NOT: "--end-group"
// CHECK036-NOT: fini.o

// RUN: %clangxx -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -nostartfiles \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK037 %s
// CHECK037: "-cc1"
// CHECK037: {{hexagon-link|ld}}
// CHECK037-NOT: crt0_standalone.o
// CHECK037-NOT: crt0.o
// CHECK037-NOT: init.o
// CHECK037: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60"
// CHECK037: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib"
// CHECK037: "{{[^"]+}}.o"
// CHECK037: "-lstdc++"
// CHECK037: "-lm"
// CHECK037: "--start-group"
// CHECK037: "-lstandalone"
// CHECK037: "-lc"
// CHECK037: "-lgcc"
// CHECK037: "--end-group"
// CHECK037-NOT: fini.o

// RUN: %clangxx -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -nodefaultlibs \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK038 %s
// CHECK038: "-cc1"
// CHECK038: {{hexagon-link|ld}}
// CHECK038: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/crt0_standalone.o"
// CHECK038: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/crt0.o"
// CHECK038: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/init.o"
// CHECK038: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60"
// CHECK038: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib"
// CHECK038: "{{[^"]+}}.o"
// CHECK038-NOT: "-lstdc++"
// CHECK038-NOT: "-lm"
// CHECK038-NOT: "--start-group"
// CHECK038-NOT: "-lstandalone"
// CHECK038-NOT: "-lc"
// CHECK038-NOT: "-lgcc"
// CHECK038-NOT: "--end-group"
// CHECK038: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/fini.o"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// -moslib
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -moslib=first -moslib=second \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK039 %s
// CHECK039: "-cc1"
// CHECK039: {{hexagon-link|ld}}
// CHECK039-NOT: "-static"
// CHECK039-NOT: "-shared"
// CHECK039-NOT: crt0_standalone.o
// CHECK039: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/crt0.o"
// CHECK039: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/init.o"
// CHECK039: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60"
// CHECK039: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib"
// CHECK039: "{{[^"]+}}.o"
// CHECK039: "--start-group"
// CHECK039: "-lfirst" "-lsecond"
// CHECK039-NOT: "-lstandalone"
// CHECK039: "-lc" "-lgcc" "--end-group"
// CHECK039: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/fini.o"

// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -moslib=first -moslib=second -moslib=standalone \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK03A %s
// CHECK03A: "-cc1"
// CHECK03A: {{hexagon-link|ld}}
// CHECK03A-NOT: "-static"
// CHECK03A-NOT: "-shared"
// CHECK03A: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/crt0_standalone.o"
// CHECK03A: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/crt0.o"
// CHECK03A: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/init.o"
// CHECK03A: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60"
// CHECK03A: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib"
// CHECK03A: "{{[^"]+}}.o"
// CHECK03A: "--start-group"
// CHECK03A: "-lfirst" "-lsecond"
// CHECK03A: "-lstandalone"
// CHECK03A: "-lc" "-lgcc" "--end-group"
// CHECK03A: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/fini.o"

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Other args to pass to linker
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// RUN: %clangxx -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -s \
// RUN:   -Tbss 0xdead -Tdata 0xbeef -Ttext 0xcafe \
// RUN:   -t \
// RUN:   -e start_here \
// RUN:   -uFoo -undefined Bar \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK03B %s
// CHECK03B: "-cc1"
// CHECK03B: {{hexagon-link|ld}}
// CHECK03B: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/crt0_standalone.o"
// CHECK03B: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/crt0.o"
// CHECK03B: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/init.o"
// CHECK03B: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60"
// CHECK03B: "-L{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib"
// CHECK03B: "-s"
// CHECK03B: "-Tbss" "0xdead" "-Tdata" "0xbeef" "-Ttext" "0xcafe"
// CHECK03B: "-t"
// CHECK03B: "-u" "Foo" "-undefined" "Bar"
// CHECK03B: "{{[^"]+}}.o"
// CHECK03B: "-lstdc++" "-lm"
// CHECK03B: "--start-group" "-lstandalone" "-lc" "-lgcc" "--end-group"
// CHECK03B: "{{.*}}/Inputs/hexagon_tree/Tools/bin/../target/hexagon/lib/v60/fini.o"

// -----------------------------------------------------------------------------
// pic, small data threshold
// -----------------------------------------------------------------------------
// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK040 %s
// CHECK040:      "-cc1"
// CHECK040: {{hexagon-link|ld}}
// CHECK040-NOT:  "-G{{[0-9]+}}"

// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -fpic \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK041 %s
// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -fPIC \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK041 %s
// CHECK041:      "-cc1"
// CHECK041-NOT:  "-mrelocation-model" "static"
// CHECK041:      "-pic-level" "{{[12]}}"
// CHECK041:      "-mllvm" "-hexagon-small-data-threshold=0"
// CHECK041: {{hexagon-link|ld}}
// CHECK041:      "-G0"

// RUN: %clang -### -target hexagon-unknown-elf -fno-integrated-as \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -G=8 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK042 %s
// RUN: %clang -### -target hexagon-unknown-elf -fno-integrated-as \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -G 8 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK042 %s
// RUN: %clang -### -target hexagon-unknown-elf -fno-integrated-as \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -msmall-data-threshold=8 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK042 %s
// CHECK042:      "-cc1"
// CHECK042:      "-mrelocation-model" "static"
// CHECK042:      "-mllvm" "-hexagon-small-data-threshold=8"
// CHECK042-NEXT: llvm-mc
// CHECK042:      "-gpsize=8"
// CHECK042: {{hexagon-link|ld}}
// CHECK042:      "-G8"

// -----------------------------------------------------------------------------
// pie
// -----------------------------------------------------------------------------
// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -pie \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK050 %s
// CHECK050:      "-cc1"
// CHECK050:      {{hexagon-link|ld}}
// CHECK050:      "-pie"

// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -pie -shared \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK051 %s
// CHECK051:      "-cc1"
// CHECK051       {{hexagon-link|ld}}
// CHECK051-NOT:  "-pie"

// -----------------------------------------------------------------------------
// Test Assembler related args
// -----------------------------------------------------------------------------
// RUN: %clang -### -target hexagon-unknown-elf -fno-integrated-as    \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   -gdwarf-2 \
// RUN:   -Wa,--noexecstack,--trap \
// RUN:   -Xassembler --keep-locals \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK060 %s
// CHECK060:      "-cc1"
// CHECK060-NEXT: llvm-mc
// CHECK060:      "--noexecstack" "--trap" "--keep-locals"
// CHECK060       {{hexagon-link|ld}}

// -----------------------------------------------------------------------------
// ffixed-r19
// -----------------------------------------------------------------------------
// RUN: %clang -### -target hexagon-unknown-elf -ffixed-r19 %s 2>&1 \
// RUN:        | FileCheck --check-prefix=CHECK070 %s
// CHECK070: "-target-feature" "+reserved-r19"
// RUN: %clang -### -target hexagon-unknown-elf %s 2>&1 \
// RUN:        | FileCheck --check-prefix=CHECK071 %s
// CHECK071-NOT: "+reserved-r19"

// -----------------------------------------------------------------------------
// Misc Defaults
// -----------------------------------------------------------------------------
// RUN: %clang -### -target hexagon-unknown-elf \
// RUN:   -ccc-install-dir %S/Inputs/hexagon_tree/Tools/bin \
// RUN:   -mcpu=hexagonv60 \
// RUN:   %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK080 %s
// CHECK080:      "-cc1"
// CHECK080:      "-Wreturn-type"
