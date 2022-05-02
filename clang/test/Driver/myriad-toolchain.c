// RUN: %clang -### --target=sparc-myriad-rtems %s \
// RUN: -ccc-install-dir %S/Inputs/basic_myriad_tree/bin \
// RUN: --gcc-toolchain=%S/Inputs/basic_myriad_tree 2>&1 | FileCheck %s -check-prefix=LINK_WITH_RTEMS
// LINK_WITH_RTEMS: Inputs{{.*}}crti.o
// LINK_WITH_RTEMS: Inputs{{.*}}crtbegin.o
// LINK_WITH_RTEMS: "-L{{.*}}Inputs/basic_myriad_tree/lib/gcc/sparc-myriad-rtems/6.3.0"
// LINK_WITH_RTEMS: "-L{{.*}}Inputs/basic_myriad_tree/bin/../sparc-myriad-rtems/lib"
// LINK_WITH_RTEMS: "--start-group" "-lc" "-lgcc" "-lrtemscpu" "-lrtemsbsp" "--end-group"
// LINK_WITH_RTEMS: Inputs{{.*}}crtend.o
// LINK_WITH_RTEMS: Inputs{{.*}}crtn.o

// RUN: %clang -c -### --target=sparc-myriad-rtems -x c++ %s \
// RUN: -stdlib=libstdc++ --gcc-toolchain=%S/Inputs/basic_myriad_tree 2>&1 | FileCheck %s -check-prefix=COMPILE_CXX
// COMPILE_CXX: "-internal-isystem" "{{.*}}/Inputs/basic_myriad_tree/lib/gcc/sparc-myriad-rtems/6.3.0/../../../../sparc-myriad-rtems/include/c++/6.3.0"
// COMPILE_CXX: "-internal-isystem" "{{.*}}/Inputs/basic_myriad_tree/lib/gcc/sparc-myriad-rtems/6.3.0/../../../../sparc-myriad-rtems/include/c++/6.3.0/sparc-myriad-rtems"
// COMPILE_CXX: "-internal-isystem" "{{.*}}/Inputs/basic_myriad_tree/lib/gcc/sparc-myriad-rtems/6.3.0/../../../../sparc-myriad-rtems/include/c++/6.3.0/backward"

// RUN: %clang -### -E --target=sparc-myriad --sysroot=/yow %s 2>&1 \
// RUN:   | FileCheck %s -check-prefix=SLASH_INCLUDE
// SLASH_INCLUDE: "-isysroot" "/yow" "-internal-isystem" "/yow/include"

// RUN: %clang -### -E --target=sparc-myriad --sysroot=/yow %s -nostdinc 2>&1 \
// RUN:   | FileCheck %s -check-prefix=NO_SLASH_INCLUDE
// NO_SLASH_INCLUDE: "-isysroot" "/yow"
// NO_SLASH_INCLUDE-NOT: "-internal-isystem" "/yow/include"

// RUN: %clang -### --target=what-myriad %s 2>&1 | FileCheck %s -check-prefix=BAD_ARCH
// BAD_ARCH: the target architecture 'what' is not supported by the target 'myriad'

// Ensure that '-target shave' picks a different compiler.
// Also check that '-I' is turned into '-i:' for the assembler.

// Note that since we don't know where movi tools are installed,
// the driver may or may not find a full path to them.
// That is, the 0th argument will be "/path/to/my/moviCompile"
// or just "moviCompile" depending on whether moviCompile is found.
// As such, we test only for a trailing quote in its rendering.
// The same goes for "moviAsm".

// RUN: %clang --target=shave-myriad -mcpu=myriad2.2 -c -### %s -isystem somewhere -Icommon -Wa,-yippee 2>&1 \
// RUN:   | FileCheck %s -check-prefix=MOVICOMPILE
// MOVICOMPILE: moviCompile{{(.exe)?}}" "-S" "-fno-exceptions" "-DMYRIAD2" "-mcpu=myriad2.2" "-isystem" "somewhere" "-I" "common"
// MOVICOMPILE: moviAsm{{(.exe)?}}" "-no6thSlotCompression" "-cv:myriad2.2" "-noSPrefixing" "-a"
// MOVICOMPILE: "-yippee" "-i:somewhere" "-i:common"

// RUN: %clang --target=shave-myriad -c -### %s -DEFINE_ME -UNDEFINE_ME 2>&1 \
// RUN:   | FileCheck %s -check-prefix=DEFINES
// DEFINES: "-D" "EFINE_ME" "-U" "NDEFINE_ME"

// RUN: %clang --target=shave-myriad -c -### %s -Icommon -iquote quotepath -isystem syspath 2>&1 \
// RUN:   | FileCheck %s -check-prefix=INCLUDES
// INCLUDES: "-iquote" "quotepath" "-isystem" "syspath"

// -fno-split-dwarf-inlining is consumed but not passed to moviCompile.
// RUN: %clang --target=shave-myriad -c -### %s -g -fno-inline-functions \
// RUN: -fno-inline-functions-called-once -Os -Wall -MF dep.d -fno-split-dwarf-inlining \
// RUN: -ffunction-sections -Xclang -xclangflag -mllvm -llvm-flag 2>&1 \
// RUN:   | FileCheck %s -check-prefix=PASSTHRU_OPTIONS
// PASSTHRU_OPTIONS: "-g" "-fno-inline-functions" "-fno-inline-functions-called-once"
// PASSTHRU_OPTIONS: "-Os" "-Wall" "-MF" "dep.d" "-ffunction-sections"
// PASSTHRU_OPTIONS: "-Xclang" "-xclangflag" "-mllvm" "-llvm-flag"

// RUN: %clang --target=shave-myriad -c %s -o foo.o -### -MD -MF dep.d 2>&1 \
// RUN:   | FileCheck %s -check-prefix=MDMF
// MDMF: "-S" "-fno-exceptions" "-DMYRIAD2" "-MD" "-MF" "dep.d" "-MT" "foo.o"

// RUN: %clang --target=shave-myriad -std=gnu++11 -mcpu=anothercpu -S %s -o foo.o -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=STDEQ
// STDEQ: "-S" "-fno-exceptions" "-DMYRIAD2" "-std=gnu++11" "-mcpu=anothercpu"

// RUN: %clang --target=shave-myriad -E -Ifoo %s -o foo.i -### 2>&1 \
// RUN:   | FileCheck %s -check-prefix=PREPROCESS
// PREPROCESS: "-E" "-DMYRIAD2" "-I" "foo"

// RUN: %clang -stdlib=platform --target=sparc-myriad -### --driver-mode=g++ %s 2>&1 | FileCheck %s --check-prefix=LIBSTDCXX
// LIBSTDCXX: "-lstdc++" "-lc" "-lgcc"

// RUN: %clang -stdlib=libc++ -### --target=sparcel-myriad -S -x c++ %s 2>&1 | FileCheck %s -check-prefix=LIBCXX
// LIBCXX: "-internal-isystem" "{{.*}}/../include/c++/v1"

// RUN: %clang --target=sparc-myriad -### -nostdlib %s 2>&1 | FileCheck %s --check-prefix=NOSTDLIB
// NOSTDLIB-NOT: crtbegin.o
// NOSTDLIB-NOT: "-lc"

// RUN: %clang -### -c -g %s --target=sparc-myriad 2>&1 | FileCheck -check-prefix=G_SPARC %s
// G_SPARC: "-debug-info-kind=constructor" "-dwarf-version=2"

// RUN: %clang -### -c %s --target=sparc-myriad-rtems -fuse-init-array 2>&1 \
// RUN: | FileCheck -check-prefix=USE-INIT-ARRAY %s
// USE-INIT-ARRAY-NOT: argument unused
