// Check to make sure clang is somewhat picky about -g options.
// rdar://10383444

// Linux.
// RUN: %clang -### -c -g %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G_LIMITED -check-prefix=G_GDB %s
// RUN: %clang -### -c -g2 %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G_LIMITED -check-prefix=G_GDB %s
// RUN: %clang -### -c -g3 %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G_LIMITED -check-prefix=G_GDB %s
// RUN: %clang -### -c -ggdb %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G_LIMITED -check-prefix=G_GDB %s
// RUN: %clang -### -c -ggdb1 %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_ONLY -check-prefix=G_GDB %s
// RUN: %clang -### -c -ggdb3 %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G_LIMITED -check-prefix=G_GDB %s
// RUN: %clang -### -c -glldb %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G_STANDALONE -check-prefix=G_LLDB %s
// RUN: %clang -### -c -gsce %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G_LIMITED -check-prefix=G_SCE %s
// RUN: %clang -### -c -gdbx %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G_LIMITED -check-prefix=G_DBX %s

// Android.
// Android should always generate DWARF4.
// RUN: %clang -### -c -g %s -target arm-linux-androideabi 2>&1 \
// RUN:             | FileCheck -check-prefix=G_LIMITED -check-prefix=G_DWARF4 %s

// Darwin.
// RUN: %clang -### -c -g %s -target x86_64-apple-darwin14 2>&1 \
// RUN:             | FileCheck -check-prefix=G_STANDALONE \
// RUN:                         -check-prefix=G_DWARF2 \
// RUN:                         -check-prefix=G_LLDB %s
// RUN: %clang -### -c -g %s -target x86_64-apple-darwin16 2>&1 \
// RUN:             | FileCheck -check-prefix=G_STANDALONE \
// RUN:                         -check-prefix=G_DWARF4 \
// RUN:                         -check-prefix=G_LLDB %s
// RUN: %clang -### -c -g2 %s -target x86_64-apple-darwin16 2>&1 \
// RUN:             | FileCheck -check-prefix=G_STANDALONE \
// RUN:                         -check-prefix=G_DWARF4 %s
// RUN: %clang -### -c -g3 %s -target x86_64-apple-darwin16 2>&1 \
// RUN:             | FileCheck -check-prefix=G_STANDALONE \
// RUN:                         -check-prefix=G_DWARF4 %s
// RUN: %clang -### -c -ggdb %s -target x86_64-apple-darwin16 2>&1 \
// RUN:             | FileCheck -check-prefix=G_STANDALONE \
// RUN:                         -check-prefix=G_DWARF4 \
// RUN:                         -check-prefix=G_GDB %s
// RUN: %clang -### -c -ggdb1 %s -target x86_64-apple-darwin16 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_ONLY %s
// RUN: %clang -### -c -ggdb3 %s -target x86_64-apple-darwin16 2>&1 \
// RUN:             | FileCheck -check-prefix=G_STANDALONE \
// RUN:                         -check-prefix=G_DWARF4 %s
// RUN: %clang -### -c -g %s -target x86_64-apple-macosx10.11 2>&1 \
// RUN:             | FileCheck -check-prefix=G_STANDALONE \
// RUN:                         -check-prefix=G_DWARF4 %s
// RUN: %clang -### -c -g %s -target x86_64-apple-macosx10.10 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY_DWARF2 %s
// RUN: %clang -### -c -g %s -target armv7-apple-ios9.0 2>&1 \
// RUN:             | FileCheck -check-prefix=G_STANDALONE \
// RUN:                         -check-prefix=G_DWARF4 %s
// RUN: %clang -### -c -g %s -target armv7-apple-ios8.0 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY_DWARF2 %s
// RUN: %clang -### -c -g %s -target armv7k-apple-watchos 2>&1 \
// RUN:             | FileCheck -check-prefix=G_STANDALONE \
// RUN:                         -check-prefix=G_DWARF4 %s
// RUN: %clang -### -c -g %s -target arm64-apple-tvos9.0 2>&1 \
// RUN:             | FileCheck -check-prefix=G_STANDALONE \
// RUN:                         -check-prefix=G_DWARF4 %s
// RUN: %clang -### -c -g %s -target x86_64-apple-driverkit19.0 2>&1 \
// RUN:             | FileCheck -check-prefix=G_STANDALONE \
// RUN:                         -check-prefix=G_DWARF4 %s
// RUN: %clang -### -c -fsave-optimization-record %s \
// RUN:        -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_ONLY %s
// RUN: %clang -### -c -g -fsave-optimization-record %s \
// RUN:        -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck -check-prefix=G_STANDALONE %s

// FreeBSD.
// RUN: %clang -### -c -g %s -target x86_64-pc-freebsd11.0 2>&1 \
// RUN:             | FileCheck -check-prefix=G_GDB \
// RUN:                         -check-prefix=G_DWARF2 %s
// RUN: %clang -### -c -g %s -target x86_64-pc-freebsd12.0 2>&1 \
// RUN:             | FileCheck -check-prefix=G_GDB \
// RUN:                         -check-prefix=G_DWARF4 %s

// Windows.
// RUN: %clang -### -c -g %s -target x86_64-w64-windows-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G_GDB %s
// RUN: %clang -### -c -g %s -target x86_64-windows-msvc 2>&1 \
// RUN:             | FileCheck -check-prefix=G_NOTUNING %s
// RUN: %clang_cl -### -c -Z7 -target x86_64-windows-msvc -- %s 2>&1 \
// RUN:             | FileCheck -check-prefix=G_NOTUNING %s

// On the PS4/PS5, -g defaults to -gno-column-info, and we always generate the
// arange section.
// RUN: %clang -### -c %s -target x86_64-scei-ps4 2>&1 \
// RUN:             | FileCheck -check-prefix=NOG_PS %s
// RUN: %clang -### -c %s -target x86_64-sie-ps5 2>&1 \
// RUN:             | FileCheck -check-prefix=NOG_PS %s
/// PS4 will stay on v4 even if the generic default version changes.
// RUN: %clang -### -c %s -g -target x86_64-scei-ps4 2>&1 \
// RUN:             | FileCheck -check-prefixes=G_DWARF4,GARANGE,G_SCE,NOCI,FWD_TMPL_PARAMS %s
// RUN: %clang -### -c %s -g -target x86_64-sie-ps5 2>&1 \
// RUN:             | FileCheck -check-prefixes=G_DWARF5,GARANGE,G_SCE,NOCI,FWD_TMPL_PARAMS %s
// RUN: %clang -### -c %s -g -gcolumn-info -target x86_64-scei-ps4 2>&1 \
// RUN:             | FileCheck -check-prefix=CI %s
// RUN: %clang -### -c %s -gsce -target x86_64-unknown-linux 2>&1 \
// RUN:             | FileCheck -check-prefix=NOCI %s

// On the AIX, -g defaults to -gdbx and limited debug info.
// RUN: %clang -### -c -g %s -target powerpc-ibm-aix-xcoff 2>&1 \
// RUN:             | FileCheck -check-prefix=G_LIMITED -check-prefix=G_DBX %s
// RUN: %clang -### -c -g %s -target powerpc64-ibm-aix-xcoff 2>&1 \
// RUN:             | FileCheck -check-prefix=G_LIMITED -check-prefix=G_DBX %s

// For DBX, -g defaults to -gstrict-dwarf.
// RUN: %clang -### -c -g %s -target powerpc-ibm-aix-xcoff 2>&1 \
// RUN:             | FileCheck -check-prefix=STRICT %s
// RUN: %clang -### -c -g %s -target powerpc64-ibm-aix-xcoff 2>&1 \
// RUN:             | FileCheck -check-prefix=STRICT %s
// RUN: %clang -### -c -g -gno-strict-dwarf %s -target powerpc-ibm-aix-xcoff \
// RUN:             2>&1 | FileCheck -check-prefix=NOSTRICT %s
// RUN: %clang -### -c -g %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=NOSTRICT %s
// RUN: %clang -### -c -g -ggdb %s -target powerpc-ibm-aix-xcoff 2>&1 \
// RUN:             | FileCheck -check-prefix=NOSTRICT %s

// On the AIX, -g defaults to -gno-column-info.
// RUN: %clang -### -c -g %s -target powerpc-ibm-aix-xcoff 2>&1 \
// RUN:             | FileCheck -check-prefix=NOCI %s
// RUN: %clang -### -c -g %s -target powerpc64-ibm-aix-xcoff 2>&1 \
// RUN:             | FileCheck -check-prefix=NOCI %s
// RUN: %clang -### -c -g %s -target powerpc-ibm-aix-xcoff -gcolumn-info 2>&1 \
// RUN:             | FileCheck -check-prefix=CI %s
// RUN: %clang -### -c -g %s -target powerpc64-ibm-aix-xcoff -gcolumn-info \
// RUN:             2>&1 | FileCheck -check-prefix=CI %s

// WebAssembly.
// WebAssembly should default to DWARF4.
// RUN: %clang -### -c -g %s -target wasm32 2>&1 \
// RUN:             | FileCheck -check-prefix=G_DWARF4 %s
// RUN: %clang -### -c -g %s -target wasm64 2>&1 \
// RUN:             | FileCheck -check-prefix=G_DWARF4 %s

// RUN: %clang -### -c -gdwarf-2 %s 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY_DWARF2 %s
//
// RUN: not %clang -### -c -gfoo %s 2>&1 | FileCheck -check-prefix=G_ERR %s
// RUN: %clang -### -c -g -g0 %s 2>&1 | FileCheck -check-prefix=G_NO %s
// RUN: %clang -### -c -ggdb0 %s 2>&1 | FileCheck -check-prefix=G_NO %s
// RUN: %clang -### -c -glldb -g0 %s 2>&1 | FileCheck -check-prefix=G_NO %s
// RUN: %clang -### -c -glldb -g1 %s 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_ONLY -check-prefix=G_LLDB %s
//
// PS4 defaults to sce; -ggdb0 changes tuning but turns off debug info,
// then -g turns it back on without affecting tuning.
// RUN: %clang -### -c -ggdb0 -g -target x86_64-scei-ps4 %s 2>&1 \
// RUN:             | FileCheck -check-prefix=G_GDB %s
//
// RUN: %clang -### -c -g1 %s 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_ONLY %s
// RUN: %clang -### -c -gmlt %s 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_ONLY %s
// RUN: %clang -### -c -gline-tables-only %s 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_ONLY %s
// RUN: %clang -### -c -gline-tables-only %s -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_ONLY %s
// RUN: %clang -### -c -gline-tables-only %s -target i686-pc-openbsd 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_ONLY_DWARF2 %s
// RUN: %clang -### -c -gline-tables-only %s -target x86_64-pc-freebsd10.0 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_ONLY_DWARF2 %s
// RUN: %clang -### -c -gline-tables-only -g %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY %s
// RUN: %clang -### -c -gline-tables-only -g %s -target x86_64-apple-darwin16 2>&1 \
// RUN:             | FileCheck -check-prefix=G_STANDALONE -check-prefix=G_DWARF4 %s
// RUN: %clang -### -c -gline-tables-only -g %s -target i686-pc-openbsd 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY_DWARF2 %s
// RUN: %clang -### -c -gline-tables-only -g %s -target x86_64-pc-freebsd10.0 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY_DWARF2 %s
// RUN: %clang -### -c -gline-tables-only -g %s -target i386-pc-solaris 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY_DWARF2 %s
// RUN: %clang -### -c -gline-tables-only -g0 %s 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_NO %s
//
// RUN: %clang -### -c -gline-directives-only %s -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck -check-prefix=GLIO_ONLY %s
// RUN: %clang -### -c -gline-directives-only %s -target i686-pc-openbsd 2>&1 \
// RUN:             | FileCheck -check-prefix=GLIO_ONLY_DWARF2 %s
// RUN: %clang -### -c -gline-directives-only %s -target x86_64-pc-freebsd10.0 2>&1 \
// RUN:             | FileCheck -check-prefix=GLIO_ONLY_DWARF2 %s
// RUN: %clang -### -c -gline-directives-only -g %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY %s
// RUN: %clang -### -c -gline-directives-only -g %s -target x86_64-apple-darwin16 2>&1 \
// RUN:             | FileCheck -check-prefix=G_STANDALONE -check-prefix=G_DWARF4 %s
// RUN: %clang -### -c -gline-directives-only -g %s -target i686-pc-openbsd 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY_DWARF2 %s
// RUN: %clang -### -c -gline-directives-only -g %s -target x86_64-pc-freebsd10.0 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY_DWARF2 %s
// RUN: %clang -### -c -gline-directives-only -g %s -target i386-pc-solaris 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY_DWARF2 %s
// RUN: %clang -### -c -gline-directives-only -g0 %s 2>&1 \
// RUN:             | FileCheck -check-prefix=GLIO_NO %s
//
// RUN: %clang -### -c -grecord-gcc-switches %s 2>&1 \
//             | FileCheck -check-prefix=GRECORD %s
// RUN: %clang -### -c -gno-record-gcc-switches %s 2>&1 \
//             | FileCheck -check-prefix=GNO_RECORD %s
// RUN: %clang -### -c -grecord-gcc-switches -gno-record-gcc-switches %s 2>&1 \
//             | FileCheck -check-prefix=GNO_RECORD %s/
// RUN: %clang -### -c -grecord-gcc-switches -o - %s 2>&1 \
//             | FileCheck -check-prefix=GRECORD_O %s
// RUN: %clang -### -c -O3 -ffunction-sections -grecord-gcc-switches %s 2>&1 \
//             | FileCheck -check-prefix=GRECORD_OPT %s
//
// RUN: %clang -### -c -grecord-command-line %s 2>&1 \
//             | FileCheck -check-prefix=GRECORD %s
// RUN: %clang -### -c -gno-record-command-line %s 2>&1 \
//             | FileCheck -check-prefix=GNO_RECORD %s
// RUN: %clang -### -c -grecord-command-line -gno-record-command-line %s 2>&1 \
//             | FileCheck -check-prefix=GNO_RECORD %s/
// RUN: %clang -### -c -grecord-command-line -o - %s 2>&1 \
//             | FileCheck -check-prefix=GRECORD_O %s
// RUN: %clang -### -c -O3 -ffunction-sections -grecord-command-line %s 2>&1 \
//             | FileCheck -check-prefix=GRECORD_OPT %s
//
// RUN: %clang -### -c -gstrict-dwarf -gno-strict-dwarf %s 2>&1 \
// RUN:        | FileCheck -check-prefix=GIGNORE %s
//
// RUN: %clang -### -c -ggnu-pubnames %s 2>&1 | FileCheck -check-prefix=GPUB %s
// RUN: %clang -### -c -ggdb %s 2>&1 | FileCheck -check-prefix=NOPUB %s
// RUN: %clang -### -c -ggnu-pubnames -gno-gnu-pubnames %s 2>&1 | FileCheck -check-prefix=NOPUB %s
// RUN: %clang -### -c -ggnu-pubnames -gno-pubnames %s 2>&1 | FileCheck -check-prefix=NOPUB %s
//
// RUN: %clang -### -c -gpubnames %s 2>&1 | FileCheck -check-prefix=PUB %s
// RUN: %clang -### -c -ggdb %s 2>&1 | FileCheck -check-prefix=NOPUB %s
// RUN: %clang -### -c -gpubnames -gno-gnu-pubnames %s 2>&1 | FileCheck -check-prefix=NOPUB %s
// RUN: %clang -### -c -gpubnames -gno-pubnames %s 2>&1 | FileCheck -check-prefix=NOPUB %s
//
// RUN: %clang -### -c -gsplit-dwarf -g -gno-pubnames %s 2>&1 | FileCheck -check-prefix=NOPUB %s
//
// RUN: %clang -### -c -fdebug-ranges-base-address %s 2>&1 | FileCheck -check-prefix=RNGBSE %s
// RUN: %clang -### -c %s 2>&1 | FileCheck -check-prefix=NORNGBSE %s
// RUN: %clang -### -c -fdebug-ranges-base-address -fno-debug-ranges-base-address %s 2>&1 | FileCheck -check-prefix=NORNGBSE %s
//
// RUN: %clang -### -c -glldb %s 2>&1 | FileCheck -check-prefix=NOPUB %s
// RUN: %clang -### -c -glldb -gno-pubnames %s 2>&1 | FileCheck -check-prefix=NOPUB %s
//
// RUN: %clang -### -c -gdwarf-aranges %s 2>&1 | FileCheck -check-prefix=GARANGE %s
//
// RUN: %clang -### -fdebug-types-section -target x86_64-unknown-linux %s 2>&1 \
// RUN:        | FileCheck -check-prefix=FDTS %s
//
// RUN: %clang -### -fdebug-types-section -fno-debug-types-section -target x86_64-unknown-linux %s 2>&1 \
// RUN:        | FileCheck -check-prefix=NOFDTS %s
//
// RUN: %clang -### -fdebug-types-section -target wasm32-unknown-unknown %s 2>&1 \
// RUN:        | FileCheck -check-prefix=FDTS %s
//
// RUN: %clang -### -fdebug-types-section -target x86_64-apple-darwin %s 2>&1 \
// RUN:        | FileCheck -check-prefix=FDTSE %s
//
// RUN: %clang -### -fdebug-types-section -fno-debug-types-section -target x86_64-apple-darwin %s 2>&1 \
// RUN:        | FileCheck -check-prefix=NOFDTSE %s
//
// RUN: %clang -### -g -gno-column-info %s 2>&1 \
// RUN:        | FileCheck -check-prefix=NOCI %s
//
// RUN: %clang -### -g -target x86_64-unknown-unknown %s 2>&1 \
//             | FileCheck -check-prefix=CI %s
//
// RUN: %clang -### -gmodules %s 2>&1 \
// RUN:        | FileCheck -check-prefix=GEXTREFS %s
//
// RUN: %clang -### -gmodules -g %s 2>&1 \
// RUN:        | FileCheck -check-prefix=GEXTREFS %s
//
// RUN: %clang -### -gline-tables-only -gmodules %s 2>&1 \
// RUN:        | FileCheck -check-prefix=GEXTREFS %s
//
// RUN: %clang -### -gmodules -gline-tables-only %s 2>&1 \
// RUN:        | FileCheck -check-prefix=GLTO_ONLY %s
//
// RUN: %clang -### -target %itanium_abi_triple -gmodules -gline-directives-only %s 2>&1 \
// RUN:        | FileCheck -check-prefix=GLIO_ONLY %s
//
// NOG_PS: "-cc1"
// NOG_PS-NOT: "-dwarf-version=
// NOG_PS: "-generate-arange-section"
// NOG_PS-NOT: "-dwarf-version=
//
// G_ERR: error: unknown argument:
//
// G_NO: "-cc1"
// G_NO-NOT: -debug-info-kind=
//
// GLTO_ONLY: "-cc1"
// GLTO_ONLY-NOT: "-dwarf-ext-refs"
// GLTO_ONLY: "-debug-info-kind=line-tables-only"
// GLTO_ONLY-NOT: "-dwarf-ext-refs"
//
// GLTO_ONLY_DWARF2: "-cc1"
// GLTO_ONLY_DWARF2: "-debug-info-kind=line-tables-only"
// GLTO_ONLY_DWARF2: "-dwarf-version=2"
//
// GLIO_ONLY: "-cc1"
// GLIO_ONLY-NOT: "-dwarf-ext-refs"
// GLIO_ONLY: "-debug-info-kind=line-directives-only"
// GLIO_ONLY-NOT: "-dwarf-ext-refs"
//
// GLIO_ONLY_DWARF2: "-cc1"
// GLIO_ONLY_DWARF2: "-debug-info-kind=line-directives-only"
// GLIO_ONLY_DWARF2: "-dwarf-version=2"
//
// G_ONLY: "-cc1"
// G_ONLY: "-debug-info-kind=constructor"
//
// These tests assert that "-gline-tables-only" "-g" uses the latter,
// but otherwise not caring about the DebugInfoKind.
// G_ONLY_DWARF2: "-cc1"
// G_ONLY_DWARF2: "-debug-info-kind={{standalone|constructor}}"
// G_ONLY_DWARF2: "-dwarf-version=2"
//
// G_STANDALONE: "-cc1"
// G_STANDALONE: "-debug-info-kind=standalone"
// G_LIMITED: "-cc1"
// G_LIMITED: "-debug-info-kind=constructor"
// G_DWARF2: "-dwarf-version=2"
// G_DWARF4-DAG: "-dwarf-version=4"
// G_DWARF5-DAG: "-dwarf-version=5"
//
// G_GDB:  "-debugger-tuning=gdb"
// G_LLDB: "-debugger-tuning=lldb"
// G_SCE-DAG:  "-debugger-tuning=sce"
// G_DBX:  "-debugger-tuning=dbx"
//
// STRICT:  "-gstrict-dwarf"
// NOSTRICT-NOT:  "-gstrict-dwarf"
//
// G_NOTUNING: "-cc1"
// G_NOTUNING-NOT: "-debugger-tuning="
//
// This tests asserts that "-gline-tables-only" "-g0" disables debug info.
// GLTO_NO: "-cc1"
// GLTO_NO-NOT: -debug-info-kind=
//
// This tests asserts that "-gline-directives-only" "-g0" disables debug info.
// GLIO_NO: "-cc1"
// GLIO_NO-NOT: -debug-info-kind=
//
// GRECORD: "-dwarf-debug-flags"
// GRECORD: -### -c -grecord-gcc-switches
//
// GNO_RECORD-NOT: "-dwarf-debug-flags"
// GNO_RECORD-NOT: -### -c -grecord-gcc-switches
//
// GRECORD_O: "-dwarf-debug-flags"
// GRECORD_O: -### -c -grecord-gcc-switches -o -
//
// GRECORD_OPT: -### -c -O3 -ffunction-sections -grecord-gcc-switches
//
// GIGNORE-NOT: "argument unused during compilation"
//
// GPUB: -ggnu-pubnames
// NOPUB-NOT: -ggnu-pubnames
// NOPUB-NOT: -gpubnames
//
// PUB: -gpubnames
//
// RNGBSE: -fdebug-ranges-base-address
// NORNGBSE-NOT: -fdebug-ranges-base-address
//
// GARANGE-DAG: -generate-arange-section
//
// FDTS: "-mllvm" "-generate-type-units"
// FDTSE: error: unsupported option '-fdebug-types-section' for target 'x86_64-apple-darwin'
//
// NOFDTS-NOT: "-mllvm" "-generate-type-units"
// NOFDTSE-NOT: error: unsupported option '-fdebug-types-section' for target 'x86_64-apple-darwin'
//
// CI-NOT: "-gno-column-info"
//
// NOCI-DAG: "-gno-column-info"
//
// GEXTREFS: "-dwarf-ext-refs" "-fmodule-format=obj"
// GEXTREFS: "-debug-info-kind={{standalone|constructor}}"

// RUN: not %clang -cc1 -debug-info-kind=watkind 2>&1 | FileCheck -check-prefix=BADSTRING1 %s
// BADSTRING1: error: invalid value 'watkind' in '-debug-info-kind=watkind'
// RUN: not %clang -cc1 -debugger-tuning=gmodal 2>&1 | FileCheck -check-prefix=BADSTRING2 %s
// BADSTRING2: error: invalid value 'gmodal' in '-debugger-tuning=gmodal'

// RUN: %clang -### -fdebug-macro    %s 2>&1 | FileCheck -check-prefix=MACRO %s
// RUN: %clang -### -fno-debug-macro %s 2>&1 | FileCheck -check-prefix=NOMACRO %s
// RUN: %clang -###                  %s 2>&1 | FileCheck -check-prefix=NOMACRO %s
// MACRO: "-debug-info-macro"
// NOMACRO-NOT: "-debug-info-macro"
//
// RUN: %clang -### -gdwarf-5 -gembed-source %s 2>&1 | FileCheck -check-prefix=GEMBED_5 %s
// RUN: %clang -### -gdwarf-2 -gembed-source %s 2>&1 | FileCheck -check-prefix=GEMBED_2 %s
// RUN: %clang -### -gdwarf-5 -gno-embed-source %s 2>&1 | FileCheck -check-prefix=NOGEMBED_5 %s
// RUN: %clang -### -gdwarf-2 -gno-embed-source %s 2>&1 | FileCheck -check-prefix=NOGEMBED_2 %s
//
// GEMBED_5:  "-gembed-source"
// GEMBED_2:  error: invalid argument '-gembed-source' only allowed with '-gdwarf-5'
// NOGEMBED_5-NOT:  "-gembed-source"
// NOGEMBED_2-NOT:  error: invalid argument '-gembed-source' only allowed with '-gdwarf-5'
//
// RUN: %clang -### -g -fno-eliminate-unused-debug-types -c %s 2>&1 \
// RUN:        | FileCheck -check-prefix=DEBUG_UNUSED_TYPES %s
// DEBUG_UNUSED_TYPES: "-debug-info-kind=unused-types"
// DEBUG_UNUSED_TYPES-NOT: "-debug-info-kind=limited"
// RUN: %clang -### -g -feliminate-unused-debug-types -c %s 2>&1 \
// RUN:        | FileCheck -check-prefix=NO_DEBUG_UNUSED_TYPES %s
// RUN: %clang -### -fno-eliminate-unused-debug-types -g1 -c %s 2>&1 \
// RUN:        | FileCheck -check-prefix=NO_DEBUG_UNUSED_TYPES %s
// NO_DEBUG_UNUSED_TYPES: "-debug-info-kind={{constructor|line-tables-only|standalone}}"
// NO_DEBUG_UNUSED_TYPES-NOT: "-debug-info-kind=unused-types"
//
// RUN: %clang -### -c -gdwarf-5 -gdwarf64 -target x86_64 %s 2>&1 | FileCheck -check-prefix=GDWARF64_ON %s
// RUN: %clang -### -c -gdwarf-4 -gdwarf64 -target x86_64 %s 2>&1 | FileCheck -check-prefix=GDWARF64_ON %s
// RUN: %clang -### -c -gdwarf-3 -gdwarf64 -target x86_64 %s 2>&1 | FileCheck -check-prefix=GDWARF64_ON %s
// RUN: %clang -### -c -gdwarf-2 -gdwarf64 -target x86_64 %s 2>&1 | FileCheck -check-prefix=GDWARF64_VER %s
// RUN: %clang -### -c -gdwarf-4 -gdwarf64 -target x86_64 -target x86_64 %s 2>&1 \
// RUN:       | FileCheck -check-prefix=GDWARF64_ON %s
// RUN: %clang -### -c -gdwarf-4 -gdwarf64 -target i386-linux-gnu %s 2>&1 \
// RUN:       | FileCheck -check-prefix=GDWARF64_32ARCH %s
// RUN: %clang -### -c -gdwarf-4 -gdwarf64 -target x86_64-apple-darwin %s 2>&1 \
// RUN:       | FileCheck -check-prefix=GDWARF64_ELF %s
//
// GDWARF64_ON:  "-gdwarf64"
// GDWARF64_VER:  error: invalid argument '-gdwarf64' only allowed with 'DWARFv3 or greater'
// GDWARF64_32ARCH: error: invalid argument '-gdwarf64' only allowed with '64 bit architecture'
// GDWARF64_ELF: error: invalid argument '-gdwarf64' only allowed with 'ELF platforms'

/// Default to -fno-dwarf-directory-asm for -fno-integrated-as before DWARF v5.
// RUN: %clang -### -target x86_64 -c -gdwarf-2 %s 2>&1 | FileCheck --check-prefix=DIRECTORY %s
// RUN: %clang -### -target x86_64 -c -gdwarf-5 %s 2>&1 | FileCheck --check-prefix=DIRECTORY %s
// RUN: %clang -### -target x86_64 -c -gdwarf-4 -fno-integrated-as %s 2>&1 | FileCheck --check-prefix=NODIRECTORY %s
// RUN: %clang -### -target x86_64 -c -gdwarf-5 -fno-integrated-as %s 2>&1 | FileCheck --check-prefix=DIRECTORY %s

// RUN: %clang -### -target x86_64 -c -gdwarf-4 -fno-dwarf-directory-asm %s 2>&1 | FileCheck --check-prefix=NODIRECTORY %s

// DIRECTORY-NOT: "-fno-dwarf-directory-asm"
// NODIRECTORY: "-fno-dwarf-directory-asm"

// RUN: %clang -### -target x86_64 -c -g -gsimple-template-names %s 2>&1 | FileCheck --check-prefixes=SIMPLE_TMPL_NAMES,FWD_TMPL_PARAMS %s
// SIMPLE_TMPL_NAMES: -gsimple-template-names=simple
// FWD_TMPL_PARAMS-DAG: -debug-forward-template-params
// RUN: not %clang -### -target x86_64 -c -g -gsimple-template-names=mangled %s 2>&1 | FileCheck --check-prefix=MANGLED_TEMP_NAMES %s
// MANGLED_TEMP_NAMES: error: unknown argument: '-gsimple-template-names=mangled'
// RUN: %clang -### -target x86_64 -c -g %s 2>&1 | FileCheck --check-prefix=FULL_TEMP_NAMES --implicit-check-not=debug-forward-template-params %s
// FULL_TEMP_NAMES-NOT: -gsimple-template-names
