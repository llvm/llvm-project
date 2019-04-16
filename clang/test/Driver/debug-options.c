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

// FreeBSD.
// RUN: %clang -### -c -g %s -target x86_64-pc-freebsd10.0 2>&1 \
// RUN:             | FileCheck -check-prefix=G_GDB %s

// Windows.
// RUN: %clang -### -c -g %s -target x86_64-w64-windows-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G_GDB %s
// RUN: %clang -### -c -g %s -target x86_64-windows-msvc 2>&1 \
// RUN:             | FileCheck -check-prefix=G_NOTUNING %s
// RUN: %clang_cl -### -c -Z7 -target x86_64-windows-msvc -- %s 2>&1 \
// RUN:             | FileCheck -check-prefix=G_NOTUNING %s

// On the PS4, -g defaults to -gno-column-info, and we always generate the
// arange section.
// RUN: %clang -### -c %s -target x86_64-scei-ps4 2>&1 \
// RUN:             | FileCheck -check-prefix=NOG_PS4 %s
// RUN: %clang -### -c %s -g -target x86_64-scei-ps4 2>&1 \
// RUN:             | FileCheck -check-prefix=G_PS4 %s
// RUN: %clang -### -c %s -g -target x86_64-scei-ps4 2>&1 \
// RUN:             | FileCheck -check-prefix=G_SCE %s
// RUN: %clang -### -c %s -g -target x86_64-scei-ps4 2>&1 \
// RUN:             | FileCheck -check-prefix=NOCI %s
// RUN: %clang -### -c %s -g -gcolumn-info -target x86_64-scei-ps4 2>&1 \
// RUN:             | FileCheck -check-prefix=CI %s
// RUN: %clang -### -c %s -gsce -target x86_64-unknown-linux 2>&1 \
// RUN:             | FileCheck -check-prefix=NOCI %s

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
// RUN:             | FileCheck -check-prefix=G -check-prefix=G_GDB %s
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
// RUN: %clang -### -c -gsplit-dwarf %s 2>&1 | FileCheck -check-prefix=GPUB %s
// RUN: %clang -### -c -gsplit-dwarf -gno-pubnames %s 2>&1 | FileCheck -check-prefix=NOPUB %s
//
// RUN: %clang -### -c -fdebug-ranges-base-address %s 2>&1 | FileCheck -check-prefix=RNGBSE %s
// RUN: %clang -### -c %s 2>&1 | FileCheck -check-prefix=NORNGBSE %s
// RUN: %clang -### -c -fdebug-ranges-base-address -fno-debug-ranges-base-address %s 2>&1 | FileCheck -check-prefix=NORNGBSE %s
//
// RUN: %clang -### -c -glldb %s 2>&1 | FileCheck -check-prefix=GPUB %s
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
// NOG_PS4: "-cc1"
// NOG_PS4-NOT "-dwarf-version=
// NOG_PS4: "-generate-arange-section"
// NOG_PS4-NOT: "-dwarf-version=
//
// G_PS4: "-cc1"
// G_PS4: "-dwarf-version=
// G_PS4: "-generate-arange-section"
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
// G_ONLY: "-debug-info-kind=limited"
//
// These tests assert that "-gline-tables-only" "-g" uses the latter,
// but otherwise not caring about the DebugInfoKind.
// G_ONLY_DWARF2: "-cc1"
// G_ONLY_DWARF2: "-debug-info-kind={{standalone|limited}}"
// G_ONLY_DWARF2: "-dwarf-version=2"
//
// G_STANDALONE: "-cc1"
// G_STANDALONE: "-debug-info-kind=standalone"
// G_LIMITED: "-cc1"
// G_LIMITED: "-debug-info-kind=limited"
// G_DWARF2: "-dwarf-version=2"
// G_DWARF4: "-dwarf-version=4"
//
// G_GDB:  "-debugger-tuning=gdb"
// G_LLDB: "-debugger-tuning=lldb"
// G_SCE:  "-debugger-tuning=sce"
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
// GARANGE: -generate-arange-section
//
// FDTS: "-mllvm" "-generate-type-units"
// FDTSE: error: unsupported option '-fdebug-types-section' for target 'x86_64-apple-darwin'
//
// NOFDTS-NOT: "-mllvm" "-generate-type-units"
// NOFDTSE-NOT: error: unsupported option '-fdebug-types-section' for target 'x86_64-apple-darwin'
//
// CI: "-dwarf-column-info"
//
// NOCI-NOT: "-dwarf-column-info"
//
// GEXTREFS: "-dwarf-ext-refs" "-fmodule-format=obj"
// GEXTREFS: "-debug-info-kind={{standalone|limited}}"

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
