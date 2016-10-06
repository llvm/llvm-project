// Check to make sure clang is somewhat picky about -g options.
// rdar://10383444

// RUN: %clang -### -c -g %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G -check-prefix=G_GDB %s
// RUN: %clang -### -c -g2 %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G %s
// RUN: %clang -### -c -g3 %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G %s
// RUN: %clang -### -c -ggdb %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G -check-prefix=G_GDB %s
// RUN: %clang -### -c -ggdb1 %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_ONLY -check-prefix=G_GDB %s
// RUN: %clang -### -c -ggdb3 %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G %s
// RUN: %clang -### -c -glldb %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G -check-prefix=G_LLDB %s
// RUN: %clang -### -c -gsce %s -target x86_64-linux-gnu 2>&1 \
// RUN:             | FileCheck -check-prefix=G -check-prefix=G_SCE %s

// RUN: %clang -### -c -g %s -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck -check-prefix=G_DARWIN -check-prefix=G_LLDB %s
// RUN: %clang -### -c -g2 %s -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck -check-prefix=G_DARWIN %s
// RUN: %clang -### -c -g3 %s -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck -check-prefix=G_DARWIN %s
// RUN: %clang -### -c -ggdb %s -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck -check-prefix=G_DARWIN -check-prefix=G_GDB %s
// RUN: %clang -### -c -ggdb1 %s -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_ONLY %s
// RUN: %clang -### -c -ggdb3 %s -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck -check-prefix=G_DARWIN %s

// RUN: %clang -### -c -g %s -target x86_64-pc-freebsd10.0 2>&1 \
// RUN:             | FileCheck -check-prefix=G_GDB %s

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

// RUN: %clang -### -c -gdwarf-2 %s 2>&1 | FileCheck -check-prefix=G_D2 %s
//
// RUN: %clang -### -c -gfoo %s 2>&1 | FileCheck -check-prefix=G_NO %s
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
// RUN: %clang -### -c -gline-tables-only -g %s -target x86_64-apple-darwin 2>&1 \
// RUN:             | FileCheck -check-prefix=G_STANDALONE_DWARF4 %s
// RUN: %clang -### -c -gline-tables-only -g %s -target i686-pc-openbsd 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY_DWARF2 %s
// RUN: %clang -### -c -gline-tables-only -g %s -target x86_64-pc-freebsd10.0 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY_DWARF2 %s
// RUN: %clang -### -c -gline-tables-only -g %s -target i386-pc-solaris 2>&1 \
// RUN:             | FileCheck -check-prefix=G_ONLY_DWARF2 %s
// RUN: %clang -### -c -gline-tables-only -g0 %s 2>&1 \
// RUN:             | FileCheck -check-prefix=GLTO_NO %s
//
// RUN: %clang -### -c -grecord-gcc-switches -gno-record-gcc-switches \
// RUN:        -gstrict-dwarf -gno-strict-dwarf %s 2>&1 \
// RUN:        | FileCheck -check-prefix=GIGNORE %s
//
// RUN: %clang -### -c -ggnu-pubnames %s 2>&1 | FileCheck -check-prefix=GOPT %s
//
// RUN: %clang -### -c -gdwarf-aranges %s 2>&1 | FileCheck -check-prefix=GARANGE %s
//
// RUN: %clang -### -fdebug-types-section %s 2>&1 \
// RUN:        | FileCheck -check-prefix=FDTS %s
//
// RUN: %clang -### -fdebug-types-section -fno-debug-types-section %s 2>&1 \
// RUN:        | FileCheck -check-prefix=NOFDTS %s
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
// G: "-cc1"
// G: "-debug-info-kind=limited"
//
// G_DARWIN: "-cc1"
// G_DARWIN: "-dwarf-version=4"
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
// G_D2: "-cc1"
// G_D2: "-dwarf-version=2"
//
// G_NO: "-cc1"
// G_NO-NOT: -debug-info-kind=
//
// GLTO_ONLY: "-cc1"
// GLTO_ONLY: "-debug-info-kind=line-tables-only"
//
// GLTO_ONLY_DWARF2: "-cc1"
// GLTO_ONLY_DWARF2: "-debug-info-kind=line-tables-only"
// GLTO_ONLY_DWARF2: "-dwarf-version=2"
//
// G_ONLY: "-cc1"
// G_ONLY: "-debug-info-kind=limited"
//
// G_GDB:  "-debugger-tuning=gdb"
// G_LLDB: "-debugger-tuning=lldb"
// G_SCE:  "-debugger-tuning=sce"
//
// These tests assert that "-gline-tables-only" "-g" uses the latter,
// but otherwise not caring about the DebugInfoKind.
// G_ONLY_DWARF2: "-cc1"
// G_ONLY_DWARF2: "-debug-info-kind={{standalone|limited}}"
// G_ONLY_DWARF2: "-dwarf-version=2"
//
// G_STANDALONE_DWARF4: "-cc1"
// G_STANDALONE_DWARF4: "-debug-info-kind=standalone"
// G_STANDALONE_DWARF4: "-dwarf-version=4"
//
// This tests asserts that "-gline-tables-only" "-g0" disables debug info.
// GLTO_NO: "-cc1"
// GLTO_NO-NOT: -debug-info-kind=
//
// GIGNORE-NOT: "argument unused during compilation"
//
// GOPT: -generate-gnu-dwarf-pub-sections
//
// GARANGE: -generate-arange-section
//
// FDTS: "-backend-option" "-generate-type-units"
//
// NOFDTS-NOT: "-backend-option" "-generate-type-units"
//
// CI: "-dwarf-column-info"
//
// NOCI-NOT: "-dwarf-column-info"
//
// GEXTREFS: "-dwarf-ext-refs" "-fmodule-format=obj" "-debug-info-kind={{standalone|limited}}"

// RUN: not %clang -cc1 -debug-info-kind=watkind 2>&1 | FileCheck -check-prefix=BADSTRING1 %s
// BADSTRING1: error: invalid value 'watkind' in '-debug-info-kind=watkind'
// RUN: not %clang -cc1 -debugger-tuning=gmodal 2>&1 | FileCheck -check-prefix=BADSTRING2 %s
// BADSTRING2: error: invalid value 'gmodal' in '-debugger-tuning=gmodal'
