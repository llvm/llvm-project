// In this example, a central resolved module, Top, reorders its input files
// under the same context hash, based on whether relocation checking is enabled.

// FIXME: The ordering should be stable. When that is fixed, collapse
// NORELOC and RELOC into one common check prefix.

// RUN: rm -rf %t
// RUN: split-file %s %t

// Build with the relocation check disabled.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fsyntax-only %t/tu.c \
// RUN:   -fmodules-cache-path=%t/cache -fmodules-strict-context-hash \
// RUN:   -isystem %t/usr/include -fno-modules-check-relocated
// RUN: %clang_cc1 -module-file-info %t/cache/*/Top-*.pcm | FileCheck %s --check-prefix=NORELOC

// RUN: ls %t/cache/*/Top-*.pcm > %t/top-pcm.txt

// Rebuild with the relocation check enabled, as the default.
// RUN: rm -rf %t/cache
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fsyntax-only %t/tu.c \
// RUN:   -fmodules-cache-path=%t/cache -fmodules-strict-context-hash \
// RUN:   -isystem %t/usr/include
// RUN: %clang_cc1 -module-file-info %t/cache/*/Top-*.pcm | FileCheck %s --check-prefix=RELOC

// Verify Top PCM resolves to the same location as in the first build.
// RUN: ls %t/cache/*/Top-*.pcm | diff %t/top-pcm.txt -

// NORELOC:  Input file: {{.*}}include{{/|\\}}module.modulemap
// NORELOC:  Input file: {{.*}}B{{/|\\}}module.modulemap
// NORELOC:  Input file: {{.*}}P{{/|\\}}module.modulemap
// NORELOC:  Input file: {{.*}}A{{/|\\}}module.modulemap

// RELOC: Input file: {{.*}}include{{/|\\}}module.modulemap
// RELOC: Input file: {{.*}}B{{/|\\}}module.modulemap
// RELOC: Input file: {{.*}}A{{/|\\}}module.modulemap
// RELOC: Input file: {{.*}}P{{/|\\}}module.modulemap

//--- usr/include/module.modulemap
module Top { header "Top.h" export * }
//--- usr/include/Top.h
#include <B/B.h>
#include <P/P.h>
#include <A/AT.h>

//--- usr/include/B/module.modulemap
module B { header "B.h" export * }
//--- usr/include/B/B.h
#include <A/A.h>

//--- usr/include/A/module.modulemap
module A { header "A.h" export * }
module AT { header "AT.h" export * }

//--- usr/include/A/A.h
int a(void);
//--- usr/include/A/AT.h
int at(void);

//--- usr/include/P/module.modulemap
module P { header "P.h" export * }
//--- usr/include/P/P.h
int p(void);

//--- tu.c
#include <Top.h>
