// Validate configuration mismatches from precompiled files are reported.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -I%t/BuildDir -fimplicit-module-maps -fmodules \
// RUN:   -fmodules-cache-path=%t/cache %t/h1.h -emit-pch -o %t/BuildDir/h1.h.pch

// Check command line diff that is reported uniquely.
// RUN: not %clang_cc1 -I%t/BuildDir -fimplicit-module-maps -fmodules \
// RUN:   -O3 \
// RUN:   -fmodules-cache-path=%t/cache -fsyntax-only -include-pch %t/BuildDir/h1.h.pch \
// RUN:   %t/client.c 2>&1 | FileCheck %s  --check-prefixes=OPTMODE,CONFIG

// Check command line difference that end up in the module hash, but is not
// uniquely reported as a mismatch.
// RUN: not %clang_cc1 -I%t/BuildDir -fimplicit-module-maps -fmodules \
// RUN:   -dwarf-ext-refs -fmodule-format=obj \
// RUN:   -debug-info-kind=standalone -dwarf-version=5 \
// RUN:   -fmodules-cache-path=%t/cache -fsyntax-only -include-pch %t/BuildDir/h1.h.pch \
// RUN:   %t/client.c 2>&1 | FileCheck %s --check-prefix=CONFIG

// Check that module cache path is uniquely reported.
// RUN: not %clang_cc1 -I%t/BuildDir -fimplicit-module-maps -fmodules \
// RUN:   -fmodules-cache-path=%t/wrong/cache -fsyntax-only \
// RUN:   -include-pch %t/BuildDir/h1.h.pch \
// RUN:   %t/client.c 2>&1 | FileCheck %s --check-prefix=CACHEPATH

// OPTMODE: OptimizationLevel differs in precompiled file
// CONFIG: h1.h.pch' cannot be loaded due to a configuration mismatch
// CACHEPATH: h1.h.pch' was compiled with module cache path '{{.*}}', but the path is currently '{{.*}}'

//--- BuildDir/A/module.modulemap
module A [system] {
  umbrella "."
}

//--- BuildDir/A/A.h
typedef int A_t;

//--- h1.h
#include <A/A.h>
#if __OPTIMIZE__
A_t foo(void);
#endif

//--- client.c
typedef int foo_t;

