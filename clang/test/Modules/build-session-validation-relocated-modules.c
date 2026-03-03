// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: touch %t/session.timestamp
// RUN: %clang -fmodules -fimplicit-module-maps  -fsyntax-only %t/tu1.c \
// RUN:   -fmodules-cache-path=%t/cache -F%t/preferred_frameworks -F%t/fallback_frameworks \
// RUN:   -fbuild-session-file=%t/session.timestamp -fmodules-validate-once-per-build-session \
// RUN:   -Rmodule-validation 2>&1 | FileCheck %s --check-prefix=NO_RELOC

// NO_RELOC-NOT: checking if module {{.*}} has relocated

// RUN: mkdir %t/preferred_frameworks/
// RUN: cp -r %t/fallback_frameworks/IndirectDep.framework %t/preferred_frameworks/

// Verify no relocation checks happen because pcms are newer than timestamp.
// RUN: %clang -fmodules -fimplicit-module-maps  -fsyntax-only %t/tu1.c \
// RUN:   -fmodules-cache-path=%t/cache -F%t/preferred_frameworks -F%t/fallback_frameworks \
// RUN:   -fbuild-session-file=%t/session.timestamp -fmodules-validate-once-per-build-session \
// RUN:   -Rmodule-validation 2>&1 | FileCheck %s --check-prefix=NO_RELOC

// Verify no relocation checks happen even when the build session is new because it is explicitly
// disabled.
// RUN: touch %t/session.timestamp
// RUN: %clang -fmodules -fimplicit-module-maps  -fsyntax-only %t/tu1.c \
// RUN:   -fmodules-cache-path=%t/cache -F%t/preferred_frameworks -F%t/fallback_frameworks \
// RUN:   -fbuild-session-file=%t/session.timestamp -fmodules-validate-once-per-build-session \
// RUN:   -Xclang -fno-modules-check-relocated -Rmodule-validation 2>&1 | FileCheck %s --check-prefix=NO_RELOC

// Ensure future new timestamp doesn't have same time as older one.
// RUN: sleep 1

// Now remove the disablement and check.
// RUN: touch %t/session.timestamp
// RUN: %clang -fmodules -fimplicit-module-maps  -fsyntax-only %t/tu1.c \
// RUN:   -fmodules-cache-path=%t/cache -F%t/preferred_frameworks -F%t/fallback_frameworks \
// RUN:   -fbuild-session-file=%t/session.timestamp -fmodules-validate-once-per-build-session \
// RUN:   -Rmodule-validation 2>&1 | FileCheck %s --check-prefix=RELOC

// NO_RELOC-NOT: checking if module {{.*}} has relocated

// RELOC: checking if module {{.*}} has relocated
// RELOC: module 'IndirectDep' relocated from {{.*}}fallback_frameworks{{.*}} to {{.*}}preferred_frameworks

//--- fallback_frameworks/DirectDep.framework/Modules/module.modulemap
framework module DirectDep { header "DirectDep.h" }
//--- fallback_frameworks/DirectDep.framework/Headers/DirectDep.h
#include <IndirectDep/IndirectDep.h>

//--- fallback_frameworks/IndirectDep.framework/Modules/module.modulemap
framework module IndirectDep { header "IndirectDep.h" }
//--- fallback_frameworks/IndirectDep.framework/Headers/IndirectDep.h
int foo(void);

//--- tu1.c
#include <DirectDep/DirectDep.h>
