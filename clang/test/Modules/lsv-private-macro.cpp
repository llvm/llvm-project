// RUN: rm -rf %t
//
// RUN: %clang_cc1 %s \
// RUN:   -fmodules-local-submodule-visibility \
// RUN:   -fmodule-map-file=%S/Inputs/lsv-private-macro/mod.map \
// RUN:   -I%S/Inputs/lsv-private-macro -fmodule-name=self \
// RUN:   -verify=expected-lsv
//
// RUN: %clang_cc1 %s \
// RUN:   -fmodules -fmodules-cache-path=%t \
// RUN:   -fmodule-map-file=%S/Inputs/lsv-private-macro/mod.map \
// RUN:   -I%S/Inputs/lsv-private-macro -fmodule-name=self \
// RUN:   -verify=expected-nolsv
//
// RUN: %clang_cc1 %s \
// RUN:   -fmodules -fmodules-cache-path=%t \
// RUN:   -fmodules-local-submodule-visibility \
// RUN:   -fmodule-map-file=%S/Inputs/lsv-private-macro/mod.map \
// RUN:   -I%S/Inputs/lsv-private-macro -fmodule-name=self \
// RUN:   -verify=expected-lsv

#include "self.h"

// With local submodule visibility enabled, private macros don't leak out of
// their respective submodules, even within the same top-level module.
// expected-lsv-no-diagnostics

// expected-nolsv-error@+2 {{SELF_PRIVATE defined}}
#ifdef SELF_PRIVATE
#error SELF_PRIVATE defined
#endif

#ifndef SELF_PUBLIC
#error SELF_PUBLIC not defined
#endif

#ifndef SELF_DEFAULT
#error SELF_DEFAULT not defined
#endif

#include "other.h"

#ifdef OTHER_PRIVATE
#error OTHER_PRIVATE defined
#endif

#ifndef OTHER_PUBLIC
#error OTHER_PUBLIC not defined
#endif

#ifndef OTHER_DEFAULT
#error OTHER_DEFAULT not defined
#endif
