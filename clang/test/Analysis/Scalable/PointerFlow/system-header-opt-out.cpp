// Synthesise an -isystem header containing a benign user-named
// symbol.

// REQUIRES: system-darwin || system-linux

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// === Case A: flag absent (default extracts from system headers). ===
// The extractor enumerates both sys_fn and user_fn; the TU summary's
// IdTable contains both names.
// RUN: %clang -c %t/test.cpp -o %t/default.o -isystem %t/sysinc \
// RUN:   --ssaf-extract-summaries=PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t/default.json \
// RUN:   --ssaf-compilation-unit-id=sys-default
// RUN: FileCheck --check-prefix=DEFAULT %s < %t/default.json
// DEFAULT-DAG: sys_fn
// DEFAULT-DAG: user_fn

// === Case B: flag present (opt-out skips system-header decls). ===
// The extractor skips sys_fn (system header) but keeps user_fn.
// The TU summary's IdTable contains user_fn but NOT sys_fn.
// RUN: %clang -c %t/test.cpp -o %t/optout.o -isystem %t/sysinc \
// RUN:   --ssaf-extract-summaries=PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t/optout.json \
// RUN:   --ssaf-no-extract-from-system-headers \
// RUN:   --ssaf-compilation-unit-id=sys-optout
// RUN: FileCheck --check-prefix=OPTOUT %s < %t/optout.json
// OPTOUT-NOT: sys_fn
// OPTOUT: user_fn

//--- sysinc/sys.h
#pragma clang system_header
int *sys_gp;
void sys_fn(int *p) { sys_gp = p; }

//--- test.cpp
#include <sys.h>

int *user_gp;
void user_fn(int *p) { user_gp = p; }
