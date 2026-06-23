// Regression for clang-reforge-7y4 / iter-03: end-to-end verification
// of the --ssaf-no-extract-from-system-headers opt-out. Spec:
// tu-summary-extraction's "System-header contributor opt-out flag"
// requirement.

// REQUIRES: system-darwin || system-linux

// Setup: synthesise an -isystem header containing a benign user-named
// symbol (no USR collision — that case is exercised end-to-end by the
// parity-finale verification step against libJP2).
// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/sysinc
// RUN: printf '#pragma clang system_header\nint *sys_gp; void sys_fn(int *p) { sys_gp = p; }\n' > %t.dir/sysinc/sys.h

// === Case A: flag absent (default extracts from system headers). ===
// The extractor enumerates both sys_fn and user_fn; the TU summary's
// IdTable contains both names.
// RUN: rm -f %t-default.json
// RUN: %clang -c %s -o %t-default.o -isystem %t.dir/sysinc \
// RUN:   --ssaf-extract-summaries=PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t-default.json \
// RUN:   --ssaf-compilation-unit-id=sys-default
// RUN: FileCheck --check-prefix=DEFAULT %s < %t-default.json
// DEFAULT-DAG: sys_fn
// DEFAULT-DAG: user_fn

// === Case B: flag present (opt-out skips system-header decls). ===
// The extractor skips sys_fn (system header) but keeps user_fn.
// The TU summary's IdTable contains user_fn but NOT sys_fn.
// RUN: rm -f %t-optout.json
// RUN: %clang -c %s -o %t-optout.o -isystem %t.dir/sysinc \
// RUN:   --ssaf-extract-summaries=PointerFlow \
// RUN:   --ssaf-tu-summary-file=%t-optout.json \
// RUN:   --ssaf-no-extract-from-system-headers \
// RUN:   --ssaf-compilation-unit-id=sys-optout
// RUN: FileCheck --check-prefix=OPTOUT %s < %t-optout.json
// OPTOUT-NOT: sys_fn
// OPTOUT: user_fn

#include <sys.h>

int *user_gp;
void user_fn(int *p) { user_gp = p; }
