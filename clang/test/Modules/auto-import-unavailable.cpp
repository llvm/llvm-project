// RUN: rm -rf %t
// RUN: not %clang_cc1 -x c++ -Rmodule-build -DMISSING_HEADER -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs/auto-import-unavailable %s 2>&1 | FileCheck %s --check-prefix=MISSING-HEADER
// RUN: %clang_cc1 -x c++ -Rmodule-build -DNONREQUIRED_MISSING_HEADER -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs/auto-import-unavailable %s 2>&1 | FileCheck %s --check-prefix=NONREQUIRED-MISSING-HEADER
// RUN: not %clang_cc1 -x c++ -Rmodule-build -DMISSING_REQUIREMENT -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs/auto-import-unavailable %s 2>&1 | FileCheck %s --check-prefix=MISSING-REQUIREMENT

#ifdef MISSING_HEADER

// Even if the header we ask for is not missing, if the top-level module
// containing it has a missing header, then the whole top-level is
// unavailable and we issue an error.

// MISSING-HEADER: module.modulemap:2:27: error: header 'missing_header/missing.h' not found
// MISSING-HEADER-DAG: auto-import-unavailable.cpp:[[@LINE+1]]:10: note: submodule of top-level module 'missing_header' implicitly imported here
#include "missing_header/not_missing.h"

// We should not attempt to build the module.
// MISSING-HEADER-NOT: remark: building module

#endif // #ifdef MISSING_HEADER


#ifdef NONREQUIRED_MISSING_HEADER

// However, if the missing header is dominated by an unsatisfied
// `requires`, then that is acceptable.
// This also tests that an unsatisfied `requires` elsewhere in the
// top-level module doesn't affect an available module.

// NONREQUIRED-MISSING-HEADER: auto-import-unavailable.cpp:[[@LINE+2]]:10: remark: building module 'nonrequired_missing_header'
// NONREQUIRED-MISSING-HEADER: auto-import-unavailable.cpp:[[@LINE+1]]:10: remark: finished building module 'nonrequired_missing_header'
#include "nonrequired_missing_header/not_missing.h"

#endif // #ifdef NONREQUIRED_MISSING_HEADER


#ifdef MISSING_REQUIREMENT

// If the header is unavailable due to a missing requirement, an error
// should be emitted if a user tries to include it.

// MISSING-REQUIREMENT:module.modulemap:16:8: error: module 'missing_requirement' requires feature 'nonexistent_feature'
// MISSING-REQUIREMENT: auto-import-unavailable.cpp:[[@LINE+1]]:10: note: submodule of top-level module 'missing_requirement' implicitly imported here
#include "missing_requirement.h"

// MISSING-REQUIREMENT-NOT: remark: building module

#endif // #ifdef MISSING_REQUIREMENT
