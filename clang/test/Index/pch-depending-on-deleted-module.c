#include "a.h"

// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: %clang_cc1 -x c-header -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -emit-pch -I %S/Inputs/Headers -o %t/use_LibA.pch %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -I %S/Inputs/Headers -verify-pch %t/use_LibA.pch
// RUN: rm -f %t/modules-cache/LibA.pcm
// RUN: not %clang_cc1 -fmodules -fimplicit-module-maps -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -I %S/Inputs/Headers -verify-pch %t/use_LibA.pch 2>&1 | FileCheck -check-prefix=VERIFY %s
// RUN: not c-index-test -test-load-source all -x c -fmodules -fimplicit-module-maps -Xclang -fdisable-module-hash -fmodules-cache-path=%t/modules-cache -I %S/Inputs/Headers -include-pch %t/use_LibA.pch %s 2>&1 | FileCheck -check-prefix=INDEX %s

// VERIFY: fatal error: module file '{{.*}}LibA.pcm' not found
// INDEX: {{^}}Failure: AST deserialization error occurred{{$}}

