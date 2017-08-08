// RUN: %clang_cc1 -I%S -include Inputs/test3.h -E -H -o /dev/null %s 2> %t.stderr
// RUN: FileCheck < %t.stderr %s

// CHECK-NOT: test3.h
// CHECK: . {{.*test.h}}
// CHECK: .. {{.*test2.h}}

// RUN: %clang_cc1 -I%S -include Inputs/test3.h -E --show-includes -o /dev/null %s | \
// RUN:     FileCheck --strict-whitespace --check-prefix=MS %s
// MS-NOT: <command line>
// MS: Note: including file: {{[^ ]*test3.h}}
// MS: Note: including file: {{[^ ]*test.h}}
// MS: Note: including file:  {{[^ ]*test2.h}}
// MS-NOT: Note

// RUN: echo "fun:foo" > %t.blacklist
// RUN: %clang_cc1 -I%S -fsanitize=address -fdepfile-entry=%t.blacklist -E --show-includes -o /dev/null %s | \
// RUN:     FileCheck --strict-whitespace --check-prefix=MS-BLACKLIST %s
// MS-BLACKLIST: Note: including file: {{[^ ]*\.blacklist}}
// MS-BLACKLIST: Note: including file: {{[^ ]*test.h}}
// MS-BLACKLIST: Note: including file:  {{[^ ]*test2.h}}
// MS-BLACKLIST-NOT: Note

#include "Inputs/test.h"
