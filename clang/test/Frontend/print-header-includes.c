// RUN: %clang_cc1 -I%S -include Inputs/test3.h -isystem %S/Inputs/SystemHeaderPrefix \
// RUN:     -E -H -o /dev/null %s 2> %t.stderr
// RUN: FileCheck < %t.stderr %s

// CHECK-NOT: test3.h
// CHECK-NOT: . {{.*noline.h}}
// CHECK: . {{.*test.h}}
// CHECK: .. {{.*test2.h}}
// CHECK-NOT: .. {{.*test2.h}}

// RUN: %clang_cc1 -I%S -isystem %S/Inputs/SystemHeaderPrefix \
// RUN:     -E -H -fshow-skipped-includes -o /dev/null %s 2> %t.stderr
// RUN: FileCheck --check-prefix=SKIPPED < %t.stderr %s

// SKIPPED-NOT: . {{.*noline.h}}
// SKIPPED: . {{.*test.h}}
// SKIPPED: .. {{.*test2.h}}
// SKIPPED: .. {{.*test2.h}}

// RUN: %clang_cc1 -isystem %S -isystem %S/Inputs/SystemHeaderPrefix \
// RUN:     -E -H -fshow-skipped-includes -sys-header-deps -o /dev/null %s 2> %t.stderr
// RUN: FileCheck --check-prefix=SKIPPED-SYS < %t.stderr %s

// SKIPPED-SYS: . {{.*noline.h}}
// SKIPPED-SYS: . {{.*test.h}}
// SKIPPED-SYS: .. {{.*test2.h}}
// SKIPPED-SYS: .. {{.*test2.h}}

// RUN: %clang_cc1 -I%S -isystem %S/Inputs/SystemHeaderPrefix -include Inputs/test.h \
// RUN:     -E -H -fshow-skipped-includes -o /dev/null %s 2> %t.stderr
// RUN: FileCheck --check-prefix=SKIPPED-PREDEFINES < %t.stderr %s

// The skipped include of test2.h from the -include test.h shouldn't be printed.
// SKIPPED-PREDEFINES-NOT: {{.*test2.h}}
// SKIPPED-PREDEFINES:     . {{.*test.h}}

// RUN: %clang_cc1 -isystem %S -isystem %S/Inputs/SystemHeaderPrefix \
// RUN:     -E -H -fshow-skipped-includes -o /dev/null %s 2> %t.stderr
// RUN: FileCheck --check-prefix=SKIPPED-NO-SYS --allow-empty < %t.stderr %s

// SKIPPED-NO-SYS-NOT: .

// RUN: %clang_cc1 -I%S -include Inputs/test3.h -isystem %S/Inputs/SystemHeaderPrefix \
// RUN:     -E -H -sys-header-deps -o /dev/null %s 2> %t.stderr
// RUN: FileCheck --check-prefix SYSHEADERS < %t.stderr %s

// SYSHEADERS-NOT: test3.h
// SYSHEADERS: . {{.*noline.h}}
// SYSHEADERS: . {{.*test.h}}
// SYSHEADERS: .. {{.*test2.h}}

// RUN: %clang_cc1 -I%S -include Inputs/test3.h -isystem %S/Inputs/SystemHeaderPrefix \
// RUN:     --show-includes -o /dev/null %s | \
// RUN:     FileCheck --strict-whitespace --check-prefix=MS-STDOUT %s
// MS-STDOUT-NOT: <command line>
// MS-STDOUT-NOT: Note: including file: {{[^ ]*noline.h}}
// MS-STDOUT: Note: including file: {{[^ ]*test3.h}}
// MS-STDOUT: Note: including file: {{[^ ]*test.h}}
// MS-STDOUT: Note: including file:  {{[^ ]*test2.h}}
// MS-STDOUT-NOT: Note

// RUN: %clang_cc1 -I%S -include Inputs/test3.h -isystem %S/Inputs/SystemHeaderPrefix \
// RUN:     -E --show-includes -o /dev/null %s 2> %t.stderr
// RUN: FileCheck --strict-whitespace --check-prefix=MS-STDERR < %t.stderr %s
// MS-STDERR-NOT: <command line>
// MS-STDERR-NOT: Note: including file: {{[^ ]*noline.h}}
// MS-STDERR: Note: including file: {{[^ ]*test3.h}}
// MS-STDERR: Note: including file: {{[^ ]*test.h}}
// MS-STDERR: Note: including file:  {{[^ ]*test2.h}}
// MS-STDERR-NOT: Note

// RUN: echo "fun:foo" > %t.ignorelist
// RUN: %clang_cc1 -I%S -isystem %S/Inputs/SystemHeaderPrefix \
// RUN:     -fsanitize=address -fdepfile-entry=%t.ignorelist \
// RUN:     --show-includes -o /dev/null %s | \
// RUN:     FileCheck --strict-whitespace --check-prefix=MS-IGNORELIST %s
// MS-IGNORELIST: Note: including file: {{[^ ]*\.ignorelist}}
// MS-IGNORELIST-NOT: Note: including file: {{[^ ]*noline.h}}
// MS-IGNORELIST: Note: including file: {{[^ ]*test.h}}
// MS-IGNORELIST: Note: including file:  {{[^ ]*test2.h}}
// MS-IGNORELIST-NOT: Note

#include <noline.h>
#include <Inputs/test.h>
