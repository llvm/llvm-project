// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='' %s -- -I %S/Inputs/file-filter -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='' -quiet %s -- -I %S/Inputs/file-filter -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck --check-prefix=CHECK-QUIET %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='.*' %s -- -I %S/Inputs/file-filter -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='.*' -quiet %s -- -I %S/Inputs/file-filter -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck --check-prefix=CHECK2-QUIET %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='header2\.h' %s -- -I %S/Inputs/file-filter -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='header2\.h' -quiet %s -- -I %S/Inputs/file-filter -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck --check-prefix=CHECK3-QUIET %s
// FIXME: "-I %S/Inputs/file-filter/system/.." must be redundant.
//       On Win32, file-filter/system\system-header1.h precedes
//       file-filter\header*.h due to code order between '/' and '\\'.
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='.*' -system-headers %s -- -I %S/Inputs/file-filter/system/.. -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='.*' -system-headers -quiet %s -- -I %S/Inputs/file-filter/system/.. -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck --check-prefix=CHECK4-QUIET %s
// RUN: clang-tidy -checks='-*,cppcoreguidelines-pro-type-cstyle-cast' -header-filter='.*' -system-headers %s -- -I %S/Inputs/file-filter/system/.. -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck --check-prefix=CHECK5 %s
// RUN: clang-tidy -checks='-*,cppcoreguidelines-pro-type-cstyle-cast' -header-filter='.*' %s -- -I %S/Inputs/file-filter/system/.. -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck --check-prefix=CHECK5-NO-SYSTEM-HEADERS %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='.*' -exclude-header-filter='header1\.h' %s -- -I %S/Inputs/file-filter/ -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck --check-prefix=CHECK6 %s

#include "header1.h"
// CHECK-NOT: warning:
// CHECK-QUIET-NOT: warning:
// CHECK2: header1.h:1:12: warning: single-argument constructors must be marked explicit
// CHECK2-QUIET: header1.h:1:12: warning: single-argument constructors must be marked explicit
// CHECK3-NOT: warning:
// CHECK3-QUIET-NOT: warning:
// CHECK4: header1.h:1:12: warning: single-argument constructors
// CHECK4-QUIET: header1.h:1:12: warning: single-argument constructors
// CHECK6-NOT: warning:

#include "header2.h"
// CHECK-NOT: warning:
// CHECK-QUIET-NOT: warning:
// CHECK2: header2.h:1:12: warning: single-argument constructors
// CHECK2-QUIET: header2.h:1:12: warning: single-argument constructors
// CHECK3: header2.h:1:12: warning: single-argument constructors
// CHECK3-QUIET: header2.h:1:12: warning: single-argument constructors
// CHECK4: header2.h:1:12: warning: single-argument constructors
// CHECK4-QUIET: header2.h:1:12: warning: single-argument constructors
// CHECK6: header2.h:1:12: warning: single-argument constructors

#include <system-header.h>
// CHECK-NOT: warning:
// CHECK-QUIET-NOT: warning:
// CHECK2-NOT: warning:
// CHECK2-QUIET-NOT: warning:
// CHECK3-NOT: warning:
// CHECK3-QUIET-NOT: warning:
// CHECK4: system-header.h:1:12: warning: single-argument constructors
// CHECK4-QUIET: system-header.h:1:12: warning: single-argument constructors
// CHECK6-NOT: warning:

class A { A(int); };
// CHECK: :[[@LINE-1]]:11: warning: single-argument constructors
// CHECK-QUIET: :[[@LINE-2]]:11: warning: single-argument constructors
// CHECK2: :[[@LINE-3]]:11: warning: single-argument constructors
// CHECK2-QUIET: :[[@LINE-4]]:11: warning: single-argument constructors
// CHECK3: :[[@LINE-5]]:11: warning: single-argument constructors
// CHECK3-QUIET: :[[@LINE-6]]:11: warning: single-argument constructors
// CHECK4: :[[@LINE-7]]:11: warning: single-argument constructors
// CHECK4-QUIET: :[[@LINE-8]]:11: warning: single-argument constructors
// CHECK6: :[[@LINE-9]]:11: warning: single-argument constructors

// CHECK-NOT: warning:
// CHECK-QUIET-NOT: warning:
// CHECK2-NOT: warning:
// CHECK2-QUIET-NOT: warning:
// CHECK3-NOT: warning:
// CHECK3-QUIET-NOT: warning:
// CHECK4-NOT: warning:
// CHECK4-QUIET-NOT: warning:

// CHECK: Use -header-filter=.* to display errors from all non-system headers.
// CHECK-QUIET-NOT: Suppressed
// CHECK2-QUIET-NOT: Suppressed
// CHECK3: Use -header-filter=.* {{.*}}
// CHECK3-QUIET-NOT: Suppressed
// CHECK4-NOT: Suppressed {{.*}} warnings
// CHECK4-NOT: Use -header-filter=.* {{.*}}
// CHECK4-QUIET-NOT: Suppressed
// CHECK6: Use -header-filter=.* {{.*}}

int x = 123;
auto x_ptr = TO_FLOAT_PTR(&x);
// CHECK5: :[[@LINE-1]]:14: warning: do not use C-style cast to convert between unrelated types
// CHECK5-NO-SYSTEM-HEADERS-NOT: :[[@LINE-2]]:14: warning: do not use C-style cast to convert between unrelated types
