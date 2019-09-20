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
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='subfolder_a' %s -- -I %S/Inputs/file-filter -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck --check-prefix=CHECK5 %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='subfolder_a' -quiet %s -- -I %S/Inputs/file-filter -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck --check-prefix=CHECK5-QUIET %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='subfolder_b' %s -- -I %S/Inputs/file-filter -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck --check-prefix=CHECK6 %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='subfolder_b' -quiet %s -- -I %S/Inputs/file-filter -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck --check-prefix=CHECK6-QUIET %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='subfolder_c' %s -- -I %S/Inputs/file-filter -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck --check-prefix=CHECK7 %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='subfolder_c' -quiet %s -- -I %S/Inputs/file-filter -isystem %S/Inputs/file-filter/system 2>&1 | FileCheck --check-prefix=CHECK7-QUIET %s

#include "header1.h"
// CHECK-NOT: warning:
// CHECK-QUIET-NOT: warning:
// CHECK2: header1.h:1:12: warning: single-argument constructors must be marked explicit
// CHECK2-QUIET: header1.h:1:12: warning: single-argument constructors must be marked explicit
// CHECK3-NOT: warning:
// CHECK3-QUIET-NOT: warning:
// CHECK4: header1.h:1:12: warning: single-argument constructors
// CHECK4-QUIET: header1.h:1:12: warning: single-argument constructors
// CHECK5-NOT: warning:
// CHECK5-QUIET-NOT: warning:
// CHECK6-NOT: warning:
// CHECK6-QUIET-NOT: warning:
// CHECK7-NOT: warning:
// CHECK7-QUIET-NOT: warning:

#include "header2.h"
// CHECK-NOT: warning:
// CHECK-QUIET-NOT: warning:
// CHECK2: header2.h:1:12: warning: single-argument constructors
// CHECK2-QUIET: header2.h:1:12: warning: single-argument constructors
// CHECK3: header2.h:1:12: warning: single-argument constructors
// CHECK3-QUIET: header2.h:1:12: warning: single-argument constructors
// CHECK4: header2.h:1:12: warning: single-argument constructors
// CHECK4-QUIET: header2.h:1:12: warning: single-argument constructors
// CHECK5-NOT: warning:
// CHECK5-QUIET-NOT: warning:
// CHECK6-NOT: warning:
// CHECK6-QUIET-NOT: warning:
// CHECK7-NOT: warning:
// CHECK7-QUIET-NOT: warning:

#include "subfolder_a/header_a.h"
// CHECK-NOT: warning:
// CHECK-QUIET-NOT: warning:
// CHECK2: header_b.h:1:12: warning: single-argument constructors must be marked explicit
// CHECK2-QUIET: header_b.h:1:12: warning: single-argument constructors must be marked explicit
// CHECK3-NOT: warning:
// CHECK3-QUIET-NOT: warning:
// CHECK4: header_b.h:1:12: warning: single-argument constructors must be marked explicit
// CHECK4-QUIET: header_b.h:1:12: warning: single-argument constructors must be marked explicit
// CHECK5: header_a.h:3:12: warning: single-argument constructors must be marked explicit
// CHECK5-QUIET: header_a.h:3:12: warning: single-argument constructors must be marked explicit
// CHECK6: header_b.h:1:12: warning: single-argument constructors must be marked explicit
// CHECK6-QUIET: header_b.h:1:12: warning: single-argument constructors must be marked explicit
// CHECK7-NOT: warning:
// CHECK7-QUIET-NOT: warning:

#include "subfolder_c/header_c.h"
// CHECK-NOT: warning:
// CHECK-QUIET-NOT: warning:
// CHECK2: header_c.h:1:12: warning: single-argument constructors must be marked explicit
// CHECK2-QUIET: header_c.h:1:12: warning: single-argument constructors must be marked explicit
// CHECK3-NOT: warning:
// CHECK3-QUIET-NOT: warning:
// CHECK4: header_c.h:1:12: warning: single-argument constructors must be marked explicit
// CHECK4-QUIET: header_c.h:1:12: warning: single-argument constructors must be marked explicit
// CHECK5-NOT: warning:
// CHECK5-QUIET-NOT: warning:
// CHECK6-NOT: warning:
// CHECK6-QUIET-NOT: warning:
// CHECK7: header_c.h:1:12: warning: single-argument constructors must be marked explicit
// CHECK7-QUIET: header_c.h:1:12: warning: single-argument constructors must be marked explicit

#include <system-header.h>
// CHECK-NOT: warning:
// CHECK-QUIET-NOT: warning:
// CHECK2-NOT: warning:
// CHECK2-QUIET-NOT: warning:
// CHECK3-NOT: warning:
// CHECK3-QUIET-NOT: warning:
// CHECK4: system-header.h:1:12: warning: single-argument constructors
// CHECK4-QUIET: system-header.h:1:12: warning: single-argument constructors
// CHECK5-NOT: warning:
// CHECK5-QUIET-NOT: warning:
// CHECK6-NOT: warning:
// CHECK6-QUIET-NOT: warning:
// CHECK7-NOT: warning:
// CHECK7-QUIET-NOT: warning:

class A { A(int); };
// CHECK: :[[@LINE-1]]:11: warning: single-argument constructors
// CHECK-QUIET: :[[@LINE-2]]:11: warning: single-argument constructors
// CHECK2: :[[@LINE-3]]:11: warning: single-argument constructors
// CHECK2-QUIET: :[[@LINE-4]]:11: warning: single-argument constructors
// CHECK3: :[[@LINE-5]]:11: warning: single-argument constructors
// CHECK3-QUIET: :[[@LINE-6]]:11: warning: single-argument constructors
// CHECK4: :[[@LINE-7]]:11: warning: single-argument constructors
// CHECK4-QUIET: :[[@LINE-8]]:11: warning: single-argument constructors
// CHECK5: :[[@LINE-9]]:11: warning: single-argument constructors
// CHECK5-QUIET: :[[@LINE-10]]:11: warning: single-argument constructors
// CHECK6: :[[@LINE-11]]:11: warning: single-argument constructors
// CHECK6-QUIET: :[[@LINE-12]]:11: warning: single-argument constructors
// CHECK7: :[[@LINE-13]]:11: warning: single-argument constructors
// CHECK7-QUIET: :[[@LINE-14]]:11: warning: single-argument constructors

// CHECK-NOT: warning:
// CHECK-QUIET-NOT: warning:
// CHECK2-NOT: warning:
// CHECK2-QUIET-NOT: warning:
// CHECK3-NOT: warning:
// CHECK3-QUIET-NOT: warning:
// CHECK4-NOT: warning:
// CHECK4-QUIET-NOT: warning:
// CHECK5-NOT: warning:
// CHECK5-QUIET-NOT: warning:
// CHECK6-NOT: warning:
// CHECK6-QUIET-NOT: warning:
// CHECK7-NOT: warning:
// CHECK7-QUIET-NOT: warning:

// CHECK: Suppressed 6 warnings (6 in non-user code)
// CHECK: Use -header-filter=.* to display errors from all non-system headers.
// CHECK-QUIET-NOT: Suppressed
// CHECK2: Suppressed 1 warnings (1 in non-user code)
// CHECK2: Use -header-filter=.* {{.*}}
// CHECK2-QUIET-NOT: Suppressed
// CHECK3: Suppressed 5 warnings (5 in non-user code)
// CHECK3: Use -header-filter=.* {{.*}}
// CHECK3-QUIET-NOT: Suppressed
// CHECK4-NOT: Suppressed {{.*}} warnings
// CHECK4-NOT: Use -header-filter=.* {{.*}}
// CHECK4-QUIET-NOT: Suppressed
// CHECK5: Suppressed 5 warnings (5 in non-user code)
// CHECK5: Use -header-filter=.* to display errors from all non-system headers.
// CHECK5-QUIET-NOT: Suppressed
// CHECK6: Suppressed 5 warnings (5 in non-user code)
// CHECK6: Use -header-filter=.* to display errors from all non-system headers.
// CHECK6-QUIET-NOT: Suppressed
// CHECK7: Suppressed 5 warnings (5 in non-user code)
// CHECK7: Use -header-filter=.* to display errors from all non-system headers.
// CHECK7-QUIET-NOT: Suppressed
