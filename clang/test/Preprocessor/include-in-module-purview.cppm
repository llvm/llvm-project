// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -E -P -I%t -o %t/tmp 2>&1 | FileCheck %t/a.cppm
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -E -P -I%t -o - 2>&1 \
// RUN:     -Wno-include-angled-in-module-purview | FileCheck %t/a.cppm --check-prefix=CHECK-NO-WARN

//--- a.h
// left empty

//--- b.h
#include <stddef.h>
// The headers not get included shouldn't be affected.
#ifdef WHATEVER
#include <stdint.h>
#endif

//--- a.cppm
module;
#include <stddef.h>
#include <a.h>
#include <b.h>
#include "a.h"
#include "b.h"
export module a;

#include <stddef.h>
#include <a.h>
#include <b.h>
#include "a.h"
#include "b.h"

// CHECK: a.cppm:9:10: warning: '#include <filename>' attaches the declarations to the named module 'a'
// CHECK: a.cppm:10:10: warning: '#include <filename>' attaches the declarations to the named module 'a'
// CHECK: a.cppm:11:10: warning: '#include <filename>' attaches the declarations to the named module 'a'
// CHECK: In file included from {{.*}}/a.cppm:11
// CHECK-NEXT: b.h:1:10: warning: '#include <filename>' attaches the declarations to the named module 'a'
// CHECK: In file included from {{.*}}/a.cppm:13
// CHECK-NEXT: b.h:1:10: warning: '#include <filename>' attaches the declarations to the named module 'a'

module :private;
#include <stddef.h>
#include <a.h>
#include <b.h>
#include "a.h"
#include "b.h"

// CHECK: a.cppm:24:10: warning: '#include <filename>' attaches the declarations to the named module 'a'
// CHECK: a.cppm:25:10: warning: '#include <filename>' attaches the declarations to the named module 'a'
// CHECK: a.cppm:26:10: warning: '#include <filename>' attaches the declarations to the named module 'a'
// CHECK: In file included from {{.*}}/a.cppm:26
// CHECK-NEXT: b.h:1:10: warning: '#include <filename>' attaches the declarations to the named module 'a'
// CHECK: In file included from {{.*}}/a.cppm:28
// CHECK-NEXT: b.h:1:10: warning: '#include <filename>' attaches the declarations to the named module 'a'

// We should have catched all warnings.
// CHECK: 10 warnings generated.

// CHECK-NO-WARN-NOT: warning
