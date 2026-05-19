// RUN: %clang_cc1 -E -DDATETIME_DEFAULT -init-datetime-macros=default %s | FileCheck %s --check-prefix CHECK-INIT-DATETIME-DEFAULT
// RUN: %clang_cc1 -E -DDATETIME_DEFAULT %s | FileCheck %s --check-prefix CHECK-INIT-DATETIME-DEFAULT
// CHECK-INIT-DATETIME-DEFAULT: date: "{{(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)}} {{.*}}"
// CHECK-INIT-DATETIME-DEFAULT: time: "{{[0-9][0-9]:[0-9][0-9]:[0-9][0-9]}}"
// CHECK-INIT-DATETIME-DEFAULT: timestamp: "{{.*}} {{[0-9][0-9]:[0-9][0-9]:[0-9][0-9]}} {{.*}}"
// CHECK-INIT-DATETIME-DEFAULT-NOT: date: 
// CHECK-INIT-DATETIME-DEFAULT-NOT: time: 
// CHECK-INIT-DATETIME-DEFAULT-NOT: timestamp: 

// RUN: %clang_cc1 -E -DDATETIME_LITERALONE -init-datetime-macros=literalone %s | FileCheck %s --check-prefix CHECK-INIT-DATETIME-LITERALONE
// CHECK-INIT-DATETIME-LITERALONE: date: "1"
// CHECK-INIT-DATETIME-LITERALONE: time: "1"
// CHECK-INIT-DATETIME-LITERALONE: timestamp: "1"
// CHECK-INIT-DATETIME-LITERALONE-NOT: date: 
// CHECK-INIT-DATETIME-LITERALONE-NOT: time: 
// CHECK-INIT-DATETIME-LITERALONE-NOT: timestamp: 

// RUN: %clang_cc1 -E -DDATETIME_CUSTOM -init-datetime-macros=undefined -D__DATE__="\"d3\"" -D__TIME__="\"t4\"" -D__TIMESTAMP__="\"ts5\"" %s | FileCheck %s --check-prefix CHECK-INIT-DATETIME-CUSTOM
// CHECK-INIT-DATETIME-CUSTOM: date: "d3"
// CHECK-INIT-DATETIME-CUSTOM: time: "t4"
// CHECK-INIT-DATETIME-CUSTOM: timestamp: "ts5"

// RUN: %clang_cc1 -DDATETIME_UNDEFINED -verify -Wall -init-datetime-macros=undefined %s

// clang-cl deterministic options checks:
//  /d1nodatetime - undefines __DATE__, __TIME__ and __TIMESTAMP__
//  /Brepro - sets __DATE__, __TIME__ and __TIMESTAMP__ to "1"

// RUN: %clang_cl -Xclang -verify /d1nodatetime /DDATETIME_UNDEFINED /c %s

// RUN: %clang_cl -E /Brepro /DDATETIME_LITERALONE /c %s | FileCheck %s --check-prefix CHECK-INIT-DATETIME-LITERALONE

#if defined(DATETIME_LITERALONE) || defined(DATETIME_DEFAULT) || defined(DATETIME_CUSTOM)
date: __DATE__
time: __TIME__
timestamp: __TIMESTAMP__
#endif

// Check we didn't break literal processing inside of PP macro expansion.
#if defined(DATETIME_DEFAULT)
#define CUST_DATE __DATE__
#define CUST_TIME __TIME__
#define CUST_TIMESTAMP __TIMESTAMP__
const char *s0 = "CD: "  CUST_DATE " CT: " CUST_TIME " CTS: " CUST_TIMESTAMP
// CHECK-INIT-DATETIME-DEFAULT: const char *s0 = "CD: " "{{(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)}} {{.*}}" " CT: " "{{[0-9][0-9]:[0-9][0-9]:[0-9][0-9]}}" " CTS: " "{{.*}} {{[0-9][0-9]:[0-9][0-9]:[0-9][0-9]}} {{.*}}" 
#endif

#if defined(DATETIME_LITERALONE)
#define CUST_DATE __DATE__
#define CUST_TIME __TIME__
#define CUST_TIMESTAMP __TIMESTAMP__
const char *s0 = "CD: "  CUST_DATE " CT: " CUST_TIME " CTS: " CUST_TIMESTAMP
// CHECK-INIT-DATETIME-LITERALONE: const char *s0 = "CD: " "1" " CT: " "1" " CTS: " "1" 
#endif

#ifdef DATETIME_UNDEFINED
const char *s1 = __DATE__; // expected-error{{use of undeclared identifier '__DATE__'}} 
const char *s2 = __TIME__; // expected-error{{use of undeclared identifier '__TIME__'}} 
const char *s3 = __TIMESTAMP__; // expected-error{{use of undeclared identifier '__TIMESTAMP__'}}
#endif
