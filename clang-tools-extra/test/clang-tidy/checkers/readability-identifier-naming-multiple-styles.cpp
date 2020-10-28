// Setup header directory

// RUN: rm -rf %theaders
// RUN: mkdir %theaders
// RUN: cp -R %S/Inputs/readability-identifier-naming/. %theaders

// C++11 isn't explicitly required, but failing to specify a standard means the
// check will run multiple times for different standards. This will cause the
// second test to fail as the header file will be changed during the first run.
// InheritParentConfig is needed to look for the clang-tidy configuration files.

// RUN: %check_clang_tidy -check-suffixes=ENABLED,SHARED -std=c++11 %s \
// RUN: readability-identifier-naming %t -- \
// RUN:  -config='{ InheritParentConfig: true, CheckOptions: [ \
// RUN:   {key: readability-identifier-naming.FunctionCase, value: camelBack}, \
// RUN:   {key: readability-identifier-naming.GetConfigPerFile, value: true} \
// RUN:  ]}' -header-filter='.*' -- -I%theaders

// On DISABLED run, everything should be made 'camelBack'.

// RUN: cp -R %S/Inputs/readability-identifier-naming/. %theaders
// RUN: %check_clang_tidy -check-suffixes=DISABLED,SHARED -std=c++11 %s \
// RUN: readability-identifier-naming %t -- \
// RUN:  -config='{ InheritParentConfig: true, CheckOptions: [ \
// RUN:   {key: readability-identifier-naming.FunctionCase, value: camelBack}, \
// RUN:   {key: readability-identifier-naming.GetConfigPerFile, value: false} \
// RUN:  ]}' -header-filter='.*' -- -I%theaders

#include "global-style-disabled/header.h"
#include "global-style1/header.h"
#include "global-style2/header.h"
// CHECK-MESSAGES-ENABLED-DAG: global-style1/header.h:5:6: warning: invalid case style for global function 'styleFirstBad'
// CHECK-MESSAGES-ENABLED-DAG: global-style2/header.h:5:6: warning: invalid case style for global function 'styleSecondBad'
// CHECK-MESSAGES-DISABLED-DAG: global-style1/header.h:3:6: warning: invalid case style for function 'style_first_good'
// CHECK-MESSAGES-DISABLED-DAG: global-style2/header.h:3:6: warning: invalid case style for function 'STYLE_SECOND_GOOD'
// CHECK-MESSAGES-DISABLED-DAG: global-style-disabled/header.h:1:6: warning: invalid case style for function 'disabled_style_1'
// CHECK-MESSAGES-DISABLED-DAG: global-style-disabled/header.h:3:6: warning: invalid case style for function 'DISABLED_STYLE_3'

void goodStyle() {
  style_first_good();
  STYLE_SECOND_GOOD();
  //      CHECK-FIXES-DISABLED: styleFirstGood();
  // CHECK-FIXES-DISABLED-NEXT: styleSecondGood();
}
// CHECK-MESSAGES-SHARED-DAG: :[[@LINE+1]]:6: warning: invalid case style for function 'bad_style'
void bad_style() {
  styleFirstBad();
  styleSecondBad();
}
//        CHECK-FIXES-SHARED: void badStyle() {
// CHECK-FIXES-DISABLED-NEXT:   styleFirstBad();
//  CHECK-FIXES-ENABLED-NEXT:   style_first_bad();
// CHECK-FIXES-DISABLED-NEXT:   styleSecondBad();
//  CHECK-FIXES-ENABLED-NEXT:   STYLE_SECOND_BAD();
//   CHECK-FIXES-SHARED-NEXT: }

void expectNoStyle() {
  disabled_style_1();
  disabledStyle2();
  DISABLED_STYLE_3();
  //      CHECK-FIXES-DISABLED: disabledStyle1();
  // CHECK-FIXES-DISABLED-NEXT: disabledStyle2();
  // CHECK-FIXES-DISABLED-NEXT: disabledStyle3();
}
