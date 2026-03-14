// RUN: %clang_analyze_cc1 %s -verify \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config core.CallAndMessage:FunctionPointer=false \
// RUN:   -analyzer-config core.CallAndMessage:ParameterCount=false \
// RUN:   -analyzer-config core.CallAndMessage:CXXThisMethodCall=false \
// RUN:   -analyzer-config core.CallAndMessage:CXXDeallocationArg=false \
// RUN:   -analyzer-config core.CallAndMessage:ArgInitializedness=false \
// RUN:   -analyzer-config core.CallAndMessage:ArgPointeeInitializedness=false \
// RUN:   -analyzer-config core.CallAndMessage:NilReceiver=true \
// RUN:   -analyzer-config core.CallAndMessage:UndefReceiver=false \
// RUN:   -analyzer-output=plist -o %t.plist
// RUN: cat %t.plist | FileCheck %s

@interface Foo
- (int &)ref;
@end

Foo *getFoo() { return 0; }

void testNullPointerSuppression() {
  getFoo().ref = 1;
}

void testPositiveNullReference() {
  Foo *x = 0;
  x.ref = 1; // expected-warning {{The receiver of message 'ref' is nil, which results in forming a null reference [core.CallAndMessage]}}
}

// TODO: If this hash ever changes, turn core.CallAndMessage:NilReceiver from a
// checker option into a checker, as described in the CallAndMessage comments!
// CHECK: <key>issue_hash_content_of_line_in_context</key>
// CHECK-SAME: <string>abe2e0574dd901094c511bae2f93f926</string>
