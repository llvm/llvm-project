@import hidden_redecls;

@interface Foo (Top)
- (void)top_method;
@end

// p1_method in protocol P1 is hidden since module_redecls.sub hasn't been
// imported yet. Check it is still indexed.

// RUN: rm -rf %t
// RUN: c-index-test -index-file-full %s -isystem %S/Inputs -fmodules -fmodules-cache-path=%t -target x86_64-apple-macosx10.7 | FileCheck %s
// CHECK: [indexDeclaration]: kind: objc-instance-method | name: p1_method | {{.*}} | loc: {{.*}}hidden-redecls-sub.h:2:9 | {{.*}} | isRedecl: 0
// CHECK: [indexDeclaration]: kind: objc-instance-method | name: p1_method | {{.*}} | loc: {{.*}}hidden-redecls-sub.h:3:9 | {{.*}} | isRedecl: 1
