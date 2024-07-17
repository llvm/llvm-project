// RUN: %clang_cc1 -fsyntax-only -verify -std=c++17 %s
// RUN: not %clang_cc1 -ast-dump %s -std=c++17 | FileCheck %s

void test() {
  while(!!!) // expected-error {{expected expression}}
    int whileBody;
  // CHECK: WhileStmt
  // CHECK: RecoveryExpr {{.*}} <line:{{.*}}:9, col:11> 'bool'
  // CHECK: whileBody 'int'

  for(!!!) // expected-error {{expected expression}} expected-error {{expected ';'}}
    int forBody;
  // CHECK: ForStmt
  // FIXME: the AST should have a RecoveryExpr to distinguish from for(;;)
  // CHECK-NOT: RecoveryExpr
  // CHECK: forBody 'int'

  for(auto c : !!!) // expected-error {{expected expression}}
    int forEachBody;
  // FIXME: parse the foreach body
  // CHECK-NOT: CXXForRangeStmt
  // CHECK-NOT: forEachBody 'int'

  do
    int doBody;
  while(!!!); // expected-error {{expected expression}}
  // CHECK: DoStmt
  // CHECK: doBody 'int'
  // CHECK: RecoveryExpr {{.*}} <line:{{.*}}:9, col:11> 'bool'

  if(!!!) // expected-error {{expected expression}}
    int ifBody;
  else
    int elseBody;
  // CHECK: IfStmt
  // CHECK: RecoveryExpr {{.*}} <line:{{.*}}:6, col:8> 'bool'
  // CHECK: ifBody 'int'
  // CHECK: elseBody 'int'

  switch(!!!) // expected-error {{expected expression}}
    int switchBody;
  // CHECK: SwitchStmt
  // CHECK: RecoveryExpr {{.*}} <line:{{.*}}:10, col:12> 'int'
  // CHECK: switchBody 'int'

  switch (;) // expected-error {{expected expression}}
    int switchBody;
  // CHECK: SwitchStmt
  // CHECK: NullStmt
  // CHECK: RecoveryExpr {{.*}} <col:11> 'int'
  // CHECK: switchBody 'int'

  switch (;;) // expected-error {{expected expression}}
    int switchBody;
  // CHECK: SwitchStmt
  // CHECK: NullStmt
  // CHECK: RecoveryExpr {{.*}} <col:11, col:12> 'int'
  // CHECK: switchBody 'int'

  switch (!!!;) // expected-error {{expected expression}}
    int switchBody;
  // CHECK: SwitchStmt
  // CHECK: RecoveryExpr {{.*}} <line:{{.*}}:11, col:14> 'int'
  // CHECK: switchBody 'int'
}
