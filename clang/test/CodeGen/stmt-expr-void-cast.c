// RUN: %clang_cc1 -emit-llvm -verify %s -o /dev/null

// Regression test for https://github.com/llvm/llvm-project/issues/173921
// Ensure that "extern void;" inside a multi-statement GNU statement expression
// does not cause a crash in CodeGen due to a leaked RecoveryExpr with
// dependent type.

extern int bar(int); // expected-note 3{{passing argument to parameter here}}

int test_switch(int x) {
  return 1 + bar(({     // expected-error {{passing 'void' to parameter of incompatible type 'int'}}
           int y;
           switch (x) {
           default:
             y = 7;
             break;
           }
           extern void; // expected-warning {{declaration does not declare anything}}
         }));
}

int test_single_stmt(int x) {
  return bar(({          // expected-error {{passing 'void' to parameter of incompatible type 'int'}}
           extern void;  // expected-warning {{declaration does not declare anything}}
         }));
}

int test_if(int x) {
  return bar(({          // expected-error {{passing 'void' to parameter of incompatible type 'int'}}
           if (x) {}
           extern void;  // expected-warning {{declaration does not declare anything}}
         }));
}
