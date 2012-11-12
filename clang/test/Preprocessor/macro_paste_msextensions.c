// RUN: %clang_cc1 -P -E -fms-extensions %s | FileCheck -strict-whitespace %s

// This horrible stuff should preprocess into (other than whitespace):
//   int foo;
//   int bar;
//   int baz;

int foo;

// CHECK: int foo;

#define comment /##/  dead tokens live here
comment This is stupidity

int bar;

// CHECK: int bar;

#define nested(x) int x comment cute little dead tokens...

nested(baz)  rise of the dead tokens

;

// CHECK: int baz
// CHECK: ;


// rdar://8197149 - VC++ allows invalid token pastes: (##baz
#define foo(x) abc(x)
#define bar(y) foo(##baz(y))
bar(q)

// CHECK: abc(baz(q))
