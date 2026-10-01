// RUN: cat %s | clang-repl | FileCheck %s

// Undoing an unscoped enumeration also undoes its enumerators, which are
// visible in the enclosing context. Only value printing is checked here:
// the output of printf() is buffered separately and may come out of order.

enum E { Red, Green };
%undo
int Red = 5;
Red
// CHECK: (int) 5

enum E2 { Blue };
%undo
enum E2 { Blue = 7 };
Blue
// CHECK-NEXT: (E2) (Blue) : unsigned int 7

namespace N { enum F { Yellow = 1 }; }
%undo
namespace N { int Yellow = 8; }
N::Yellow
// CHECK-NEXT: (int) 8

%quit
