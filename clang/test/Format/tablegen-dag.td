// RUN: grep -Ev "// *[A-Z-]+:" %s \
// RUN:   | clang-format  -assume-filename=%t/foo.td \
// RUN:   | FileCheck -strict-whitespace %s

def foo;
def bar;
def baz;

defvar x = [(foo(bar baz:$str))];
// CHECK: x = [(foo (bar baz:$str))];
