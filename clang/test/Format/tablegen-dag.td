// RUN: clang-format  %s -style="{Language: TableGen}" | FileCheck %s --strict-whitespace

def foo;
def bar;
def baz;

defvar x = [(foo(bar baz:$str))];.
// CHECK: x = [(foo(bar baz:$str))];
