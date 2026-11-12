// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins 2>&1 \
// RUN:   | FileCheck %s -DFILE=%s

include "clang/Basic/BuiltinsBase.td"

def : Builtin {
// CHECK: [[FILE]]:[[#@LINE+1]]:7: error: Expected a type after 'long'
  let Prototype = "long int(long)";
  let Spellings = ["__builtin_long"];
}

// `long long` is only parsed as a modifier for OpenCL targets.
def : Builtin {
// CHECK: [[FILE]]:[[#@LINE+2]]:7: error: Expected a type after 'long long'
  let EnableOpenCLLong = 1;
  let Prototype = "int(long long)";
  let Spellings = ["__builtin_long_long"];
}

def : Builtin {
// CHECK: [[FILE]]:[[#@LINE+1]]:7: error: Expected a type after 'signed'
  let Prototype = "int(signed)";
  let Spellings = ["__builtin_signed"];
}

def : Builtin {
// CHECK: [[FILE]]:[[#@LINE+1]]:7: error: Expected a type after 'unsigned'
  let Prototype = "int(unsigned)";
  let Spellings = ["__builtin_unsigned"];
}

def : Builtin {
// CHECK: [[FILE]]:[[#@LINE+1]]:7: error: Expected a type after '_Constant'
  let Prototype = "int(_Constant)";
  let Spellings = ["__builtin_constant"];
}
