// RUN: clang-tblgen -I %p/../../include/ %s --gen-clang-builtins | FileCheck %s
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_INDEX 2>&1 | FileCheck %s --check-prefix=ERROR_INDEX
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_NEGATIVE_INDEX 2>&1 | FileCheck %s --check-prefix=ERROR_NEGATIVE_INDEX
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_NOT_CONSTANT 2>&1 | FileCheck %s --check-prefix=ERROR_NOT_CONSTANT
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_EMPTY 2>&1 | FileCheck %s --check-prefix=ERROR_EMPTY
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_DUPLICATE 2>&1 | FileCheck %s --check-prefix=ERROR_DUPLICATE
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_UPPER_OVERFLOW 2>&1 | FileCheck %s --check-prefix=ERROR_UPPER_OVERFLOW
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_LOWER_OVERFLOW 2>&1 | FileCheck %s --check-prefix=ERROR_LOWER_OVERFLOW

include "clang/Basic/BuiltinsBase.td"

def : Builtin {
// CHECK: Builtin::Info{{.*}} __builtin_one_range {{.*}} /* nR<1:0:4> */
  let Prototype = "int(int, _Constant int)";
  let Spellings = ["__builtin_one_range"];
  let Attributes = [NoThrow, ArgRange<1, 0, 4>];
}

def : Builtin {
// CHECK: Builtin::Info{{.*}} __builtin_two_ranges {{.*}} /* cR<0:-8:8,2:1:65> */
  let Prototype = "void(_Constant int, _ExtVector<4, int>, _Constant unsigned int)";
  let Spellings = ["__builtin_two_ranges"];
  let Attributes = [Const, ArgRange<0, -8, 8>, ArgRange<2, 1, 65>];
}

#ifdef ERROR_INDEX
def : Builtin {
// ERROR_INDEX: :[[# @LINE - 1]]:1: error: ArgRange index 2 out of bounds
  let Prototype = "int(int, _Constant int)";
  let Spellings = ["__builtin_error_index"];
  let Attributes = [ArgRange<2, 0, 4>];
}
#endif

#ifdef ERROR_NEGATIVE_INDEX
def : Builtin {
// ERROR_NEGATIVE_INDEX: :[[# @LINE - 1]]:1: error: ArgRange index -1 out of bounds
  let Prototype = "int(int, _Constant int)";
  let Spellings = ["__builtin_error_negative_index"];
  let Attributes = [ArgRange<-1, 0, 4>];
}
#endif

#ifdef ERROR_NOT_CONSTANT
def : Builtin {
// ERROR_NOT_CONSTANT: :[[# @LINE - 1]]:1: error: ArgRange argument 0 must be _Constant
  let Prototype = "int(int, _Constant int)";
  let Spellings = ["__builtin_error_not_constant"];
  let Attributes = [ArgRange<0, 0, 4>];
}
#endif

#ifdef ERROR_EMPTY
def : Builtin {
// ERROR_EMPTY: :[[# @LINE - 1]]:1: error: ArgRange range is empty
  let Prototype = "int(int, _Constant int)";
  let Spellings = ["__builtin_error_empty"];
  let Attributes = [ArgRange<1, 4, 4>];
}
#endif

#ifdef ERROR_DUPLICATE
def : Builtin {
// ERROR_DUPLICATE: :[[# @LINE - 1]]:1: error: duplicate ArgRange for argument 1
  let Prototype = "int(int, _Constant int)";
  let Spellings = ["__builtin_error_duplicate"];
  let Attributes = [ArgRange<1, 0, 4>, ArgRange<1, 0, 8>];
}
#endif

#ifdef ERROR_UPPER_OVERFLOW
def : Builtin {
// ERROR_UPPER_OVERFLOW: :[[# @LINE - 1]]:1: error: ArgRange bounds must fit in a 32-bit int
  let Prototype = "int(int, _Constant unsigned int)";
  let Spellings = ["__builtin_error_upper_overflow"];
  let Attributes = [ArgRange<1, 0, 4294967296>];
}
#endif

#ifdef ERROR_LOWER_OVERFLOW
def : Builtin {
// ERROR_LOWER_OVERFLOW: :[[# @LINE - 1]]:1: error: ArgRange bounds must fit in a 32-bit int
  let Prototype = "int(int, _Constant int)";
  let Spellings = ["__builtin_error_lower_overflow"];
  let Attributes = [ArgRange<1, -2147483649, 0>];
}
#endif
