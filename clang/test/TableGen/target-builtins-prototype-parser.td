// RUN: clang-tblgen -I %p/../../include/ %s --gen-clang-builtins | FileCheck %s
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_EXPECTED_LANES 2>&1 | FileCheck %s --check-prefix=ERROR_EXPECTED_LANES
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_EXPECTED_COMMA 2>&1 | FileCheck %s --check-prefix=ERROR_EXPECTED_COMMA
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_EXPECTED_TYPE 2>&1 | FileCheck %s --check-prefix=ERROR_EXPECTED_TYPE
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_EXPECTED_A 2>&1 | FileCheck %s --check-prefix=ERROR_EXPECTED_A
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_EXPECTED_B 2>&1 | FileCheck %s --check-prefix=ERROR_EXPECTED_B
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_EXPECTED_C 2>&1 | FileCheck %s --check-prefix=ERROR_EXPECTED_C
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_EXPECTED_D 2>&1 | FileCheck %s --check-prefix=ERROR_EXPECTED_D

include "clang/Basic/BuiltinsBase.td"

def : Builtin {
// CHECK: BUILTIN(__builtin_01, "E8idE4b", "")
  let Prototype = "_ExtVector<8,int>(double, _ExtVector<4,        bool>)";
  let Spellings = ["__builtin_01"];
}

def : Builtin {
// CHECK: BUILTIN(__builtin_02, "E8UiE4s", "")
  let Prototype = "_ExtVector<8,unsigned int>(_ExtVector<4, short>)";
  let Spellings = ["__builtin_02"];
}

def : Builtin {
// CHECK: BUILTIN(__builtin_03, "di", "")
  let Prototype = "double(int)";
  let Spellings = ["__builtin_03"];
}

def : Builtin {
// CHECK: BUILTIN(__builtin_04, "diIUi", "")
 let Prototype = "double(int, _Constant unsigned int)";
  let Spellings = ["__builtin_04"];
}

def : Builtin {
// CHECK: BUILTIN(__builtin_05, "v&v&", "")
 let Prototype = "void&(void&)";
  let Spellings = ["__builtin_05"];
}

def : Builtin {
// CHECK: BUILTIN(__builtin_06, "v*v*cC*.", "")
 let Prototype = "void*(void*, char const*, ...)";
  let Spellings = ["__builtin_06"];
}

def : Builtin {
// CHECK: BUILTIN(__builtin_07, "E8iE4dE4b.", "")
  let Prototype = "_ExtVector<8, int>(_ExtVector<4,double>, _ExtVector<4, bool>, ...)";
  let Spellings = ["__builtin_07"];
}

def : Builtin {
// CHECK: BUILTIN(__builtin_08, "di*R", "")
  let Prototype = "double(int * restrict)";
  let Spellings = ["__builtin_08"];
}

def : Builtin {
// CHECK: BUILTIN(__builtin_09, "V2yy", "")
  let Prototype = "_Vector<2, __bf16>(__bf16)";
  let Spellings = ["__builtin_09"];
}

#ifdef ERROR_EXPECTED_LANES
def : Builtin {
// ERROR_EXPECTED_LANES: :[[# @LINE + 1]]:7: error: Expected number of lanes after '_ExtVector<'
  let Prototype = "_ExtVector<int>(double)";
  let Spellings = ["__builtin_test_use_clang_extended_vectors"];
}
#endif

#ifdef ERROR_EXPECTED_COMMA
def : Builtin {
// ERROR_EXPECTED_COMMA: :[[# @LINE + 1]]:7: error: Expected ',' after number of lanes in '_ExtVector<'
  let Prototype = "_ExtVector<8 int>(double)";
  let Spellings = ["__builtin_test_use_clang_extended_vectors"];
}
#endif

#ifdef ERROR_EXPECTED_TYPE
def : Builtin {
// ERROR_EXPECTED_TYPE: :[[# @LINE + 1]]:7: error: Expected '>' after scalar type in '_ExtVector<N, type>'
  let Prototype = "_ExtVector<8, int (double)";
  let Spellings = ["__builtin_test_use_clang_extended_vectors"];
}
#endif

#ifdef ERROR_EXPECTED_A
def : Builtin {
// ERROR_EXPECTED_A: :[[# @LINE + 1]]:7: error: Expected '<' after '_ExtVector'
  let Prototype = "_ExtVector(8, int) (double)";
  let Spellings = ["__builtin_test_use_clang_extended_vectors"];
}
#endif

#ifdef ERROR_EXPECTED_B
def : Builtin {
// ERROR_EXPECTED_B: :[[# @LINE + 1]]:7: error: Expected '<' after '_ExtVector'
  let Prototype = "double(_ExtVector(8, int))";
  let Spellings = ["__builtin_test_use_clang_extended_vectors"];
}
#endif

#ifdef ERROR_EXPECTED_C
def : Builtin {
// ERROR_EXPECTED_C: :[[# @LINE + 1]]:7: error: Unknown Type: _EtxVector<8, int>
  let Prototype = "_EtxVector<8, int>(void)";
  let Spellings = ["__builtin_test_use_clang_extended_vectors"];
}
#endif

#ifdef ERROR_EXPECTED_D
def : Builtin {
// ERROR_EXPECTED_D: :[[# @LINE + 1]]:7: error: Expected number of lanes after '_ExtVector<'
  let Prototype = "_ExtVector<>(void)";
  let Spellings = ["__builtin_test_use_clang_extended_vectors"];
}
#endif
