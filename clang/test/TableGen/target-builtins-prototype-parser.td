// RUN: clang-tblgen -I %p/../../include/ %s --gen-clang-builtins | FileCheck %s
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_EXPECTED_LANES 2>&1 | FileCheck %s --check-prefix=ERROR_EXPECTED_LANES
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_EXPECTED_COMMA 2>&1 | FileCheck %s --check-prefix=ERROR_EXPECTED_COMMA
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_EXPECTED_TYPE 2>&1 | FileCheck %s --check-prefix=ERROR_EXPECTED_TYPE
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_EXPECTED_A 2>&1 | FileCheck %s --check-prefix=ERROR_EXPECTED_A
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_EXPECTED_B 2>&1 | FileCheck %s --check-prefix=ERROR_EXPECTED_B
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_EXPECTED_C 2>&1 | FileCheck %s --check-prefix=ERROR_EXPECTED_C
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_EXPECTED_D 2>&1 | FileCheck %s --check-prefix=ERROR_EXPECTED_D
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_EXPECTED_E 2>&1 | FileCheck %s --check-prefix=ERROR_EXPECTED_E
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_EXPECTED_F 2>&1 | FileCheck %s --check-prefix=ERROR_EXPECTED_F
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_EXPECTED_G 2>&1 | FileCheck %s --check-prefix=ERROR_EXPECTED_G
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_EXPECTED_H 2>&1 | FileCheck %s --check-prefix=ERROR_EXPECTED_H
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_EXPECTED_I 2>&1 | FileCheck %s --check-prefix=ERROR_EXPECTED_I
// RUN: not clang-tblgen -I %p/../../include/ %s --gen-clang-builtins -DERROR_EXPECTED_J 2>&1 | FileCheck %s --check-prefix=ERROR_EXPECTED_J

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

def : Builtin {
// CHECK: BUILTIN(__builtin_test_addrspace_attribute_01, "di*3", "")
  let Prototype = "double( [[addrspace[3]]] int*)";
  let Spellings = ["__builtin_test_addrspace_attribute_01"];
}

def : Builtin {
// CHECK: BUILTIN(__builtin_test_addrspace_attribute_02, "Ii*5i*d", "")
  let Prototype = "_Constant [[addrspace[5]]] int* (int*, double)";
  let Spellings = ["__builtin_test_addrspace_attribute_02"];
}

def : Builtin {
// CHECK: BUILTIN(__builtin_test_addrspace_attribute_04, "Ii&4id*7", "")
  let Prototype = "_Constant [[addrspace[4]]] int& (int , [[addrspace[7]]] double*)";
  let Spellings = ["__builtin_test_addrspace_attribute_04"];
}

#ifdef ERROR_EXPECTED_E
def : Builtin {
// ERROR_EXPECTED_E: :[[# @LINE + 1]]:7: error: Expected opening bracket '[' after 'addrspace'
  let Prototype = "_Constant [[addrspace]] int& (int , double*)";
  let Spellings = ["__builtin_test_addrspace_attribute_04"];
}
#endif

#ifdef ERROR_EXPECTED_F
def : Builtin {
// ERROR_EXPECTED_F: :[[# @LINE + 1]]:7: error: Address space attribute can only be specified with a pointer or reference type
  let Prototype = "_Constant [[addrspace[4]]] int (int , double*)";
  let Spellings = ["__builtin_test_addrspace_attribute_04"];
}
#endif

#ifdef ERROR_EXPECTED_G
def : Builtin {
// ERROR_EXPECTED_G: :[[# @LINE + 1]]:7: error: Expecetd valid integer for 'addrspace' attribute
  let Prototype = "_Constant [[addrspace[k]]] int* (int , double*)";
  let Spellings = ["__builtin_test_addrspace_attribute_04"];
}
#endif

#ifdef ERROR_EXPECTED_H
def : Builtin {
// ERROR_EXPECTED_H: :[[# @LINE + 1]]:7: error: Expected closing bracket ']' after address space specification
  let Prototype = "_Constant [[addrspace[6 int* (int , double*)";
  let Spellings = ["__builtin_test_addrspace_attribute_04"];
}
#endif

#ifdef ERROR_EXPECTED_I
def : Builtin {
// ERROR_EXPECTED_I: :[[# @LINE + 1]]:7: error: Expected closing brackets ']]' for attribute list
  let Prototype = "_Constant [[addrspace[6] int* (int , double*)";
  let Spellings = ["__builtin_test_addrspace_attribute_04"];
}
#endif

#ifdef ERROR_EXPECTED_J
def : Builtin {
// ERROR_EXPECTED_J: :[[# @LINE + 1]]:7: error: Unknown attribute name specified
  let Prototype = "_Constant [[invalid]] int* (int , double*)";
  let Spellings = ["__builtin_test_addrspace_attribute_04"];
}
#endif
