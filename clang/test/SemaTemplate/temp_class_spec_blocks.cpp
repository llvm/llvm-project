// RUN: %clang_cc1 -fsyntax-only -verify %s -fblocks
// expected-no-diagnostics
template<typename T>
struct is_unary_block {
  static const bool value = false;
};

template<typename T, typename U>
struct is_unary_block<T (^)(U)> {
  static const bool value = true;
};

int is_unary_block0[is_unary_block<int>::value ? -1 : 1];
int is_unary_block1[is_unary_block<int (^)()>::value ? -1 : 1];
int is_unary_block2[is_unary_block<int (^)(int, bool)>::value ? -1 : 1];
int is_unary_block3[is_unary_block<int (^)(bool)>::value ? 1 : -1];
int is_unary_block4[is_unary_block<int (^)(int)>::value ? 1 : -1];

template<typename T>
struct is_unary_block_with_same_return_type_as_argument_type {
  static const bool value = false;
};

template<typename T>
struct is_unary_block_with_same_return_type_as_argument_type<T (^)(T)> {
  static const bool value = true;
};

int is_unary_block5[is_unary_block_with_same_return_type_as_argument_type<int>::value ? -1 : 1];
int is_unary_block6[is_unary_block_with_same_return_type_as_argument_type<int (^)()>::value ? -1 : 1];
int is_unary_block7[is_unary_block_with_same_return_type_as_argument_type<int (^)(int, bool)>::value ? -1 : 1];
int is_unary_block8[is_unary_block_with_same_return_type_as_argument_type<int (^)(bool)>::value ? -1 : 1];
int is_unary_block9[is_unary_block_with_same_return_type_as_argument_type<int (^)(int)>::value ? 1 : -1];
int is_unary_block10[is_unary_block_with_same_return_type_as_argument_type<int (^)(int, ...)>::value ? -1 : 1];
int is_unary_block11[is_unary_block_with_same_return_type_as_argument_type<int (^ const)(int)>::value ? -1 : 1];
