// RUN: %check_clang_tidy %s readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: {}}' \
// RUN:   -- -fno-delayed-template-parsing

// Empty options effectively disable the check, allowing PascalCase...
#define MyMacro

namespace MyNamespace {

using MyAlias = int;

int MyGlobal;

struct MyStruct {
  int MyField;
};

template <typename MyTypename>
int MyFunction(int MyArgument) {
  int MyVariable = MyArgument;
  return MyVariable;
}

template int MyFunction<int>(int);

}  // namespace MyNamespace

// ...or lower_case for the same set of symbol types.
#define my_macro_2

namespace my_namespace_2 {

using my_alias = int;

my_alias my_global;

struct my_struct {
  int my_field;
};

template <typename my_typename>
int my_function(int my_argument) {
  int my_variable = my_argument;
  return my_variable;
}

template int my_function<int>(int);

}  // namespace my_namespace_2
