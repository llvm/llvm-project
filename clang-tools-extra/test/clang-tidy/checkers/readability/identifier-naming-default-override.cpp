// RUN: %check_clang_tidy %s readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-identifier-naming.DefaultCase: "lower_case", \
// RUN:     readability-identifier-naming.MacroDefinitionCase: "UPPER_CASE", \
// RUN:     readability-identifier-naming.TemplateParameterCase: "UPPER_CASE", \
// RUN:   }}' \
// RUN:   -- -fno-delayed-template-parsing

#define MyMacro
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for macro definition 'MyMacro' [readability-identifier-naming]
// CHECK-FIXES: #define MY_MACRO

namespace MyNamespace {
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for identifier 'MyNamespace' [readability-identifier-naming]
// CHECK-FIXES: namespace my_namespace {

using MyAlias = int;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for identifier 'MyAlias' [readability-identifier-naming]
// CHECK-FIXES: using my_alias = int;

int MyGlobal;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for identifier 'MyGlobal' [readability-identifier-naming]
// CHECK-FIXES: int my_global;

struct MyStruct {
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for identifier 'MyStruct' [readability-identifier-naming]
// CHECK-FIXES: struct my_struct {
  int MyField;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for identifier 'MyField' [readability-identifier-naming]
// CHECK-FIXES: int my_field;
};

template <typename MyTypename>
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: invalid case style for template parameter 'MyTypename' [readability-identifier-naming]
// CHECK-FIXES: template <typename MY_TYPENAME>
int MyFunction(int MyArgument) {
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for identifier 'MyFunction' [readability-identifier-naming]
// CHECK-MESSAGES: :[[@LINE-2]]:20: warning: invalid case style for identifier 'MyArgument' [readability-identifier-naming]
// CHECK-FIXES: int my_function(int my_argument) {
  int MyVariable = MyArgument;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for identifier 'MyVariable' [readability-identifier-naming]
// CHECK-FIXES: int my_variable = my_argument;
  return MyVariable;
// CHECK-FIXES: return my_variable;
}

template int MyFunction<int>(int);

}  // namespace MyNamespace

#define my_macro_2
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for macro definition 'my_macro_2' [readability-identifier-naming]
// CHECK-FIXES: #define MY_MACRO_2

namespace my_namespace_2 {

using my_alias = int;

my_alias my_global;

struct my_struct {
  int my_field;
};

template <typename my_typename>
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: invalid case style for template parameter 'my_typename' [readability-identifier-naming]
// CHECK-FIXES: template <typename MY_TYPENAME>
int my_function(int my_argument) {
  int my_variable = my_argument;
  return my_variable;
}

template int my_function<int>(int);

}  // namespace my_namespace_2
