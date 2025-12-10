// RUN: %check_clang_tidy %s readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-identifier-naming.DefaultCase: "lower_case" }}'

int BadGlobal;

int good_global;

struct BadStruct {
  int BadField;
};

struct good_struct {
  int good_field;
};

int BadFunction(int BadParameter) {
  int BadVariable = BadParameter;
  return BadVariable;
}

int good_function(int good_parameter) {
  int good_variable = good_parameter;
  return good_variable;
}

