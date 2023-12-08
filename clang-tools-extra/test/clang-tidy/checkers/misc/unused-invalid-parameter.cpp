// RUN: %check_clang_tidy -fix-errors %s misc-unused-parameters %t

namespace GH56152 {
// There's no way to know whether the parameter is used or not if the parameter
// is an invalid declaration. Ensure the diagnostic is suppressed in this case.
void func(unknown_type value) { // CHECK-MESSAGES: :[[@LINE]]:11: error: unknown type name 'unknown_type'
  value += 1;
}
}

