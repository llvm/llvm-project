// RUN: not %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck -strict-whitespace %s

void foo() {
  (void)"\q \u123z \x \U \U123 \U12345 \u123 \xyzzy \777 \U"
  // CHECK: {{^  \(void\)"\\q \\u123z \\x \\U \\U123 \\U12345 \\u123 \\xyzzy \\777 \\U"$}}
  //
  //              (void)"\q \u123z \x \U \U123 \U12345 \u123 \xyzzy \777 \U"
  // CHECK: {{^         \^~$}}
  // CHECK: {{^            \^~~~~$}}
  // CHECK: {{^                   \^~$}}
  // CHECK: {{^                      \^~$}}
  // CHECK: {{^                         \^~~~~$}}
  // CHECK: {{^                               \^~~~~~~$}}
  // CHECK: {{^                                       \^~~~~$}}
  // CHECK: {{^                                             \^~$}}
  // CHECK: {{^                                                    \^~~~$}}
  // CHECK: {{^                                                         \^~$}}

  "123 \x \z";
  // CHECK: {{^  "123 \\x \\z";$}}
  //
  //              "123 \x \z";
  // CHECK: {{^       \^~$}}
  // CHECK: {{^          \^~$}}
}

#define foo() lots and lots of tokens, need at least 8 to fill up the smallvector buffer #BadThingsHappenNow
