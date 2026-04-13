// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits -fno-diagnostics-show-line-numbers %s 2>&1 | FileCheck %s -strict-whitespace

[[nodiscard]] enum class E1 { };
// expected-error@-1 {{misplaced attributes; expected attributes here}}
// CHECK: {{^}}{{\[\[}}nodiscard]] enum class E1 { };
// CHECK: {{^}}~~~~~~~~~~~~~           ^
// CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:1-[[@LINE-4]]:15}:""
// CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:25-[[@LINE-5]]:25}:"{{\[\[}}nodiscard]]"

[[nodiscard]] enum struct E2 { };
// expected-error@-1 {{misplaced attributes; expected attributes here}}
// CHECK: {{^}}{{\[\[}}nodiscard]] enum struct E2 { };
// CHECK: {{^}}~~~~~~~~~~~~~            ^
// CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:1-[[@LINE-4]]:15}:""
// CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:26-[[@LINE-5]]:26}:"{{\[\[}}nodiscard]]"

[[nodiscard]] enum          class E3 { };
// expected-error@-1 {{misplaced attributes; expected attributes here}}
// CHECK: {{^}}{{\[\[}}nodiscard]] enum          class E3 { };
// CHECK: {{^}}~~~~~~~~~~~~~                    ^
// CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:1-[[@LINE-4]]:15}:""
// CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:34-[[@LINE-5]]:34}:"{{\[\[}}nodiscard]]"

[[nodiscard]] enum  /*comment*/ class E4 { };
// expected-error@-1 {{misplaced attributes; expected attributes here}}
// CHECK: {{^}}{{\[\[}}nodiscard]] enum  /*comment*/ class E4 { };
// CHECK: {{^}}~~~~~~~~~~~~~                        ^
// CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:1-[[@LINE-4]]:15}:""
// CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:38-[[@LINE-5]]:38}:"{{\[\[}}nodiscard]]"

[[nodiscard]] enum { A = 0 };
// expected-error@-1 {{misplaced attributes; expected attributes here}}
// CHECK: {{^}}{{\[\[}}nodiscard]] enum { A = 0 };
// CHECK: {{^}}~~~~~~~~~~~~~     ^
// CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:1-[[@LINE-4]]:15}:""
// CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:19-[[@LINE-5]]:19}:"{{\[\[}}nodiscard]]"

namespace NS {
  enum class E5;
}

[[nodiscard]] enum class NS::E5 { };
// expected-error@-1 {{misplaced attributes; expected attributes here}}
// CHECK: {{^}}{{\[\[}}nodiscard]] enum class NS::E5 { };
// CHECK: {{^}}~~~~~~~~~~~~~           ^
// CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:1-[[@LINE-4]]:15}:""
// CHECK: fix-it:"{{.*}}":{[[@LINE-5]]:25-[[@LINE-5]]:25}:"{{\[\[}}nodiscard]]"
