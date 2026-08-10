// RUN: %clang_cc1 -verify -std=c++11 %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -x c++ -std=c++11 -fixit %t
// RUN: %clang_cc1 -Wall -pedantic -x c++ -std=c++11 %t
// RUN: not %clang_cc1 -std=c++11 -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

namespace ClassSpecifier {
  class [[]] [[]]
    attr_after_class_name_decl [[]] [[]]; // expected-error {{an attribute list cannot appear here}}
    // CHECK: fix-it:{{.*}}:{9:5-9:5}
    // CHECK: fix-it:{{.*}}:{9:32-9:41}

  class [[]] [[]]
   attr_after_class_name_definition [[]] [[]] [[]]{}; // expected-error {{an attribute list cannot appear here}}
   // CHECK: fix-it:{{.*}}:{14:4-14:4}
   // CHECK: fix-it:{{.*}}:{14:37-14:51}

  class base {};
  class [[]] [[]] final_class 
    alignas(float) [[]] final // expected-error {{an attribute list cannot appear here}}
    alignas(float) [[]] [[]] alignas(float): base{}; // expected-error {{an attribute list cannot appear here}}
    // CHECK: fix-it:{{.*}}:{19:19-19:19}
    // CHECK: fix-it:{{.*}}:{20:5-20:25}
    // CHECK: fix-it:{{.*}}:{19:19-19:19}
    // CHECK: fix-it:{{.*}}:{21:5-21:44}

  class [[]] [[]] final_class_another 
    [[]] [[]] alignas(16) final // expected-error {{an attribute list cannot appear here}}
    [[]] [[]] alignas(16) [[]]{}; // expected-error {{an attribute list cannot appear here}}
    // CHECK: fix-it:{{.*}}:{27:19-27:19}
    // CHECK: fix-it:{{.*}}:{28:5-28:27}
    // CHECK: fix-it:{{.*}}:{27:19-27:19}
    // CHECK: fix-it:{{.*}}:{29:5-29:31}
}

namespace BaseSpecifier {
  struct base1 {};
  struct base2 {};
  class with_base_spec : public [[a]] // expected-error {{an attribute list cannot appear here}} expected-warning {{unknown}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:26-[[@LINE-1]]:26}:"[{{\[}}a]]"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:33-[[@LINE-2]]:39}:""
                         virtual [[b]] base1, // expected-error {{an attribute list cannot appear here}} expected-warning {{unknown}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:26-[[@LINE-4]]:26}:"[{{\[}}b]]"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:34-[[@LINE-2]]:40}:""
                         virtual [[c]] // expected-error {{an attribute list cannot appear here}} expected-warning {{unknown}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:26-[[@LINE-1]]:26}:"[{{\[}}c]]"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:34-[[@LINE-2]]:40}:""
                         public [[d]] base2 {}; // expected-error {{an attribute list cannot appear here}} expected-warning {{unknown}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:26-[[@LINE-4]]:26}:"[{{\[}}d]]"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:33-[[@LINE-2]]:39}:""
}

[[__clang__::annotate("test")]] void annotate3();  // expected-warning {{'__clang__' is a predefined macro name, not an attribute scope specifier; did you mean '_Clang' instead?}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:12}:"_Clang"
