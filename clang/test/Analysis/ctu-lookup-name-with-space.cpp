// RUN: rm -rf %t
// RUN: mkdir %t

// RUN: echo '"%/S/Inputs/ctu-lookup-name-with-space.cpp" : ["g++", "-c", "%/S/Inputs/ctu-lookup-name-with-space.cpp"]' > %t/invocations.yaml
// RUN: %clang_extdef_map %S/Inputs/ctu-lookup-name-with-space.cpp -- > %t/externalDefMap.txt

// RUN: cd %t && %clang_cc1 -fsyntax-only -analyze \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=. \
// RUN:   -analyzer-config ctu-invocation-list=invocations.yaml \
// RUN:   -verify %s

void importee();

void trigger() {
  // Call an external function to trigger the parsing process of CTU index.
  // Refer to file Inputs/ctu-lookup-name-with-space.cpp for more details.

  importee(); // expected-no-diagnostics
}
