// RUN: rm -fr %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/foo.cppm -emit-module-interface -o %t/foo.pcm
// RUN: %clang_cc1 -std=c++20 %t/bar.cppm -emit-module-interface -fprebuilt-module-path=%t -o %t/bar.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -verify %t/Use.cpp -fsyntax-only
//
//--- foo.cppm
export module foo;
export class foo {
};

//--- bar.cppm
export module bar;
import foo;
export auto bar() {
  return foo{};
}

//--- Use.cpp
// expected-no-diagnostics
import bar;
auto foo() {
  // [module.reach]Note1:
  // While module interface units are reachable even when they
  // are only transitively imported via a non-exported import declaration,
  // namespace-scope names from such module interface units are not found
  // by name lookup ([basic.lookup]).
  auto b = bar(); // foo should be reachable here.
}
