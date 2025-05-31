// RUN: %clang_cc1 -emit-llvm %s -verify -o /dev/null

// Previously, delayed typo correction would cause a crash because codegen is
// run on each top-level declaration as we finish with it unless an
// unrecoverable error occured. However, with delayed typo correction, the
// error is not emit until the end of the TU, so CodeGen would be run on
// invalid declarations.
namespace GH140461 {
auto s{new auto(one)}; // expected-error {{use of undeclared identifier 'one'}}
} // namespace GH140461

