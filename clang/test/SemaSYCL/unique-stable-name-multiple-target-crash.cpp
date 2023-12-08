// RUN: %clang_cc1 %s %s -std=c++17 -triple x86_64-linux-gnu -fsycl-is-device -verify -fsyntax-only -Wno-unused

// This would crash due to the double-inputs, since the 'magic static' use in
// the AST Context SCYL Filtering would end up caching an old version of the
// ASTContext object, which no longer exists in the second file's invocation.
//
// expected-no-diagnostics
class Empty {};
template <typename, typename F> __attribute__((sycl_kernel)) void kernel(F) {
    __builtin_sycl_unique_stable_name(F);
}

void use() {
  [](Empty) {
    auto lambda = []{};
    kernel<class i>(lambda);
  };
}
