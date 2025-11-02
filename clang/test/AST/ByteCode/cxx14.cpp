// RUN: %clang_cc1 -std=c++14 -verify=both,expected %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++14 -verify=both,ref      %s



constexpr int(*null_ptr)() = nullptr;
constexpr int test4 = (*null_ptr)(); // both-error {{must be initialized by a constant expression}} \
                                     // both-note {{evaluates to a null function pointer}}

