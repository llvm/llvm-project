// RUN: %clang_analyze_cc1 -analyzer-checker=optin.cplusplus.UninitializedObject -verify %s
// expected-no-diagnostics

struct S
{
    S(bool b)
    : b(b)
    {}
    bool b{false};
    long long : 7; // padding
};

void f()
{
    S s(true);
}
