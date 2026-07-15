// RUN: %clang_analyze_cc1 -std=c++20 %s -analyzer-checker=core.NullDereference -analyzer-output=text -verify

// expected-no-diagnostics

struct S { int x; };

void f(int x) { (void)x; }

int main()
{
    S s{42};
    auto& [x] = s;
    auto g = [x](){ f(x); }; // no warning
    g();
}
