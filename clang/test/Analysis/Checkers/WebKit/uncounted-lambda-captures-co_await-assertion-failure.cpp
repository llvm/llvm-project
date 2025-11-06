// RUN: %clang_analyze_cc1 -analyzer-checker=webkit.UncountedLambdaCapturesChecker -std=c++20 -verify %s
// expected-no-diagnostics

template<typename Arg>
void foo(Arg&& arg)
{
    [&]{
        co_await [&](auto&&... args) {
        }(arg);
    }();
}
