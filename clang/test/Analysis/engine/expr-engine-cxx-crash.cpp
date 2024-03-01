// RUN: %clang_analyze_cc1 -analyzer-checker=core.DivideZero -std=c++23 -verify %s
// expected-no-diagnostics

struct S
{
    constexpr auto operator==(this auto, S)
    {
        return true;
    }
};

int main()
{
    return S {} == S {};
}
