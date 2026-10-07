// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s
// expected-no-diagnostics

struct X {};
namespace NS {
    bool operator==(X, X);
}
using namespace NS;
struct Y {
    X x;
    friend bool operator==(Y, Y);
};
bool operator==(Y, Y) = default;