// RUN: clang-include-cleaner %s --

namespace std {
class Foo {};
bool operator==(Foo, int) { return false; }
}
// no crash on a reference to a non-identifier symbol (operator ==).
bool s = std::operator==(std::Foo(), 1);
