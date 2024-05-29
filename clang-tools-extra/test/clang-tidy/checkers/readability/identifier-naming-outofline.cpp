// RUN: %check_clang_tidy %s readability-identifier-naming %t -std=c++20 \
// RUN:   --config='{CheckOptions: { \
// RUN:     readability-identifier-naming.MethodCase: CamelCase, \
// RUN:  }}'

namespace SomeNamespace {
namespace Inner {

class SomeClass {
public:
    template <typename T>
    int someMethod();
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for method 'someMethod' [readability-identifier-naming]
// CHECK-FIXES: {{^}}    int SomeMethod();
};
template <typename T>
int SomeClass::someMethod() {
// CHECK-FIXES: {{^}}int SomeClass::SomeMethod() {
    return 5;
}

} // namespace Inner

void someFunc() {
    Inner::SomeClass S;
    S.someMethod<int>();
// CHECK-FIXES: {{^}}    S.SomeMethod<int>();
}

} // namespace SomeNamespace
