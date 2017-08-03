
namespace ns { // CHECK4: rename [[@LINE]]:11 -> [[@LINE]]:13

struct Struct { }; // CHECK1: rename [[@LINE]]:8 -> [[@LINE]]:14

void func(); // CHECK2: rename [[@LINE]]:6 -> [[@LINE]]:10

void overload1(int x);
void overload1(double y); // CHECK3: rename [[@LINE]]:6 -> [[@LINE]]:15

}

using ns::Struct; // CHECK1: rename [[@LINE]]:11 -> [[@LINE]]:17

// RUN: clang-refactor-test rename-initiate -at=%s:13:11 -new-name=x %s | FileCheck --check-prefix=CHECK1 %s

using ns::func; // CHECK2: rename [[@LINE]]:11 -> [[@LINE]]:15

// RUN: clang-refactor-test rename-initiate -at=%s:17:11 -new-name=x %s | FileCheck --check-prefix=CHECK2 %s

using ns::overload1; // CHECK3: rename [[@LINE]]:11 -> [[@LINE]]:20

// RUN: clang-refactor-test rename-initiate -at=%s:21:11 -new-name=x %s | FileCheck --check-prefix=CHECK3 %s

using namespace ns; // CHECK4: rename [[@LINE]]:17 -> [[@LINE]]:19

// RUN: clang-refactor-test rename-initiate -at=%s:25:17 -new-name=x %s | FileCheck --check-prefix=CHECK4 %s

struct UsingConstructor {
  template<class T>
  UsingConstructor(T, typename T::Q);
};
struct UsingConstructorHere : UsingConstructor {
using UsingConstructor::UsingConstructor; // CHECK5: rename [[@LINE]]:7 -> [[@LINE]]:23
// CHECK5: rename [[@LINE-1]]:25 -> [[@LINE-1]]:41
};
// RUN: clang-refactor-test rename-initiate -at=%s:34:25 -new-name=x %s -std=c++14 | FileCheck --check-prefix=CHECK5 %s

template<typename T>
struct S : T {
  using T::T; // CHECK6: rename local [[@LINE]]:12 -> [[@LINE]]:13
// RUN: clang-refactor-test rename-initiate -at=%s:41:12 -new-name=x %s -std=c++14 | FileCheck --check-prefix=CHECK6 %s
  using typename T::Foo;
// RUN: not clang-refactor-test rename-initiate -at=%s:43:12 -new-name=x %s -std=c++14 2>&1 | FileCheck --check-prefix=CHECK-UNRESOLVED %s
// CHECK-UNRESOLVED: error: could not rename symbol at the given location
};
