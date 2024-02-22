class Foo /* Test 1 */ {              // CHECK: rename [[@LINE]]:7 -> [[@LINE]]:10
public:
  void foo(int x);
};

void Foo::foo(int x) /* Test 2 */ {}  // CHECK: rename [[@LINE]]:6 -> [[@LINE]]:9

// Test 1.
// RUN: clang-refactor-test rename-initiate -at=%s:1:7 -new-name=Bar %s | FileCheck %s
// Test 2.
// RUN: clang-refactor-test rename-initiate -at=%s:6:6 -new-name=Bar %s | FileCheck %s

struct ForwardDeclaration; // CHECK2: rename [[@LINE]]:8 -> [[@LINE]]:26

ForwardDeclaration *variable; // CHECK2: rename [[@LINE]]:1 -> [[@LINE]]:19

// RUN: clang-refactor-test rename-initiate -at=%s:13:8 -at=%s:15:1 -new-name=Bar %s | FileCheck --check-prefix=CHECK2 %s

template<typename T>
struct SpecializationWithoutDefinition { }; // CHECK3: rename [[@LINE]]:8 -> [[@LINE]]:39

template<>
struct SpecializationWithoutDefinition<int> { }; // CHECK3: rename [[@LINE]]:8 -> [[@LINE]]:39

template<>
struct SpecializationWithoutDefinition<float>; // CHECK3: rename [[@LINE]]:8 -> [[@LINE]]:39

// RUN: clang-refactor-test rename-initiate -at=%s:20:8 -at=%s:23:8 -new-name=Bar %s | FileCheck --check-prefix=CHECK3 %s

template<typename T>
struct Class {
  void method();
};

template<typename T>        // CHECK4: [[@LINE]]:19 -> [[@LINE]]:20
void Class<T>::method() { } // CHECK4: [[@LINE]]:12 -> [[@LINE]]:13

// RUN: clang-refactor-test rename-initiate -at=%s:35:19 -new-name=Bar %s | FileCheck --check-prefix=CHECK4 %s
