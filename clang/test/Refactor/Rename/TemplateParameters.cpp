template <typename T, int x> // CHECK1: rename local [[@LINE]]:20 -> [[@LINE]]:21
class Foo { // CHECK2: rename local [[@LINE-1]]:27 -> [[@LINE-1]]:28

T func(); // CHECK1-NEXT: rename local [[@LINE]]:1 -> [[@LINE]]:2

int array[x]; // CHECK2-NEXT: rename local [[@LINE]]:11 -> [[@LINE]]:12
};

// CHECK1-NOT: rename
// CHECK2-NOT: rename

// RUN: clang-refactor-test rename-initiate -at=%s:1:20 -new-name=U %s -fno-delayed-template-parsing | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:1:27 -new-name=U %s -fno-delayed-template-parsing | FileCheck --check-prefix=CHECK2 %s

template<typename T, int x>
T Foo<T, x>::func() { return T(); }

template <template<typename> class H, typename S> // CHECK3: rename local [[@LINE]]:36 -> [[@LINE]]:37
// CHECK4: rename local [[@LINE-1]]:48 -> [[@LINE-1]]:49
void templateTemplateParam(const H<S> &value) { // CHECK3-NEXT: rename local [[@LINE]]:34 -> [[@LINE]]:35
// CHECK4-NEXT: rename local [[@LINE-1]]:36 -> [[@LINE-1]]:37
}

// RUN: clang-refactor-test rename-initiate -at=%s:18:36 -at=%s:20:34 -new-name=U %s -fno-delayed-template-parsing | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-initiate -at=%s:18:48 -at=%s:20:36 -new-name=U %s -fno-delayed-template-parsing | FileCheck --check-prefix=CHECK4 %s
