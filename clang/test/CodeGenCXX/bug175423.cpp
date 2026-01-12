// RUN: %clang_cc1 -std=c++20 -emit-llvm -o - %s -w | FileCheck %s
// CHECK: define{{.*}} i32 @main

template <typename, typename>
inline constexpr bool is_same_v = false;

template <typename T>
inline constexpr bool is_same_v<T, T> = true;

template <int &T> void FuncTemplate() { T; }

template <typename T> struct Container {
  static void Execute() {
    if (is_same_v<T, int>)
      ;
    static int InternalVar;
    FuncTemplate<InternalVar>;
  }
};

int main() { Container<char>::Execute; }
