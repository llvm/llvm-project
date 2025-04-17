// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header %t/hu-01.h \
// RUN:  -o %t/hu-01.pcm

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header %t/hu-02.h \
// RUN:  -Wno-experimental-header-units -fmodule-file=%t/hu-01.pcm -o %t/hu-02.pcm

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header %t/hu-03.h \
// RUN:  -Wno-experimental-header-units \
// RUN:  -fmodule-file=%t/hu-01.pcm -o %t/hu-03.pcm

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header %t/hu-04.h \
// RUN:  -Wno-experimental-header-units -fmodule-file=%t/hu-02.pcm \
// RUN:  -fmodule-file=%t/hu-03.pcm -o %t/hu-04.pcm

// RUN: %clang_cc1 -std=c++20 -emit-obj %t/main.cpp \
// RUN:  -Wno-experimental-header-units -fmodule-file=%t/hu-04.pcm
//--- hu-01.h
template <typename T>
struct A {
  ~A() { f(); }
  auto f() const { return 0; }
};

template <typename T>
struct B {
  int g() const { return a.f(); }
  A<T> a;
};

//--- hu-02.h
import "hu-01.h";

template <typename = void>
struct C {
  void h() {
    B<int>().g();
  }
};

template struct A<double>;

//--- hu-03.h
import "hu-01.h";

inline B<int> b() {
  return {};
}

//--- hu-04.h
import "hu-02.h";
import "hu-03.h";

inline void f4() {
  C{}.h();
}

//--- main.cpp
import "hu-04.h";

int main() {
  f4();
}
