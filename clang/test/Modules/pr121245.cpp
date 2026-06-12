// If this test fails, it should be investigated under Debug builds.
// Before the PR, this test was encountering an `llvm_unreachable()`.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header %t/hu-01.h \
// RUN:  -fcxx-exceptions -o %t/hu-01.pcm

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header %t/hu-02.h \
// RUN:  -Wno-experimental-header-units -fcxx-exceptions \
// RUN:  -fmodule-file=%t/hu-01.pcm -o %t/hu-02.pcm

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header %t/hu-03.h \
// RUN:  -Wno-experimental-header-units -fcxx-exceptions \
// RUN:  -fmodule-file=%t/hu-01.pcm -o %t/hu-03.pcm

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header %t/hu-04.h \
// RUN:  -Wno-experimental-header-units -fcxx-exceptions \
// RUN:  -fmodule-file=%t/hu-01.pcm -o %t/hu-04.pcm

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header %t/hu-05.h \
// RUN:  -Wno-experimental-header-units -fcxx-exceptions \
// RUN:  -fmodule-file=%t/hu-03.pcm -fmodule-file=%t/hu-04.pcm \
// RUN:  -fmodule-file=%t/hu-01.pcm -o %t/hu-05.pcm

// RUN: %clang_cc1 -std=c++20 -emit-obj %t/main.cpp \
// RUN:  -Wno-experimental-header-units -fcxx-exceptions \
// RUN:  -fmodule-file=%t/hu-02.pcm -fmodule-file=%t/hu-05.pcm \
// RUN:  -fmodule-file=%t/hu-04.pcm -fmodule-file=%t/hu-03.pcm \
// RUN:  -fmodule-file=%t/hu-01.pcm

//--- hu-01.h
template <typename T>
struct A {
  A() {}
  ~A() {}
};

template <typename T>
struct EBO : T {
  EBO() = default;
};

template <typename T>
struct HT : EBO<A<T>> {};

//--- hu-02.h
import "hu-01.h";

inline void f() {
  HT<int>();
}

//--- hu-03.h
import "hu-01.h";

struct C {
  C();

  HT<long> _;
};

//--- hu-04.h
import "hu-01.h";

void g(HT<long> = {});

//--- hu-05.h
import "hu-03.h";
import "hu-04.h";
import "hu-01.h";

struct B {
  virtual ~B() = default;

  virtual void f() {
    HT<long>();
  }
};

//--- main.cpp
import "hu-02.h";
import "hu-05.h";
import "hu-03.h";

int main() {
  f();
  C();
  B();
}
