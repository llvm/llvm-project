static int staticIsGlobalVar = 0; // CHECK1: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:1:12 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK1 %s

int globalVar = 0; // CHECK2: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:4:5 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK2 %s

struct GlobalFoo { // CHECK3: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:7:8 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK3 %s

  struct GlobalBar { // CHECK4: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:10:10 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK4 %s
  };

  void foo() { // CHECK5: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:14:8 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK5 %s

    struct LocalFoo { }; // CHECK6: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:17:12 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK6 %s
  }

  virtual void bar() { } // CHECK-BAR: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:21:16 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK-BAR %s

  int globalField; // CHECK7: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:24:7 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK7 %s
};

enum GlobalEnum { // CHECK8: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:28:6 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK8 %s

  GlobalEnumCase = 0 // CHECK9: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:31:3 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK9 %s
};

namespace {
  struct AnonymousIsGlobalFoo { }; // CHECK10: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:36:10 -new-name=name %s -std=c++14 -x c++-header | FileCheck --check-prefix=CHECK10 %s

  void globalFoo() { } // CHECK11: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:39:8 -new-name=name %s -std=c++14 -x c++-header | FileCheck --check-prefix=CHECK11 %s
}

namespace globalNamespace { // CHECK12: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:43:11 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK12 %s
}

void func(int localParam) { // CHECK13: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:47:15 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK13 %s

  int localVar = 0; // CHECK14: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:50:7 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK14 %s

  enum LocalEnum { // CHECK15: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:53:8 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK15 %s

    LocalEnumCase = 0 // CHECK16: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:56:5 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK16 %s
  };

  struct LocalFoo: GlobalFoo { // CHECK17: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:60:10 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK17 %s

    struct LocalBar { }; // CHECK18: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:63:12 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK18 %s

    void foo() { } // CHECK19: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:66:10 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK19 %s

    // Bar is global since it overrides GlobalFoo::bar
    void bar() { } // CHECK-BAR: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:70:10 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK-BAR %s

    int localField; // CHECK20: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:73:10 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK20 %s
  };
}

auto escapable1 = []() -> auto {
  struct Global {  // ESCAPES1: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:79:10 -new-name=name %s -std=c++14 -x c++-header | FileCheck --check-prefix=ESCAPES1 %s

    int field = 2; // ESCAPES2: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:82:9 -new-name=name %s -std=c++14 -x c++-header | FileCheck --check-prefix=ESCAPES2 %s

    void foo() { } // ESCAPES3: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:85:10 -new-name=name %s -std=c++14 -x c++-header | FileCheck --check-prefix=ESCAPES3 %s
  };
  return Global();
};

auto escapable2() {
  struct Global {  // ESCAPES4: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:92:10 -new-name=name %s -std=c++14 -x c++-header | FileCheck --check-prefix=ESCAPES4 %s

    int field = 2; // ESCAPES5: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:95:9 -new-name=name %s -std=c++14 -x c++-header | FileCheck --check-prefix=ESCAPES5 %s

    void foo() { } // ESCAPES6: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:98:10 -new-name=name %s -std=c++14 -x c++-header | FileCheck --check-prefix=ESCAPES6 %s
  };
  return Global();
}

template<typename T>
struct C {
  T x;
};

decltype(auto) escapable4() {
  struct Global {  // ESCAPES7: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:110:10 -new-name=name %s -std=c++14 -x c++-header | FileCheck --check-prefix=ESCAPES7 %s
  };
  return C<Global>();
}

auto escapable5() -> decltype(auto) {
  struct Foo {
    struct Global { // ESCAPES8: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:118:12 -new-name=name %s -std=c++14 -x c++-header | FileCheck --check-prefix=ESCAPES8 %s
    };
  };
  return C<Foo::Global>();
}

auto escapable6() {
  struct Foo {
    struct Global { // ESCAPES9: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:127:12 -new-name=name %s -std=c++14 -x c++-header | FileCheck --check-prefix=ESCAPES9 %s
    };
  };
  return Foo::Global();
}

auto escapableOuter1() {
  struct Foo {
    auto escapableInner() {
      struct Global {  // ESCAPES10: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:137:14 -new-name=name %s -std=c++14 -x c++-header | FileCheck --check-prefix=ESCAPES10 %s
      };
      return Global();
    }
  }
  return Foo().escapableInner();
}

auto escapableOuter2() {
  auto escapableInner = []() -> auto {
    struct Global {  // ESCAPES11: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:148:12 -new-name=name %s -std=c++14 -x c++-header | FileCheck --check-prefix=ESCAPES11 %s
    };
    return Global();
  }
  return escapableInner();
}

void outer() {
  auto escapableInner2 = []() -> auto {
    struct Local {  // ESCAPES12: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:158:12 -new-name=name %s -std=c++14 -x c++-header | FileCheck --check-prefix=ESCAPES12 %s
    };
    return Local();
  }
  struct Foo {
    auto foo() {
      struct Local {  // ESCAPES13: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:165:14 -new-name=name %s -std=c++14 -x c++-header | FileCheck --check-prefix=ESCAPES13 %s
      };
      return Local();
    }
  };
}

struct Escapable {
  auto escapable() {
    struct Global {  // ESCAPES14: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:175:12 -new-name=name %s -std=c++14 -x c++-header | FileCheck --check-prefix=ESCAPES14 %s
    };
    return Global();
  }
};

auto escapableViaTypedef() {
  struct Global {  // ESCAPES15: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:183:10 -new-name=name %s -std=c++14 -x c++-header | FileCheck --check-prefix=ESCAPES15 %s
  };
  typedef Foo Global;
  return Foo();
}

auto nonescapable1 = []() -> auto {
  struct Local {  // NOESCAPE1: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:191:10 -new-name=name %s -std=c++14 -x c++ | FileCheck --check-prefix=NOESCAPE1 %s

    int field = 2; // NOESCAPE2: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:194:9 -new-name=name %s -std=c++14 -x c++ | FileCheck --check-prefix=NOESCAPE2 %s

    void foo() { } // NOESCAPE3: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:197:10 -new-name=name %s -std=c++14 -x c++ | FileCheck --check-prefix=NOESCAPE3 %s
  };
  return Local();
};

static void localOrGlobal1() { }; // ISGLOBAL1: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:203:13 -new-name=name %s -std=c++14 -x c++ | FileCheck --check-prefix=ISGLOBAL1 %s
// RUN: clang-refactor-test rename-initiate -at=%s:203:13 -new-name=name %s -std=c++14 -x c++-header | FileCheck --check-prefix=ISGLOBAL1 %s

namespace {

struct LocalOrGlobal { }; // ISGLOBAL2: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:209:13 -new-name=name %s -std=c++14 -x c++ | FileCheck --check-prefix=ISGLOBAL2 %s
// RUN: clang-refactor-test rename-initiate -at=%s:209:13 -new-name=name %s -std=c++14 -x c++-header | FileCheck --check-prefix=ISGLOBAL2 %s


void localOrGlobal2() { }; // ISGLOBAL3: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:214:13 -new-name=name %s -std=c++14 | FileCheck --check-prefix=ISGLOBAL3 %s
}


struct LocalOrGlobalWrapper1 {

  static void foo() { } // ISGLOBAL4: rename [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:221:15 -new-name=name %s -std=c++14 -x c++ | FileCheck --check-prefix=ISGLOBAL4 %s
// RUN: clang-refactor-test rename-initiate -at=%s:221:15 -new-name=name %s -std=c++14 -x c++-header | FileCheck --check-prefix=ISGLOBAL4 %s
};

void func2(int x) {
  auto lambda1 = [x] // CHECK21: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:226:16 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK21 %s
    (int z) { // CHECK22: rename local [[@LINE]]
// RUN: clang-refactor-test rename-initiate -at=%s:229:10 -new-name=name %s -std=c++14 | FileCheck --check-prefix=CHECK22 %s
  }
}
