// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -mconstructor-aliases -fclangir -clangir-disable-emit-cxx-default -fclangir-lifetime-check="history=all" -clangir-verify-diagnostics -emit-cir %s -o %t.cir

class _j {};
typedef _j* jobj;

typedef enum SType {
  INFO_ENUM_0 = 9,
  INFO_ENUM_1 = 2020,
} SType;

typedef SType ( *FnPtr2)(unsigned session, jobj* surface);

struct X {
  struct entries {
    FnPtr2 wildfn = nullptr;
  };
  static entries e;
};

void nullpassing() {
  jobj o = nullptr;
  X::e.wildfn(0, &o);
}