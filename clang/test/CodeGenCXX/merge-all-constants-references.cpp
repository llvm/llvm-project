// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fmerge-all-constants %s -o /dev/null

struct A {
};

struct B {
  const struct A& a = {};
};

void Test(const struct B&);

void Run() {
  Test({});
}
