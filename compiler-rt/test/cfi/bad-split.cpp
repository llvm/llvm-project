// GlobalSplit used to lose type metadata for classes with virtual bases but no virtual methods.
// RUN: %clangxx_cfi -o %t1 %s && %run %t1

// UNSUPPORTED: windows-msvc

struct Z {
};

struct ZZ : public virtual Z {
};

struct A : public ZZ {
};

struct B : public A {
};

int main() {
  A* a = new B();
  B *b = (B*)a;
}
