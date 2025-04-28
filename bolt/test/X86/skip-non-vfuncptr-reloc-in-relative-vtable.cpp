// Test the fix that BOLT should skip speical handling of any non virtual
// function pointer relocations in relative vtable.

// RUN: %clang -fuse-ld=lld -o %t.so %s -Wl,-q \
// RUN:     -fexperimental-relative-c++-abi-vtables
// RUN: llvm-bolt %t.so -o %t.bolted.so

extern "C" unsigned long long _ZTVN10__cxxabiv117__class_type_infoE = 0;
extern "C" unsigned long long _ZTVN10__cxxabiv120__si_class_type_infoE = 0;

class A {
public:
  virtual void foo() {}
};

class B : public A {
  virtual void foo() override {}
};

int main() {
  B b;
  A *p = &b;
  p->foo();
  return 0;
}
