// RUN: %clang_cc1 -emit-llvm -fexceptions -o - %s -triple x86_64-linux-gnu | FileCheck -check-prefixes=EH,CHECK %s
// RUN: %clang_cc1 -emit-llvm -o - %s -triple x86_64-linux-gnu | FileCheck -check-prefixes=NOEH,CHECK %s
namespace std {
  typedef decltype(sizeof(int)) size_t;

  // libc++'s implementation
  template <class _E>
  class initializer_list
  {
    const _E* __begin_;
    size_t    __size_;
    initializer_list(const _E* __b, size_t __s);
  };
}

class Object {
public:
  Object() = default;
  struct KV;
  explicit Object(std::initializer_list<KV> Properties);
};

class Value {
public:
  Value(std::initializer_list<Value> Elements);
  Value(const char *V);
  ~Value();
};

class ObjectKey {
public:
  ObjectKey(const char *S);
  ~ObjectKey();
};

struct Object::KV {
  ObjectKey K;
  Value V;
};

bool foo();
void bar() {
  // Verify we use conditional cleanups.
  foo() ? Object{{"key1", {"val1", "val2"}}} : Object{{"key2", "val2"}};
  // CHECK:   cond.true:
  // EH:        invoke void @_ZN9ObjectKeyC1EPKc
  // NOEH:      call void @_ZN9ObjectKeyC1EPKc
  // CHECK:     store ptr %K, ptr %cond-cleanup.save
}
