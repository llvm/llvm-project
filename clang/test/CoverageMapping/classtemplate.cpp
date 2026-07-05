// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -triple %itanium_abi_triple -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name classtemplate.cpp %s > %tmapping
// RUN: FileCheck -input-file %tmapping %s --check-prefix=CHECK-CONSTRUCTOR
// RUN: FileCheck -input-file %tmapping %s --check-prefix=CHECK-GETTER
// RUN: FileCheck -input-file %tmapping %s --check-prefix=CHECK-SETTER
// RUN: FileCheck -input-file %tmapping %s --check-prefix=CHECK-INIT-LIST

template<class TT>
class Test {
public:
  enum BaseType {
    A, C, G, T, Invalid
  };
  const static int BaseCount = 4;
  double bases[BaseCount];

                                        // CHECK-CONSTRUCTOR: _ZN4TestIjEC
  Test() { }                            // CHECK-CONSTRUCTOR: File 0, [[@LINE]]:10 -> [[@LINE]]:13 = #0

  // FIXME: It would be nice to emit no-coverage for get, but trying to do this
  // runs afoul of cases like Test3::unmangleable below.
                                        // FIXME-GETTER: _ZNK4TestIjE3get
  double get(TT position) const {       // FIXME-GETTER: File 0, [[@LINE]]:33 -> [[@LINE+2]]:4 = 0
    return bases[position];
  }
                                        // CHECK-SETTER: _ZN4TestIjE3set
  void set(TT position, double value) { // CHECK-SETTER: File 0, [[@LINE]]:39 -> [[@LINE+2]]:4 = #0
    bases[position] = value;
  }
};

class Test2 {
                                        // CHECK-CONSTRUCTOR: _ZN5Test2C
  Test2() { }                           // CHECK-CONSTRUCTOR: File 0, [[@LINE]]:11 -> [[@LINE]]:14 = 0
                                        // CHECK-GETTER: _ZNK5Test23get
  double get(unsigned position) const { // CHECK-GETTER: File 0, [[@LINE]]:39 -> [[@LINE+2]]:4 = 0
    return 0.0;
  }
};

// Test3::unmangleable can't be mangled, since there isn't a complete type for
// the __is_final type trait expression. This would cause errors if we try to
// emit a no-coverage mapping for the method.
template <class T, bool = __is_final(T)> class UninstantiatedClassWithTraits {};
template <class T> class Test3 {
  void unmangleable(UninstantiatedClassWithTraits<T> x) {}
};

void abort() __attribute__((noreturn));

namespace std {
typedef decltype(sizeof(int)) size_t;

template <typename E> struct initializer_list {
  const E *p;
  size_t n;
  initializer_list(const E *p, size_t n) : p(p), n(n) {}
};

template <typename F, typename S> struct pair {
  F f;
  S s;
  pair(const F &f, const S &s) : f(f), s(s) {}
};

struct string {
  const char *str;
  string() { abort(); }
  string(const char *S) : str(S) {}
  ~string() { abort(); }
};

template<typename K, typename V>
struct map {
  using T = pair<K, V>;
  map(initializer_list<T> i, const string &s = string()) {}
  ~map() { abort(); }
};

}; // namespace std

// CHECK-INIT-LIST-LABEL: _Z5Test4v:
std::map<int, int> Test4() { // CHECK-INIT-LIST: File 0, [[@LINE]]:28 -> [[@LINE+3]]:2 = #0
  abort();                           // CHECK-INIT-LIST-NEXT: Gap,File 0, [[@LINE]]:11 -> [[@LINE+1]]:3 = 0
  return std::map<int, int>{{0, 0}}; // CHECK-INIT-LIST-NEXT: [[@LINE]]:3 -> [[@LINE]]:36 = 0
}

int main() {
  Test<unsigned> t;
  t.set(Test<unsigned>::A, 5.5);
  t.set(Test<unsigned>::T, 5.6);
  t.set(Test<unsigned>::G, 5.7);
  t.set(Test<unsigned>::C, 5.8);
  Test4();
  return 0;
}
