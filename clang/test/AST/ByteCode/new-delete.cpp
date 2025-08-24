// RUN: %clang_cc1            -verify=expected,both                        -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -std=c++20 -verify=expected,both                        -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -std=c++20 -verify=expected,both -triple=i686-linux-gnu -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1            -verify=ref,both                                                                     %s
// RUN: %clang_cc1 -std=c++20 -verify=ref,both                                                                     %s
// RUN: %clang_cc1 -std=c++20 -verify=ref,both      -triple=i686-linux-gnu                                         %s

#if __cplusplus >= 202002L

constexpr int *Global = new int(12); // both-error {{must be initialized by a constant expression}} \
                                     // both-note {{pointer to heap-allocated object}} \
                                     // both-note {{heap allocation performed here}}

static_assert(*(new int(12)) == 12); // both-error {{not an integral constant expression}} \
                                     // both-note {{allocation performed here was not deallocated}}


constexpr int a() {
  new int(12); // both-note {{allocation performed here was not deallocated}}
  return 1;
}
static_assert(a() == 1, ""); // both-error {{not an integral constant expression}}

constexpr int b() {
  int *i = new int(12);
  int m = *i;
  delete(i);
  return m;
}
static_assert(b() == 12, "");


struct S {
  int a;
  int b;

  static constexpr S *create(int a, int b) {
    return new S(a, b);
  }
};

constexpr int c() {
  S *s = new S(12, 13);

  int i = s->a;
  delete s;

  return i;
}
static_assert(c() == 12, "");

/// Dynamic allocation in function ::create(), freed in function d().
constexpr int d() {
  S* s = S::create(12, 14);

  int sum = s->a + s->b;
  delete s;
  return sum;
}
static_assert(d() == 26);


/// Test we emit the right diagnostic for several allocations done on
/// the same site.
constexpr int loop() {
  for (int i = 0; i < 10; ++i) {
    int *a = new int[10]; // both-note {{not deallocated (along with 9 other memory leaks)}}
  }

  return 1;
}
static_assert(loop() == 1, ""); // both-error {{not an integral constant expression}}

/// No initializer.
constexpr int noInit() {
  int *i = new int;
  delete i;
  return 0;
}
static_assert(noInit() == 0, "");

/// Try to delete a pointer that hasn't been heap allocated.
constexpr int notHeapAllocated() { // both-error {{never produces a constant expression}}
  int A = 0; // both-note 2{{declared here}}
  delete &A; // both-note 2{{delete of pointer '&A' that does not point to a heap-allocated object}}

  return 1;
}
static_assert(notHeapAllocated() == 1, ""); // both-error {{not an integral constant expression}} \
                                            // both-note {{in call to 'notHeapAllocated()'}}

consteval int deleteNull() {
  int *A = nullptr;
  delete A;
  return 1;
}
static_assert(deleteNull() == 1, "");

consteval int doubleDelete() { // both-error {{never produces a constant expression}}
  int *A = new int;
  delete A;
  delete A; // both-note 2{{delete of pointer that has already been deleted}}
  return 1;
}
static_assert(doubleDelete() == 1); // both-error {{not an integral constant expression}} \
                                    // both-note {{in call to 'doubleDelete()'}}

constexpr int AutoArray() {
  auto array = new int[]{0, 1, 2, 3};
  int ret = array[3];
  delete [] array;
  return ret;
}

static_assert(AutoArray() == 3);

#if 0
consteval int largeArray1(bool b) {
  if (b) {
    int *a = new int[1ull<<32]; // both-note {{cannot allocate array; evaluated array bound 4294967296 is too large}}
    delete[] a;
  }
  return 1;
}
static_assert(largeArray1(false) == 1, "");
static_assert(largeArray1(true) == 1, ""); // both-error {{not an integral constant expression}} \
                                           // both-note {{in call to 'largeArray1(true)'}}

consteval int largeArray2(bool b) {
  if (b) {
    S *a = new S[1ull<<32]; // both-note {{cannot allocate array; evaluated array bound 4294967296 is too large}}
    delete[] a;
  }
  return 1;
}
static_assert(largeArray2(false) == 1, "");
static_assert(largeArray2(true) == 1, ""); // both-error {{not an integral constant expression}} \
                                           // both-note {{in call to 'largeArray2(true)'}}
#endif
namespace Arrays {
  constexpr int d() {
    int *Arr = new int[12];

    Arr[0] = 1;
    Arr[1] = 5;

    int sum = Arr[0] + Arr[1];
    delete[] Arr;
    return sum;
  }
  static_assert(d() == 6);


  constexpr int mismatch1() { // both-error {{never produces a constant expression}}
    int *i = new int(12); // both-note {{allocated with 'new' here}} \
                          // both-note 2{{heap allocation performed here}}
    delete[] i; // both-warning {{'delete[]' applied to a pointer that was allocated with 'new'}} \
                // both-note 2{{array delete used to delete pointer to non-array object of type 'int'}}
    return 6;
  }
  static_assert(mismatch1() == 6); // both-error {{not an integral constant expression}} \
                                   // both-note {{in call to 'mismatch1()'}}

  constexpr int mismatch2() { // both-error {{never produces a constant expression}}
    int *i = new int[12]; // both-note {{allocated with 'new[]' here}} \
                          // both-note 2{{heap allocation performed here}}
    delete i; // both-warning {{'delete' applied to a pointer that was allocated with 'new[]'}} \
              // both-note 2{{non-array delete used to delete pointer to array object of type 'int[12]'}}
    return 6;
  }
  static_assert(mismatch2() == 6); // both-error {{not an integral constant expression}} \
                                   // both-note {{in call to 'mismatch2()'}}

  constexpr int mismatch3() { // both-error {{never produces a constant expression}}
    int a = 0;
    struct S {};
    struct T : S {};
    T *p = new T[3]{}; // both-note 2{{heap allocation performed here}}
    delete (S*)p; // both-note 2{{non-array delete used to delete pointer to array object of type 'T[3]'}}

    return 0;

  }
  static_assert(mismatch3() == 0); // both-error {{not an integral constant expression}} \
                                   // both-note {{in call to}}

  /// Array of composite elements.
  constexpr int foo() {
    S *ss = new S[12];

    ss[0].a = 12;

    int m = ss[0].a;

    delete[] ss;
    return m;
  }
  static_assert(foo() == 12);



  constexpr int ArrayInit() {
    auto array = new int[4]{0, 1, 2, 3};
    int ret = array[0];
    delete [] array;
    return ret;
  }
  static_assert(ArrayInit() == 0, "");

  struct S {
    float F;
  };
  constexpr float ArrayInit2() {
    auto array = new S[4]{};
    float ret = array[0].F;
    delete [] array;
    return ret;
  }
  static_assert(ArrayInit2() == 0.0f, "");
}

namespace std {
  struct type_info;
  struct destroying_delete_t {
    explicit destroying_delete_t() = default;
  } inline constexpr destroying_delete{};
  struct nothrow_t {
    explicit nothrow_t() = default;
  } inline constexpr nothrow{};
  using size_t = decltype(sizeof(0));
  enum class align_val_t : size_t {};
};

[[nodiscard]] void *operator new(std::size_t, const std::nothrow_t&) noexcept;
[[nodiscard]] void *operator new(std::size_t, std::align_val_t, const std::nothrow_t&) noexcept;
[[nodiscard]] void *operator new[](std::size_t, const std::nothrow_t&) noexcept;
[[nodiscard]] void *operator new[](std::size_t, std::align_val_t, const std::nothrow_t&) noexcept;
[[nodiscard]] void *operator new[](std::size_t, std::align_val_t);
void operator delete(void*, const std::nothrow_t&) noexcept;
void operator delete(void*, std::align_val_t, const std::nothrow_t&) noexcept;
void operator delete[](void*, const std::nothrow_t&) noexcept;
void operator delete[](void*, std::align_val_t, const std::nothrow_t&) noexcept;

struct placement_new_arg {};
void *operator new(std::size_t, placement_new_arg);
void operator delete(void*, placement_new_arg);


constexpr void *operator new(std::size_t, void *p) { return p; }
namespace std {
  template<typename T> constexpr T *construct(T *p) { return new (p) T; }
  template<typename T> constexpr void destroy(T *p) { p->~T(); }
}



namespace PlacementNew {
  constexpr int foo() { // both-error {{never produces a constant expression}}
    char c[sizeof(int)];
    new (c) int{12}; // both-note {{this placement new expression is not supported in constant expressions before C++2c}}
    return 0;
  }
}

namespace NowThrowNew {
  constexpr bool erroneous_array_bound_nothrow(long long n) {
    int *p = new (std::nothrow) int[n];
    bool result = p != nullptr;
    delete[] p;
    return result;
  }
  static_assert(erroneous_array_bound_nothrow(3));
  static_assert(erroneous_array_bound_nothrow(0));
  static_assert(erroneous_array_bound_nothrow(-1) == 0);
  static_assert(!erroneous_array_bound_nothrow(1LL << 62));

  struct S { int a; };
  constexpr bool erroneous_array_bound_nothrow2(long long n) {
    S *p = new (std::nothrow) S[n];
    bool result = p != nullptr;
    delete[] p;
    return result;
  }
  static_assert(erroneous_array_bound_nothrow2(3));
  static_assert(erroneous_array_bound_nothrow2(0));
  static_assert(erroneous_array_bound_nothrow2(-1) == 0);
  static_assert(!erroneous_array_bound_nothrow2(1LL << 62));

  constexpr bool erroneous_array_bound(long long n) {
    delete[] new int[n]; // both-note {{array bound -1 is negative}} both-note {{array bound 4611686018427387904 is too large}}
    return true;
  }
  static_assert(erroneous_array_bound(3));
  static_assert(erroneous_array_bound(0));
  static_assert(erroneous_array_bound(-1)); // both-error {{constant expression}} both-note {{in call}}
  static_assert(erroneous_array_bound(1LL << 62)); // both-error {{constant expression}} both-note {{in call}}

  constexpr bool evaluate_nothrow_arg() {
    bool ok = false;
    delete new ((ok = true, std::nothrow)) int;
    return ok;
  }
  static_assert(evaluate_nothrow_arg());
}

namespace placement_new_delete {
  struct ClassSpecificNew {
    void *operator new(std::size_t);
  };
  struct ClassSpecificDelete {
    void operator delete(void*);
  };
  struct DestroyingDelete {
    void operator delete(DestroyingDelete*, std::destroying_delete_t);
  };
  struct alignas(64) Overaligned {};

  constexpr bool ok() {
    delete new Overaligned;
    delete ::new ClassSpecificNew;
    ::delete new ClassSpecificDelete;
    ::delete new DestroyingDelete;
    return true;
  }
  static_assert(ok());

  constexpr bool bad(int which) {
    switch (which) {
    case 0:
      delete new (placement_new_arg{}) int; // both-note {{this placement new expression is not supported in constant expressions}}
      break;

    case 1:
      delete new ClassSpecificNew; // both-note {{call to class-specific 'operator new'}}
      break;

    case 2:
      delete new ClassSpecificDelete; // both-note {{call to class-specific 'operator delete'}}
      break;

    case 3:
      delete new DestroyingDelete; // both-note {{call to class-specific 'operator delete'}}
      break;

    case 4:
      // FIXME: This technically follows the standard's rules, but it seems
      // unreasonable to expect implementations to support this.
      delete new (std::align_val_t{64}) Overaligned; // both-note {{this placement new expression is not supported in constant expressions}}
      break;
    }

    return true;
  }
  static_assert(bad(0)); // both-error {{constant expression}} \
                         // both-note {{in call}}
  static_assert(bad(1)); // both-error {{constant expression}} both-note {{in call}}
  static_assert(bad(2)); // both-error {{constant expression}} both-note {{in call}}
  static_assert(bad(3)); // both-error {{constant expression}} both-note {{in call}}
  static_assert(bad(4)); // both-error {{constant expression}} \
                         // both-note {{in call}}
}




namespace delete_random_things {
  static_assert((delete new int, true));
  static_assert((delete (int*)0, true));
  int n; // both-note {{declared here}}
  static_assert((delete &n, true)); // both-error {{}} \
                                    // both-note {{delete of pointer '&n' that does not point to a heap-allocated object}}
  struct A { int n; };
  static_assert((delete &(new A)->n, true)); // both-error {{}} \
                                             // both-note {{delete of pointer to subobject }}
  static_assert((delete (new int + 1), true)); // both-error {{}} \
                                               // both-note {{delete of pointer '&{*new int#0} + 1' that does not point to complete object}}
  static_assert((delete[] (new int[3] + 1), true)); // both-error {{}} \
                                                    // both-note {{delete of pointer to subobject}}
  static_assert((delete &(int&)(int&&)0, true)); // both-error {{}} \
                                                 // both-note {{delete of pointer '&0' that does not point to a heap-allocated object}} \
                                                 // both-note {{temporary created here}}
}

namespace value_dependent_delete {
  template<typename T> void f(T *p) {
    int arr[(delete p, 0)];
  }
}

namespace memory_leaks {
  static_assert(*new bool(true)); // both-error {{}} both-note {{allocation performed here was not deallocated}}

  constexpr bool *f() { return new bool(true); } // both-note {{allocation performed here was not deallocated}}
  static_assert(*f()); // both-error {{}}

  struct UP {
    bool *p;
    constexpr ~UP() { delete p; }
    constexpr bool &operator*() { return *p; }
  };
  constexpr UP g() { return {new bool(true)}; }
  static_assert(*g()); // ok

  constexpr bool h(UP p) { return *p; }
  static_assert(h({new bool(true)})); // ok
}

/// From test/SemaCXX/cxx2a-consteval.cpp

namespace std {
template <typename T> struct remove_reference { using type = T; };
template <typename T> struct remove_reference<T &> { using type = T; };
template <typename T> struct remove_reference<T &&> { using type = T; };
template <typename T>
constexpr typename std::remove_reference<T>::type&& move(T &&t) noexcept {
  return static_cast<typename std::remove_reference<T>::type &&>(t);
}
}

namespace cxx2a {
struct A {
  int* p = new int(42); // both-note 3{{heap allocation performed here}}
  consteval int ret_i() const { return p ? *p : 0; }
  consteval A ret_a() const { return A{}; }
  constexpr ~A() { delete p; }
};

consteval int by_value_a(A a) { return a.ret_i(); }

consteval int const_a_ref(const A &a) {
  return a.ret_i();
}

consteval int rvalue_ref(const A &&a) {
  return a.ret_i();
}

consteval const A &to_lvalue_ref(const A &&a) {
  return a;
}

void test() {
  constexpr A a{ nullptr };
  { int k = A().ret_i(); }

  { A k = A().ret_a(); } // both-error {{'cxx2a::A::ret_a' is not a constant expression}} \
                         // both-note {{heap-allocated object is not a constant expression}}
  { A k = to_lvalue_ref(A()); } // both-error {{'cxx2a::to_lvalue_ref' is not a constant expression}} \
                                // both-note {{reference to temporary is not a constant expression}} \
                                // both-note {{temporary created here}}
  { A k = to_lvalue_ref(A().ret_a()); } // both-error {{'cxx2a::to_lvalue_ref' is not a constant expression}} \
                                        // both-note {{reference to temporary is not a constant expression}} \
                                        // both-note {{temporary created here}}
  { int k = A().ret_a().ret_i(); } // both-error {{'cxx2a::A::ret_a' is not a constant expression}} \
                                   // both-note {{heap-allocated object is not a constant expression}}
  { int k = by_value_a(A()); }
  { int k = const_a_ref(A()); }
  { int k = const_a_ref(a); }
  { int k = rvalue_ref(A()); }
  { int k = rvalue_ref(std::move(a)); }
  { int k = const_a_ref(A().ret_a()); }
  { int k = const_a_ref(to_lvalue_ref(A().ret_a())); }
  { int k = const_a_ref(to_lvalue_ref(std::move(a))); }
  { int k = by_value_a(A().ret_a()); }
  { int k = by_value_a(to_lvalue_ref(static_cast<const A&&>(a))); }
  { int k = (A().ret_a(), A().ret_i()); } // both-error {{'cxx2a::A::ret_a' is not a constant expression}} \
                                          // both-note {{is not a constant expression}} \
                                          // both-warning {{left operand of comma operator has no effect}}
  { int k = (const_a_ref(A().ret_a()), A().ret_i()); } // both-warning {{left operand of comma operator has no effect}}
}
}

constexpr int *const &p = new int; // both-error {{must be initialized by a constant expression}} \
                                   // both-note {{pointer to heap-allocated object}} \
                                   // both-note {{allocation performed here}}

constexpr const int *A[] = {nullptr, nullptr, new int{12}}; // both-error {{must be initialized by a constant expression}} \
                                                            // both-note {{pointer to heap-allocated object}} \
                                                            // both-note {{allocation performed here}}

struct Sp {
  const int *p;
};
constexpr Sp ss[] = {Sp{new int{154}}}; // both-error {{must be initialized by a constant expression}} \
                                        // both-note {{pointer to heap-allocated object}} \
                                        // both-note {{allocation performed here}}

namespace DeleteRunsDtors {
  struct InnerFoo {
    int *mem;
    constexpr ~InnerFoo() {
      delete mem;
    }
  };

  struct Foo {
    int *a;
    InnerFoo IF;

    constexpr Foo() {
      a = new int(13);
      IF.mem = new int(100);
    }
    constexpr ~Foo() { delete a; }
  };

  constexpr int abc() {
    Foo *F = new Foo();
    int n = *F->a;
    delete F;

    return n;
  }
  static_assert(abc() == 13);

  constexpr int abc2() {
    Foo *f = new Foo[3];

    delete[] f;

    return 1;
  }
  static_assert(abc2() == 1);
}

/// FIXME: There is a slight difference in diagnostics here.
namespace FaultyDtorCalledByDelete {
  struct InnerFoo {
    int *mem;
    constexpr ~InnerFoo() {
      if (mem) {
        (void)(1/0); // both-warning {{division by zero is undefined}} \
                     // both-note {{division by zero}}
      }
      delete mem;
    }
  };

  struct Foo {
    int *a;
    InnerFoo IF;

    constexpr Foo() {
      a = new int(13);
      IF.mem = new int(100);
    }
    constexpr ~Foo() { delete a; }
  };

  constexpr int abc() {
    Foo *F = new Foo();
    int n = *F->a;
    delete F; // both-note 2{{in call to}}

    return n;
  }
  static_assert(abc() == 13); // both-error {{not an integral constant expression}} \
                              // both-note {{in call to 'abc()'}}
}

namespace DeleteThis {
  constexpr bool super_secret_double_delete() {
    struct A {
      constexpr ~A() { delete this; } // both-note {{destruction of object that is already being destroyed}} \
                                      // ref-note {{in call to}}
    };
    delete new A; // both-note {{in call to}}
    return true;
  }
  static_assert(super_secret_double_delete()); // both-error {{not an integral constant expression}} \
                                               // both-note {{in call to 'super_secret_double_delete()'}}

  struct B {
    constexpr void reset() { delete this; }
  };
  static_assert(((new B)->reset(), true));
}

namespace CastedDelete {
  struct S {
    constexpr S(int *p) : p(p) {}
    constexpr virtual ~S() { *p = 1; }
    int *p;
  };
  struct T: S {
    // implicit destructor defined eagerly because it is constexpr and virtual
    using S::S;
  };

  constexpr int vdtor_1() {
    int a;
    delete (S*)new T(&a);
    return a;
  }
  static_assert(vdtor_1() == 1);

  constexpr int foo() { // both-error {{never produces a constant expression}}
      struct S {};
      struct T : S {};
      S *p = new T();
      delete p; // both-note 2{{delete of object with dynamic type 'T' through pointer to base class type 'S' with non-virtual destructor}}
      return 1;
  }
  static_assert(foo() == 1); // both-error {{not an integral constant expression}} \
                             // both-note {{in call to}}
}

constexpr void use_after_free_2() { // both-error {{never produces a constant expression}}
  struct X { constexpr void f() {} };
  X *p = new X;
  delete p;
  p->f(); // both-note {{member call on heap allocated object that has been deleted}}
}

/// std::allocator definition
namespace std {
  using size_t = decltype(sizeof(0));
  template<typename T> struct allocator {
    constexpr T *allocate(size_t N) {
      return (T*)__builtin_operator_new(sizeof(T) * N); // #alloc
    }
    constexpr void deallocate(void *p) {
      __builtin_operator_delete(p); // both-note 2{{std::allocator<...>::deallocate' used to delete pointer to object allocated with 'new'}} \
                                    // both-note {{used to delete a null pointer}} \
                                    // both-note {{delete of pointer '&no_deallocate_nonalloc' that does not point to a heap-allocated object}}
    }
  };
  template<typename T, typename ...Args>
  constexpr void construct_at(void *p, Args &&...args) { // #construct
    new (p) T((Args&&)args...);
  }
}

constexpr int *escape = std::allocator<int>().allocate(3); // both-error {{constant expression}} \
                                                           // both-note {{pointer to subobject of heap-allocated}} \
                                                           // both-note {{heap allocation performed here}}

/// Specialization for float, using operator new/delete.
namespace std {
  using size_t = decltype(sizeof(0));
  template<> struct allocator<float> {
    constexpr float *allocate(size_t N) {
      return (float*)operator new (sizeof(float) * N);
    }
    constexpr void deallocate(void *p) {
      operator delete(p);
    }
  };
}

namespace OperatorNewDelete {

  constexpr bool mismatched(int alloc_kind, int dealloc_kind) {
    int *p;
    switch (alloc_kind) {
    case 0:
      p = new int; // both-note {{heap allocation performed here}}
      break;
    case 1:
      p = new int[1]; // both-note {{heap allocation performed here}}
      break;
    case 2:
      p = std::allocator<int>().allocate(1); // both-note 2{{heap allocation performed here}}
      break;
    }
    switch (dealloc_kind) {
    case 0:
      delete p; // both-note {{'delete' used to delete pointer to object allocated with 'std::allocator<...>::allocate'}}
      break;
    case 1:
      delete[] p; // both-note {{'delete' used to delete pointer to object allocated with 'std::allocator<...>::allocate'}}
      break;
    case 2:
      std::allocator<int>().deallocate(p); // both-note 2{{in call}}
      break;
    }
    return true;
  }
  static_assert(mismatched(0, 2)); // both-error {{constant expression}} \
                                   // both-note {{in call to}}
  static_assert(mismatched(1, 2)); // both-error {{constant expression}} \
                                   // both-note {{in call to}}
  static_assert(mismatched(2, 0)); // both-error {{constant expression}} \
                                   // both-note {{in call}}
  static_assert(mismatched(2, 1)); // both-error {{constant expression}} \
                                   // both-note {{in call}}
  static_assert(mismatched(2, 2));

  constexpr bool zeroAlloc() {
    int *F = std::allocator<int>().allocate(0);
    std::allocator<int>().deallocate(F);
    return true;
  }
  static_assert(zeroAlloc());

  constexpr int arrayAlloc() {
    int *F = std::allocator<int>().allocate(2);
    F[0] = 10; // both-note {{assignment to object outside its lifetime is not allowed in a constant expression}}
    F[1] = 13;
    int Res = F[1] + F[0];
    std::allocator<int>().deallocate(F);
    return Res;
  }
  static_assert(arrayAlloc() == 23); // both-error {{not an integral constant expression}} \
                                     // both-note {{in call to}}

  struct S {
    int i;
    constexpr S(int i) : i(i) {}
    constexpr ~S() { }
  };

  /// FIXME: This is broken in the current interpreter.
  constexpr bool structAlloc() {
    S *s = std::allocator<S>().allocate(1);

    s->i = 12; // ref-note {{assignment to object outside its lifetime is not allowed in a constant expression}}

    bool Res = (s->i == 12);
    std::allocator<S>().deallocate(s);

    return Res;
  }
  static_assert(structAlloc()); // ref-error {{not an integral constant expression}} \
                                // ref-note {{in call to}}

  constexpr bool structAllocArray() {
    S *s = std::allocator<S>().allocate(9);

    s[2].i = 12; // ref-note {{assignment to object outside its lifetime is not allowed in a constant expression}}
    bool Res = (s[2].i == 12);
    std::allocator<S>().deallocate(s);

    return Res;
  }
  static_assert(structAllocArray()); // ref-error {{not an integral constant expression}} \
                                     // ref-note {{in call to}}

  constexpr bool alloc_from_user_code() {
    void *p = __builtin_operator_new(sizeof(int)); // both-note {{cannot allocate untyped memory in a constant expression; use 'std::allocator<T>::allocate'}}
    __builtin_operator_delete(p);
    return true;
  }
  static_assert(alloc_from_user_code()); // both-error {{constant expression}} \
                                         // both-note {{in call to}}


  constexpr int no_deallocate_nullptr = (std::allocator<int>().deallocate(nullptr), 1); // both-error {{constant expression}} \
                                                                                        // both-note {{in call}}

  static_assert((std::allocator<float>().deallocate(std::allocator<float>().allocate(10)), 1) == 1);
}

namespace Limits {
  template<typename T>
  constexpr T dynarray(int elems, int i) {
    T *p;
    if constexpr (sizeof(T) == 1)
      p = new T[elems]{"fox"};
    else
      p = new T[elems]{1, 2, 3};
    T n = p[i];
    delete [] p;
    return n;
  }
  static_assert(dynarray<char>(5, 0) == 'f');


#if __LP64__
  template <typename T>
  struct S {
      constexpr S(unsigned long long N)
      : data(nullptr){
          data = alloc.allocate(N); // both-note {{in call to 'this->alloc.allocate(18446744073709551615)}}
      }
      constexpr T operator[](std::size_t i) const {
        return data[i];
      }

      constexpr ~S() {
          alloc.deallocate(data);
      }
      std::allocator<T> alloc;
      T* data;
  };

  constexpr std::size_t s = S<std::size_t>(~0UL)[42]; // both-error {{constexpr variable 's' must be initialized by a constant expression}} \
                                                      // both-note@#alloc {{cannot allocate array; evaluated array bound 2305843009213693951 is too large}} \
                                                      // both-note {{in call to}}
#endif
}

/// Just test that we reject placement-new expressions before C++2c.
/// Tests for successful expressions are in placement-new.cpp
namespace Placement {
  consteval auto ok1() { // both-error {{never produces a constant expression}}
    bool b;
    new (&b) bool(true); // both-note 2{{this placement new expression is not supported in constant expressions before C++2c}}
    return b;
  }
  static_assert(ok1()); // both-error {{not an integral constant expression}} \
                        // both-note {{in call to}}

  /// placement-new should be supported before C++26 in std functions.
  constexpr int ok2() {
    int *I = new int;
    std::construct_at<int>(I);
    int r = *I;
    delete I;
    return r;
  }
  static_assert(ok2()== 0);
}

constexpr bool virt_delete(bool global) {
  struct A {
    virtual constexpr ~A() {}
  };
  struct B : A {
    void operator delete(void *);
    constexpr ~B() {}
  };

  A *p = new B;
  if (global)
    ::delete p;
  else
    delete p; // both-note {{call to class-specific 'operator delete'}}
  return true;
}
static_assert(virt_delete(true));
static_assert(virt_delete(false)); // both-error {{not an integral constant expression}} \
                                   // both-note {{in call to}}


namespace ToplevelScopeInTemplateArg {
  class string {
  public:
    char *mem;
    constexpr string() {
      this->mem = new char(1);
    }
    constexpr ~string() {
      delete this->mem;
    }
    constexpr unsigned size() const { return 4; }
  };


  template <unsigned N>
  void test() {};

  void f() {
      test<string().size()>();
      static_assert(string().size() == 4);
  }
}

template <typename T>
struct SS {
    constexpr SS(unsigned long long N)
    : data(nullptr){
        data = alloc.allocate(N);
        for(std::size_t i = 0; i < N; i ++)
            std::construct_at<T>(data + i, i);
    }

    constexpr SS()
    : data(nullptr){
        data = alloc.allocate(1);
        std::construct_at<T>(data);
    }

    constexpr T operator[](std::size_t i) const {
      return data[i];
    }

    constexpr ~SS() {
        alloc.deallocate(data);
    }
    std::allocator<T> alloc;
    T* data;
};
constexpr unsigned short ssmall = SS<unsigned short>(100)[42];
constexpr auto Ss = SS<S>()[0];


namespace IncompleteArray {
  struct A {
    int b = 10;
  };
  constexpr int test1() {
    int n = 5;
    int* a = new int[n];
    int c = a[0]; // both-note {{read of uninitialized object}}
    delete[] a;
    return c;
  }
  static_assert(test1() == 10); // both-error {{not an integral constant expression}} \
                                // both-note {{in call to}}

  constexpr int test2() {
    int n = 0;
    int* a = new int[n];
    delete[] a;
    return 10;
  }
  static_assert(test2() == 10);

  /// In this case, the type of the initializer is A[2], while the full size of the
  /// allocated array is of course 5. The remaining 3 elements need to be initialized
  /// using A's constructor.
  constexpr int test3() {
    int n = 3;
    A* a = new A[n]{5, 1};
    int c = a[0].b + a[1].b + a[2].b;
    delete[] a;
    return c;
  }
  static_assert(test3() == (5 + 1 + 10));

  constexpr int test4() {
    auto n = 3;
    int *a = new int[n]{12};
    int c =  a[0] + a[1];
    delete[] a;
    return c;
  }
  static_assert(test4() == 12);


  constexpr char *f(int n) {
    return new char[n]();
  }
  static_assert((delete[] f(2), true));
}

namespace NonConstexprArrayCtor {
  struct S {
    S() {} // both-note 2{{declared here}}
  };

  constexpr bool test() { // both-error {{never produces a constant expression}}
     auto s = new S[1]; // both-note 2{{non-constexpr constructor}}
     return true;
  }
  static_assert(test()); // both-error {{not an integral constant expression}} \
                         // both-note {{in call to}}
}

namespace ArrayBaseCast {
  struct A {};
  struct B : A {};
  constexpr bool test() {
    B *b = new B[2];

    A* a = b;

    delete[] b;
    return true;
  }
  static_assert(test());
}

namespace PR45350 {
  int q;
  struct V { int n; int *p = &n; constexpr ~V() { *p = *p * 10 + n; }};
  constexpr int f(int n) {
    int k = 0;
    V *p = new V[n];
    for (int i = 0; i != n; ++i) {
      if (p[i].p != &p[i].n) return -1;
      p[i].n = i;
      p[i].p = &k;
    }
    delete[] p;
    return k;
  }
  // [expr.delete]p6:
  //   In the case of an array, the elements will be destroyed in order of
  //   decreasing address
  static_assert(f(6) == 543210);
}

namespace ZeroSizeSub {
  consteval unsigned ptr_diff1() {
    int *b = new int[0];
    unsigned d = 0;
    d = b - b;
    delete[] b;

    return d;
  }
  static_assert(ptr_diff1() == 0);


  consteval unsigned ptr_diff2() { // both-error {{never produces a constant expression}}
    int *a = new int[0];
    int *b = new int[0];

    unsigned d = a - b; // both-note 2{{arithmetic involving unrelated objects}}
    delete[] b;
    delete[] a;
    return d;
  }
  static_assert(ptr_diff2() == 0); // both-error {{not an integral constant expression}} \
                                   // both-note {{in call to}}
}

namespace WrongFrame {
  constexpr int foo() {
    int *p = nullptr;
    __builtin_operator_delete(p); // both-note {{subexpression not valid in a constant expression}}

    return 1;
  }
  static_assert(foo()); // both-error {{not an integral constant expression}} \
                        // both-note {{in call to}}

}

constexpr int no_deallocate_nonalloc = (std::allocator<int>().deallocate((int*)&no_deallocate_nonalloc), 1); // both-error {{constant expression}} \
                                                                                                             // both-note {{in call}} \
                                                                                                             // both-note {{declared here}}

namespace OpNewNothrow {
  constexpr int f() {
      int *v = (int*)operator new(sizeof(int), std::align_val_t(2), std::nothrow); // both-note {{cannot allocate untyped memory in a constant expression; use 'std::allocator<T>::allocate' to allocate memory of type 'T'}}
      operator delete(v, std::align_val_t(2), std::nothrow);
      return 1;
  }
  static_assert(f()); // both-error {{not an integral constant expression}} \
                      // both-note {{in call to}}
}

namespace BaseCompare {
  struct Cmp {
    void *p;

    template<typename T>
    constexpr Cmp(T *t) : p(t) {}

    constexpr friend bool operator==(Cmp a, Cmp b) {
      return a.p == b.p;
    }
  };

  class Base {};
  class Derived : public Base {};
  constexpr bool foo() {
    Derived *D = std::allocator<Derived>{}.allocate(1);;
    std::construct_at<Derived>(D);

    Derived *d = D;
    Base    *b = D;

    Cmp ca(d);
    Cmp cb(b);

    if (ca == cb) {
      std::allocator<Derived>{}.deallocate(D);
      return true;
    }
    std::allocator<Derived>{}.deallocate(D);

    return false;

  }
  static_assert(foo());
}

#else
/// Make sure we reject this prior to C++20
constexpr int a() { // both-error {{never produces a constant expression}}
  delete new int(12); // both-note 2{{dynamic memory allocation is not permitted in constant expressions until C++20}}
  return 1;
}
static_assert(a() == 1, ""); // both-error {{not an integral constant expression}} \
                             // both-note {{in call to 'a()'}}


static_assert(true ? *new int : 4, ""); // both-error {{expression is not an integral constant expression}} \
                                        // both-note {{read of uninitialized object is not allowed in a constant expression}}

#endif
