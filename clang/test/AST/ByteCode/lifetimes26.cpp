// RUN: %clang_cc1 -verify=expected,both -std=c++26 %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -verify=ref,both      -std=c++26 %s

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

  using size_t = decltype(sizeof(0));
  template<typename T> struct allocator {
    constexpr T *allocate(size_t N) {
      return (T*)__builtin_operator_new(sizeof(T) * N);
    }
    constexpr void deallocate(void *p) {
      __builtin_operator_delete(p);
    }
  };
  template<typename T, typename ...Args>
  constexpr void construct_at(void *p, Args &&...args) {
    new (p) T((Args&&)args...);
  }
};

constexpr void *operator new(std::size_t, void *p) { return p; }
namespace std {
  template<typename T> constexpr T *construct_at(T *p) { return new (p) T; }
  template<typename T> constexpr void destroy_at(T *p) { p->~T(); } // #destroy
}

constexpr bool foo() {
  using T = bool;
  bool b = true;
  b.~T();
  new (&b) bool(false);
  return b;
}
static_assert(!foo());

struct S {};
constexpr bool foo2() {
  S s;
  s.~S();
  new (&s) S{};
  return true;
}
static_assert(foo2());

constexpr void destroy_pointer() {
  using T = int*;
  T p;
  p.~T();
  std::construct_at(&p);
}
static_assert((destroy_pointer(), true));


namespace DestroyArrayElem {

  constexpr int test1() {
    int a[4] = {};
    std::destroy_at(&a); // both-error@#destroy {{object expression of non-scalar type 'int[4]' cannot be used in a pseudo-destructor expression}} \
                         // both-note {{in instantiation of function template specialization 'std::destroy_at<int[4]>' requested here}} \
                         // both-note {{subexpression not valid}}
    return 0;
  }
  static_assert(test1() == 0); // both-error {{not an integral constant expression}} \
                               // both-note {{in call to}}

  constexpr int test2() {

    int a[4] = {};
    std::destroy_at(&a[1]);
    int r = a[1]; // both-note {{read of object outside its lifetime}}
    std::construct_at(&a[1]);
    return 0;
  }
  static_assert(test2() == 0); // both-error {{not an integral constant expression}} \
                               // both-note {{in call to}}

  constexpr int test3() {
    int a[4]; /// Array with no init map.
    std::construct_at(&a[3]);
    return a[3]; // both-note {{read of uninitialized object}}
  }
  static_assert(test3() == 0); // both-error {{not an integral constant expression}} \
                               // both-note {{in call to}}

  constexpr int test4(bool b) {
    int a[4];
    a[0] = 12;
    std::construct_at(&a[3]);
    return b ? a[0] : a[3]; // both-note {{read of uninitialized object}}
  }
  static_assert(test4(true) == 12);
  static_assert(test4(false) == 0); // both-error {{not an integral constant expression}} \
                                    // both-note {{in call to}}


  constexpr int test5() {
    int *a = std::allocator<int>{}.allocate(3);
    a[0] = 1; // both-note {{assignment to object outside its lifetime}}
    int b = a[1];
    std::allocator<int>{}.deallocate(a);
    return b;
  }
  static_assert(test5() == 1); // both-error {{not an integral constant expression}} \
                               // both-note {{in call to}}

  constexpr int test6() {
    int *a = std::allocator<int>{}.allocate(3);
    std::construct_at<int>(&a[0]);
    a[0] = 1;
    std::construct_at<int>(&a[1]);
    a[1] = 2;
    std::construct_at<int>(&a[2]);
    a[2] = 3;

    int b = a[1];
    std::allocator<int>{}.deallocate(a);
    return b;
  }
  static_assert(test6() == 2);

  constexpr int test7() {
    int *a = std::allocator<int>{}.allocate(3);
    int *b = a + 3;
    new (a + 0) int();
    new (a + 1) int();
    new (a + 2) int();

    b = b - 1;
    new (b) int(0);

    bool r = a[0] == 0;

    std::allocator<int>{}.deallocate(a);
    return r;
  }
  static_assert(test7());

  constexpr int test8() {
    int *a = std::allocator<int>{}.allocate(3);
    int *b = a + 3;
    new (a + 0) int();
    new (a + 1) int();
    new (a + 2) int();

    new (a) int();
    std::destroy_at(b - 1);

    bool r = *a == 0;
    std::allocator<int>{}.deallocate(a);
    return r;
  }
  static_assert(test8());
}
