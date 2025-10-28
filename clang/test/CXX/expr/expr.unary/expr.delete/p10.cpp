// RUN: %clang_cc1 -std=c++20 -verify %s

using size_t = decltype(sizeof(0));
namespace std {
  enum class align_val_t : size_t {};
  struct destroying_delete_t {
    explicit destroying_delete_t() = default;
  };

  inline constexpr destroying_delete_t destroying_delete{};
}

// Aligned version is preferred over unaligned version,
// unsized version is preferred over sized version.
template<unsigned Align>
struct alignas(Align) A {
  void operator delete(void*);
  void operator delete(void*, std::align_val_t) = delete; // expected-note {{here}}

  void operator delete(void*, size_t) = delete;
  void operator delete(void*, size_t, std::align_val_t) = delete;
};
void f(A<__STDCPP_DEFAULT_NEW_ALIGNMENT__> *p) { delete p; }
void f(A<__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2> *p) { delete p; } // expected-error {{deleted}}

template<unsigned Align>
struct alignas(Align) B {
  void operator delete(void*, size_t);
  void operator delete(void*, size_t, std::align_val_t) = delete; // expected-note {{here}}
};
void f(B<__STDCPP_DEFAULT_NEW_ALIGNMENT__> *p) { delete p; }
void f(B<__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2> *p) { delete p; } // expected-error {{deleted}}

// Ensure that a deleted destructor is acceptable when the selected overload
// for operator delete is a destroying delete. See the comments in GH118660.
struct S {
  ~S() = delete;
  void operator delete(S *, std::destroying_delete_t) noexcept {}
};

struct T {
  void operator delete(T *, std::destroying_delete_t) noexcept {}
private:
  ~T();
};

void foo(S *s, T *t) {
  delete s; // Was rejected, is intended to be accepted.
  delete t; // Was rejected, is intended to be accepted.
}

// However, if the destructor is virtual, then it has to be accessible because
// the behavior depends on which operator delete is selected and that is based
// on the dynamic type of the pointer.
struct U {
  virtual ~U() = delete; // expected-note {{here}}
  void operator delete(U *, std::destroying_delete_t) noexcept {}
};

struct V {
  void operator delete(V *, std::destroying_delete_t) noexcept {}
private:
  virtual ~V(); // expected-note {{here}}
};

void bar(U *u, V *v) {
  // Both should be rejected because they have virtual destructors.
  delete u; // expected-error {{attempt to use a deleted function}}
  delete v; // expected-error {{calling a private destructor of class 'V'}}
}
