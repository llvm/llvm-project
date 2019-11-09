// RUN: %clang_cc1 -std=c++14 -verify %s

// pr33561
class ArrayBuffer;
template <typename T> class Trans_NS_WTF_RefPtr {
public:
  ArrayBuffer *operator->() { return nullptr; }
};
Trans_NS_WTF_RefPtr<ArrayBuffer> get();
template <typename _Visitor>
constexpr void visit(_Visitor __visitor) {
  __visitor(get()); // expected-note {{in instantiation}}
}
class ArrayBuffer {
  char data() {
    visit([](auto buffer) -> char { // expected-note {{in instantiation}}
      buffer->data();
    }); // expected-warning {{non-void lambda does not return a value}}
  } // expected-warning {{non-void function does not return a value}}
};

// pr34185
template <typename Promise> struct coroutine_handle {
  Promise &promise() const { return
    *static_cast<Promise *>(nullptr); // expected-warning {{binding dereferenced null}}
  }
};

template <typename Promise> auto GetCurrenPromise() {
  struct Awaiter { // expected-note {{in instantiation}}
    void await_suspend(coroutine_handle<Promise> h) {
      h.promise(); // expected-note {{in instantiation}}
    }
  };
  return Awaiter{};
}

void foo() {
  auto &&p = GetCurrenPromise<int>(); // expected-note {{in instantiation}}
}
