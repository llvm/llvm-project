// RUN: %clang_cc1 -std=gnu++20 -fsanitize=cfi-icall -fsanitize-minimal-runtime \
// RUN: -fsanitize-cfi-icall-experimental-normalize-integers \
// RUN: -fsized-deallocation  -flto=thin -std=gnu++20 -S %s -o- 

template <class _Fn>
void invoke(_Fn) {
  return;
}

template <class _Iter>
struct reverse_iterator {
  _Iter operator*() {
    _Iter __tmp;
    *__tmp;
    return __tmp;
  }
  void operator++();
};

template <class _Iter1, class _Iter2> bool operator!=(_Iter1, _Iter2);

template <typename _Fn>
struct __attribute__((__abi_tag__("v1"))) transform_view {
  void operator*() { invoke(__func_); }
  _Fn __func_;
};

template <typename It>
struct range {
  It begin();
  It end();
};

template <class>
struct VarargsNodeTMixin {
  auto args() {
    return transform_view([] {});
  }
};

struct Call : VarargsNodeTMixin<int> {
  void GenerateCode();
};

void Call::GenerateCode() {
  for (auto item : range<reverse_iterator<decltype(args())>>{}) { }
}
