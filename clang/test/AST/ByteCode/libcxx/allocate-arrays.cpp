// RUN: %clang_cc1 -std=c++2c -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++2c  -verify=ref,both %s


/// This example used to cause an invalid read because allocating
/// an array needs to return a pointer to the first element,
/// not to the array.

namespace std {
  using size_t = decltype(sizeof(0));

  template <class _Tp>
  class allocator {
  public:
    typedef size_t size_type;
    typedef _Tp value_type;
    constexpr _Tp *allocate(size_t __n) {
      return static_cast<_Tp *>(::operator new(__n * sizeof(_Tp)));
    }
  };
}

void *operator new(std::size_t, void *p) { return p; }
void* operator new[] (std::size_t, void* p) {return p;}

namespace std {
  template <class _Ep>
  class initializer_list {
    const _Ep *__begin_;
    __SIZE_TYPE__ __size_;

  public:
    typedef _Ep value_type;
    typedef const _Ep &reference;
    constexpr __SIZE_TYPE__ size() const noexcept { return __size_; }
    constexpr const _Ep *begin() const noexcept { return __begin_; }
    constexpr const _Ep *end() const noexcept { return __begin_ + __size_; }
  };
}

template<typename T>
class vector {
public:
  constexpr vector(std::initializer_list<T> Ts) {
    A = B = std::allocator<T>{}.allocate(Ts.size()); // both-note {{heap allocation performed here}}

    new (A) T(*Ts.begin());
  }
private:
  T *A = nullptr;
  T *B = nullptr;
};

constexpr vector<vector<int>> ints = {{3}, {4}}; // both-error {{must be initialized by a constant expression}} \
                                                 // both-note {{pointer to}}
