// This is a fake system header with divide-by-zero bugs introduced in
// c++ std library functions. We use these bugs to test hard-coded
// suppression of diagnostics within standard library functions that are known
// to produce false positives.

#pragma clang system_header

typedef unsigned char uint8_t;

typedef __typeof__(sizeof(int)) size_t;
void *memmove(void *s1, const void *s2, size_t n);

namespace std {

  template <class _Tp>
  class allocator {
  public:
    void deallocate(void *p) {
      ::delete p;
    }
  };

  template <class _Alloc>
  class allocator_traits {
  public:
    static void deallocate(void *p) {
      _Alloc().deallocate(p);
    }
  };

  template <class _Tp, class _Alloc>
  class __list_imp
  {};

  template <class _Tp, class _Alloc = allocator<_Tp> >
  class list
  : private __list_imp<_Tp, _Alloc>
  {
  public:
    void pop_front() {
      // Fake use-after-free.
      // No warning is expected as we are suppressing warning coming
      // out of std::list.
      int z = 0;
      z = 5/z;
    }
    bool empty() const;
  };

  // basic_string
  template<class _CharT, class _Alloc = allocator<_CharT> >
  class __attribute__ ((__type_visibility__("default"))) basic_string {
    bool isLong;
    union {
      _CharT localStorage[4];
      _CharT *externalStorage;

      void assignExternal(_CharT *newExternal) {
        externalStorage = newExternal;
      }
    } storage;

    typedef allocator_traits<_Alloc> __alloc_traits;

  public:
    basic_string();

    void push_back(int c) {
      // Fake error trigger.
      // No warning is expected as we are suppressing warning coming
      // out of std::basic_string.
      int z = 0;
      z = 5/z;
    }

    _CharT *getBuffer() {
      return isLong ? storage.externalStorage : storage.localStorage;
    }

    basic_string &operator +=(int c) {
      // Fake deallocate stack-based storage.
      // No warning is expected as we are suppressing warnings within
      // std::basic_string.
      __alloc_traits::deallocate(getBuffer());
    }

    basic_string &operator =(const basic_string &other) {
      // Fake deallocate stack-based storage, then use the variable in the
      // same union.
      // No warning is expected as we are suppressing warnings within
      // std::basic_string.
      __alloc_traits::deallocate(getBuffer());
      storage.assignExternal(new _CharT[4]);
    }
  };

template<class _Engine, class _UIntType>
class __independent_bits_engine {
public:
  // constructors and seeding functions
  __independent_bits_engine(_Engine& __e, size_t __w);
};

template<class _Engine, class _UIntType>
__independent_bits_engine<_Engine, _UIntType>
    ::__independent_bits_engine(_Engine& __e, size_t __w)
{
  // Fake error trigger.
  // No warning is expected as we are suppressing warning coming
  // out of std::__independent_bits_engine.
  int z = 0;
  z = 5/z;
}

#if __has_feature(cxx_decltype)
typedef decltype(nullptr) nullptr_t;

template<class _Tp>
class shared_ptr
{
public:
  constexpr shared_ptr(nullptr_t);
  explicit shared_ptr(_Tp* __p);

  shared_ptr(shared_ptr&& __r) { }

  ~shared_ptr();

  shared_ptr& operator=(shared_ptr&& __r) {
    // Fake error trigger.
    // No warning is expected as we are suppressing warning coming
    // out of std::shared_ptr.
    int z = 0;
    z = 5/z;
  }
};

template<class _Tp>
inline
constexpr
shared_ptr<_Tp>::shared_ptr(nullptr_t) {
}

#endif // __has_feature(cxx_decltype)
}

