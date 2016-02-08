// Like the compiler, the static analyzer treats some functions differently if
// they come from a system header -- for example, it is assumed that system
// functions do not arbitrarily free() their parameters, and that some bugs
// found in system headers cannot be fixed by the user and should be
// suppressed.
#pragma clang system_header

typedef unsigned char uint8_t;

typedef __typeof__(sizeof(int)) size_t;
void *memmove(void *s1, const void *s2, size_t n);

namespace std {
  template <class T1, class T2>
  struct pair {
    T1 first;
    T2 second;
    
    pair() : first(), second() {}
    pair(const T1 &a, const T2 &b) : first(a), second(b) {}
    
    template<class U1, class U2>
    pair(const pair<U1, U2> &other) : first(other.first), second(other.second) {}
  };
  
  typedef __typeof__(sizeof(int)) size_t;
  
  template<typename T>
  class vector {
    T *_start;
    T *_finish;
    T *_end_of_storage;
  public:
    vector() : _start(0), _finish(0), _end_of_storage(0) {}
    ~vector();
    
    size_t size() const {
      return size_t(_finish - _start);
    }
    
    void push_back();
    T pop_back();

    T &operator[](size_t n) {
      return _start[n];
    }
    
    const T &operator[](size_t n) const {
      return _start[n];
    }
    
    T *begin() { return _start; }
    const T *begin() const { return _start; }

    T *end() { return _finish; }
    const T *end() const { return _finish; }
  };
  
  class exception {
  public:
    exception() throw();
    virtual ~exception() throw();
    virtual const char *what() const throw() {
      return 0;
    }
  };

  class bad_alloc : public exception {
    public:
    bad_alloc() throw();
    bad_alloc(const bad_alloc&) throw();
    bad_alloc& operator=(const bad_alloc&) throw();
    virtual const char* what() const throw() {
      return 0;
    }
  };

  struct nothrow_t {};

  extern const nothrow_t nothrow;

  // libc++'s implementation
  template <class _E>
  class initializer_list
  {
    const _E* __begin_;
    size_t    __size_;

    initializer_list(const _E* __b, size_t __s)
      : __begin_(__b),
        __size_(__s)
    {}

  public:
    typedef _E        value_type;
    typedef const _E& reference;
    typedef const _E& const_reference;
    typedef size_t    size_type;

    typedef const _E* iterator;
    typedef const _E* const_iterator;

    initializer_list() : __begin_(0), __size_(0) {}

    size_t    size()  const {return __size_;}
    const _E* begin() const {return __begin_;}
    const _E* end()   const {return __begin_ + __size_;}
  };

  template <bool, class _Tp = void> struct enable_if {};
  template <class _Tp> struct enable_if<true, _Tp> {typedef _Tp type;};

  template <class _Tp, _Tp __v>
  struct integral_constant
  {
      static const _Tp      value = __v;
      typedef _Tp               value_type;
      typedef integral_constant type;

     operator value_type() const {return value;}

     value_type operator ()() const {return value;}
  };

  template <class _Tp, _Tp __v>
  const _Tp integral_constant<_Tp, __v>::value;

    template <class _Tp, class _Arg>
    struct is_trivially_assignable
      : integral_constant<bool, __is_trivially_assignable(_Tp, _Arg)>
    {
    };

  typedef integral_constant<bool,true>  true_type;
  typedef integral_constant<bool,false> false_type;

  template <class _Tp> struct is_const            : public false_type {};
  template <class _Tp> struct is_const<_Tp const> : public true_type {};

  template <class _Tp> struct  is_reference        : public false_type {};
  template <class _Tp> struct  is_reference<_Tp&>  : public true_type {};

  template <class _Tp, class _Up> struct  is_same           : public false_type {};
  template <class _Tp>            struct  is_same<_Tp, _Tp> : public true_type {};

  template <class _Tp, bool = is_const<_Tp>::value || is_reference<_Tp>::value    >
  struct __add_const             {typedef _Tp type;};

  template <class _Tp>
  struct __add_const<_Tp, false> {typedef const _Tp type;};

  template <class _Tp> struct add_const {typedef typename __add_const<_Tp>::type type;};

  template <class _Tp> struct  remove_const            {typedef _Tp type;};
  template <class _Tp> struct  remove_const<const _Tp> {typedef _Tp type;};

  template <class _Tp> struct  add_lvalue_reference    {typedef _Tp& type;};

  template <class _Tp> struct is_trivially_copy_assignable
      : public is_trivially_assignable<typename add_lvalue_reference<_Tp>::type,
            typename add_lvalue_reference<typename add_const<_Tp>::type>::type> {};

    template<class InputIter, class OutputIter>
    OutputIter __copy(InputIter II, InputIter IE, OutputIter OI) {
      while (II != IE)
        *OI++ = *II++;

      return OI;
    }

  template <class _Tp, class _Up>
  inline
  typename enable_if
  <
      is_same<typename remove_const<_Tp>::type, _Up>::value &&
      is_trivially_copy_assignable<_Up>::value,
      _Up*
  >::type __copy(_Tp* __first, _Tp* __last, _Up* __result) {
      size_t __n = __last - __first;

      if (__n > 0)
        memmove(__result, __first, __n * sizeof(_Up));

      return __result + __n;
    }

  template<class InputIter, class OutputIter>
  OutputIter copy(InputIter II, InputIter IE, OutputIter OI) {
    return __copy(II, IE, OI);
  }

  template <class _BidirectionalIterator, class _OutputIterator>
  inline
  _OutputIterator
  __copy_backward(_BidirectionalIterator __first, _BidirectionalIterator __last,
                  _OutputIterator __result)
  {
      while (__first != __last)
          *--__result = *--__last;
      return __result;
  }

  template <class _Tp, class _Up>
  inline
  typename enable_if
  <
      is_same<typename remove_const<_Tp>::type, _Up>::value &&
      is_trivially_copy_assignable<_Up>::value,
      _Up*
  >::type __copy_backward(_Tp* __first, _Tp* __last, _Up* __result) {
      size_t __n = __last - __first;

    if (__n > 0)
    {
        __result -= __n;
        memmove(__result, __first, __n * sizeof(_Up));
    }
    return __result;
  }

  template<class InputIter, class OutputIter>
  OutputIter copy_backward(InputIter II, InputIter IE, OutputIter OI) {
    return __copy_backward(II, IE, OI);
  }

  struct input_iterator_tag { };
  struct output_iterator_tag { };
  struct forward_iterator_tag : public input_iterator_tag { };
  struct bidirectional_iterator_tag : public forward_iterator_tag { };
  struct random_access_iterator_tag : public bidirectional_iterator_tag { };

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
  // out of std::basic_string.
  int z = 0;
  z = 5/z;
}

}

void* operator new(std::size_t, const std::nothrow_t&) throw();
void* operator new[](std::size_t, const std::nothrow_t&) throw();
void operator delete(void*, const std::nothrow_t&) throw();
void operator delete[](void*, const std::nothrow_t&) throw();

void* operator new (std::size_t size, void* ptr) throw() { return ptr; };
void* operator new[] (std::size_t size, void* ptr) throw() { return ptr; };
void operator delete (void* ptr, void*) throw() {};
void operator delete[] (void* ptr, void*) throw() {};
