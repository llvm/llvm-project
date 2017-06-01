// Like the compiler, the static analyzer treats some functions differently if
// they come from a system header -- for example, it is assumed that system
// functions do not arbitrarily free() their parameters, and that some bugs
// found in system headers cannot be fixed by the user and should be
// suppressed.
#pragma clang system_header

typedef unsigned char uint8_t;

typedef __typeof__(sizeof(int)) size_t;
void *memmove(void *s1, const void *s2, size_t n);

template <typename T, typename Ptr, typename Ref> struct __iterator {
  typedef __iterator<T, T *, T &> iterator;
  typedef __iterator<T, const T *, const T &> const_iterator;

  __iterator(const Ptr p) : ptr(p) {}

  __iterator<T, Ptr, Ref> operator++() { return *this; }
  __iterator<T, Ptr, Ref> operator++(int) { return *this; }
  __iterator<T, Ptr, Ref> operator--() { return *this; }
  __iterator<T, Ptr, Ref> operator--(int) { return *this; }
  Ref operator*() const { return *ptr; }
  Ptr operator->() const { return *ptr; }

  bool operator==(const iterator &rhs) const { return ptr == rhs.ptr; }
  bool operator==(const const_iterator &rhs) const { return ptr == rhs.ptr; }

  bool operator!=(const iterator &rhs) const { return ptr != rhs.ptr; }
  bool operator!=(const const_iterator &rhs) const { return ptr != rhs.ptr; }

private:
  Ptr ptr;
};

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
    typedef __iterator<T, T *, T &> iterator;
    typedef __iterator<T, const T *, const T &> const_iterator;

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
    
    iterator begin() { return iterator(_start); }
    const_iterator begin() const { return const_iterator(_start); }
    iterator end() { return iterator(_finish); }
    const_iterator end() const { return const_iterator(_finish); }
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

  template <class InputIterator, class T>
  InputIterator find(InputIterator first, InputIterator last, const T &val);
  template <class ForwardIterator1, class ForwardIterator2>
  ForwardIterator1 find_end(ForwardIterator1 first1, ForwardIterator1 last1,
                            ForwardIterator2 first2, ForwardIterator2 last2);
  template <class ForwardIterator1, class ForwardIterator2>
  ForwardIterator1 find_first_of(ForwardIterator1 first1,
                                 ForwardIterator1 last1,
                                 ForwardIterator2 first2,
                                 ForwardIterator2 last2);
  template <class InputIterator, class UnaryPredicate>
  InputIterator find_if(InputIterator first, InputIterator last,
                        UnaryPredicate pred);
  template <class InputIterator, class UnaryPredicate>
  InputIterator find_if_not(InputIterator first, InputIterator last,
                            UnaryPredicate pred);
  template <class InputIterator, class T>
  InputIterator lower_bound(InputIterator first, InputIterator last,
                            const T &val);
  template <class InputIterator, class T>
  InputIterator upper_bound(InputIterator first, InputIterator last,
                            const T &val);
  template <class ForwardIterator1, class ForwardIterator2>
  ForwardIterator1 search(ForwardIterator1 first1, ForwardIterator1 last1,
                          ForwardIterator2 first2, ForwardIterator2 last2);
  template <class ForwardIterator1, class ForwardIterator2>
  ForwardIterator1 search_n(ForwardIterator1 first1, ForwardIterator1 last1,
                            ForwardIterator2 first2, ForwardIterator2 last2);

  struct input_iterator_tag { };
  struct output_iterator_tag { };
  struct forward_iterator_tag : public input_iterator_tag { };
  struct bidirectional_iterator_tag : public forward_iterator_tag { };
  struct random_access_iterator_tag : public bidirectional_iterator_tag { };

}

void* operator new(std::size_t, const std::nothrow_t&) throw();
void* operator new[](std::size_t, const std::nothrow_t&) throw();
void operator delete(void*, const std::nothrow_t&) throw();
void operator delete[](void*, const std::nothrow_t&) throw();

void* operator new (std::size_t size, void* ptr) throw() { return ptr; };
void* operator new[] (std::size_t size, void* ptr) throw() { return ptr; };
void operator delete (void* ptr, void*) throw() {};
void operator delete[] (void* ptr, void*) throw() {};

namespace __cxxabiv1 {
extern "C" {
extern char *__cxa_demangle(const char *mangled_name,
                            char *output_buffer,
                            size_t *length,
                            int *status);
}}
namespace abi = __cxxabiv1;
