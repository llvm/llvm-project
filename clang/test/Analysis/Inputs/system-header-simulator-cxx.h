// Like the compiler, the static analyzer treats some functions differently if
// they come from a system header -- for example, it is assumed that system
// functions do not arbitrarily free() their parameters, and that some bugs
// found in system headers cannot be fixed by the user and should be
// suppressed.
#pragma clang system_header

typedef unsigned char uint8_t;

typedef __typeof__(sizeof(int)) size_t;
typedef __typeof__((char*)0-(char*)0) ptrdiff_t;
void *memmove(void *s1, const void *s2, size_t n);

namespace std {
  struct input_iterator_tag { };
  struct output_iterator_tag { };
  struct forward_iterator_tag : public input_iterator_tag { };
  struct bidirectional_iterator_tag : public forward_iterator_tag { };
  struct random_access_iterator_tag : public bidirectional_iterator_tag { };

  template <typename Iterator> struct iterator_traits {
    typedef typename Iterator::difference_type difference_type;
    typedef typename Iterator::value_type value_type;
    typedef typename Iterator::pointer pointer;
    typedef typename Iterator::reference reference;
    typedef typename Iterator::iterator_category iterator_category;
  };
}

template <typename T, typename Ptr, typename Ref> struct __vector_iterator {
  typedef __vector_iterator<T, T *, T &> iterator;
  typedef __vector_iterator<T, const T *, const T &> const_iterator;

  typedef ptrdiff_t difference_type;
  typedef T value_type;
  typedef Ptr pointer;
  typedef Ref reference;
  typedef std::random_access_iterator_tag iterator_category;

  __vector_iterator(const Ptr p = 0) : ptr(p) {}
  __vector_iterator(const iterator &rhs): ptr(rhs.base()) {}
  __vector_iterator<T, Ptr, Ref> operator++() { ++ ptr; return *this; }
  __vector_iterator<T, Ptr, Ref> operator++(int) {
    auto tmp = *this;
    ++ ptr;
    return tmp;
  }
  __vector_iterator<T, Ptr, Ref> operator--() { -- ptr; return *this; }
  __vector_iterator<T, Ptr, Ref> operator--(int) {
    auto tmp = *this; -- ptr;
    return tmp;
  }
  __vector_iterator<T, Ptr, Ref> operator+(difference_type n) {
    return ptr + n;
  }
  __vector_iterator<T, Ptr, Ref> operator-(difference_type n) {
    return ptr - n;
  }
  __vector_iterator<T, Ptr, Ref> operator+=(difference_type n) {
    return ptr += n;
  }
  __vector_iterator<T, Ptr, Ref> operator-=(difference_type n) {
    return ptr -= n;
  }

  Ref operator*() const { return *ptr; }
  Ptr operator->() const { return *ptr; }

  bool operator==(const iterator &rhs) const { return ptr == rhs.ptr; }
  bool operator==(const const_iterator &rhs) const { return ptr == rhs.ptr; }

  bool operator!=(const iterator &rhs) const { return ptr != rhs.ptr; }
  bool operator!=(const const_iterator &rhs) const { return ptr != rhs.ptr; }

  const Ptr& base() const { return ptr; }

private:
  Ptr ptr;
};

template <typename T, typename Ptr, typename Ref> struct __deque_iterator {
  typedef __deque_iterator<T, T *, T &> iterator;
  typedef __deque_iterator<T, const T *, const T &> const_iterator;

  typedef ptrdiff_t difference_type;
  typedef T value_type;
  typedef Ptr pointer;
  typedef Ref reference;
  typedef std::random_access_iterator_tag iterator_category;

  __deque_iterator(const Ptr p = 0) : ptr(p) {}
  __deque_iterator(const iterator &rhs): ptr(rhs.base()) {}
  __deque_iterator<T, Ptr, Ref> operator++() { ++ ptr; return *this; }
  __deque_iterator<T, Ptr, Ref> operator++(int) {
    auto tmp = *this;
    ++ ptr;
    return tmp;
  }
  __deque_iterator<T, Ptr, Ref> operator--() { -- ptr; return *this; }
  __deque_iterator<T, Ptr, Ref> operator--(int) {
    auto tmp = *this; -- ptr;
    return tmp;
  }
  __deque_iterator<T, Ptr, Ref> operator+(difference_type n) {
    return ptr + n;
  }
  __deque_iterator<T, Ptr, Ref> operator-(difference_type n) {
    return ptr - n;
  }
  __deque_iterator<T, Ptr, Ref> operator+=(difference_type n) {
    return ptr += n;
  }
  __deque_iterator<T, Ptr, Ref> operator-=(difference_type n) {
    return ptr -= n;
  }

  Ref operator*() const { return *ptr; }
  Ptr operator->() const { return *ptr; }

  bool operator==(const iterator &rhs) const { return ptr == rhs.ptr; }
  bool operator==(const const_iterator &rhs) const { return ptr == rhs.ptr; }

  bool operator!=(const iterator &rhs) const { return ptr != rhs.ptr; }
  bool operator!=(const const_iterator &rhs) const { return ptr != rhs.ptr; }

  const Ptr& base() const { return ptr; }

private:
  Ptr ptr;
};

template <typename T, typename Ptr, typename Ref> struct __list_iterator {
  typedef __list_iterator<T, __typeof__(T::data) *, __typeof__(T::data) &> iterator;
  typedef __list_iterator<T, const __typeof__(T::data) *, const __typeof__(T::data) &> const_iterator;

  typedef ptrdiff_t difference_type;
  typedef T value_type;
  typedef Ptr pointer;
  typedef Ref reference;
  typedef std::bidirectional_iterator_tag iterator_category;

  __list_iterator(T* it = 0) : item(it) {}
  __list_iterator(const iterator &rhs): item(rhs.base()) {}
  __list_iterator<T, Ptr, Ref> operator++() { item = item->next; return *this; }
  __list_iterator<T, Ptr, Ref> operator++(int) {
    auto tmp = *this;
    item = item->next;
    return tmp;
  }
  __list_iterator<T, Ptr, Ref> operator--() { item = item->prev; return *this; }
  __list_iterator<T, Ptr, Ref> operator--(int) {
    auto tmp = *this;
    item = item->prev;
    return tmp;
  }

  Ref operator*() const { return item->data; }
  Ptr operator->() const { return item->data; }

  bool operator==(const iterator &rhs) const { return item == rhs->item; }
  bool operator==(const const_iterator &rhs) const { return item == rhs->item; }

  bool operator!=(const iterator &rhs) const { return item != rhs->item; }
  bool operator!=(const const_iterator &rhs) const { return item != rhs->item; }

  const T* &base() const { return item; }

private:
  T* item;
};

template <typename T, typename Ptr, typename Ref> struct __fwdl_iterator {
  typedef __fwdl_iterator<T, __typeof__(T::data) *, __typeof__(T::data) &> iterator;
  typedef __fwdl_iterator<T, const __typeof__(T::data) *, const __typeof__(T::data) &> const_iterator;

  typedef ptrdiff_t difference_type;
  typedef T value_type;
  typedef Ptr pointer;
  typedef Ref reference;
  typedef std::forward_iterator_tag iterator_category;

  __fwdl_iterator(T* it = 0) : item(it) {}
  __fwdl_iterator(const iterator &rhs): item(rhs.base()) {}
  __fwdl_iterator<T, Ptr, Ref> operator++() { item = item->next; return *this; }
  __fwdl_iterator<T, Ptr, Ref> operator++(int) {
    auto tmp = *this;
    item = item->next;
    return tmp;
  }
  Ref operator*() const { return item->data; }
  Ptr operator->() const { return item->data; }

  bool operator==(const iterator &rhs) const { return item == rhs->item; }
  bool operator==(const const_iterator &rhs) const { return item == rhs->item; }

  bool operator!=(const iterator &rhs) const { return item != rhs->item; }
  bool operator!=(const const_iterator &rhs) const { return item != rhs->item; }

  const T* &base() const { return item; }

private:
  T* item;
};

namespace std {
  template <class T1, class T2>
  struct pair {
    T1 first;
    T2 second;
    
    pair() : first(), second() {}
    pair(const T1 &a, const T2 &b) : first(a), second(b) {}
    
    template<class U1, class U2>
    pair(const pair<U1, U2> &other) : first(other.first),
                                      second(other.second) {}
  };
  
  typedef __typeof__(sizeof(int)) size_t;

  template <class T> class initializer_list;
  
  template< class T > struct remove_reference      {typedef T type;};
  template< class T > struct remove_reference<T&>  {typedef T type;};
  template< class T > struct remove_reference<T&&> {typedef T type;};

  template<class T> 
  typename remove_reference<T>::type&& move(T&& a) {
    typedef typename remove_reference<T>::type&& RvalRef;
    return static_cast<RvalRef>(a);
  }

  template<typename T>
  class vector {
    typedef T value_type;
    typedef size_t size_type;
    typedef __vector_iterator<T, T *, T &> iterator;
    typedef __vector_iterator<T, const T *, const T &> const_iterator;

    T *_start;
    T *_finish;
    T *_end_of_storage;
  public:
    vector() : _start(0), _finish(0), _end_of_storage(0) {}
    template <typename InputIterator>
    vector(InputIterator first, InputIterator last);
    vector(const vector &other);
    vector(vector &&other);
    ~vector();
    
    size_t size() const {
      return size_t(_finish - _start);
    }

    void clear();

    void push_back(const T &value);
    void push_back(T &&value);
    void pop_back();

    T &operator[](size_t n) {
      return _start[n];
    }
    
    const T &operator[](size_t n) const {
      return _start[n];
    }
    
    iterator begin() { return iterator(_start); }
    const_iterator begin() const { return const_iterator(_start); }
    const_iterator cbegin() const { return const_iterator(_start); }
    iterator end() { return iterator(_finish); }
    const_iterator end() const { return const_iterator(_finish); }
    const_iterator cend() const { return const_iterator(_finish); }
    T& front() { return *begin(); }
    const T& front() const { return *begin(); }
    T& back() { return *(end() - 1); }
    const T& back() const { return *(end() - 1); }
  };
  
  template<typename T>
  class list {
    struct __item {
      T data;
      __item *prev, *next;
    } *_start, *_finish;
  public:
    typedef T value_type;
    typedef size_t size_type;
    typedef __list_iterator<__item, T *, T &> iterator;
    typedef __list_iterator<__item, const T *, const T &> const_iterator;

    list() : _start(0), _finish(0) {}
    template <typename InputIterator>
    list(InputIterator first, InputIterator last);
    list(const list &other);
    list(list &&other);
    ~list();
    
    list& operator=(const list &other);
    list& operator=(list &&other);
    list& operator=(std::initializer_list<T> ilist);

    void clear();

    iterator begin() { return iterator(_start); }
    const_iterator begin() const { return const_iterator(_start); }
    const_iterator cbegin() const { return const_iterator(_start); }
    iterator end() { return iterator(_finish); }
    const_iterator end() const { return const_iterator(_finish); }
    const_iterator cend() const { return const_iterator(_finish); }

    T& front() { return *begin(); }
    const T& front() const { return *begin(); }
    T& back() { return *--end(); }
    const T& back() const { return *--end(); }
  };

  template<typename T>
  class deque {
    typedef T value_type;
    typedef size_t size_type;
    typedef __deque_iterator<T, T *, T &> iterator;
    typedef __deque_iterator<T, const T *, const T &> const_iterator;

    T *_start;
    T *_finish;
    T *_end_of_storage;
  public:
    deque() : _start(0), _finish(0), _end_of_storage(0) {}
    template <typename InputIterator>
    deque(InputIterator first, InputIterator last);
    deque(const deque &other);
    deque(deque &&other);
    ~deque();
    
    size_t size() const {
      return size_t(_finish - _start);
    }
    
    void clear();

    void push_back(const T &value);
    void push_back(T &&value);
    void pop_back();

    void push_front(const T &value);
    void push_front(T &&value);
    void pop_front();

    T &operator[](size_t n) {
      return _start[n];
    }
    
    const T &operator[](size_t n) const {
      return _start[n];
    }
    
    iterator begin() { return iterator(_start); }
    const_iterator begin() const { return const_iterator(_start); }
    const_iterator cbegin() const { return const_iterator(_start); }
    iterator end() { return iterator(_finish); }
    const_iterator end() const { return const_iterator(_finish); }
    const_iterator cend() const { return const_iterator(_finish); }
    T& front() { return *begin(); }
    const T& front() const { return *begin(); }
    T& back() { return *(end() - 1); }
    const T& back() const { return *(end() - 1); }
  };
  
  template<typename T>
  class forward_list {
    struct __item {
      T data;
      __item *next;
    } *_start;
  public:
    typedef T value_type;
    typedef size_t size_type;
    typedef __fwdl_iterator<__item, T *, T &> iterator;
    typedef __fwdl_iterator<__item, const T *, const T &> const_iterator;

    forward_list() : _start(0) {}
    template <typename InputIterator>
    forward_list(InputIterator first, InputIterator last);
    forward_list(const forward_list &other);
    forward_list(forward_list &&other);
    ~forward_list();
    
    void clear();

    void push_front(const T &value);
    void push_front(T &&value);
    void pop_front();

    iterator begin() { return iterator(_start); }
    const_iterator begin() const { return const_iterator(_start); }
    const_iterator cbegin() const { return const_iterator(_start); }
    iterator end() { return iterator(); }
    const_iterator end() const { return const_iterator(); }
    const_iterator cend() const { return const_iterator(); }

    T& front() { return *begin(); }
    const T& front() const { return *begin(); }
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
}

template <class BidirectionalIterator, class Distance>
void __advance (BidirectionalIterator& it, Distance n,
                std::bidirectional_iterator_tag) {
  if (n >= 0) while(n-- > 0) ++it; else while (n++<0) --it;
}

template <class RandomAccessIterator, class Distance>
void __advance (RandomAccessIterator& it, Distance n,
                std::random_access_iterator_tag) {
  it += n;
}

namespace std {
  template <class InputIterator, class Distance>
  void advance (InputIterator& it, Distance n) {
    __advance(it, n, typename InputIterator::iterator_category());
  }

  template <class BidirectionalIterator>
  BidirectionalIterator
  prev (BidirectionalIterator it,
        typename iterator_traits<BidirectionalIterator>::difference_type n =
        1) {
    advance(it, -n);
    return it;
  }

  template <class InputIterator, class T>
  InputIterator find(InputIterator first, InputIterator last, const T &val);

  template <class InputIterator, class OutputIterator>
  OutputIterator copy(InputIterator first, InputIterator last,
                      OutputIterator result);

}

#ifdef TEST_INLINABLE_ALLOCATORS
namespace std {
  void *malloc(size_t);
  void free(void *);
}
void* operator new(std::size_t size, const std::nothrow_t&) throw() { return std::malloc(size); }
void* operator new[](std::size_t size, const std::nothrow_t&) throw() { return std::malloc(size); }
void operator delete(void* ptr, const std::nothrow_t&) throw() { std::free(ptr); }
void operator delete[](void* ptr, const std::nothrow_t&) throw() { std::free(ptr); }
#else
void* operator new(std::size_t, const std::nothrow_t&) throw();
void* operator new[](std::size_t, const std::nothrow_t&) throw();
void operator delete(void*, const std::nothrow_t&) throw();
void operator delete[](void*, const std::nothrow_t&) throw();
#endif

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
