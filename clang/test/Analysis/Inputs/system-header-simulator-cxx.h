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
  typedef size_t size_type;
#if __cplusplus >= 201103L
  using nullptr_t = decltype(nullptr);
#endif
}

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
  __vector_iterator<T, Ptr, Ref>& operator++() { ++ ptr; return *this; }
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
  friend __vector_iterator<T, Ptr, Ref> operator+(
      difference_type n,
      const __vector_iterator<T, Ptr, Ref> &iter) {
    return n + iter.ptr;
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

  template<typename U, typename Ptr2, typename Ref2>
  difference_type operator-(const __vector_iterator<U, Ptr2, Ref2> &rhs);

  Ref operator*() const { return *ptr; }
  Ptr operator->() const { return ptr; }

  Ref operator[](difference_type n) {
    return *(ptr+n);
  }
  
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
  __deque_iterator<T, Ptr, Ref>& operator++() { ++ ptr; return *this; }
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
  friend __deque_iterator<T, Ptr, Ref> operator+(
      difference_type n,
      const __deque_iterator<T, Ptr, Ref> &iter) {
    return n + iter.ptr;
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
  Ptr operator->() const { return ptr; }

  Ref operator[](difference_type n) {
    return *(ptr+n);
  }
  
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
  __list_iterator(const iterator &rhs): item(rhs.item) {}
  __list_iterator<T, Ptr, Ref>& operator++() { item = item->next; return *this; }
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
  Ptr operator->() const { return &item->data; }

  bool operator==(const iterator &rhs) const { return item == rhs->item; }
  bool operator==(const const_iterator &rhs) const { return item == rhs->item; }

  bool operator!=(const iterator &rhs) const { return item != rhs->item; }
  bool operator!=(const const_iterator &rhs) const { return item != rhs->item; }

  const T* &base() const { return item; }

  template <typename UT, typename UPtr, typename URef>
  friend struct __list_iterator;

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
  __fwdl_iterator(const iterator &rhs): item(rhs.item) {}
  __fwdl_iterator<T, Ptr, Ref>& operator++() { item = item->next; return *this; }
  __fwdl_iterator<T, Ptr, Ref> operator++(int) {
    auto tmp = *this;
    item = item->next;
    return tmp;
  }
  Ref operator*() const { return item->data; }
  Ptr operator->() const { return &item->data; }

  bool operator==(const iterator &rhs) const { return item == rhs->item; }
  bool operator==(const const_iterator &rhs) const { return item == rhs->item; }

  bool operator!=(const iterator &rhs) const { return item != rhs->item; }
  bool operator!=(const const_iterator &rhs) const { return item != rhs->item; }

  const T* &base() const { return item; }

  template <typename UT, typename UPtr, typename URef>
  friend struct __fwdl_iterator;

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

  template<class T2, class T1>
  T2& get(pair<T1, T2>& p) ;
  template<class T1, class T2>
  T1& get(const pair<T1, T2>& p) ;
  
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

  template< class T >
  using remove_reference_t = typename remove_reference<T>::type;

  template <class T>
  void swap(T &a, T &b) {
    T c(std::move(a));
    a = std::move(b);
    b = std::move(c);
  }

  template<typename T>
  class vector {
    T *_start;
    T *_finish;
    T *_end_of_storage;

  public:
    typedef T value_type;
    typedef size_t size_type;
    typedef __vector_iterator<T, T *, T &> iterator;
    typedef __vector_iterator<T, const T *, const T &> const_iterator;

    vector() : _start(0), _finish(0), _end_of_storage(0) {}
    template <typename InputIterator>
    vector(InputIterator first, InputIterator last);
    vector(const vector &other);
    vector(vector &&other);
    ~vector();
    
    size_t size() const {
      return size_t(_finish - _start);
    }

    vector& operator=(const vector &other);
    vector& operator=(vector &&other);
    vector& operator=(std::initializer_list<T> ilist);

    void assign(size_type count, const T &value);
    template <typename InputIterator >
    void assign(InputIterator first, InputIterator last);
    void assign(std::initializer_list<T> ilist);

    void clear();

    void push_back(const T &value);
    void push_back(T &&value);
    template<class... Args>
    void emplace_back(Args&&... args);
    void pop_back();

    iterator insert(const_iterator position, const value_type &val);
    iterator insert(const_iterator position, size_type n,
                    const value_type &val);
    template <typename InputIterator>
    iterator insert(const_iterator position, InputIterator first,
                    InputIterator last);
    iterator insert(const_iterator position, value_type &&val);
    iterator insert(const_iterator position, initializer_list<value_type> il);

    template <class... Args>
    iterator emplace(const_iterator position, Args&&... args);

    iterator erase(const_iterator position);
    iterator erase(const_iterator first, const_iterator last);

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

    void assign(size_type count, const T &value);
    template <typename InputIterator >
    void assign(InputIterator first, InputIterator last);
    void assign(std::initializer_list<T> ilist);

    void clear();

    void push_back(const T &value);
    void push_back(T &&value);
    template<class... Args>
    void emplace_back(Args&&... args);
    void pop_back();

    void push_front(const T &value);
    void push_front(T &&value);
    template<class... Args>
    void emplace_front(Args&&... args);
    void pop_front();

    iterator insert(const_iterator position, const value_type &val);
    iterator insert(const_iterator position, size_type n,
                    const value_type &val);
    template <typename InputIterator>
    iterator insert(const_iterator position, InputIterator first,
                    InputIterator last);
    iterator insert(const_iterator position, value_type &&val);
    iterator insert(const_iterator position, initializer_list<value_type> il);

    template <class... Args>
    iterator emplace(const_iterator position, Args&&... args);

    iterator erase(const_iterator position);
    iterator erase(const_iterator first, const_iterator last);

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
    T *_start;
    T *_finish;
    T *_end_of_storage;

  public:
    typedef T value_type;
    typedef size_t size_type;
    typedef __deque_iterator<T, T *, T &> iterator;
    typedef __deque_iterator<T, const T *, const T &> const_iterator;

    deque() : _start(0), _finish(0), _end_of_storage(0) {}
    template <typename InputIterator>
    deque(InputIterator first, InputIterator last);
    deque(const deque &other);
    deque(deque &&other);
    ~deque();
    
    size_t size() const {
      return size_t(_finish - _start);
    }
    
    deque& operator=(const deque &other);
    deque& operator=(deque &&other);
    deque& operator=(std::initializer_list<T> ilist);

    void assign(size_type count, const T &value);
    template <typename InputIterator >
    void assign(InputIterator first, InputIterator last);
    void assign(std::initializer_list<T> ilist);

    void clear();

    void push_back(const T &value);
    void push_back(T &&value);
    template<class... Args>
    void emplace_back(Args&&... args);
    void pop_back();

    void push_front(const T &value);
    void push_front(T &&value);
    template<class... Args>
    void emplace_front(Args&&... args);
    void pop_front();

    iterator insert(const_iterator position, const value_type &val);
    iterator insert(const_iterator position, size_type n,
                    const value_type &val);
    template <typename InputIterator>
    iterator insert(const_iterator position, InputIterator first,
                    InputIterator last);
    iterator insert(const_iterator position, value_type &&val);
    iterator insert(const_iterator position, initializer_list<value_type> il);

    template <class... Args>
    iterator emplace(const_iterator position, Args&&... args);

    iterator erase(const_iterator position);
    iterator erase(const_iterator first, const_iterator last);

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
    
    forward_list& operator=(const forward_list &other);
    forward_list& operator=(forward_list &&other);
    forward_list& operator=(std::initializer_list<T> ilist);

    void assign(size_type count, const T &value);
    template <typename InputIterator >
    void assign(InputIterator first, InputIterator last);
    void assign(std::initializer_list<T> ilist);

    void clear();

    void push_front(const T &value);
    void push_front(T &&value);
    template<class... Args>
    void emplace_front(Args&&... args);
    void pop_front();

    iterator insert_after(const_iterator position, const value_type &val);
    iterator insert_after(const_iterator position, value_type &&val);
    iterator insert_after(const_iterator position, size_type n,
                          const value_type &val);
    template <typename InputIterator>
    iterator insert_after(const_iterator position, InputIterator first,
                          InputIterator last);
    iterator insert_after(const_iterator position,
                          initializer_list<value_type> il);

    template <class... Args>
    iterator emplace_after(const_iterator position, Args&&... args);

    iterator erase_after(const_iterator position);
    iterator erase_after(const_iterator first, const_iterator last);

    iterator begin() { return iterator(_start); }
    const_iterator begin() const { return const_iterator(_start); }
    const_iterator cbegin() const { return const_iterator(_start); }
    iterator end() { return iterator(); }
    const_iterator end() const { return const_iterator(); }
    const_iterator cend() const { return const_iterator(); }

    T& front() { return *begin(); }
    const T& front() const { return *begin(); }
  };

  template <typename CharT>
  class basic_string {
    class Allocator {};

  public:
    basic_string() : basic_string(Allocator()) {}
    explicit basic_string(const Allocator &alloc);
    basic_string(size_type count, CharT ch,
                 const Allocator &alloc = Allocator());
    basic_string(const basic_string &other,
                 size_type pos,
                 const Allocator &alloc = Allocator());
    basic_string(const basic_string &other,
                 size_type pos, size_type count,
                 const Allocator &alloc = Allocator());
    basic_string(const CharT *s, size_type count,
                 const Allocator &alloc = Allocator());
    basic_string(const CharT *s,
                 const Allocator &alloc = Allocator());
    template <class InputIt>
    basic_string(InputIt first, InputIt last,
                 const Allocator &alloc = Allocator());
    basic_string(const basic_string &other);
    basic_string(const basic_string &other,
                 const Allocator &alloc);
    basic_string(basic_string &&other);
    basic_string(basic_string &&other,
                 const Allocator &alloc);
    basic_string(std::initializer_list<CharT> ilist,
                 const Allocator &alloc = Allocator());
    template <class T>
    basic_string(const T &t, size_type pos, size_type n,
                 const Allocator &alloc = Allocator());
    // basic_string(std::nullptr_t) = delete;

    ~basic_string();
    void clear();

    basic_string &operator=(const basic_string &str);
    basic_string &operator+=(const basic_string &str);

    const CharT *c_str() const;
    const CharT *data() const;
    CharT *data();

    const char *begin() const;
    const char *end() const;

    basic_string &append(size_type count, CharT ch);
    basic_string &assign(size_type count, CharT ch);
    basic_string &erase(size_type index, size_type count);
    basic_string &insert(size_type index, size_type count, CharT ch);
    basic_string &replace(size_type pos, size_type count, const basic_string &str);
    void pop_back();
    void push_back(CharT ch);
    void reserve(size_type new_cap);
    void resize(size_type count);
    void shrink_to_fit();
    void swap(basic_string &other);
  };

  typedef basic_string<char> string;
  typedef basic_string<wchar_t> wstring;
#if __cplusplus >= 201103L
  typedef basic_string<char16_t> u16string;
  typedef basic_string<char32_t> u32string;
#endif

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

  enum class align_val_t : size_t {};

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

  #if __cplusplus >= 201703L
  template< class T, class U >
  inline constexpr bool is_same_v = is_same<T, U>::value;
  #endif

  template <class _Tp, bool = is_const<_Tp>::value || is_reference<_Tp>::value    >
  struct __add_const             {typedef _Tp type;};

  template <class _Tp>
  struct __add_const<_Tp, false> {typedef const _Tp type;};

  template <class _Tp> struct add_const {typedef typename __add_const<_Tp>::type type;};

  template <class _Tp> struct  remove_const            {typedef _Tp type;};
  template <class _Tp> struct  remove_const<const _Tp> {typedef _Tp type;};

  template< class T >
  using remove_const_t = typename remove_const<T>::type;

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

  template< bool B, class T = void >
  using enable_if_t = typename enable_if<B,T>::type;

  template<class InputIter, class OutputIter>
  OutputIter copy_backward(InputIter II, InputIter IE, OutputIter OI) {
    return __copy_backward(II, IE, OI);
  }
}

template <class BidirectionalIterator, class Distance>
void __advance(BidirectionalIterator& it, Distance n,
               std::bidirectional_iterator_tag)
#if !defined(STD_ADVANCE_INLINE_LEVEL) || STD_ADVANCE_INLINE_LEVEL > 2
{
  if (n >= 0) while(n-- > 0) ++it; else while (n++<0) --it;
}
#else
    ;
#endif

template <class RandomAccessIterator, class Distance>
void __advance(RandomAccessIterator& it, Distance n,
               std::random_access_iterator_tag)
#if !defined(STD_ADVANCE_INLINE_LEVEL) || STD_ADVANCE_INLINE_LEVEL > 2
{
  it += n;
}
#else
    ;
#endif

namespace std {

template <class InputIterator, class Distance>
void advance(InputIterator& it, Distance n)
#if !defined(STD_ADVANCE_INLINE_LEVEL) || STD_ADVANCE_INLINE_LEVEL > 1
{
  __advance(it, n, typename InputIterator::iterator_category());
}
#else
    ;
#endif

template <class BidirectionalIterator>
BidirectionalIterator
prev(BidirectionalIterator it,
     typename iterator_traits<BidirectionalIterator>::difference_type n =
         1)
#if !defined(STD_ADVANCE_INLINE_LEVEL) || STD_ADVANCE_INLINE_LEVEL > 0
{
  advance(it, -n);
  return it;
}
#else
    ;
#endif

template <class ForwardIterator>
ForwardIterator
next(ForwardIterator it,
     typename iterator_traits<ForwardIterator>::difference_type n =
         1)
#if !defined(STD_ADVANCE_INLINE_LEVEL) || STD_ADVANCE_INLINE_LEVEL > 0
{
  advance(it, n);
  return it;
}
#else
    ;
#endif

  template <class InputIt, class T>
  InputIt find(InputIt first, InputIt last, const T& value);

  template <class ExecutionPolicy, class ForwardIt, class T>
  ForwardIt find(ExecutionPolicy&& policy, ForwardIt first, ForwardIt last,
                 const T& value);

  template <class InputIt, class UnaryPredicate>
  InputIt find_if (InputIt first, InputIt last, UnaryPredicate p);

  template <class ExecutionPolicy, class ForwardIt, class UnaryPredicate>
  ForwardIt find_if (ExecutionPolicy&& policy, ForwardIt first, ForwardIt last,
                     UnaryPredicate p);

  template <class InputIt, class UnaryPredicate>
  InputIt find_if_not (InputIt first, InputIt last, UnaryPredicate q);

  template <class ExecutionPolicy, class ForwardIt, class UnaryPredicate>
  ForwardIt find_if_not (ExecutionPolicy&& policy, ForwardIt first,
                         ForwardIt last, UnaryPredicate q);

  template <class InputIt, class ForwardIt>
  InputIt find_first_of(InputIt first, InputIt last,
                         ForwardIt s_first, ForwardIt s_last);

  template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2>
  ForwardIt1 find_first_of (ExecutionPolicy&& policy,
                            ForwardIt1 first, ForwardIt1 last,
                            ForwardIt2 s_first, ForwardIt2 s_last);

  template <class InputIt, class ForwardIt, class BinaryPredicate>
  InputIt find_first_of (InputIt first, InputIt last,
                         ForwardIt s_first, ForwardIt s_last,
                         BinaryPredicate p );

  template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2,
            class BinaryPredicate>
  ForwardIt1 find_first_of (ExecutionPolicy&& policy,
                            ForwardIt1 first, ForwardIt1 last,
                            ForwardIt2 s_first, ForwardIt2 s_last,
                            BinaryPredicate p );

  template <class InputIt, class ForwardIt>
  InputIt find_end(InputIt first, InputIt last,
                   ForwardIt s_first, ForwardIt s_last);

  template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2>
  ForwardIt1 find_end (ExecutionPolicy&& policy,
                       ForwardIt1 first, ForwardIt1 last,
                       ForwardIt2 s_first, ForwardIt2 s_last);

  template <class InputIt, class ForwardIt, class BinaryPredicate>
  InputIt find_end (InputIt first, InputIt last,
                    ForwardIt s_first, ForwardIt s_last,
                    BinaryPredicate p );

  template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2,
            class BinaryPredicate>
  ForwardIt1 find_end (ExecutionPolicy&& policy,
                       ForwardIt1 first, ForwardIt1 last,
                       ForwardIt2 s_first, ForwardIt2 s_last,
                       BinaryPredicate p );

  template <class ForwardIt, class T>
  ForwardIt lower_bound (ForwardIt first, ForwardIt last, const T& value);

  template <class ForwardIt, class T, class Compare>
  ForwardIt lower_bound (ForwardIt first, ForwardIt last, const T& value,
                         Compare comp);

  template <class ForwardIt, class T>
  ForwardIt upper_bound (ForwardIt first, ForwardIt last, const T& value);

  template <class ForwardIt, class T, class Compare>
  ForwardIt upper_bound (ForwardIt first, ForwardIt last, const T& value,
                         Compare comp);

  template <class ForwardIt1, class ForwardIt2>
  ForwardIt1 search (ForwardIt1 first, ForwardIt1 last,
                     ForwardIt2 s_first, ForwardIt2 s_last);

  template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2>
  ForwardIt1 search (ExecutionPolicy&& policy,
                     ForwardIt1 first, ForwardIt1 last,
                     ForwardIt2 s_first, ForwardIt2 s_last);

  template <class ForwardIt1, class ForwardIt2, class BinaryPredicate>
  ForwardIt1 search (ForwardIt1 first, ForwardIt1 last,
                     ForwardIt2 s_first, ForwardIt2 s_last, BinaryPredicate p);

  template <class ExecutionPolicy, class ForwardIt1, class ForwardIt2,
            class BinaryPredicate >
  ForwardIt1 search (ExecutionPolicy&& policy,
                     ForwardIt1 first, ForwardIt1 last,
                     ForwardIt2 s_first, ForwardIt2 s_last, BinaryPredicate p);

  template <class ForwardIt, class Searcher>
  ForwardIt search (ForwardIt first, ForwardIt last, const Searcher& searcher);

  template <class ForwardIt, class Size, class T>
  ForwardIt search_n (ForwardIt first, ForwardIt last, Size count,
                      const T& value);

  template <class ExecutionPolicy, class ForwardIt, class Size, class T>
  ForwardIt search_n (ExecutionPolicy&& policy, ForwardIt first, ForwardIt last,
                      Size count, const T& value);

  template <class ForwardIt, class Size, class T, class BinaryPredicate>
  ForwardIt search_n (ForwardIt first, ForwardIt last, Size count,
                      const T& value, BinaryPredicate p);

  template <class ExecutionPolicy, class ForwardIt, class Size, class T,
            class BinaryPredicate>
  ForwardIt search_n (ExecutionPolicy&& policy, ForwardIt first, ForwardIt last,
                      Size count, const T& value, BinaryPredicate p);

  template <class InputIterator, class OutputIterator>
  OutputIterator copy(InputIterator first, InputIterator last,
                      OutputIterator result);

}

#if __cplusplus >= 201103L
namespace std {
template <typename T> // TODO: Implement the stub for deleter.
class unique_ptr {
public:
  unique_ptr() noexcept {}
  unique_ptr(T *) noexcept {}
  unique_ptr(const unique_ptr &) noexcept = delete;
  unique_ptr(unique_ptr &&) noexcept;

  T *get() const noexcept;
  T *release() noexcept;
  void reset(T *p = nullptr) noexcept;
  void swap(unique_ptr<T> &p) noexcept;

  typename std::add_lvalue_reference<T>::type operator*() const;
  T *operator->() const noexcept;
  operator bool() const noexcept;
  unique_ptr<T> &operator=(unique_ptr<T> &&p) noexcept;
  unique_ptr<T> &operator=(nullptr_t) noexcept;
};

// TODO :: Once the deleter parameter is added update with additional template parameter.
template <typename T>
void swap(unique_ptr<T> &x, unique_ptr<T> &y) noexcept {
  x.swap(y);
}

template <typename T1, typename T2>
bool operator==(const unique_ptr<T1> &x, const unique_ptr<T2> &y);

template <typename T1, typename T2>
bool operator!=(const unique_ptr<T1> &x, const unique_ptr<T2> &y);

template <typename T1, typename T2>
bool operator<(const unique_ptr<T1> &x, const unique_ptr<T2> &y);

template <typename T1, typename T2>
bool operator>(const unique_ptr<T1> &x, const unique_ptr<T2> &y);

template <typename T1, typename T2>
bool operator<=(const unique_ptr<T1> &x, const unique_ptr<T2> &y);

template <typename T1, typename T2>
bool operator>=(const unique_ptr<T1> &x, const unique_ptr<T2> &y);

template <typename T>
bool operator==(const unique_ptr<T> &x, nullptr_t y);

template <typename T>
bool operator!=(const unique_ptr<T> &x, nullptr_t y);

template <typename T>
bool operator<(const unique_ptr<T> &x, nullptr_t y);

template <typename T>
bool operator>(const unique_ptr<T> &x, nullptr_t y);

template <typename T>
bool operator<=(const unique_ptr<T> &x, nullptr_t y);

template <typename T>
bool operator>=(const unique_ptr<T> &x, nullptr_t y);

template <typename T>
bool operator==(nullptr_t x, const unique_ptr<T> &y);

template <typename T>
bool operator!=(nullptr_t x, const unique_ptr<T> &y);

template <typename T>
bool operator>(nullptr_t x, const unique_ptr<T> &y);

template <typename T>
bool operator<(nullptr_t x, const unique_ptr<T> &y);

template <typename T>
bool operator>=(nullptr_t x, const unique_ptr<T> &y);

template <typename T>
bool operator<=(nullptr_t x, const unique_ptr<T> &y);

template <class T, class... Args>
unique_ptr<T> make_unique(Args &&...args);

#if __cplusplus >= 202002L

template <class T>
unique_ptr<T> make_unique_for_overwrite();

#endif

} // namespace std
#endif

namespace std {
template <class CharT>
class basic_ostream;

using ostream = basic_ostream<char>;

extern std::ostream cout;

ostream &operator<<(ostream &, const string &);

#if __cplusplus >= 202002L
template <class T>
ostream &operator<<(ostream &, const std::unique_ptr<T> &);
#endif

template <class CharT>
class basic_istream;

using istream = basic_istream<char>;

extern std::istream cin;

istream &getline(istream &, string &, char);
istream &getline(istream &, string &);
} // namespace std

namespace std {
  void *malloc(size_t);
  void free(void *);
} // namespace std

#ifdef TEST_INLINABLE_ALLOCATORS
void* operator new(std::size_t size, const std::nothrow_t&) throw() { return std::malloc(size); }
void* operator new[](std::size_t size, const std::nothrow_t&) throw() { return std::malloc(size); }
void operator delete(void* ptr, const std::nothrow_t&) throw() { std::free(ptr); }
void operator delete[](void* ptr, const std::nothrow_t&) throw() { std::free(ptr); }
#else
// C++20 standard draft 17.6.1, from "Header <new> synopsis", but with throw()
// instead of noexcept:

void *operator new(std::size_t size);
void *operator new(std::size_t size, std::align_val_t alignment);
void *operator new(std::size_t size, const std::nothrow_t &) throw();
void *operator new(std::size_t size, std::align_val_t alignment,
                   const std::nothrow_t &) throw();
void operator delete(void *ptr) throw();
void operator delete(void *ptr, std::size_t size) throw();
void operator delete(void *ptr, std::align_val_t alignment) throw();
void operator delete(void *ptr, std::size_t size, std::align_val_t alignment) throw();
void operator delete(void *ptr, const std::nothrow_t &)throw();
void operator delete(void *ptr, std::align_val_t alignment,
                     const std::nothrow_t &)throw();
void *operator new[](std::size_t size);
void *operator new[](std::size_t size, std::align_val_t alignment);
void *operator new[](std::size_t size, const std::nothrow_t &) throw();
void *operator new[](std::size_t size, std::align_val_t alignment,
                     const std::nothrow_t &) throw();
void operator delete[](void *ptr) throw();
void operator delete[](void *ptr, std::size_t size) throw();
void operator delete[](void *ptr, std::align_val_t alignment) throw();
void operator delete[](void *ptr, std::size_t size, std::align_val_t alignment) throw();
void operator delete[](void *ptr, const std::nothrow_t &) throw();
void operator delete[](void *ptr, std::align_val_t alignment,
                       const std::nothrow_t &) throw();
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

namespace std {
  template<class ForwardIt>
  bool is_sorted(ForwardIt first, ForwardIt last);

  template <class RandomIt>
  void nth_element(RandomIt first, RandomIt nth, RandomIt last);

  template<class RandomIt>
  void partial_sort(RandomIt first, RandomIt middle, RandomIt last);

  template<class RandomIt>
  void sort (RandomIt first, RandomIt last);

  template<class RandomIt>
  void stable_sort(RandomIt first, RandomIt last);

  template<class BidirIt, class UnaryPredicate>
  BidirIt partition(BidirIt first, BidirIt last, UnaryPredicate p);

  template<class BidirIt, class UnaryPredicate>
  BidirIt stable_partition(BidirIt first, BidirIt last, UnaryPredicate p);
}

namespace std {

template< class T = void >
struct less;

template< class T >
struct allocator;

template< class Key >
struct hash;

template<
  class Key,
  class Compare = std::less<Key>,
  class Alloc = std::allocator<Key>
> class set {
  public:
    set(initializer_list<Key> __list) {}

    class iterator {
    public:
      iterator(Key *key): ptr(key) {}
      iterator& operator++() { ++ptr; return *this; }
      bool operator!=(const iterator &other) const { return ptr != other.ptr; }
      const Key &operator*() const { return *ptr; }
    private:
      Key *ptr;
    };

  public:
    Key *val;
    iterator begin() const { return iterator(val); }
    iterator end() const { return iterator(val + 1); }
};

template<
  class Key,
  class Hash = std::hash<Key>,
  class Compare = std::less<Key>,
  class Alloc = std::allocator<Key>
> class unordered_set {
  public:
    unordered_set(initializer_list<Key> __list) {}

    class iterator {
    public:
      iterator(Key *key): ptr(key) {}
      iterator& operator++() { ++ptr; return *this; }
      bool operator!=(const iterator &other) const { return ptr != other.ptr; }
      const Key &operator*() const { return *ptr; }
    private:
      Key *ptr;
    };

  public:
    Key *val;
    iterator begin() const { return iterator(val); }
    iterator end() const { return iterator(val + 1); }
};

template <typename T>
class atomic {
public:
  T operator++();
  T operator--();
};

namespace execution {
class sequenced_policy {};
}

template <class T = void> struct equal_to {};

template <class ForwardIt, class BinaryPredicate = std::equal_to<> >
class default_searcher {
public:
  default_searcher (ForwardIt pat_first,
                    ForwardIt pat_last,
                    BinaryPredicate pred = BinaryPredicate());
  template <class ForwardIt2>
  std::pair <ForwardIt2, ForwardIt2>
  operator()( ForwardIt2 first, ForwardIt2 last ) const;
};

template <typename> class packaged_task;
template <typename Ret, typename... Args> class packaged_task<Ret(Args...)> {
  // TODO: Add some actual implementation.
};

  #if __cplusplus >= 201703L

  namespace detail
  {
    template<class T>
    struct type_identity { using type = T; }; // or use std::type_identity (since C++20)
 
    template<class T>
    auto try_add_pointer(int) -> type_identity<typename std::remove_reference<T>::type*>;
    template<class T>
    auto try_add_pointer(...) -> type_identity<T>;
  } // namespace detail
 
  template<class T>
  struct add_pointer : decltype(detail::try_add_pointer<T>(0)) {};

  template< class T >
  using add_pointer_t = typename add_pointer<T>::type;

  template<class T> struct remove_cv { typedef T type; };
  template<class T> struct remove_cv<const T> { typedef T type; };
  template<class T> struct remove_cv<volatile T> { typedef T type; };
  template<class T> struct remove_cv<const volatile T> { typedef T type; };

  template< class T >
  using remove_cv_t = typename remove_cv<T>::type;

  // This decay does not behave exactly like std::decay, but this is enough
  // for testing the std::variant checker
  template<class T>
  struct decay{typedef remove_cv_t<remove_reference_t<T>> type;};
  template<class T>
  using decay_t = typename decay<T>::type;
  
  // variant
  template <class... Types> class variant;
  // variant helper classes
  template <class T> struct variant_size;
  template <class T> struct variant_size<const T>;
  template <class T> struct variant_size<volatile T>;
  template <class T> struct variant_size<const volatile T>;
  template <class T> inline constexpr size_t variant_size_v = variant_size<T>::value;
  template <class... Types>
  struct variant_size<variant<Types...>>;
  template <size_t I, class T> struct variant_alternative;
  template <size_t I, class T> struct variant_alternative<I, const T>;
  template <size_t I, class T> struct variant_alternative<I, volatile T>;
  template <size_t I, class T> struct variant_alternative<I, const volatile T>;
  template <size_t I, class T>
  using variant_alternative_t = typename variant_alternative<I, T>::type;
  template <size_t I, class... Types>
  struct variant_alternative<I, variant<Types...>>;
  inline constexpr size_t variant_npos = -1;
  template <size_t I, class... Types>
  constexpr variant_alternative_t<I, variant<Types...>>&
    get(variant<Types...>&);
  template <size_t I, class... Types>
  constexpr variant_alternative_t<I, variant<Types...>>&&
    get(variant<Types...>&&);
  template <size_t I, class... Types>
  constexpr const variant_alternative_t<I, variant<Types...>>&
    get(const variant<Types...>&);
  template <size_t I, class... Types>
  constexpr const variant_alternative_t<I, variant<Types...>>&&
    get(const variant<Types...>&&);
  template <class T, class... Types>
  constexpr T& get(variant<Types...>&);
  template <class T, class... Types>
  constexpr T&& get(variant<Types...>&&);
  template <class T, class... Types>
  constexpr const T& get(const variant<Types...>&);
  template <class T, class... Types>
  constexpr const T&& get(const variant<Types...>&&);
  template <size_t I, class... Types>
  constexpr add_pointer_t<variant_alternative_t<I, variant<Types...>>>
    get_if(variant<Types...>*) noexcept;
  template <size_t I, class... Types>
  constexpr add_pointer_t<const variant_alternative_t<I, variant<Types...>>>
    get_if(const variant<Types...>*) noexcept;
  template <class T, class... Types>
  constexpr add_pointer_t<T> get_if(variant<Types...>*) noexcept;
  template <class T, class... Types>
  constexpr add_pointer_t<const T> get_if(const variant<Types...>*) noexcept;

  template <class... Types>
  class variant {
  public:
    // constructors
    constexpr variant()= default ;
    constexpr variant(const variant&);
    constexpr variant(variant&&);
    template<typename T,
            typename = std::enable_if_t<!is_same_v<std::variant<Types...>, decay_t<T>>>>
	  constexpr variant(T&&);
    // assignment
    variant& operator=(const variant&);
    variant& operator=(variant&&) ;
    template<typename T,
            typename = std::enable_if_t<!is_same_v<std::variant<Types...>, decay_t<T>>>>
    variant& operator=(T&&);
  };
  #endif

} // namespace std
