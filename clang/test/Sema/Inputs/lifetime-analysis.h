
namespace __gnu_cxx {
template <typename T>
struct basic_iterator {
  basic_iterator operator++();
  T& operator*() const;
  T* operator->() const;
};

template<typename T>
bool operator!=(basic_iterator<T>, basic_iterator<T>);
}

namespace std {
template<typename T> struct remove_reference       { typedef T type; };
template<typename T> struct remove_reference<T &>  { typedef T type; };
template<typename T> struct remove_reference<T &&> { typedef T type; };

template<typename T>
typename remove_reference<T>::type &&move(T &&t) noexcept;

template <typename C>
auto data(const C &c) -> decltype(c.data());

template <typename C>
auto begin(C &c) -> decltype(c.begin());

template<typename T, int N>
T *begin(T (&array)[N]);

using size_t = decltype(sizeof(0));

template<typename T>
struct initializer_list {
  const T* ptr; size_t sz;
};
template<typename T> class allocator {};
template <typename T, typename Alloc = allocator<T>>
struct vector {
  typedef __gnu_cxx::basic_iterator<T> iterator;
  iterator begin();
  iterator end();
  const T *data() const;
  vector();
  vector(initializer_list<T> __l,
         const Alloc& alloc = Alloc());

  template<typename InputIterator>
	vector(InputIterator first, InputIterator __last);

  T &at(int n);

  void push_back(const T&);
  void push_back(T&&);
  
  void insert(iterator, T&&);
};

template<typename T>
struct basic_string_view {
  basic_string_view();
  basic_string_view(const T *);
  const T *begin() const;
};
using string_view = basic_string_view<char>;

template<class _Mystr> struct iter {
    iter& operator-=(int);

    iter operator-(int _Off) const {
        iter _Tmp = *this;
        return _Tmp -= _Off;
    }
};

template<typename T>
struct basic_string {
  basic_string();
  basic_string(const T *);
  const T *c_str() const;
  operator basic_string_view<T> () const;
  using const_iterator = iter<T>;
};
using string = basic_string<char>;

template<typename T>
struct unique_ptr {
  T &operator*();
  T *get() const;
};

template<typename T>
struct optional {
  optional();
  optional(const T&);

  template<typename U = T>
  optional(U&& t);

  template<typename U>
  optional(optional<U>&& __t);

  T &operator*() &;
  T &&operator*() &&;
  T &value() &;
  T &&value() &&;
};
template<typename T>
optional<__decay(T)> make_optional(T&&);


template<typename T>
struct stack {
  T &top();
};

struct any {};

template<typename T>
T any_cast(const any& operand);

template<typename T>
struct reference_wrapper {
  template<typename U>
  reference_wrapper(U &&);
};

template<typename T>
reference_wrapper<T> ref(T& t) noexcept;

template <typename T>
struct [[gsl::Pointer]] iterator {
  T& operator*() const;
};

struct false_type {
    static constexpr bool value = false;
    constexpr operator bool() const noexcept { return value; }
};
struct true_type {
    static constexpr bool value = true;
    constexpr operator bool() const noexcept { return value; }
};

template<class T> struct is_pointer : false_type {};
template<class T> struct is_pointer<T*> : true_type {};
template<class T> struct is_pointer<T* const> : true_type {};
}
