
namespace __gnu_cxx {
template <typename T>
struct basic_iterator {
  basic_iterator operator++();
  T& operator*() const;
  T* operator->() const;
};

template<typename T>
bool operator==(basic_iterator<T>, basic_iterator<T>);
template<typename T>
bool operator!=(basic_iterator<T>, basic_iterator<T>);
}

namespace std {
template<typename T> struct remove_reference       { typedef T type; };
template<typename T> struct remove_reference<T &>  { typedef T type; };
template<typename T> struct remove_reference<T &&> { typedef T type; };

template< class InputIt, class T >
InputIt find( InputIt first, InputIt last, const T& value );

template< class ForwardIt1, class ForwardIt2 >
ForwardIt1 search( ForwardIt1 first, ForwardIt1 last,
                   ForwardIt2 s_first, ForwardIt2 s_last );

template<typename T>
typename remove_reference<T>::type &&move(T &&t) noexcept;

template <typename C>
auto data(const C &c) -> decltype(c.data());

template <typename C>
auto begin(C &c) -> decltype(c.begin());
template <typename C>
auto end(C &c) -> decltype(c.end());

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
  ~vector();
  vector(initializer_list<T> __l,
         const Alloc& alloc = Alloc());

  template<typename InputIterator>
	vector(InputIterator first, InputIterator __last);

  T& operator[](unsigned);

  T &  at(int n) &;
  T && at(int n) &&;

  void push_back(const T&);
  void push_back(T&&);
  const T& back() const;
  void pop_back();
  iterator insert(iterator, T&&);
  void resize(size_t);
  void erase(iterator);
  void clear();
};

template<class T>
void swap( T& a, T& b );

template<typename A, typename B>
struct pair {
  A first;
  B second;
};

template<class Key,class T>
struct unordered_map {
  using iterator = __gnu_cxx::basic_iterator<std::pair<const Key, T>>;
  T& operator[](const Key& key);
  iterator begin();
  iterator end();
  iterator erase(iterator);
};

template<typename T>
struct basic_string_view {
  basic_string_view();
  basic_string_view(const T *);
  const T *begin() const;
  const T *data() const;
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
  basic_string(const basic_string<T> &);
  basic_string(basic_string<T> &&);
  basic_string(const T *);
  ~basic_string();
  basic_string& operator=(const basic_string&);
  basic_string& operator+=(const basic_string&);
  basic_string& operator+=(const T*);
  const T *c_str() const;
  operator basic_string_view<T> () const;
  using const_iterator = iter<T>;
  const T *data() const;
};
using string = basic_string<char>;

template<typename T>
struct unique_ptr {
  unique_ptr();
  unique_ptr(unique_ptr<T>&&);
  ~unique_ptr();
  T* release();
  T &operator*();
  T *get() const;
};

template<typename T>
struct optional {
  optional();
  optional(const T&);

  ~optional();

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

struct any {
  // FIXME: CFG based analysis should be able to catch bugs without need of ctor and dtor.
  any();
  ~any();
};

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
