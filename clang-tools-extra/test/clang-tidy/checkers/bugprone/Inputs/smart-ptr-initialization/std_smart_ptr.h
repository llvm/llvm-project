namespace std {

typedef decltype(nullptr) nullptr_t;
typedef unsigned long size_t;

template <typename T>
struct default_delete {
  void operator()(T* p) const;
};

template <typename T>
struct default_delete<T[]> {
  void operator()(T* p) const;
};

template <typename T, typename Deleter = default_delete<T>>
class unique_ptr {
public:
  unique_ptr();
  explicit unique_ptr(T* p);
  unique_ptr(T* p, Deleter d) {}
  unique_ptr(std::nullptr_t);
  
  T* release();
  
  void reset(T* p = nullptr);
  
  template <typename D>
  void reset(T* p, D d) {}
};

template <typename T, typename Deleter>
class unique_ptr<T[], Deleter> {
public:
  unique_ptr();
  template <typename U>
  explicit unique_ptr(U* p);
  template <typename U>
  unique_ptr(U* p, Deleter d) {}
  unique_ptr(std::nullptr_t);
  
  T* release();
  
  void reset(T* p = nullptr);
  
  template <typename D>
  void reset(T* p, D d) {}
};

template <typename T>
class shared_ptr {
public:
  shared_ptr();
  explicit shared_ptr(T* p);
  template <typename Deleter>
  shared_ptr(T* p, Deleter d) {}
  shared_ptr(std::nullptr_t);
  
  T* release();
  
  void reset(T* p = nullptr);
  
  template <typename Deleter>
  void reset(T* p, Deleter d) {}
};

template <typename T>
class shared_ptr<T[]> {
public:
  shared_ptr();
  template <typename U>
  explicit shared_ptr(U* p);
  template <typename U, typename Deleter>
  shared_ptr(U* p, Deleter d) {}
  shared_ptr(std::nullptr_t);
  
  T* release();
  
  void reset(T* p = nullptr);
  
  template <typename Deleter>
  void reset(T* p, Deleter d) {}
};

template <typename T>
shared_ptr<T> make_shared();

template <typename T>
shared_ptr<T[]> make_shared(std::size_t n);

template <typename T>
unique_ptr<T> make_unique();

template <typename T>
unique_ptr<T[]> make_unique(std::size_t n);

} // namespace std
