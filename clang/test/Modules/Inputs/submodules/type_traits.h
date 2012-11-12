template<typename T>
struct remove_reference {
  typedef T type;
};

template<typename T>
struct remove_reference<T&> {
  typedef T type;
};

#define HAVE_TYPE_TRAITS

