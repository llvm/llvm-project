template<typename T> struct is_floating {
  enum { value = 0 };
  typedef int type;
};
