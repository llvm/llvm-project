template <typename> struct _Vector_base {};
struct vector {
  vector() {}
  vector(_Vector_base<int>);
};
