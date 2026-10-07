enum MyEnum {
  A = 1,
  B = 2
};

static inline MyEnum bad_cast(int x) {
  return (MyEnum)x;
}
