template <typename T>
struct Box {
  T value;

  const T& get_value() const { return value; }
  const T* get_ptr() const { return &value; }
};

using FloatBox = Box<float>;
using IntBox = Box<int>;

template <typename T>
struct MoveOnly {
  T value;
};

template <>
struct MoveOnly<float> {
  double value;
};

struct MoveOnlyBox {
  MoveOnly<int> value1;
  MoveOnly<float> value2;
};
