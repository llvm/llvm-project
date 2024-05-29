template <typename T>
struct Box {
  T value;

  const T& get_value() const { return value; }
  const T* get_ptr() const { return &value; }
};

using IntBox = Box<int>;
