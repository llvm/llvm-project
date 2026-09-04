template <typename T>
concept Addable = requires(T a, T b) {
  { a + b };
};

template <typename T>
  requires Addable<T>
struct MyClass;
