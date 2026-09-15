template <typename T>
concept Incrementable = requires(T x) {
  ++x;
  x++;
};

template <typename T>
void increment(T t)
  requires Incrementable<T>;

template <Incrementable T> Incrementable auto incrementTwo(T t);
