template <typename T>
concept Sizable = requires(T t) { t.size(); };

template <typename T>
concept Container = Sizable<T> && requires(T t) { t.begin(); };

template <bool> struct BoolConstant {};
using FalseCheck = BoolConstant<Container<int>>;

void importee() {
  FalseCheck f{};
  (void)f;
}
