template <typename... T>
void f1() {}
void a() {
  auto Lambda = [] {};
  f1<decltype(Lambda)>();
}
