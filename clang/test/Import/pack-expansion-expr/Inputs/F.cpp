template <typename... T>
void sink(T... a);

template <typename... T>
void packfuncT(T... a) {
  sink(a...);
}

void f() {
  packfuncT(1, 2, 3);
}
