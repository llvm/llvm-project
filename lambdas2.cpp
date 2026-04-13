template <typename T>
void f1(T) {}
inline void f1() {
    f1([] {});
    f1([] {});
    f1([](int) {});
    f1([](int) {});
}
void f2() {
    f1();
}
void f3() {
    f1([] {});
    f1([](int) {});
}
