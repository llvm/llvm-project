template<typename, typename = int> struct A;
template<typename T> struct B;

template<typename, typename> struct A {};
template<typename T> struct B : A<T> {};

inline void f() {
  B<int> bi;
}
