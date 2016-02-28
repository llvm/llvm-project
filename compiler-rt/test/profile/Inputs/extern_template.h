template <typename T> struct Test {
  Test() : M(10) {}
  void doIt(int N) { // CHECK: 2| [[@LINE]]|  void doIt
    if (N > 10) {    // CHECK: 2| [[@LINE]]|    if (N > 10) {
      M += 2;        // CHECK: 1| [[@LINE]]|      M += 2;
    } else           // CHECK: 1| [[@LINE]]|    } else
      M -= 2;        // CHECK: 1| [[@LINE]]|      M -= 2;
  }
  T M;
};

#ifdef USE
extern template struct Test<int>;
#endif
#ifdef DEF
template struct Test<int>;
#endif
