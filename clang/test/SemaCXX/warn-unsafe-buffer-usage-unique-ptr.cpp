// RUN: %clang_cc1 -Wno-unused-value -Wunsafe-buffer-usage -fsafe-buffer-usage-suggestions -std=c++20 -verify=expected %s

namespace std {
inline namespace __1 {
template <class T> class unique_ptr {
public:
  T &operator[](long long i) const;
};
} // namespace __1
} // namespace std

int get_index() {
  return 4;
}

void basic_unique_ptr() {
  std::unique_ptr<int[]> p1;
  int i = 2;
  const int j = 3;
  int k = 0;

  p1[0];  // This is allowed

  p1[k];  // expected-warning{{direct access using operator[] on std::unique_ptr<T[]> is unsafe due to lack of bounds checking}}

  p1[1];  // expected-warning{{direct access using operator[] on std::unique_ptr<T[]> is unsafe due to lack of bounds checking}}

  p1[1L];  // expected-warning{{direct access using operator[] on std::unique_ptr<T[]> is unsafe due to lack of bounds checking}}

  p1[1LL];  // expected-warning{{direct access using operator[] on std::unique_ptr<T[]> is unsafe due to lack of bounds checking}}

  p1[3 * 5];  // expected-warning{{direct access using operator[] on std::unique_ptr<T[]> is unsafe due to lack of bounds checking}}

  p1[i];  // expected-warning{{direct access using operator[] on std::unique_ptr<T[]> is unsafe due to lack of bounds checking}}

  p1[j];  // expected-warning{{direct access using operator[] on std::unique_ptr<T[]> is unsafe due to lack of bounds checking}}

  p1[i + 5]; // expected-warning{{direct access using operator[] on std::unique_ptr<T[]> is unsafe due to lack of bounds checking}}

  p1[get_index()];  // expected-warning{{direct access using operator[] on std::unique_ptr<T[]> is unsafe due to lack of bounds checking}}

}

