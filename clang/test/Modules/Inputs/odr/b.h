struct Y {
  int m;
  double f;
} y2;
enum E { e2 };

template<typename T>
struct F {
  int n;
  friend bool operator==(const F &a, const F &b) { return a.n == b.n; }
};

int g() {
  return y2.m + e2 + y2.f + (F<int>{0} == F<int>{1});
}
