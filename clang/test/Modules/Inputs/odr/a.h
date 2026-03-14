extern struct Y {
  int n;
  float f;
} y1;
enum E { e1 };

struct X {
  int n;
} x1;

template<typename T>
struct F {
  int n;
  friend bool operator==(const F &a, const F &b) { return a.n == b.n; }
};

int f() {
  return y1.n + e1 + y1.f + x1.n;
}
