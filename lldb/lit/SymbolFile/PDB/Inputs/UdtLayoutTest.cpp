struct A {
  explicit A(int u) { _u._u3 = u; }
  A(const A &) = default;
  virtual ~A() = default;

private:
  union U {
    char _u1;
    short _u2;
    int _u3;
  };

  A::U _u;
};

#pragma pack(push, 1)
template <int I> struct B : public virtual A {
  B(char a, unsigned short b, int c) : A(a + b + c), _a(a), _b(b), _c(c) {}

private:
  char _a;
  unsigned short : 3;
  unsigned short _b : 6;
  unsigned short : 4;
  int _c;
};
#pragma pack(pop)

#pragma pack(push, 16)
class C : private virtual B<0>, public virtual B<1>, private B<2>, public B<3> {
public:
  C(char x, char y, char z)
      : A(x - y + z), B<0>(x, y, z), B<1>(x * 2, y * 2, z * 2),
        B<2>(x * 3, y * 3, z * 3), B<3>(x * 4, y * 4, z * 4), _x(x * 5),
        _y(y * 5), _z(z * 5) {}

  static int abc;

private:
  int _x;
  short _y;
  char _z;
};
int C::abc = 123;
#pragma pack(pop)

class List {
public:
  List() = default;
  List(List *p, List *n, C v) : Prev(p), Next(n), Value(v) {}

private:
  List *Prev = nullptr;
  List *Next = nullptr;
  C Value{1, 2, 3};
};

int main() {
  List ls[16];
  return 0;
}
