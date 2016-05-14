class A_base
{
public:
  int x;
  A_base(int _x) : x(_x) {
  }
};

class A : public A_base
{
public:
  int y;
  struct { int z; };
  int array[2];
  A(int _x) : A_base(_x), y(0), z(1), array{{2},{3}} {
  }
};
