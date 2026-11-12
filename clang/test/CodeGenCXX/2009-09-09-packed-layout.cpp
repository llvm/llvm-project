// RUN: %clang_cc1 -emit-llvm -triple i386-apple-darwin11 %s -o /dev/null
class X {
 public:
  virtual ~X();
  short y;
};
#pragma pack(push, 1)
class Z : public X {
 public: enum { foo = ('x') };
 virtual int y() const;
};
#pragma pack(pop)
class Y : public X {
public: enum { foo = ('y'), bar = 0 };
};
X x;
Y y;
Z z;
