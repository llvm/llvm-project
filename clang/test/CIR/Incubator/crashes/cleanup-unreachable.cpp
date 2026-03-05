// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// XFAIL: *
//
// Branch-through cleanups NYI
// Location: CIRGenCleanup.cpp:527
//
// Original failure: cleanup_527 from LLVM build
// Reduced from /tmp/MicrosoftDemangleNodes-acf44f.cpp

class c {
public:
  ~c();
};
struct d {
  template <typename> using ac = c;
};
struct e {
  typedef d::ac<int> ae;
};
class f {
public:
  e::ae ak;
  template <typename g> f(g, g);
};
struct h {
  f i() const;
};
class j {
public:
  ~j();
};
f h::i() const {
  j a;
  f b(0, 0);
  return b;
}
