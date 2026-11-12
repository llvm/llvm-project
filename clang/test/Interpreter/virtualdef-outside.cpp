// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl | FileCheck %s
// virtual functions defined outside of class had duplicate symbols:
//     duplicate definition of symbol '__ZTV3Two' (i.e., vtable for Two)
// see https://github.com/llvm/llvm-project/issues/141039.
// fixed in PR #185648

extern "C" int printf(const char *, ...);

struct X1 { virtual void vi() { printf("1vi\n"); } };
X1().vi();
// CHECK: 1vi

struct X2 { virtual void vo(); };
void X2::vo() { printf("2vo\n"); }
X2().vo();
// CHECK: 2vo

struct X3 { \
  void ni() { printf("3ni\n"); } \
  void no(); \
  virtual void vi() { printf("3vi\n"); } \
  virtual void vo(); \
  virtual ~X3() { printf("3d\n"); } \
};
void X3::no() { printf("3no\n"); }
void X3::vo() { printf("3vo\n"); }
auto x3 = new X3;
x3->ni();
// CHECK: 3ni
x3->no();
// CHECK: 3no
x3->vi();
// CHECK: 3vi
x3->vo();
// CHECK: 3vo
delete x3;
// CHECK: 3d

%quit
