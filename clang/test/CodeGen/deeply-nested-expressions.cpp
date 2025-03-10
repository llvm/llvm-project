// RUN: %clang_cc1 %s -emit-llvm -o - -Wstack-exhausted
// RUN: %clang_cc1 %s -emit-llvm -o - -Wstack-exhausted -O2

class AClass {
public:
  AClass() {}
  AClass &f() { return *this; }
};

#define CALLS1 f
#define CALLS2 CALLS1().CALLS1
#define CALLS4 CALLS2().CALLS2
#define CALLS8 CALLS4().CALLS4
#define CALLS16 CALLS8().CALLS8
#define CALLS32 CALLS16().CALLS16
#define CALLS64 CALLS32().CALLS32
#define CALLS128 CALLS64().CALLS64
#define CALLS256 CALLS128().CALLS128
#define CALLS512 CALLS256().CALLS256
#define CALLS1024 CALLS512().CALLS512
#define CALLS2048 CALLS1024().CALLS1024
#define CALLS4096 CALLS2048().CALLS2048
#define CALLS8192 CALLS4096().CALLS4096
#define CALLS16384 CALLS8192().CALLS8192
#define CALLS32768 CALLS16384().CALLS16384

void test_bar() {
  AClass a;
  a.CALLS32768();
}
