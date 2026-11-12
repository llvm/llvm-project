// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics

@interface I {
  int position;
}
@property(nonatomic) int position;
@end

struct S {
  void *operator new(__SIZE_TYPE__, int);
};

template <typename T>
struct TS {
  void *operator new(__SIZE_TYPE__, T);
};

I *GetI();

int main() {
  @autoreleasepool {
    auto* i = GetI();
    i.position = 42;
    new (i.position) S;
    new (i.position) TS<double>;
  }
}
