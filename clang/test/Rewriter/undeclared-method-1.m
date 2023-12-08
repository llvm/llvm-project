// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o -

@interface Derived @end

int main(void) {
  Derived *v ;
  [v free];
  return 0;
}
