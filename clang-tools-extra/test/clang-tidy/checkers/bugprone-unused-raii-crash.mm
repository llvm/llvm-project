// RUN: clang-tidy %s -checks=-*,bugprone-unused-raii -- | count 0

struct CxxClass {
  ~CxxClass() {}
};

@interface ObjcClass {
}
- (CxxClass)set:(int)p;
@end

void test(ObjcClass *s) {
  [s set:1]; // ok, no crash, no diagnostic emitted.
  return;
}
