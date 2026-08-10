// RUN: %clang_cc1 -emit-llvm %s -o -
struct TRunSoon {
  template <class P1> static void Post() {}
};

@implementation TPrivsTableViewMainController
- (void) applyToEnclosed {
  TRunSoon::Post<int>();
}
@end
