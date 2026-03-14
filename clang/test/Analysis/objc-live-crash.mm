// RUN: %clang --analyze %s -fblocks

// https://reviews.llvm.org/D82598#2171312

@interface Item
// ...
@end

@interface Collection
// ...
@end

typedef void (^Blk)();

struct RAII {
  Blk blk;

public:
  RAII(Blk blk): blk(blk) {}
  ~RAII() { blk(); }
};

void foo(Collection *coll) {
  RAII raii(^{});
  for (Item *item in coll) {}
  int i;
  {
    int j;
  }
}
