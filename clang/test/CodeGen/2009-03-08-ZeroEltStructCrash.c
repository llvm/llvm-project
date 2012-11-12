// RUN: %clang_cc1 -emit-llvm %s -o - 
// PR3744
struct Empty {};
struct Union {
 union {
   int zero_arr[0];
 } contents;
};
static inline void Foo(struct Union *u) {
 int *array = u->contents.zero_arr;
}
static void Bar(struct Union *u) {
 Foo(u);
}
