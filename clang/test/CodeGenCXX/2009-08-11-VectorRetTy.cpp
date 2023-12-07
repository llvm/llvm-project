// RUN: %clang_cc1 %s -emit-llvm -o /dev/null
typedef void (*Func) ();
typedef long long m64 __attribute__((__vector_size__(8), __may_alias__));
static inline m64 __attribute__((__always_inline__, __nodebug__)) _mm_set1_pi16() {}
template <class MM>
static void Bork() {
  const m64 mmx_0x00ff = _mm_set1_pi16();
}
struct A {};
Func arr[] = {
  Bork<A>
};
