// RUN: %clang_cc1 -std=c++17 -debug-info-kind=standalone -O2 \
// RUN:   -triple x86_64-linux-gnu -emit-llvm %s -o /dev/null
//
// REQUIRES: asserts
//
// Verify we don't crash when generating debug info for a polymorphic
// template class that is still being defined when its type identifier is
// requested.
//
// getTypeIdentifier() called getVTableLinkage() on a record that was only
// "being defined" (not yet complete). getVTableLinkage() in turn calls
// ItaniumCXXABI::canSpeculativelyEmitVTable(), which builds the vtable
// layout via FinalOverriders, which calls getASTRecordLayout() on the
// incomplete type, triggering:
//   Assertion `D->isCompleteDefinition() &&
//              "Cannot layout type before complete!"' failed.
// In release (no-asserts) builds the same path silently caches a layout
// with zero fields, leading to an out-of-bounds field offset crash later
// in CollectRecordFields.

struct l;
template <typename> struct b;
template <typename> struct f {
  struct e;
  b<l> *p;
};
template <typename e> struct b {
  virtual ~b();
  typename f<e>::e x();
};
f<l> *s;
extern template struct b<l>;
