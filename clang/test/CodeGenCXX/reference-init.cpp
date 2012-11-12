// RUN: %clang_cc1 -emit-llvm-only -verify %s
// expected-no-diagnostics

struct XPTParamDescriptor {};
struct nsXPTParamInfo {
  nsXPTParamInfo(const XPTParamDescriptor& desc);
};
void a(XPTParamDescriptor *params) {
  const nsXPTParamInfo& paramInfo = params[0];
}

// CodeGen of reference initialized const arrays.
namespace PR5911 {
  template <typename T, int N> int f(const T (&a)[N]) { return N; }
  int iarr[] = { 1 };
  int test() { return f(iarr); }
}

// radar 7574896
struct Foo { int foo; };
Foo& ignoreSetMutex = *(new Foo);

// Binding to a bit-field that requires a temporary. 
struct { int bitfield : 3; } s = { 3 };
const int &s2 = s.bitfield;
