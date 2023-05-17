// RUN: %clang_cc1 -triple x86_64-apple-darwin12 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin12 -emit-llvm -std=c++98 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin12 -emit-llvm -std=c++11 -o - %s | FileCheck %s

class A { protected: virtual ~A() {} };
class B { protected: virtual ~B() {} };

class C : A { char x; };
class D : public A { short y; };
class E : public A, public B { int z; };
class F : public virtual A { long long w; };
class G : virtual A { long long w; };

class H : public E { int a; };
class I : public F { char b; };

class J : public H { char q; };
class K : public C, public B { char q; };

class XA : public A { };
class XB : public A { };
class XC : public virtual A { };
class X : public XA, public XB, public XC { };

void test(A *a, B *b) {
  volatile C *ac = dynamic_cast<C *>(a);
// CHECK: ptr @_ZTI1A, ptr @_ZTI1C, i64 -2)
  volatile D *ad = dynamic_cast<D *>(a);
// CHECK: ptr @_ZTI1A, ptr @_ZTI1D, i64 0)
  volatile E *ae = dynamic_cast<E *>(a);
// CHECK: ptr @_ZTI1A, ptr @_ZTI1E, i64 0)
  volatile F *af = dynamic_cast<F *>(a);
// CHECK: ptr @_ZTI1A, ptr @_ZTI1F, i64 -1)
  volatile G *ag = dynamic_cast<G *>(a);
// CHECK: ptr @_ZTI1A, ptr @_ZTI1G, i64 -2)
  volatile H *ah = dynamic_cast<H *>(a);
// CHECK: ptr @_ZTI1A, ptr @_ZTI1H, i64 0)
  volatile I *ai = dynamic_cast<I *>(a);
// CHECK: ptr @_ZTI1A, ptr @_ZTI1I, i64 -1)
  volatile J *aj = dynamic_cast<J *>(a);
// CHECK: ptr @_ZTI1A, ptr @_ZTI1J, i64 0)
  volatile K *ak = dynamic_cast<K *>(a);
// CHECK: ptr @_ZTI1A, ptr @_ZTI1K, i64 -2)
  volatile X *ax = dynamic_cast<X *>(a);
// CHECK: ptr @_ZTI1A, ptr @_ZTI1X, i64 -1)

  volatile E *be = dynamic_cast<E *>(b);
// CHECK: ptr @_ZTI1B, ptr @_ZTI1E, i64 8)
  volatile G *bg = dynamic_cast<G *>(b);
// CHECK: ptr @_ZTI1B, ptr @_ZTI1G, i64 -2)
  volatile J *bj = dynamic_cast<J *>(b);
// CHECK: ptr @_ZTI1B, ptr @_ZTI1J, i64 8)
  volatile K *bk = dynamic_cast<K *>(b);
// CHECK: ptr @_ZTI1B, ptr @_ZTI1K, i64 16)
}
