// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s | FileCheck %s

struct X { int i; float f; };
struct Y { X x; };

// CHECK-LABEL: define{{.*}} void @_Z21reinterpret_cast_testRiRfR1X
void reinterpret_cast_test(int &ir, float &fr, X &xr) {
  // CHECK: load float*, float**
  // CHECK: bitcast float*
  // CHECK: load i32, i32*
  ir = reinterpret_cast<int&>(fr);
  // CHECK: load
  // CHECK: {{bitcast.*to i32\*}}
  // CHECK: load i32, i32*
  ir = reinterpret_cast<int&>(xr);
  // CHECK: load i32
  // CHECK: {{bitcast.*to float\*}}
  // CHECK: load float, float*
  fr = reinterpret_cast<float&>(ir);
  // CHECK: load
  // CHECK: {{bitcast.*to float\*}}
  // CHECK: load float, float*
  fr = reinterpret_cast<float&>(xr);
  // CHECK: load i32*, i32**
  // CHECK: bitcast i32*
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
  xr = reinterpret_cast<X&>(ir);
  // CHECK: load float*, float**
  // CHECK: bitcast float*
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
  xr = reinterpret_cast<X&>(fr);
  _Complex float cf;
  _Complex float &cfr = cf;
  // CHECK: load i32*, i32**
  // CHECK: bitcast i32*
  // CHECK: load float, float*
  // CHECK: load float, float*
  cfr = reinterpret_cast<_Complex float&>(ir);
  // CHECK: load float*, float**
  // CHECK: bitcast float*
  // CHECK: load float, float*
  // CHECK: load float, float*
  cfr = reinterpret_cast<_Complex float&>(fr);
  // CHECK: bitcast
  // CHECK: load float, float*
  // CHECK: load float, float*
  cfr = reinterpret_cast<_Complex float&>(xr);
  // CHECK: ret void
}

// CHECK-LABEL: define{{.*}} void @_Z6c_castRiRfR1X
void c_cast(int &ir, float &fr, X &xr) {
  // CHECK: load float*, float**
  // CHECK: bitcast float*
  // CHECK: load i32, i32*
  ir = (int&)fr;
  // CHECK: load
  // CHECK: {{bitcast.*to i32\*}}
  // CHECK: load i32, i32*
  ir = (int&)xr;
  // CHECK: load i32
  // CHECK: {{bitcast.*to float\*}}
  // CHECK: load float, float*
  fr = (float&)ir;
  // CHECK: load
  // CHECK: {{bitcast.*to float\*}}
  // CHECK: load float, float*
  fr = (float&)xr;
  // CHECK: load i32*, i32**
  // CHECK: bitcast i32*
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
  xr = (X&)ir;
  // CHECK: load float*, float**
  // CHECK: bitcast float*
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
  xr = (X&)fr;
  _Complex float cf;
  _Complex float &cfr = cf;
  // CHECK: load i32*, i32**
  // CHECK: bitcast i32*
  // CHECK: load float, float*
  // CHECK: load float, float*
  cfr = (_Complex float&)ir;
  // CHECK: load float*, float**
  // CHECK: bitcast float*
  // CHECK: load float, float*
  // CHECK: load float, float*
  cfr = (_Complex float&)fr;
  // CHECK: bitcast
  // CHECK: load float, float*
  // CHECK: load float, float*
  cfr = (_Complex float&)xr;
  // CHECK: ret void
}

// CHECK-LABEL: define{{.*}} void @_Z15functional_castRiRfR1X
void functional_cast(int &ir, float &fr, X &xr) {
  typedef int &intref;
  typedef float &floatref;
  typedef X &Xref;
  // CHECK: load float*, float**
  // CHECK: bitcast float*
  // CHECK: load i32, i32*
  ir = intref(fr);
  // CHECK: load
  // CHECK: {{bitcast.*to i32\*}}
  // CHECK: load i32, i32*
  ir = intref(xr);
  // CHECK: load i32
  // CHECK: {{bitcast.*to float\*}}
  // CHECK: load float, float*
  fr = floatref(ir);
  // CHECK: load
  // CHECK: {{bitcast.*to float\*}}
  // CHECK: load float, float*
  fr = floatref(xr);
  // CHECK: load i32*, i32**
  // CHECK: bitcast i32*
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
  xr = Xref(ir);
  // CHECK: load float*, float**
  // CHECK: bitcast float*
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
  xr = Xref(fr);
  typedef _Complex float &complex_float_ref;
  _Complex float cf;
  _Complex float &cfr = cf;
  // CHECK: load i32*, i32**
  // CHECK: bitcast i32*
  // CHECK: load float, float*
  // CHECK: load float, float*
  cfr = complex_float_ref(ir);
  // CHECK: load float*, float**
  // CHECK: bitcast float*
  // CHECK: load float, float*
  // CHECK: load float, float*
  cfr = complex_float_ref(fr);
  // CHECK: bitcast
  // CHECK: load float, float*
  // CHECK: load float, float*
  cfr = complex_float_ref(xr);
  // CHECK: ret void
}

namespace PR6437 {
  struct in_addr {};
  void copy( const struct in_addr &new_addr ) {
    int addr = (int&)new_addr;
  }
}

namespace PR7593 {
  void foo(double &X, char *A) {
    X = reinterpret_cast<double&>(A[4]);
  }
}

namespace PR7344 {
  void serialize_annotatable_id( void*& id )
  {
    unsigned long l_id = (unsigned long&)id;
  }
}
