// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -O1
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s


// CIR: #tbaa[[tbaa_NYI:.*]] = #cir.tbaa

_BitInt(33) a;
_BitInt(31) b;
void c() {
  // CIR: %{{.*}} = cir.load %{{.*}} : !cir.ptr<!cir.int<s, 33>>, !cir.int<s, 33> tbaa(#tbaa[[tbaa_NYI]])
  // CIR: cir.store %{{.*}}, %{{.*}} : !cir.int<s, 31>, !cir.ptr<!cir.int<s, 31>> tbaa(#tbaa[[tbaa_NYI]])
  b = a;
}
