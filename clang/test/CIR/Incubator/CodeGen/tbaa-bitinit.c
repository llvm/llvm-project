// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -O1
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll -O1 -disable-llvm-passes
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// CIR: #tbaa[[BitInt33:.*]] = #cir.tbaa_scalar<id = "_BitInt(33)", type = !cir.int<s, 33>>
// CIR: #tbaa[[BitInt31:.*]] = #cir.tbaa_scalar<id = "_BitInt(31)", type = !cir.int<s, 31>>

_BitInt(33) a;
_BitInt(31) b;
void c() {
  // CIR-LABEL: cir.func {{.*}} @c()
  // CIR: %{{.*}} = cir.load{{.*}} %{{.*}} : !cir.ptr<!cir.int<s, 33>>, !cir.int<s, 33> tbaa(#tbaa[[BitInt33]])
  // CIR: cir.store{{.*}} %{{.*}}, %{{.*}} : !cir.int<s, 31>, !cir.ptr<!cir.int<s, 31>> tbaa(#tbaa[[BitInt31]])

  // LLVM-LABEL: define {{.*}} void @c()
  // LLVM: %{{.*}} = load i33, ptr @a, align 8, !tbaa [[tbaa_tag_bitint_33:!.*]]
  // LLVM: store i31 %{{.*}}, ptr @b, align 4, !tbaa [[tbaa_tag_bitint_31:!.*]]
  b = a;
}
// LLVM: [[tbaa_tag_bitint_33]] = !{[[TYPE_bitint_33:!.*]], [[TYPE_bitint_33]], i64 0}
// LLVM: [[TYPE_bitint_33]] = !{!"_BitInt(33)", [[TYPE_char:!.*]], i64 0}
// LLVM: [[TYPE_char]] = !{!"omnipotent char", [[TAG_c_tbaa:!.*]], i64 0}
// LLVM: [[TAG_c_tbaa]] = !{!"Simple C/C++ TBAA"}
// LLVM: [[tbaa_tag_bitint_31]] = !{[[TYPE_bitint_31:!.*]], [[TYPE_bitint_31]], i64 0}
// LLVM: [[TYPE_bitint_31]] = !{!"_BitInt(31)", [[TYPE_char:!.*]], i64 0}
