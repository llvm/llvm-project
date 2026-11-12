// RUN: %clang_cc1 -triple %itanium_abi_triple -fvisibility=hidden -fms-extensions -fsanitize=cfi-vcall -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=NOBL %s

// Check that ignore listing cfi and cfi-vcall work correctly
// RUN: echo "[cfi-vcall]" > %t.vcall.txt
// RUN: echo "type:std::*" >> %t.vcall.txt
// RUN: %clang_cc1 -triple %itanium_abi_triple -fvisibility=hidden -fms-extensions -fsanitize=cfi-vcall -fsanitize-ignorelist=%t.vcall.txt -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=NOSTD %s
//
// RUN: echo "[cfi]" > %t.cfi.txt
// RUN: echo "type:std::*" >> %t.cfi.txt
// RUN: %clang_cc1 -triple %itanium_abi_triple -fvisibility=hidden -fms-extensions -fsanitize=cfi-vcall -fsanitize-ignorelist=%t.cfi.txt -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=NOSTD %s

// Check that ignore listing non-vcall modes does not affect vcalls
// RUN: echo "[cfi-icall|cfi-nvcall|cfi-cast-strict|cfi-derived-cast|cfi-unrelated-cast]" > %t.other.txt
// RUN: echo "type:std::*" >> %t.other.txt
// RUN: %clang_cc1 -triple %itanium_abi_triple -fvisibility=hidden -fms-extensions -fsanitize=cfi-vcall -fsanitize-ignorelist=%t.other.txt -emit-llvm -o - %s | FileCheck --check-prefix=CHECK --check-prefix=NOBL %s

// Check ignore listing with -fwhole-program-vtables and -fvirtual-function-elimination.
// RUN: %clang_cc1 -triple x86_64-unknown-linux -flto -flto-unit -fvisibility=hidden -fms-extensions -fsanitize=cfi-vcall -fwhole-program-vtables -fsanitize-ignorelist=%t.vcall.txt -emit-llvm -o - %s | FileCheck --check-prefix=NOSTD-WPD %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -flto -flto-unit -fvisibility=hidden -fms-extensions -fsanitize=cfi-vcall -fwhole-program-vtables -fvirtual-function-elimination -fsanitize-ignorelist=%t.vcall.txt -emit-llvm -o - %s | FileCheck --check-prefix=NOSTD-VFE %s

struct S1 {
  virtual void f();
};

namespace std {

struct S2 {
  virtual void f();
};

}

// CHECK: define{{.*}}s1f
// NOSTD-WPD-LABEL: define{{.*}}s1f
// NOSTD-VFE-LABEL: define{{.*}}s1f
// NOBL: llvm.type.test
// NOSTD: llvm.type.test
// NOSTD-WPD: call { ptr, i1 } @llvm.type.checked.load
// NOSTD-WPD: call void @__ubsan_handle_cfi_check_fail_abort(ptr
// NOSTD-VFE: call { ptr, i1 } @llvm.type.checked.load
// NOSTD-VFE: call void @__ubsan_handle_cfi_check_fail_abort(ptr
void s1f(S1 *s1) {
  s1->f();
}

// CHECK: define{{.*}}s2f
// NOSTD-WPD-LABEL: define{{.*}}s2f
// NOSTD-VFE-LABEL: define{{.*}}s2f
// NOBL: llvm.type.test
// NOSTD-NOT: llvm.type.test
// NOSTD-WPD-NOT: llvm.type.checked.load
// NOSTD-WPD-NOT: llvm.type.test
// NOSTD-WPD-NOT: call void @__ubsan_handle_cfi_check_fail
// NOSTD-WPD: ret void
// NOSTD-VFE: call { ptr, i1 } @llvm.type.checked.load
// NOSTD-VFE-NOT: extractvalue { ptr, i1 } {{.*}}, 1
// NOSTD-VFE-NOT: call void @__ubsan_handle_cfi_check_fail
// NOSTD-VFE: ret void
void s2f(std::S2 *s2) {
  s2->f();
}

struct DerivedFromS2 : std::S2 {};

// CHECK-LABEL: define{{.*}}d2f
// NOSTD-WPD-LABEL: define{{.*}}d2f
// NOSTD-VFE-LABEL: define{{.*}}d2f
// NOBL: llvm.type.test{{.*}}!"_ZTSSt2S2"
// NOSTD-NOT: llvm.type.test
// CHECK: ret void
// NOSTD-WPD-NOT: llvm.type.checked.load
// NOSTD-WPD-NOT: llvm.type.test
// NOSTD-WPD-NOT: call void @__ubsan_handle_cfi_check_fail
// NOSTD-WPD: ret void
// NOSTD-VFE: call { ptr, i1 } @llvm.type.checked.load(ptr {{.*}}, i32 0, metadata !"_ZTSSt2S2")
// NOSTD-VFE-NOT: extractvalue { ptr, i1 } {{.*}}, 1
// NOSTD-VFE-NOT: call void @__ubsan_handle_cfi_check_fail
// NOSTD-VFE: ret void
void d2f(DerivedFromS2 *d) {
  d->f();
}
