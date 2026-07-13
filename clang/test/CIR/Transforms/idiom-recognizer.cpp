// RUN: %clang_cc1 -fclangir -emit-cir -mmlir --mlir-print-ir-after-all -clangir-enable-idiom-recognizer %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CIR
// CIR: IR Dump After IdiomRecognizer: cir-idiom-recognizer

// The implicit-check-not on the RAISED run makes the original std::find call an
// error anywhere in the post-pass dump, so the test only passes if that call was
// raised to cir.std.find rather than left in place.
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -clangir-enable-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=RAISED '--implicit-check-not=cir.call @_ZSt4findIPccET_S1_S1_RKT0_'
// RUN: FileCheck %s --check-prefix=FINAL --input-file=%t.cir

namespace std {
template <class Iter, class T>
__attribute__((pure)) Iter find(Iter, Iter, const T &) noexcept;
}

char *test_find(char *first, char *last, const char &value) {
  return std::find(first, last, value);
}
// Raised, then lowered back in operand order with its call attributes.
// RAISED: cir.std.find(
// RAISED-SAME: @_ZSt4find
// FINAL: %[[FIRST:.*]] = cir.load{{.*}} : !cir.ptr<!cir.ptr<!s8i>>
// FINAL: %[[LAST:.*]] = cir.load{{.*}} : !cir.ptr<!cir.ptr<!s8i>>
// FINAL: %[[VALUE:.*]] = cir.load{{.*}} : !cir.ptr<!cir.ptr<!s8i>>
// FINAL: cir.call @_ZSt4find{{.*}}(%[[FIRST]], %[[LAST]], %[[VALUE]])
// FINAL-SAME: nothrow side_effect(pure)
// FINAL-SAME: {llvm.noundef}
// FINAL-SAME: -> (!cir.ptr<!s8i> {llvm.noundef})
// FINAL-NOT: cir.call @_ZSt4find
// FINAL-NOT: cir.std.

// A function merely named like the std one is not raised.
char *find(char *first, char *last, const char &value);
char *test_non_std_find(char *first, char *last, const char &value) {
  return find(first, last, value);
}
// RAISED: cir.call @_Z4find

// A member function named find in std is not std::find. The types are chosen
// so the call reaches the member exclusion itself.
namespace std {
struct string {
  string *find(string *first, string *last);
};
}

std::string *test_member_find(std::string &s, std::string *f, std::string *l) {
  return s.find(f, l);
}
// RAISED: cir.call @_ZNSt6string4find
// RAISED-NOT: cir.std.find
