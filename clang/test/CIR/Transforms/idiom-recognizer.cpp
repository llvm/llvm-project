// RUN: %clang_cc1 -fclangir -emit-cir -mmlir --mlir-print-ir-after-all -clangir-enable-idiom-recognizer %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CIR
// CIR: IR Dump After IdiomRecognizer: cir-idiom-recognizer

// The implicit-check-not on the RAISED run makes any surviving std::find call
// an error in the post-pass dump, so the test only passes if it was raised to
// cir.std.find. The FINAL run checks the lowered output and its implicit-check-not
// proves no raised operation leaked past LoweringPrepare.
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -clangir-enable-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=RAISED '--implicit-check-not=cir.call @_ZSt4find'
// RUN: FileCheck %s --check-prefix=FINAL --input-file=%t.cir --implicit-check-not=cir.std.

namespace std {
template <class Iter, class T>
__attribute__((pure)) Iter find(Iter, Iter, const T &) noexcept;
}

char *test_find(char *first, char *last, const char &value) {
  return std::find(first, last, value);
}
// Raised to cir.std.find, then lowered back to the exact same call with its
// operands in source order and its attributes.
// RAISED: cir.std.find(
// RAISED-SAME: @_ZSt4findIPccET_S1_S1_RKT0_
// FINAL: %[[FIRST_ADDR:.*]] = cir.alloca "first"
// FINAL: %[[LAST_ADDR:.*]] = cir.alloca "last"
// FINAL: %[[VALUE_ADDR:.*]] = cir.alloca "value"
// FINAL: %[[FIRST:.*]] = cir.load{{.*}} %[[FIRST_ADDR]] :
// FINAL: %[[LAST:.*]] = cir.load{{.*}} %[[LAST_ADDR]] :
// FINAL: %[[VALUE:.*]] = cir.load{{.*}} %[[VALUE_ADDR]] :
// FINAL: cir.call @_ZSt4findIPccET_S1_S1_RKT0_(%[[FIRST]], %[[LAST]], %[[VALUE]])
// FINAL-SAME: nothrow side_effect(pure)
// FINAL-SAME: {llvm.noundef}
// FINAL-SAME: -> (!cir.ptr<!s8i> {llvm.noundef})
// FINAL-NOT: cir.call @_ZSt4find

// A function merely named like the std one is not raised, and it survives the
// whole pipeline as the same plain call.
char *find(char *first, char *last, const char &value);
char *test_non_std_find(char *first, char *last, const char &value) {
  return find(first, last, value);
}
// RAISED: cir.call @_Z4findPcS_RKc
// RAISED-NOT: cir.std.find
// FINAL: cir.call @_Z4findPcS_RKc

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
// RAISED: cir.call @_ZNSt6string4findEPS_S0_
// RAISED-NOT: cir.std.find
// FINAL: cir.call @_ZNSt6string4findEPS_S0_
