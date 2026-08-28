// RUN: %clang_cc1 -fclangir -emit-cir -mmlir --mlir-print-ir-after-all -clangir-enable-idiom-recognizer %s -o %t.cir 2>&1 | FileCheck %s -check-prefix=CIR
// CIR: IR Dump After IdiomRecognizer: cir-idiom-recognizer

// The implicit-check-not on the RAISED run makes any surviving std::find call
// an error in the post-pass dump, so the test only passes if it was raised to
// cir.std.find. The FINAL run checks the lowered output and its implicit-check-not
// proves no raised operation leaked past LoweringPrepare.
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -clangir-enable-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o %t.cir 2>&1 | FileCheck %s --check-prefix=RAISED '--implicit-check-not=cir.call @_ZSt4find'
// RUN: FileCheck %s --check-prefix=FINAL --input-file=%t.cir --implicit-check-not=cir.std.

// On targets where plain char is unsigned the pointer lowers to !u8i, and
// recognition works the same.
// RUN: %clang_cc1 -std=c++17 -triple aarch64-unknown-linux-gnu -fno-signed-char -fclangir -clangir-enable-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o %t.aarch64.cir 2>&1 | FileCheck %s --check-prefix=RAISED

// A no builtin list for another function survives the round trip on the
// rebuilt call.
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fno-builtin-memcpy -fclangir -clangir-enable-idiom-recognizer -emit-cir %s -o %t.no-builtin-memcpy.cir
// RUN: FileCheck %s --check-prefix=NO-BUILTIN-MEMCPY --input-file=%t.no-builtin-memcpy.cir

// With builtins disabled the strlen call is left alone, while the tagged
// std::find is unaffected and still raises.
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fno-builtin -fclangir -clangir-enable-idiom-recognizer -emit-cir -mmlir --mlir-print-ir-after=cir-idiom-recognizer %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NO-BUILTINS --implicit-check-not=cir.std.strlen

namespace std {
template <class Iter, class T>
__attribute__((pure)) Iter find(Iter, Iter, const T &) noexcept;
}
extern "C" unsigned long strlen(const char *);

char *test_find(char *first, char *last, const char &value) {
  return std::find(first, last, value);
}
// Raised to cir.std.find, then lowered back to the exact same call with its
// operands in source order and its attributes.
// RAISED: cir.std.find(
// RAISED-SAME: @_ZSt4findIPccET_S1_S1_RKT0_
// NO-BUILTINS: cir.std.find(
// NO-BUILTINS-SAME: @_ZSt4findIPccET_S1_S1_RKT0_
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

unsigned long test_strlen(const char *s) { return strlen(s); }
// RAISED: cir.std.strlen(
// RAISED-SAME: @strlen
// RAISED-SAME: -> !u64i
// FINAL: %[[S_ADDR:.*]] = cir.alloca "s"
// FINAL: %[[S:.*]] = cir.load{{.*}} %[[S_ADDR]] :
// FINAL: cir.call @strlen(%[[S]]) nothrow
// FINAL-SAME: (!cir.ptr<!s8i> {llvm.noundef})
// FINAL-SAME: -> !u64i
// NO-BUILTIN-MEMCPY: cir.call @strlen
// NO-BUILTIN-MEMCPY-SAME: nobuiltins = ["memcpy"]
// NO-BUILTINS: cir.call @strlen

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
