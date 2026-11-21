// RUN: clang-tblgen -gen-arm-sve-builtin-codegen \
// RUN:   -I %S/../../../llvm/include \
// RUN:   -I %S/../../include/clang/Basic %s -o /dev/null
// RUN: clang-tblgen -gen-arm-sve-builtin-codegen -DSTRONGER_GUARD \
// RUN:   -I %S/../../../llvm/include \
// RUN:   -I %S/../../include/clang/Basic %s -o /dev/null
// RUN: not clang-tblgen -gen-arm-sve-builtin-codegen -DBAD_GUARD \
// RUN:   -I %S/../../../llvm/include \
// RUN:   -I %S/../../include/clang/Basic %s -o /dev/null 2>&1 | FileCheck %s
// RUN: not clang-tblgen -gen-arm-sve-builtin-codegen -DBAD_BF16_GUARD \
// RUN:   -I %S/../../../llvm/include \
// RUN:   -I %S/../../include/clang/Basic %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=BF16
// RUN: not clang-tblgen -gen-arm-sve-builtin-codegen -DBAD_PREDICATE_TYPE \
// RUN:   -I %S/../../../llvm/include \
// RUN:   -I %S/../../include/clang/Basic %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=PREDICATE
// RUN: not clang-tblgen -gen-arm-sve-builtin-codegen -DBAD_MULTI_VECTOR_TYPE \
// RUN:   -I %S/../../../llvm/include \
// RUN:   -I %S/../../include/clang/Basic %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=MULTIVECTOR
// RUN: not clang-tblgen -gen-arm-sve-builtin-codegen -DBAD_NONOVERLOADED_TYPE \
// RUN:   -I %S/../../../llvm/include \
// RUN:   -I %S/../../include/clang/Basic %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NONOVERLOADED

include "arm_sve_intrinsic_guards.td"

// The P prototype modifier creates a predicate with an ordinary type spec.
let TargetGuards = [Variant<"sve", "sme", [[llvm_nxv16i1_ty]]>] in
def int_aarch64_sve_guard_predicate_proto
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty], [llvm_nxv16i8_ty],
                              [IntrNoMem]>;

def GOOD_PREDICATE_PROTO : SInst<"svguard_predicate_proto[_{d}]", "Pd", "c",
                                 MergeNone, "aarch64_sve_guard_predicate_proto">;

let TargetGuards = [Variant<"sve", "sme", EmptyOverload>] in
def int_aarch64_sve_guard_nonoverloaded
    : GuardedDefaultIntrinsic<[llvm_nxv16i8_ty], [llvm_nxv16i8_ty],
                              [IntrNoMem]>;

def GOOD_NONOVERLOADED_TYPE
    : SInst<"svguard_nonoverloaded[_{d}]", "dd", "c", MergeNone,
            "aarch64_sve_guard_nonoverloaded", [IsOverloadNone]>;

#ifdef BAD_NONOVERLOADED_TYPE
def BAD_NONOVERLOADED_TYPE
    : SInst<"svguard_wrong_overload[_{d}]", "dd", "c", MergeNone,
            "aarch64_sve_guard_nonoverloaded">;
#endif

// NONOVERLOADED: cannot identify target guard variant for builtin svguard_wrong_overload_s8

let TargetGuards = [Variant<"sve", "sme", [[LLVMType<nxv32i8>]]>] in
def int_aarch64_sve_guard_multivector
    : GuardedDefaultIntrinsic<[llvm_anyvector_ty],
                              [LLVMMatchType<0>, LLVMMatchType<0>],
                              [IntrNoMem]>;

#ifdef BAD_MULTI_VECTOR_TYPE
def BAD_MULTI_VECTOR_TYPE : SInst<"svguard_nonmultivector[_{d}]", "ddd", "c",
                                  MergeNone, "aarch64_sve_guard_multivector">;
#endif

// MULTIVECTOR: cannot identify target guard variant for builtin svguard_nonmultivector_s8

#ifdef STRONGER_GUARD
let SVETargetGuard = "sve2p2|sme2p2", SMETargetGuard = "sme2p2" in
def STRONGER_GUARD : SInst<"svguard_stronger[_{d}]", "ddd", "i", MergeNone,
                           "aarch64_sve_addp">;
#endif

#ifdef BAD_GUARD
def BAD_GUARD : SInst<"svguard_mismatch[_{d}]", "ddd", "i", MergeNone,
                        "aarch64_sve_addp">;
#endif

// CHECK: builtin svguard_mismatch_s32 targeting aarch64_sve_addp has guards ['sve', 'sme']; intrinsic variants have ['sve,(sve2|sme)', 'sme']

#ifdef BAD_BF16_GUARD
def BAD_BF16_GUARD : SInst<"svguard_bf16[_{d}]", "dddd", "b", MergeNone,
                             "aarch64_sve_fadd">;
#endif

// BF16: builtin svguard_bf16_bf16 targeting aarch64_sve_fadd has guards ['sve', 'sme']; intrinsic variants have ['sve,sve-b16b16', 'sme2,sve-b16b16']

#ifdef BAD_PREDICATE_TYPE
let SVETargetGuard = "sve2p1", SMETargetGuard = "sve2p1|sme2" in
def BAD_PREDICATE_TYPE : SInst<"svguard_nonpredicate[_{d}]", "}}i", "Qc",
                              MergeNone, "aarch64_sve_pext">;
#endif

// PREDICATE: cannot identify target guard variant for builtin svguard_nonpredicate_c8
