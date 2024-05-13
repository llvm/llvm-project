// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -fzero-call-used-regs=skip -o - | FileCheck %s --check-prefix CHECK-SKIP
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -fzero-call-used-regs=used-gpr-arg -o - | FileCheck %s --check-prefix CHECK-USED-GPR-ARG
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -fzero-call-used-regs=used-gpr -o - | FileCheck %s --check-prefix CHECK-USED-GPR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -fzero-call-used-regs=used-arg -o - | FileCheck %s --check-prefix CHECK-USED-ARG
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -fzero-call-used-regs=used -o - | FileCheck %s --check-prefix CHECK-USED
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -fzero-call-used-regs=all-gpr-arg -o - | FileCheck %s --check-prefix CHECK-ALL-GPR-ARG
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -fzero-call-used-regs=all-gpr -o - | FileCheck %s --check-prefix CHECK-ALL-GPR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -fzero-call-used-regs=all-arg -o - | FileCheck %s --check-prefix CHECK-ALL-ARG
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu %s -emit-llvm -fzero-call-used-regs=all -o - | FileCheck %s --check-prefix CHECK-ALL

// -fzero-call-used-regs=skip:
//
// CHECK-SKIP:               define {{.*}} @no_attribute({{.*}} #[[ATTR_NUM:[0-9]*]]
// CHECK-SKIP:               define {{.*}} @skip_test({{.*}} #[[ATTR_NUM_SKIP:[0-9]*]]
// CHECK-SKIP:               define {{.*}} @used_gpr_arg_test({{.*}} #[[ATTR_NUM_USED_GPR_ARG:[0-9]*]]
// CHECK-SKIP:               define {{.*}} @used_gpr_test({{.*}} #[[ATTR_NUM_USED_GPR:[0-9]*]]
// CHECK-SKIP:               define {{.*}} @used_arg_test({{.*}} #[[ATTR_NUM_USED_ARG:[0-9]*]]
// CHECK-SKIP:               define {{.*}} @used_test({{.*}} #[[ATTR_NUM_USED:[0-9]*]]
// CHECK-SKIP:               define {{.*}} @all_gpr_arg_test({{.*}} #[[ATTR_NUM_ALL_GPR_ARG:[0-9]*]]
// CHECK-SKIP:               define {{.*}} @all_gpr_test({{.*}} #[[ATTR_NUM_ALL_GPR:[0-9]*]]
// CHECK-SKIP:               define {{.*}} @all_arg_test({{.*}} #[[ATTR_NUM_ALL_ARG:[0-9]*]]
// CHECK-SKIP:               define {{.*}} @all_test({{.*}} #[[ATTR_NUM_ALL:[0-9]*]]
//
// CHECK-SKIP-NOT:           attributes #[[ATTR_NUM]] = {{.*}} "zero-call-used-regs"=
// CHECK-SKIP:               attributes #[[ATTR_NUM_SKIP]] = {{.*}} "zero-call-used-regs"="skip"
// CHECK-SKIP:               attributes #[[ATTR_NUM_USED_GPR_ARG]] = {{.*}} "zero-call-used-regs"="used-gpr-arg"
// CHECK-SKIP:               attributes #[[ATTR_NUM_USED_GPR]] = {{.*}} "zero-call-used-regs"="used-gpr"
// CHECK-SKIP:               attributes #[[ATTR_NUM_USED_ARG]] = {{.*}} "zero-call-used-regs"="used-arg"
// CHECK-SKIP:               attributes #[[ATTR_NUM_USED]] = {{.*}} "zero-call-used-regs"="used"
// CHECK-SKIP:               attributes #[[ATTR_NUM_ALL_GPR_ARG]] = {{.*}} "zero-call-used-regs"="all-gpr-arg"
// CHECK-SKIP:               attributes #[[ATTR_NUM_ALL_GPR]] = {{.*}} "zero-call-used-regs"="all-gpr"
// CHECK-SKIP:               attributes #[[ATTR_NUM_ALL_ARG]] = {{.*}} "zero-call-used-regs"="all-arg"
// CHECK-SKIP:               attributes #[[ATTR_NUM_ALL]] = {{.*}} "zero-call-used-regs"="all"

// -fzero-call-used-regs=used-gpr-arg:
//
// CHECK-USED-GPR-ARG:       define {{.*}} @no_attribute({{.*}} #[[ATTR_NUM_USED_GPR_ARG:[0-9]*]]
// CHECK-USED-GPR-ARG:       define {{.*}} @skip_test({{.*}} #[[ATTR_NUM_SKIP:[0-9]*]]
// CHECK-USED-GPR-ARG:       define {{.*}} @used_gpr_arg_test({{.*}} #[[ATTR_NUM_USED_GPR_ARG]]
// CHECK-USED-GPR-ARG:       define {{.*}} @used_gpr_test({{.*}} #[[ATTR_NUM_USED_GPR:[0-9]*]]
// CHECK-USED-GPR-ARG:       define {{.*}} @used_arg_test({{.*}} #[[ATTR_NUM_USED_ARG:[0-9]*]]
// CHECK-USED-GPR-ARG:       define {{.*}} @used_test({{.*}} #[[ATTR_NUM_USED:[0-9]*]]
// CHECK-USED-GPR-ARG:       define {{.*}} @all_gpr_arg_test({{.*}} #[[ATTR_NUM_ALL_GPR_ARG:[0-9]*]]
// CHECK-USED-GPR-ARG:       define {{.*}} @all_gpr_test({{.*}} #[[ATTR_NUM_ALL_GPR:[0-9]*]]
// CHECK-USED-GPR-ARG:       define {{.*}} @all_arg_test({{.*}} #[[ATTR_NUM_ALL_ARG:[0-9]*]]
// CHECK-USED-GPR-ARG:       define {{.*}} @all_test({{.*}} #[[ATTR_NUM_ALL:[0-9]*]]
//
// CHECK-USED-GPR-ARG:       attributes #[[ATTR_NUM_USED_GPR_ARG]] = {{.*}} "zero-call-used-regs"="used-gpr-arg"
// CHECK-USED-GPR-ARG:       attributes #[[ATTR_NUM_SKIP]] = {{.*}} "zero-call-used-regs"="skip"
// CHECK-USED-GPR-ARG:       attributes #[[ATTR_NUM_USED_GPR]] = {{.*}} "zero-call-used-regs"="used-gpr"
// CHECK-USED-GPR-ARG:       attributes #[[ATTR_NUM_USED_ARG]] = {{.*}} "zero-call-used-regs"="used-arg"
// CHECK-USED-GPR-ARG:       attributes #[[ATTR_NUM_USED]] = {{.*}} "zero-call-used-regs"="used"
// CHECK-USED-GPR-ARG:       attributes #[[ATTR_NUM_ALL_GPR_ARG]] = {{.*}} "zero-call-used-regs"="all-gpr-arg"
// CHECK-USED-GPR-ARG:       attributes #[[ATTR_NUM_ALL_GPR]] = {{.*}} "zero-call-used-regs"="all-gpr"
// CHECK-USED-GPR-ARG:       attributes #[[ATTR_NUM_ALL_ARG]] = {{.*}} "zero-call-used-regs"="all-arg"
// CHECK-USED-GPR-ARG:       attributes #[[ATTR_NUM_ALL]] = {{.*}} "zero-call-used-regs"="all"

// -fzero-call-used-regs=used-gpr:
//
// CHECK-USED-GPR:           define {{.*}} @no_attribute({{.*}} #[[ATTR_NUM_USED_GPR:[0-9]*]]
// CHECK-USED-GPR:           define {{.*}} @skip_test({{.*}} #[[ATTR_NUM_SKIP:[0-9]*]]
// CHECK-USED-GPR:           define {{.*}} @used_gpr_arg_test({{.*}} #[[ATTR_NUM_USED_GPR_ARG:[0-9]*]]
// CHECK-USED-GPR:           define {{.*}} @used_gpr_test({{.*}} #[[ATTR_NUM_USED_GPR]]
// CHECK-USED-GPR:           define {{.*}} @used_arg_test({{.*}} #[[ATTR_NUM_USED_ARG:[0-9]*]]
// CHECK-USED-GPR:           define {{.*}} @used_test({{.*}} #[[ATTR_NUM_USED:[0-9]*]]
// CHECK-USED-GPR:           define {{.*}} @all_gpr_arg_test({{.*}} #[[ATTR_NUM_ALL_GPR_ARG:[0-9]*]]
// CHECK-USED-GPR:           define {{.*}} @all_gpr_test({{.*}} #[[ATTR_NUM_ALL_GPR:[0-9]*]]
// CHECK-USED-GPR:           define {{.*}} @all_arg_test({{.*}} #[[ATTR_NUM_ALL_ARG:[0-9]*]]
// CHECK-USED-GPR:           define {{.*}} @all_test({{.*}} #[[ATTR_NUM_ALL:[0-9]*]]
//
// CHECK-USED-GPR:           attributes #[[ATTR_NUM_USED_GPR]] = {{.*}} "zero-call-used-regs"="used-gpr"
// CHECK-USED-GPR:           attributes #[[ATTR_NUM_SKIP]] = {{.*}} "zero-call-used-regs"="skip"
// CHECK-USED-GPR:           attributes #[[ATTR_NUM_USED_GPR_ARG]] = {{.*}} "zero-call-used-regs"="used-gpr-arg"
// CHECK-USED-GPR:           attributes #[[ATTR_NUM_USED_ARG]] = {{.*}} "zero-call-used-regs"="used-arg"
// CHECK-USED-GPR:           attributes #[[ATTR_NUM_USED]] = {{.*}} "zero-call-used-regs"="used"
// CHECK-USED-GPR:           attributes #[[ATTR_NUM_ALL_GPR_ARG]] = {{.*}} "zero-call-used-regs"="all-gpr-arg"
// CHECK-USED-GPR:           attributes #[[ATTR_NUM_ALL_GPR]] = {{.*}} "zero-call-used-regs"="all-gpr"
// CHECK-USED-GPR:           attributes #[[ATTR_NUM_ALL_ARG]] = {{.*}} "zero-call-used-regs"="all-arg"
// CHECK-USED-GPR:           attributes #[[ATTR_NUM_ALL]] = {{.*}} "zero-call-used-regs"="all"

// -fzero-call-used-regs=used-arg:
//
// CHECK-USED-ARG:           define {{.*}} @no_attribute({{.*}} #[[ATTR_NUM_USED_ARG:[0-9]*]]
// CHECK-USED-ARG:           define {{.*}} @skip_test({{.*}} #[[ATTR_NUM_SKIP:[0-9]*]]
// CHECK-USED-ARG:           define {{.*}} @used_gpr_arg_test({{.*}} #[[ATTR_NUM_USED_GPR_ARG:[0-9]*]]
// CHECK-USED-ARG:           define {{.*}} @used_gpr_test({{.*}} #[[ATTR_NUM_USED_GPR:[0-9]*]]
// CHECK-USED-ARG:           define {{.*}} @used_arg_test({{.*}} #[[ATTR_NUM_USED_ARG]]
// CHECK-USED-ARG:           define {{.*}} @used_test({{.*}} #[[ATTR_NUM_USED:[0-9]*]]
// CHECK-USED-ARG:           define {{.*}} @all_gpr_arg_test({{.*}} #[[ATTR_NUM_ALL_GPR_ARG:[0-9]*]]
// CHECK-USED-ARG:           define {{.*}} @all_gpr_test({{.*}} #[[ATTR_NUM_ALL_GPR:[0-9]*]]
// CHECK-USED-ARG:           define {{.*}} @all_arg_test({{.*}} #[[ATTR_NUM_ALL_ARG:[0-9]*]]
// CHECK-USED-ARG:           define {{.*}} @all_test({{.*}} #[[ATTR_NUM_ALL:[0-9]*]]
//
// CHECK-USED-ARG:           attributes #[[ATTR_NUM_USED_ARG]] = {{.*}} "zero-call-used-regs"="used-arg"
// CHECK-USED-ARG:           attributes #[[ATTR_NUM_SKIP]] = {{.*}} "zero-call-used-regs"="skip"
// CHECK-USED-ARG:           attributes #[[ATTR_NUM_USED_GPR_ARG]] = {{.*}} "zero-call-used-regs"="used-gpr-arg"
// CHECK-USED-ARG:           attributes #[[ATTR_NUM_USED_GPR]] = {{.*}} "zero-call-used-regs"="used-gpr"
// CHECK-USED-ARG:           attributes #[[ATTR_NUM_USED]] = {{.*}} "zero-call-used-regs"="used"
// CHECK-USED-ARG:           attributes #[[ATTR_NUM_ALL_GPR_ARG]] = {{.*}} "zero-call-used-regs"="all-gpr-arg"
// CHECK-USED-ARG:           attributes #[[ATTR_NUM_ALL_GPR]] = {{.*}} "zero-call-used-regs"="all-gpr"
// CHECK-USED-ARG:           attributes #[[ATTR_NUM_ALL_ARG]] = {{.*}} "zero-call-used-regs"="all-arg"
// CHECK-USED-ARG:           attributes #[[ATTR_NUM_ALL]] = {{.*}} "zero-call-used-regs"="all"

// -fzero-call-used-regs=used:
//
// CHECK-USED:               define {{.*}} @no_attribute({{.*}} #[[ATTR_NUM_USED:[0-9]*]]
// CHECK-USED:               define {{.*}} @skip_test({{.*}} #[[ATTR_NUM_SKIP:[0-9]*]]
// CHECK-USED:               define {{.*}} @used_gpr_arg_test({{.*}} #[[ATTR_NUM_USED_GPR_ARG:[0-9]*]]
// CHECK-USED:               define {{.*}} @used_gpr_test({{.*}} #[[ATTR_NUM_USED_GPR:[0-9]*]]
// CHECK-USED:               define {{.*}} @used_arg_test({{.*}} #[[ATTR_NUM_USED_ARG:[0-9]*]]
// CHECK-USED:               define {{.*}} @used_test({{.*}} #[[ATTR_NUM_USED]]
// CHECK-USED:               define {{.*}} @all_gpr_arg_test({{.*}} #[[ATTR_NUM_ALL_GPR_ARG:[0-9]*]]
// CHECK-USED:               define {{.*}} @all_gpr_test({{.*}} #[[ATTR_NUM_ALL_GPR:[0-9]*]]
// CHECK-USED:               define {{.*}} @all_arg_test({{.*}} #[[ATTR_NUM_ALL_ARG:[0-9]*]]
// CHECK-USED:               define {{.*}} @all_test({{.*}} #[[ATTR_NUM_ALL:[0-9]*]]
//
// CHECK-USED:               attributes #[[ATTR_NUM_USED]] = {{.*}} "zero-call-used-regs"="used"
// CHECK-USED:               attributes #[[ATTR_NUM_SKIP]] = {{.*}} "zero-call-used-regs"="skip"
// CHECK-USED:               attributes #[[ATTR_NUM_USED_GPR_ARG]] = {{.*}} "zero-call-used-regs"="used-gpr-arg"
// CHECK-USED:               attributes #[[ATTR_NUM_USED_GPR]] = {{.*}} "zero-call-used-regs"="used-gpr"
// CHECK-USED:               attributes #[[ATTR_NUM_USED_ARG]] = {{.*}} "zero-call-used-regs"="used-arg"
// CHECK-USED:               attributes #[[ATTR_NUM_ALL_GPR_ARG]] = {{.*}} "zero-call-used-regs"="all-gpr-arg"
// CHECK-USED:               attributes #[[ATTR_NUM_ALL_GPR]] = {{.*}} "zero-call-used-regs"="all-gpr"
// CHECK-USED:               attributes #[[ATTR_NUM_ALL_ARG]] = {{.*}} "zero-call-used-regs"="all-arg"
// CHECK-USED:               attributes #[[ATTR_NUM_ALL]] = {{.*}} "zero-call-used-regs"="all"

// -fzero-call-used-regs=all-gpr-arg:
//
// CHECK-ALL-GPR-ARG:        define {{.*}} @no_attribute({{.*}} #[[ATTR_NUM_ALL_GPR_ARG:[0-9]*]]
// CHECK-ALL-GPR-ARG:        define {{.*}} @skip_test({{.*}} #[[ATTR_NUM_SKIP:[0-9]*]]
// CHECK-ALL-GPR-ARG:        define {{.*}} @used_gpr_arg_test({{.*}} #[[ATTR_NUM_USED_GPR_ARG:[0-9]*]]
// CHECK-ALL-GPR-ARG:        define {{.*}} @used_gpr_test({{.*}} #[[ATTR_NUM_USED_GPR:[0-9]*]]
// CHECK-ALL-GPR-ARG:        define {{.*}} @used_arg_test({{.*}} #[[ATTR_NUM_USED_ARG:[0-9]*]]
// CHECK-ALL-GPR-ARG:        define {{.*}} @used_test({{.*}} #[[ATTR_NUM_USED:[0-9]*]]
// CHECK-ALL-GPR-ARG:        define {{.*}} @all_gpr_arg_test({{.*}} #[[ATTR_NUM_ALL_GPR_ARG]]
// CHECK-ALL-GPR-ARG:        define {{.*}} @all_gpr_test({{.*}} #[[ATTR_NUM_ALL_GPR:[0-9]*]]
// CHECK-ALL-GPR-ARG:        define {{.*}} @all_arg_test({{.*}} #[[ATTR_NUM_ALL_ARG:[0-9]*]]
// CHECK-ALL-GPR-ARG:        define {{.*}} @all_test({{.*}} #[[ATTR_NUM_ALL:[0-9]*]]
//
// CHECK-ALL-GPR-ARG:        attributes #[[ATTR_NUM_ALL_GPR_ARG]] = {{.*}} "zero-call-used-regs"="all-gpr-arg"
// CHECK-ALL-GPR-ARG:        attributes #[[ATTR_NUM_SKIP]] = {{.*}} "zero-call-used-regs"="skip"
// CHECK-ALL-GPR-ARG:        attributes #[[ATTR_NUM_USED_GPR_ARG]] = {{.*}} "zero-call-used-regs"="used-gpr-arg"
// CHECK-ALL-GPR-ARG:        attributes #[[ATTR_NUM_USED_GPR]] = {{.*}} "zero-call-used-regs"="used-gpr"
// CHECK-ALL-GPR-ARG:        attributes #[[ATTR_NUM_USED_ARG]] = {{.*}} "zero-call-used-regs"="used-arg"
// CHECK-ALL-GPR-ARG:        attributes #[[ATTR_NUM_USED]] = {{.*}} "zero-call-used-regs"="used"
// CHECK-ALL-GPR-ARG:        attributes #[[ATTR_NUM_ALL_GPR]] = {{.*}} "zero-call-used-regs"="all-gpr"
// CHECK-ALL-GPR-ARG:        attributes #[[ATTR_NUM_ALL_ARG]] = {{.*}} "zero-call-used-regs"="all-arg"
// CHECK-ALL-GPR-ARG:        attributes #[[ATTR_NUM_ALL]] = {{.*}} "zero-call-used-regs"="all"

// -fzero-call-used-regs=all-gpr:
//
// CHECK-ALL-GPR:            define {{.*}} @no_attribute({{.*}} #[[ATTR_NUM_ALL_GPR:[0-9]*]]
// CHECK-ALL-GPR:            define {{.*}} @skip_test({{.*}} #[[ATTR_NUM_SKIP:[0-9]*]]
// CHECK-ALL-GPR:            define {{.*}} @used_gpr_arg_test({{.*}} #[[ATTR_NUM_USED_GPR_ARG:[0-9]*]]
// CHECK-ALL-GPR:            define {{.*}} @used_gpr_test({{.*}} #[[ATTR_NUM_USED_GPR:[0-9]*]]
// CHECK-ALL-GPR:            define {{.*}} @used_arg_test({{.*}} #[[ATTR_NUM_USED_ARG:[0-9]*]]
// CHECK-ALL-GPR:            define {{.*}} @used_test({{.*}} #[[ATTR_NUM_USED:[0-9]*]]
// CHECK-ALL-GPR:            define {{.*}} @all_gpr_arg_test({{.*}} #[[ATTR_NUM_ALL_GPR_ARG:[0-9]*]]
// CHECK-ALL-GPR:            define {{.*}} @all_gpr_test({{.*}} #[[ATTR_NUM_ALL_GPR]]
// CHECK-ALL-GPR:            define {{.*}} @all_arg_test({{.*}} #[[ATTR_NUM_ALL_ARG:[0-9]*]]
// CHECK-ALL-GPR:            define {{.*}} @all_test({{.*}} #[[ATTR_NUM_ALL:[0-9]*]]
//
// CHECK-ALL-GPR:            attributes #[[ATTR_NUM_ALL_GPR]] = {{.*}} "zero-call-used-regs"="all-gpr"
// CHECK-ALL-GPR:            attributes #[[ATTR_NUM_SKIP]] = {{.*}} "zero-call-used-regs"="skip"
// CHECK-ALL-GPR:            attributes #[[ATTR_NUM_USED_GPR_ARG]] = {{.*}} "zero-call-used-regs"="used-gpr-arg"
// CHECK-ALL-GPR:            attributes #[[ATTR_NUM_USED_GPR]] = {{.*}} "zero-call-used-regs"="used-gpr"
// CHECK-ALL-GPR:            attributes #[[ATTR_NUM_USED_ARG]] = {{.*}} "zero-call-used-regs"="used-arg"
// CHECK-ALL-GPR:            attributes #[[ATTR_NUM_USED]] = {{.*}} "zero-call-used-regs"="used"
// CHECK-ALL-GPR:            attributes #[[ATTR_NUM_ALL_GPR_ARG]] = {{.*}} "zero-call-used-regs"="all-gpr-arg"
// CHECK-ALL-GPR:            attributes #[[ATTR_NUM_ALL_ARG]] = {{.*}} "zero-call-used-regs"="all-arg"
// CHECK-ALL-GPR:            attributes #[[ATTR_NUM_ALL]] = {{.*}} "zero-call-used-regs"="all"

// -fzero-call-used-regs=all-arg:
//
// CHECK-ALL-ARG:            define {{.*}} @no_attribute({{.*}} #[[ATTR_NUM_ALL_ARG:[0-9]*]]
// CHECK-ALL-ARG:            define {{.*}} @skip_test({{.*}} #[[ATTR_NUM_SKIP:[0-9]*]]
// CHECK-ALL-ARG:            define {{.*}} @used_gpr_arg_test({{.*}} #[[ATTR_NUM_USED_GPR_ARG:[0-9]*]]
// CHECK-ALL-ARG:            define {{.*}} @used_gpr_test({{.*}} #[[ATTR_NUM_USED_GPR:[0-9]*]]
// CHECK-ALL-ARG:            define {{.*}} @used_arg_test({{.*}} #[[ATTR_NUM_USED_ARG:[0-9]*]]
// CHECK-ALL-ARG:            define {{.*}} @used_test({{.*}} #[[ATTR_NUM_USED:[0-9]*]]
// CHECK-ALL-ARG:            define {{.*}} @all_gpr_arg_test({{.*}} #[[ATTR_NUM_ALL_GPR_ARG:[0-9]*]]
// CHECK-ALL-ARG:            define {{.*}} @all_gpr_test({{.*}} #[[ATTR_NUM_ALL_GPR:[0-9]*]]
// CHECK-ALL-ARG:            define {{.*}} @all_arg_test({{.*}} #[[ATTR_NUM_ALL_ARG]]
// CHECK-ALL-ARG:            define {{.*}} @all_test({{.*}} #[[ATTR_NUM_ALL:[0-9]*]]
//
// CHECK-ALL-ARG:            attributes #[[ATTR_NUM_ALL_ARG]] = {{.*}} "zero-call-used-regs"="all-arg"
// CHECK-ALL-ARG:            attributes #[[ATTR_NUM_SKIP]] = {{.*}} "zero-call-used-regs"="skip"
// CHECK-ALL-ARG:            attributes #[[ATTR_NUM_USED_GPR_ARG]] = {{.*}} "zero-call-used-regs"="used-gpr-arg"
// CHECK-ALL-ARG:            attributes #[[ATTR_NUM_USED_GPR]] = {{.*}} "zero-call-used-regs"="used-gpr"
// CHECK-ALL-ARG:            attributes #[[ATTR_NUM_USED_ARG]] = {{.*}} "zero-call-used-regs"="used-arg"
// CHECK-ALL-ARG:            attributes #[[ATTR_NUM_USED]] = {{.*}} "zero-call-used-regs"="used"
// CHECK-ALL-ARG:            attributes #[[ATTR_NUM_ALL_GPR_ARG]] = {{.*}} "zero-call-used-regs"="all-gpr-arg"
// CHECK-ALL-ARG:            attributes #[[ATTR_NUM_ALL_GPR]] = {{.*}} "zero-call-used-regs"="all-gpr"
// CHECK-ALL-ARG:            attributes #[[ATTR_NUM_ALL]] = {{.*}} "zero-call-used-regs"="all"

// -fzero-call-used-regs=all:
//
// CHECK-ALL:                define {{.*}} @no_attribute({{.*}} #[[ATTR_NUM_ALL:[0-9]*]]
// CHECK-ALL:                define {{.*}} @skip_test({{.*}} #[[ATTR_NUM_SKIP:[0-9]*]]
// CHECK-ALL:                define {{.*}} @used_gpr_arg_test({{.*}} #[[ATTR_NUM_USED_GPR_ARG:[0-9]*]]
// CHECK-ALL:                define {{.*}} @used_gpr_test({{.*}} #[[ATTR_NUM_USED_GPR:[0-9]*]]
// CHECK-ALL:                define {{.*}} @used_arg_test({{.*}} #[[ATTR_NUM_USED_ARG:[0-9]*]]
// CHECK-ALL:                define {{.*}} @used_test({{.*}} #[[ATTR_NUM_USED:[0-9]*]]
// CHECK-ALL:                define {{.*}} @all_gpr_arg_test({{.*}} #[[ATTR_NUM_ALL_GPR_ARG:[0-9]*]]
// CHECK-ALL:                define {{.*}} @all_gpr_test({{.*}} #[[ATTR_NUM_ALL_GPR:[0-9]*]]
// CHECK-ALL:                define {{.*}} @all_arg_test({{.*}} #[[ATTR_NUM_ALL_ARG:[0-9]*]]
// CHECK-ALL:                define {{.*}} @all_test({{.*}} #[[ATTR_NUM_ALL]]
//
// CHECK-ALL:                attributes #[[ATTR_NUM_ALL]] = {{.*}} "zero-call-used-regs"="all"
// CHECK-ALL:                attributes #[[ATTR_NUM_SKIP]] = {{.*}} "zero-call-used-regs"="skip"
// CHECK-ALL:                attributes #[[ATTR_NUM_USED_GPR_ARG]] = {{.*}} "zero-call-used-regs"="used-gpr-arg"
// CHECK-ALL:                attributes #[[ATTR_NUM_USED_GPR]] = {{.*}} "zero-call-used-regs"="used-gpr"
// CHECK-ALL:                attributes #[[ATTR_NUM_USED_ARG]] = {{.*}} "zero-call-used-regs"="used-arg"
// CHECK-ALL:                attributes #[[ATTR_NUM_USED]] = {{.*}} "zero-call-used-regs"="used"
// CHECK-ALL:                attributes #[[ATTR_NUM_ALL_GPR_ARG]] = {{.*}} "zero-call-used-regs"="all-gpr-arg"
// CHECK-ALL:                attributes #[[ATTR_NUM_ALL_GPR]] = {{.*}} "zero-call-used-regs"="all-gpr"
// CHECK-ALL:                attributes #[[ATTR_NUM_ALL_ARG]] = {{.*}} "zero-call-used-regs"="all-arg"

#define __zero_call_used_regs(kind) __attribute__((zero_call_used_regs(kind)))

void no_attribute(void) {
}

void __zero_call_used_regs("skip") skip_test(void) {
}

void __zero_call_used_regs("used-gpr-arg") used_gpr_arg_test(void) {
}

void __zero_call_used_regs("used-gpr") used_gpr_test(void) {
}

void __zero_call_used_regs("used-arg") used_arg_test(void) {
}

void __zero_call_used_regs("used") used_test(void) {
}

void __zero_call_used_regs("all-gpr-arg") all_gpr_arg_test(void) {
}

void __zero_call_used_regs("all-gpr") all_gpr_test(void) {
}

void __zero_call_used_regs("all-arg") all_arg_test(void) {
}

void __zero_call_used_regs("all") all_test(void) {
}
