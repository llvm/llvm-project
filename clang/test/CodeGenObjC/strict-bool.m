// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -load-bool-from-mem=strict -O1 -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-UNDEF
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -load-bool-from-mem=strict -O0 -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-NONZERO
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -load-bool-from-mem=nonstrict -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-NONZERO
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -load-bool-from-mem=truncate -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-TRUNCATE
// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -load-bool-from-mem=nonzero -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-NONZERO

@interface BoolProp
@property (nonatomic) _Bool b;
@end

@implementation BoolProp
@synthesize b;
@end

// CHECK-LABEL: -[BoolProp b]
// CHECK-UNDEF: [[BOOL:%.+]] = trunc nuw i8 %{{.+}} to i1
// CHECK-TRUNCATE: [[BOOL:%.+]] = trunc i8 %{{.+}} to i1
// CHECK-NONZERO: [[BOOL:%.+]] = icmp ne i8 %{{.+}}, 0
// CHECK: ret i1 [[BOOL]]
