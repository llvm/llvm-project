// RUN: clang-format -dump-config %s | FileCheck %s

// CHECK: Language: ObjC
NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXTERN int kConstant;

NS_ASSUME_NONNULL_END
