// RUN: rm -rf %t && mkdir -p %t
// RUN: clang-doc --output=%t --format=json --executor=standalone %s
// RUN: FileCheck %s < %t/json/index.json

static void myFunction() {}

void noExceptFunction() noexcept {}

inline void inlineFunction() {}

extern void externFunction() {}

constexpr void constexprFunction() {}

// CHECK:          "Functions": [
// CHECK-NEXT:       {
// CHECK:              "IsStatic": true,
// COM:                FIXME: Emit ExceptionSpecificationType
// CHECK-NOT:          "ExceptionSpecifcation" : "noexcept",
// COM:                FIXME: Emit inline
// CHECK-NOT:          "IsInline": true,
// COM:                FIXME: Emit extern
// CHECK-NOT:          "IsExtern": true,
// COM:                FIXME: Emit constexpr
// CHECK-NOT:          "IsConstexpr": true,
