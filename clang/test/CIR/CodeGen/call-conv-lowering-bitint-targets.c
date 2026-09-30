// RUN: %clang_cc1 -triple i386-unknown-linux-gnu \
// RUN:   -fexperimental-max-bitint-width=8388608 -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=I386
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu \
// RUN:   -fexperimental-max-bitint-width=8388608 -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=PPC64
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu \
// RUN:   -fexperimental-max-bitint-width=8388608 -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=AARCH64
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu \
// RUN:   -fexperimental-max-bitint-width=8388608 -fclangir -emit-cir %s -o %t.cir
// RUN: cir-translate -cir-to-llvmir --disable-cc-lowering %t.cir -o - \
// RUN:   | FileCheck %s --check-prefix=AARCH64

typedef signed _BitInt(129) i129;
typedef i129 (*fn_t)(i129);

extern i129 external(i129);

i129 direct(i129 value) { return external(value); }

// I386-LABEL: define dso_local void @direct(
// I386-SAME: ptr dead_on_unwind noalias writable sret([20 x i8]) align 4 %{{[^,]+}},
// I386-SAME: ptr nofreeobj noundef align 4 dead_on_return dereferenceable(20) %{{[^)]+}})
// I386: load i160, ptr %{{[^,]+}}, align 4
// I386: call void @external(
// I386-SAME: ptr dead_on_unwind writable sret([20 x i8]) align 4 %{{[^,]+}},
// I386-SAME: ptr nofreeobj noundef align 4 dead_on_return dereferenceable(20) %{{[^)]+}})
// I386: declare void @external(
// I386-SAME: ptr dead_on_unwind writable sret([20 x i8]) align 4,
// I386-SAME: ptr nofreeobj noundef align 4 dead_on_return dereferenceable(20))

// PPC64-LABEL: define dso_local void @direct(
// PPC64-SAME: ptr dead_on_unwind noalias writable sret([24 x i8]) align 8 %{{[^,]+}},
// PPC64-SAME: ptr noundef byval([24 x i8]) align 8 %{{[^)]+}})
// PPC64: load i192, ptr %{{[^,]+}}, align 8
// PPC64: call void @external(
// PPC64-SAME: ptr dead_on_unwind writable sret([24 x i8]) align 8 %{{[^,]+}},
// PPC64-SAME: ptr noundef byval([24 x i8]) align 8 %{{[^)]+}})
// PPC64: declare void @external(
// PPC64-SAME: ptr dead_on_unwind writable sret([24 x i8]) align 8,
// PPC64-SAME: ptr noundef byval([24 x i8]) align 8)

// AARCH64-LABEL: define dso_local void @direct(
// AARCH64-SAME: ptr dead_on_unwind noalias writable sret(i256) align 16 %{{[^,]+}},
// AARCH64-SAME: ptr nofreeobj noundef align 16 dead_on_return dereferenceable(32) %{{[^)]+}})
// AARCH64: load i256, ptr %{{[^,]+}}, align 16
// AARCH64: call void @external(
// AARCH64-SAME: ptr dead_on_unwind writable sret(i256) align 16 %{{[^,]+}},
// AARCH64-SAME: ptr nofreeobj noundef align 16 dead_on_return dereferenceable(32) %{{[^)]+}})
// AARCH64: declare void @external(
// AARCH64-SAME: ptr dead_on_unwind writable sret(i256) align 16,
// AARCH64-SAME: ptr nofreeobj noundef align 16 dead_on_return dereferenceable(32))

i129 indirect(fn_t fn, i129 value) { return fn(value); }

// I386-LABEL: define dso_local void @indirect(
// I386: call void %{{[^ (]+}}(
// I386-SAME: ptr dead_on_unwind writable sret([20 x i8]) align 4 %{{[^,]+}},
// I386-SAME: ptr nofreeobj noundef align 4 dead_on_return dereferenceable(20) %{{[^)]+}})

// PPC64-LABEL: define dso_local void @indirect(
// PPC64: call void %{{[^ (]+}}(
// PPC64-SAME: ptr dead_on_unwind writable sret([24 x i8]) align 8 %{{[^,]+}},
// PPC64-SAME: ptr noundef byval([24 x i8]) align 8 %{{[^)]+}})

// AARCH64-LABEL: define dso_local void @indirect(
// AARCH64: call void %{{[^ (]+}}(
// AARCH64-SAME: ptr dead_on_unwind writable sret(i256) align 16 %{{[^,]+}},
// AARCH64-SAME: ptr nofreeobj noundef align 16 dead_on_return dereferenceable(32) %{{[^)]+}})

signed _BitInt(31) extend31(signed _BitInt(31) value) { return value; }
signed _BitInt(33) extend33(signed _BitInt(33) value) { return value; }
_Bool extend_bool(_Bool value) { return value; }

// I386-LABEL: define dso_local signext i31 @extend31(i31 noundef signext
// I386-LABEL: define dso_local i33 @extend33(i33 noundef
// I386-LABEL: define dso_local zeroext i1 @extend_bool(i1 noundef zeroext
// PPC64-LABEL: define dso_local signext i31 @extend31(i31 noundef signext
// PPC64-LABEL: define dso_local signext i33 @extend33(i33 noundef signext
// PPC64-LABEL: define dso_local zeroext i1 @extend_bool(i1 noundef zeroext
// AARCH64-LABEL: define dso_local i31 @extend31(i31 noundef
// AARCH64-LABEL: define dso_local i33 @extend33(i33 noundef
// AARCH64-LABEL: define dso_local i1 @extend_bool(i1 noundef

signed _BitInt(64) boundary64(signed _BitInt(64) value) { return value; }
signed _BitInt(65) boundary65(signed _BitInt(65) value) { return value; }
signed _BitInt(128) boundary128(signed _BitInt(128) value) { return value; }

// I386-LABEL: define dso_local i64 @boundary64(i64 noundef
// I386-LABEL: define dso_local void @boundary65(
// I386-SAME: ptr dead_on_unwind noalias writable sret([12 x i8]) align 4 %{{[^,]+}},
// I386-SAME: ptr nofreeobj noundef align 4 dead_on_return dereferenceable(12) %{{[^)]+}})
// I386-LABEL: define dso_local void @boundary128(
// I386-SAME: ptr dead_on_unwind noalias writable sret(i128) align 4 %{{[^,]+}},
// I386-SAME: ptr nofreeobj noundef align 4 dead_on_return dereferenceable(16) %{{[^)]+}})
// PPC64-LABEL: define dso_local i64 @boundary64(i64 noundef
// PPC64-LABEL: define dso_local i65 @boundary65(i65 noundef
// PPC64-LABEL: define dso_local i128 @boundary128(i128 noundef
// AARCH64-LABEL: define dso_local i64 @boundary64(i64 noundef
// AARCH64-LABEL: define dso_local i65 @boundary65(i65 noundef
// AARCH64-LABEL: define dso_local i128 @boundary128(i128 noundef
