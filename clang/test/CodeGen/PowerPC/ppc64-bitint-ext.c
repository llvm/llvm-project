// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,UNDEF
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,UNDEF
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -fclang-abi-compat=23 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,COMPAT23
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -fclang-abi-compat=23 \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,COMPAT23

// The bits of a _BitInt above its width are unspecified when it is passed or
// returned in a register, as in GCC. Clang 23 and earlier extended a _BitInt
// narrower than 64 bits.

// UNDEF:    define{{.*}} void @p15(i15 noundef %{{.+}}, i15 noundef %{{.+}})
// COMPAT23: define{{.*}} void @p15(i15 noundef signext %{{.+}}, i15 noundef zeroext %{{.+}})
void p15(_BitInt(15) a, unsigned _BitInt(15) b) {}

// UNDEF:    define{{.*}} void @p33(i33 noundef %{{.+}}, i33 noundef %{{.+}})
// COMPAT23: define{{.*}} void @p33(i33 noundef signext %{{.+}}, i33 noundef zeroext %{{.+}})
void p33(_BitInt(33) a, unsigned _BitInt(33) b) {}

// UNDEF:    define{{.*}} void @p63(i63 noundef %{{.+}}, i63 noundef %{{.+}})
// COMPAT23: define{{.*}} void @p63(i63 noundef signext %{{.+}}, i63 noundef zeroext %{{.+}})
void p63(_BitInt(63) a, unsigned _BitInt(63) b) {}

// CHECK: define{{.*}} void @p64(i64 noundef %{{.+}}, i64 noundef %{{.+}})
void p64(_BitInt(64) a, unsigned _BitInt(64) b) {}

// UNDEF:    define{{.*}} i15 @r15s(
// COMPAT23: define{{.*}} signext i15 @r15s(
_BitInt(15) r15s(void) { return 0; }

// UNDEF:    define{{.*}} i33 @r33u(
// COMPAT23: define{{.*}} zeroext i33 @r33u(
unsigned _BitInt(33) r33u(void) { return 0; }

// UNDEF:    define{{.*}} i63 @r63s(
// COMPAT23: define{{.*}} signext i63 @r63s(
_BitInt(63) r63s(void) { return 0; }

// Ordinary integer types are still extended.
// CHECK: define{{.*}} void @pint(i32 noundef signext %{{.+}}, i32 noundef zeroext %{{.+}}, i16 noundef signext %{{.+}})
void pint(int a, unsigned b, short c) {}
