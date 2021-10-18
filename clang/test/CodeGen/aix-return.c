// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc-unknown-aix \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=AIX,AIX32
// RUN: %clang_cc1 -triple powerpc64-unknown-aix \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=AIX,AIX64

// AIX-LABEL: define void @retVoid()
void retVoid(void) {}

// AIX-LABEL: define signext i8 @retChar(i8 signext %x)
char retChar(char x) { return x; }

// AIX-LABEL: define signext i16 @retShort(i16 signext %x)
short retShort(short x) { return x; }

// AIX32-LABEL: define i32 @retInt(i32 %x)
// AIX64-LABEL: define signext i32 @retInt(i32 signext %x)
int retInt(int x) { return 1; }

// AIX-LABEL: define i64 @retLongLong(i64 %x)
long long retLongLong(long long x) { return x; }

// AIX-LABEL: define signext i8 @retEnumChar(i8 signext %x)
enum EnumChar : char { IsChar };
enum EnumChar retEnumChar(enum EnumChar x) {
  return x;
}

// AIX32-LABEL: define i32 @retEnumInt(i32 %x)
// AIX64-LABEL: define signext i32 @retEnumInt(i32 signext %x)
enum EnumInt : int { IsInt };
enum EnumInt retEnumInt(enum EnumInt x) {
  return x;
}
