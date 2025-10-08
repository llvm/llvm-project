// RUN: %clang_cc1 -emit-llvm -triple powerpc64-ibm-aix-xcoff \
// RUN:   %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -triple powerpc-ibm-aix-xcoff \
// RUN:   %s -o - | FileCheck %s
// This test file checks that we represent common C inline assembly
// PowerPC load address mode operations accureately in the platform
// agnostic LLVM IR.

char loadAddressAConstrained(char* ptr) {
// CHECK-LABEL: define{{.*}} i8 @loadAddressAConstrained(ptr noundef %ptr)
// CHECK:  %1 = call ptr asm "addi $0,$1, 0", "=r,a"(ptr %0)
  char* result;
  asm ("addi %0,%1, 0" : "=r"(result) : "a"(ptr) :);
  return *result;
}

char loadAddressZyConstrained(char* ptr) {
// CHECK-LABEL: define{{.*}} i8 @loadAddressZyConstrained(ptr noundef %ptr)
// CHECK:  %1 = call ptr asm "add $0,${1:y}", "=r,*Z"(ptr elementtype(i8) %0)
  char* result;
  asm ("add %0,%y1" : "=r"(result) : "Z"(*ptr) :);
  return *result;
}

char xFormRegImmLoadAConstrained(char* ptr) {
// CHECK-LABEL: define{{.*}} i8 @xFormRegImmLoadAConstrained(ptr noundef %ptr)
// CHECK:  %1 = call ptr asm "addi $0,$1,$2", "=r,a,I"(ptr %0, i32 10000)
  char* result;
  asm ("addi %0,%1,%2" : "=r"(result) : "a"(ptr), "I"(10000) :);
  return *result;
}

char loadIndirectAddressZConstrained(char* ptr) {
// CHECK-LABEL: define{{.*}} i8 @loadIndirectAddressZConstrained(ptr noundef %ptr)
// CHECK:  %1 = call ptr asm "ld $0,$1", "=r,*Z"(ptr elementtype(i8) %arrayidx)
  char* result;
  asm ("ld %0,%1" : "=r"(result) : "Z"(ptr[100]) :);
  return *result;
}

char loadIndirectAddressAConstrained(char** ptr, unsigned index) {
// CHECK-LABEL: define{{.*}} i8 @loadIndirectAddressAConstrained(ptr noundef %ptr, i32 noundef{{[ zeroext]*}} %index)
// CHECK:  %2 = call ptr asm "ldx $0,$1,$2", "=r,a,r"(ptr %0, i32 %1)
  char* result;
  asm ("ldx %0,%1,%2" : "=r"(result) : "a"(ptr), "r"(index) :);
  return *result;
}

char dFormLoadZConstrained(char* ptr) {
// CHECK-LABEL: define{{.*}} i8 @dFormLoadZConstrained(ptr noundef %ptr)
// CHECK:  %1 = call i8 asm "lbz $0,$1", "=r,*Z"(ptr elementtype(i8) %arrayidx)
  char result;
  asm ("lbz %0,%1" : "=r"(result) : "Z"(ptr[8]) :);
  return result;
}

char xFormRegRegLoadZyConstrained(char* ptr, unsigned index) {
// CHECK-LABEL: define{{.*}} i8 @xFormRegRegLoadZyConstrained(ptr noundef %ptr, i32 noundef{{[ zeroext]*}} %index)
// CHECK:  %2 = call i8 asm "lbzx $0, ${1:y}", "=r,*Z"(ptr elementtype(i8) %arrayidx)
  char result;
  asm("lbzx %0, %y1" : "=r"(result) : "Z"(ptr[index]) :);
  return result;
}

char xFormRegRegLoadAConstrained(char* ptr, unsigned index) {
// CHECK-LABEL: define{{.*}} i8 @xFormRegRegLoadAConstrained(ptr noundef %ptr, i32 noundef{{[ zeroext]*}} %index)
// CHECK:  %2 = call i8 asm "lbzx $0,$1,$2", "=r,a,r"(ptr %0, i32 %1)
  char result;
  asm ("lbzx %0,%1,%2" : "=r"(result) : "a"(ptr), "r"(index) :);
  return result;
}
