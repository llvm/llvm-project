// REQUIRES: arm-registered-target

// RUN: %clang --target=arm-none-eabi -mcpu=cortex-m33 -mfloat-abi=hard -O1 %s -S -o - -emit-llvm -DCALL_LIB -DDEFINE_LIB | FileCheck %s

// RUN: %clang --target=arm-none-eabi -mcpu=cortex-m33 -mfloat-abi=hard -O1 %s -flto=full -c -o %t.call_full.bc -DCALL_LIB
// RUN: %clang --target=arm-none-eabi -mcpu=cortex-m33 -mfloat-abi=hard -O1 %s -flto=full -c -o %t.define_full.bc -DDEFINE_LIB
// RUN: llvm-lto2 run -o %t.lto_full -save-temps %t.call_full.bc %t.define_full.bc \
// RUN:  -r %t.call_full.bc,fn,px \
// RUN:  -r %t.call_full.bc,fwrite,l \
// RUN:  -r %t.call_full.bc,putchar,l \
// RUN:  -r %t.call_full.bc,stdout,px \
// RUN:  -r %t.define_full.bc,fwrite,px \
// RUN:  -r %t.define_full.bc,putchar,px \
// RUN:  -r %t.define_full.bc,otherfn,px
// RUN: llvm-dis %t.lto_full.0.4.opt.bc -o - | FileCheck %s

// RUN: %clang --target=arm-none-eabi -mcpu=cortex-m33 -mfloat-abi=hard -O1 %s -flto=thin -c -o %t.call_thin.bc -DCALL_LIB
// RUN: %clang --target=arm-none-eabi -mcpu=cortex-m33 -mfloat-abi=hard -O1 %s -flto=thin -c -o %t.define_thin.bc -DDEFINE_LIB
// RUN: llvm-lto2 run -o %t.lto_thin -save-temps %t.call_thin.bc %t.define_thin.bc \
// RUN:  -r %t.call_thin.bc,fn,px \
// RUN:  -r %t.call_thin.bc,fwrite,l \
// RUN:  -r %t.call_thin.bc,putchar,l \
// RUN:  -r %t.call_thin.bc,stdout,px \
// RUN:  -r %t.define_thin.bc,fwrite,px \
// RUN:  -r %t.define_thin.bc,putchar,px \
// RUN:  -r %t.define_thin.bc,otherfn,px
// RUN: llvm-dis %t.lto_thin.1.4.opt.bc -o - | FileCheck %s

// We expect that the fprintf is optimised to fwrite, and the printf is
// optimised to putchar. Check that we don't have a mismatch in calling
// conventions causing the call to be replaced by a trap.
// CHECK-LABEL: define{{.*}}void @fn()
// CHECK-NOT: call void @llvm.trap()

typedef struct FILE FILE;
typedef unsigned int size_t;
extern FILE *stdout;
extern int fprintf(FILE *, const char *, ...);
extern int printf(const char *, ...);
extern void otherfn(const void *);

#ifdef CALL_LIB

void fn() {
  fprintf(stdout, "hello world");
  printf("a");
}

#endif

#ifdef DEFINE_LIB

size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream) {
  otherfn(ptr);
  return 0;
}

int putchar(int c) {
  otherfn(&c);
  return 0;
}

#endif
