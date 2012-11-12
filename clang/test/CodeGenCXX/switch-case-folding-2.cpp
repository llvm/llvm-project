// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// CHECK that we don't crash.

extern int printf(const char*, ...);
int test(int val){
 switch (val) {
 case 4:
   do {
     switch (6) {
       case 6: do { case 5: printf("bad\n"); } while (0);
     };
   } while (0);
 }
 return 0;
}

int main(void) {
 return test(5);
}

// CHECK: call i32 (i8*, ...)* @_Z6printfPKcz
