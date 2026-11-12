// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

int printf(const char *, ...);
int foo(void);

int main(void) {
  while (foo()) {
     switch (foo()) {
     case 0:
     case 1:
     case 2:
     case 3:
       printf("3");
     case 4: printf("4");
     case 5:
     case 6:
     default:
       break;
     }
   }
   return 0;
}
