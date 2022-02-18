// RUN: %clang_cc1 -emit-llvm %s -o %t

char *globalString = "abc";
char *globalStringArray[5] = { "123", "abc" };
char *anotherGlobalString = "123";

int main(void) {
    printf("123");
}
