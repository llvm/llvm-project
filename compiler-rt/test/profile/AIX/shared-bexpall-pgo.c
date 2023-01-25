// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_pgogen foo.c -c -o foo.o
// RUN: %clang_pgogen -shared foo.o -o libfoo.so -bexpall
// RUN: %clang_pgogen -L%t user.c libfoo.so -o user1
// RUN: ./user1

//--- foo.c
void foo() {}

//--- user.c
void foo();
int main() { foo(); }
