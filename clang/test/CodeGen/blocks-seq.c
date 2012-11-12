// FIXME: We forcibly strip the names so that the test doesn't vary between
// builds with and without asserts. We need a better solution for this.

// RUN: %clang_cc1 -fblocks -triple x86_64-apple-darwin10 -emit-llvm-bc -o - %s | opt -strip | llvm-dis > %t
// RUN: grep '%6 = call i32 (...)\* @rhs()' %t | count 1
// RUN: grep '%7 = getelementptr inbounds %0\* %1, i32 0, i32 1' %t | count 1
// RUN: grep '%8 = load %0\*\* %7' %t | count 1
// RUN: grep '%10 = call i32 (...)\* @rhs()' %t | count 1
// RUN: grep '%11 = getelementptr inbounds %0\* %1, i32 0, i32 1' %t | count 1
// RUN: grep '%12 = load %0\*\* %11' %t | count 1

int rhs();

void foo() {
  __block int i;
  i = rhs();
  i += rhs();
}
