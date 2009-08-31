// FIXME: We forcibly strip the names so that the test doesn't vary between
// builds with and without asserts. We need a better solution for this.

// RUN: clang-cc -fblocks -triple x86_64-apple-darwin10 -emit-llvm-bc -o - %s | opt -strip | llvm-dis > %t &&
// RUN: grep '%7 = call i32 (...)\* @rhs()' %t | count 1 &&
// RUN: grep '%8 = getelementptr inbounds %0\* %1, i32 0, i32 1' %t | count 1 &&
// RUN: grep '%9 = bitcast i8\*\* %8 to %0\*\*' %t | count 1 &&
// RUN: grep '%10 = load %0\*\* %9' %t | count 1 &&
// RUN: grep '%12 = call i32 (...)\* @rhs()' %t | count 1 &&
// RUN: grep '%13 = getelementptr inbounds %0\* %1, i32 0, i32 1' %t | count 1 &&
// RUN: grep '%14 = bitcast i8\*\* %13 to %0\*\*' %t | count 1 &&
// RUN: grep '%15 = load %0\*\* %14' %t | count 1

int rhs();

void foo() {
  __block int i;
  i = rhs();
  i += rhs();
}
