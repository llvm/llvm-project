// RUN: %clang -target i386-apple-darwin9 -m32 -Xarch_i386 -O2 %s -S -### 2> %t.log
// RUN: grep ' "-O2" ' %t.log | count 1
// RUN: %clang -target i386-apple-darwin9 -m64 -Xarch_i386 -O2 %s -S -### 2> %t.log
// RUN: grep ' "-O2" ' %t.log | count 0
// RUN: grep "argument unused during compilation: '-Xarch_i386 -O2'" %t.log
// RUN: not %clang -target i386-apple-darwin9 -m32 -Xarch_i386 -o -Xarch_i386 -S %s -S -Xarch_i386 -o 2> %t.log
// RUN: grep "error: invalid Xarch argument: '-Xarch_i386 -o'" %t.log | count 2
// RUN: grep "error: invalid Xarch argument: '-Xarch_i386 -S'" %t.log

