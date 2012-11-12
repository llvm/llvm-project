// RUN: %clang -target x86_64-unknown-unknown -c -x assembler %s -### 2> %t.log
// RUN: grep '.*gcc.*"-m64"' %t.log
// RUN: %clang -target x86_64-unknown-unknown -c -x assembler %s -### -m32 2> %t.log
// RUN: grep '.*gcc.*"-m32"' %t.log
// RUN: %clang -target i386-unknown-unknown -c -x assembler %s -### 2> %t.log
// RUN: grep '.*gcc.*"-m32"' %t.log
// RUN: %clang -target i386-unknown-unknown -c -x assembler %s -### -m64 2> %t.log
// RUN: grep '.*gcc.*"-m64"' %t.log
