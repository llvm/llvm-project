// RUN: %clang -target i386-unknown-unknown -### -S %s -msse -msse4 -mno-sse -mno-mmx -msse 2> %t
// RUN: grep '"pentium4" "-target-feature" "+sse4" "-target-feature" "-mmx" "-target-feature" "+sse"' %t
// Note that we filter out all but the last -m(no)sse.
