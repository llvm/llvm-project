// RUN: clang-format -verbose 2> %t.stderr
// RUN: not grep "Formatting" %t.stderr
// RUN: clang-format %s 2> %t.stderr
// RUN: not grep "Formatting" %t.stderr
// RUN: clang-format %s -verbose 2> %t.stderr
// RUN: grep -E "Formatting (.*)verbose.cpp(.*)" %t.stderr
// RUN: clang-format %s -verbose=false 2> %t.stderr
// RUN: not grep "Formatting" %t.stderr

int a;
