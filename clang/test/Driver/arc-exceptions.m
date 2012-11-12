// RUN: %clang -### -x objective-c -arch x86_64 -fobjc-arc -fsyntax-only %s 2> %t.log
// RUN: grep objective-c %t.log
// RUN: not grep "fobjc-arc-exceptions" %t.log
// RUN: %clang -### -x objective-c++ -arch x86_64 -fobjc-arc -fsyntax-only %s 2> %t.log
// RUN: grep "fobjc-arc-exceptions" %t.log
