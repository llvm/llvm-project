// RUN: %clang -fsyntax-only -iframework %S/Inputs %s -Xclang -verify
// RUN: %clang -fsyntax-only -isysroot %S -iframeworkwithsysroot /Inputs %s -Xclang -verify
// expected-no-diagnostics

#include <TestFramework/TestFramework.h>
