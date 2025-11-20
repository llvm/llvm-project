// RUN: %clang_cc1 -triple x86_64-apple-darwin -fsyntax-only -verify -Wformat-pedantic %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -fsyntax-only -fdiagnostics-parseable-fixits -Wformat-pedantic %s 2>&1 | FileCheck %s

extern "C" int printf(const char * restrict, ...);

#if __LP64__
typedef long CFIndex;
typedef long NSInteger;
typedef unsigned long NSUInteger;
#else
typedef int CFIndex;
typedef int NSInteger;
typedef unsigned int NSUInteger;
#endif

enum class CFIndexEnum : CFIndex { One };
enum class NSIntegerEnum : NSInteger { Two };
enum class NSUIntegerEnum : NSUInteger { Three };

void f() {
  printf("%d", CFIndexEnum::One); // expected-warning{{format specifies type 'int' but the argument has type 'CFIndexEnum'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%ld"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:16-[[@LINE-2]]:16}:"static_cast<long>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:32-[[@LINE-3]]:32}:")"

  printf("%d", NSIntegerEnum::Two); // expected-warning{{format specifies type 'int' but the argument has type 'NSIntegerEnum'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%ld"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:16-[[@LINE-2]]:16}:"static_cast<long>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:34-[[@LINE-3]]:34}:")"

  printf("%d", NSUIntegerEnum::Three); // expected-warning{{format specifies type 'int' but the argument has type 'NSUIntegerEnum'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:11-[[@LINE-1]]:13}:"%lu"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:16-[[@LINE-2]]:16}:"static_cast<unsigned long>("
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:37-[[@LINE-3]]:37}:")"
}
