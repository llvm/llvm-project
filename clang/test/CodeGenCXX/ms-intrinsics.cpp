// RUN: %clang_cc1 -fms-compatibility -fsyntax-only %s -verify
// expected-no-diagnostics

struct S {
  mutable long _Spinlock = 0;
  void _Unlock() {
    __iso_volatile_store32(&_Spinlock, 0);
  }
  int _Reset() {
    long v = __iso_volatile_load32(&_Spinlock);
    __iso_volatile_store32(&_Spinlock, 0);
    return v;
  }
};

S s;

