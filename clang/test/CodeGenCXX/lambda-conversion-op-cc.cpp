// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-linux-gnu | FileCheck %s --check-prefixes=CHECK,LIN64
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-linux-gnu -DCC="__attribute__((vectorcall))" | FileCheck %s --check-prefixes=CHECK,VECCALL
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-windows-pc -DWIN32 | FileCheck %s --check-prefixes=WIN32

#ifndef CC
#define CC
#endif

void usage() {
  auto lambda = [](int i, float f, double d) CC { return i + f + d; };

  double (*CC fp)(int, float, double) = lambda;
  fp(0, 1.1, 2.2);
#ifdef WIN32
  double (*__attribute__((thiscall)) fp2)(int, float, double) = lambda;
  fp2(0, 1.1, 2.2);
#endif // WIN32
}

// void usage function, calls convrsion operator.
// LIN64: define void @_Z5usagev()
// VECCALL: define void @_Z5usagev()
// WIN32: define dso_local void @"?usage@@YAXXZ"()
// CHECK: call double (i32, float, double)* @"_ZZ5usagevENK3$_0cvPFdifdEEv"
// WIN32: call x86_thiscallcc double (i32, float, double)* @"??B<lambda_0>@?0??usage@@YAXXZ@QBEP6A?A?<auto>@@HMN@ZXZ"
// WIN32: call x86_thiscallcc double (i32, float, double)* @"??B<lambda_0>@?0??usage@@YAXXZ@QBEP6E?A?<auto>@@HMN@ZXZ"
//
// Conversion operator, returns __invoke.
// CHECK: define internal double (i32, float, double)* @"_ZZ5usagevENK3$_0cvPFdifdEEv"
// CHECK: ret double (i32, float, double)* @"_ZZ5usagevEN3$_08__invokeEifd"
// WIN32: define internal x86_thiscallcc double (i32, float, double)* @"??B<lambda_0>@?0??usage@@YAXXZ@QBEP6A?A?<auto>@@HMN@ZXZ"
// WIN32: ret double (i32, float, double)* @"?__invoke@<lambda_0>@?0??usage@@YAXXZ@CA?A?<auto>@@HMN@Z"
// WIN32: define internal x86_thiscallcc double (i32, float, double)* @"??B<lambda_0>@?0??usage@@YAXXZ@QBEP6E?A?<auto>@@HMN@ZXZ"
// WIN32: ret double (i32, float, double)* @"?__invoke@<lambda_0>@?0??usage@@YAXXZ@CE?A?<auto>@@HMN@Z"
//
// __invoke function, calls operator(). Win32 should call both.
// LIN64: define internal double @"_ZZ5usagevEN3$_08__invokeEifd"
// LIN64: call double @"_ZZ5usagevENK3$_0clEifd"
// VECCALL: define internal x86_vectorcallcc double @"_ZZ5usagevEN3$_08__invokeEifd"
// VECCALL: call x86_vectorcallcc double @"_ZZ5usagevENK3$_0clEifd"
// WIN32: define internal double @"?__invoke@<lambda_0>@?0??usage@@YAXXZ@CA?A?<auto>@@HMN@Z"
// WIN32: call x86_thiscallcc double @"??R<lambda_0>@?0??usage@@YAXXZ@QBE?A?<auto>@@HMN@Z"
// WIN32: define internal x86_thiscallcc double @"?__invoke@<lambda_0>@?0??usage@@YAXXZ@CE?A?<auto>@@HMN@Z"
// WIN32: call x86_thiscallcc double @"??R<lambda_0>@?0??usage@@YAXXZ@QBE?A?<auto>@@HMN@Z"
