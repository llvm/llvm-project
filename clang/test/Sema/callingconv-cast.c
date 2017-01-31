// RUN: %clang_cc1 -fms-extensions -triple i686-pc-windows-msvc -Wcast-calling-convention -DMSVC -Wno-pointer-bool-conversion -verify -x c %s
// RUN: %clang_cc1 -fms-extensions -triple i686-pc-windows-msvc -Wcast-calling-convention -DMSVC -Wno-pointer-bool-conversion -verify -x c++ %s
// RUN: %clang_cc1 -fms-extensions -triple i686-pc-windows-msvc -Wcast-calling-convention -DMSVC -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s --check-prefix=MSFIXIT
// RUN: %clang_cc1 -triple i686-pc-windows-gnu -Wcast-calling-convention -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s --check-prefix=GNUFIXIT

// expected-note@+1 {{consider defining 'mismatched_before_winapi' with the 'stdcall' calling convention}}
void mismatched_before_winapi(int x) {}

#ifdef MSVC
#define WINAPI __stdcall
#else
#define WINAPI __attribute__((stdcall))
#endif

// expected-note@+1 3 {{consider defining 'mismatched' with the 'stdcall' calling convention}}
void mismatched(int x) {}

// expected-note@+1 {{consider defining 'mismatched_declaration' with the 'stdcall' calling convention}}
void mismatched_declaration(int x);

// expected-note@+1 {{consider defining 'suggest_fix_first_redecl' with the 'stdcall' calling convention}}
void suggest_fix_first_redecl(int x);
void suggest_fix_first_redecl(int x);

typedef void (WINAPI *callback_t)(int);
void take_callback(callback_t callback);

void WINAPI mismatched_stdcall(int x) {}

void take_opaque_fn(void (*callback)(int));

int main() {
  // expected-warning@+1 {{cast between incompatible calling conventions 'cdecl' and 'stdcall'}}
  take_callback((callback_t)mismatched);

  // expected-warning@+1 {{cast between incompatible calling conventions 'cdecl' and 'stdcall'}}
  callback_t callback = (callback_t)mismatched; // warns
  (void)callback;

  // expected-warning@+1 {{cast between incompatible calling conventions 'cdecl' and 'stdcall'}}
  callback = (callback_t)&mismatched; // warns

  // No warning, just to show we don't drill through other kinds of unary operators.
  callback = (callback_t)!mismatched;

  // expected-warning@+1 {{cast between incompatible calling conventions 'cdecl' and 'stdcall'}}
  callback = (callback_t)&mismatched_before_winapi; // warns

  // Probably a bug, but we don't warn.
  void (*callback2)(int) = mismatched;
  take_callback((callback_t)callback2);

  // Another way to suppress the warning.
  take_callback((callback_t)(void*)mismatched);

  // Warn on declarations as well as definitions.
  // expected-warning@+1 {{cast between incompatible calling conventions 'cdecl' and 'stdcall'}}
  take_callback((callback_t)mismatched_declaration);
  // expected-warning@+1 {{cast between incompatible calling conventions 'cdecl' and 'stdcall'}}
  take_callback((callback_t)suggest_fix_first_redecl);

  // Don't warn, because we're casting from stdcall to cdecl. Usually that means
  // the programmer is rinsing the function pointer through some kind of opaque
  // API.
  take_opaque_fn((void (*)(int))mismatched_stdcall);
}

// MSFIXIT: fix-it:"{{.*}}callingconv-cast.c":{16:6-16:6}:"WINAPI "
// MSFIXIT: fix-it:"{{.*}}callingconv-cast.c":{16:6-16:6}:"WINAPI "
// MSFIXIT: fix-it:"{{.*}}callingconv-cast.c":{16:6-16:6}:"WINAPI "
// MSFIXIT: fix-it:"{{.*}}callingconv-cast.c":{7:6-7:6}:"__stdcall "

// GNUFIXIT: fix-it:"{{.*}}callingconv-cast.c":{16:6-16:6}:"WINAPI "
// GNUFIXIT: fix-it:"{{.*}}callingconv-cast.c":{16:6-16:6}:"WINAPI "
// GNUFIXIT: fix-it:"{{.*}}callingconv-cast.c":{16:6-16:6}:"WINAPI "
// GNUFIXIT: fix-it:"{{.*}}callingconv-cast.c":{7:6-7:6}:"__attribute__((stdcall)) "
