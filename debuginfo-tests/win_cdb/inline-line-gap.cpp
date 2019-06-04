// RUN: %clang_cl -MD -Od %s -o %t.exe -fuse-ld=lld -Z7
// RUN: grep DE[B]UGGER: %s | sed -e 's/.*DE[B]UGGER: //' > %t.script
// RUN: %cdb -cf %t.script %t.exe | FileCheck %s --check-prefixes=DEBUGGER,CHECK
//
// RUN: %clang_cl -MD -O2 %s -o %t.exe -fuse-ld=lld -Z7
// RUN: grep DE[B]UGGER: %s | sed -e 's/.*DE[B]UGGER: //' > %t.script
// RUN: %cdb -cf %t.script %t.exe | FileCheck %s --check-prefixes=DEBUGGER,CHECK

// This code is structured to have an early exit with an epilogue in the middle
// of the function, which creates a gap between the beginning of the inlined
// code region and the end. Previously, this confused cdb.

volatile bool shutting_down_ = true;
volatile bool tearing_down_ = true;

void __attribute__((optnone)) setCrashString(const char *) {}
void __attribute__((optnone)) doTailCall() {}
extern "C" void __declspec(noreturn) abort();

void __forceinline inlineCrashFrame() {
  if (shutting_down_ || tearing_down_) {
    setCrashString("crashing");
    __debugbreak();
    // MSVC lays out calls to abort out of line, gets the layout we want.
    abort();
  }
}

void __declspec(noinline) callerOfInlineCrashFrame(bool is_keeping_alive) {
  if (is_keeping_alive)
    inlineCrashFrame();
  else
    doTailCall();
}

int __attribute__((optnone)) main() {
  callerOfInlineCrashFrame(true);
}

// DEBUGGER: g
// DEBUGGER: k3
// CHECK: {{.*}}!inlineCrashFrame
// CHECK: {{.*}}!callerOfInlineCrashFrame
// CHECK: {{.*}}!main
// DEBUGGER: q
