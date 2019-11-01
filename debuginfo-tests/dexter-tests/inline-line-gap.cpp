// REQUIRES: system-windows
//
// RUN: %dexter --fail-lt 1.0 -w --builder 'clang-cl_vs2015' \
// RUN:      --debugger 'dbgeng' --cflags '/Od /Z7 /Zi' \
// RUN:      --ldflags '/Od /Z7 /Zi' -- %s
//
// RUN: %dexter --fail-lt 1.0 -w --builder 'clang-cl_vs2015' \
// RUN:      --debugger 'dbgeng' --cflags '/O2 /Z7 /Zi' \
// RUN:      --ldflags '/O2 /Z7 /Zi' -- %s

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
    // MSVC lays out calls to abort out of line, gets the layout we want.
    abort(); // DexLabel('stop')
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

/*
DexExpectProgramState({'frames':[
     {'function': 'inlineCrashFrame', 'location':{'lineno' : 'stop'} },
     {'function': 'callerOfInlineCrashFrame'},
     {'function': 'main'}
]})
*/
