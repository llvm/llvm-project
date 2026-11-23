// REQUIRES: system-darwin
// UNSUPPORTED: darwin-remote, i386-darwin

// RUN: %clangxx_asan -O0 -g -fno-omit-frame-pointer %s -o %t

// --- Diagnostics preamble (never gates; the FileCheck at the end is the gate).
// RUN: echo '==== atos identity (runtime picks FIRST on PATH) ====' ; xcrun --find atos 2>&1 ; which -a atos 2>&1 ; true
// RUN: which atos | tr -d '\n' > %t.atos_path
// RUN: echo '==== atos version / codesign ====' ; atos -V 2>&1 ; codesign -dv --verbose=4 %{readfile:%t.atos_path} 2>&1 ; codesign -v --verbose=2 %{readfile:%t.atos_path} 2>&1 ; true
// RUN: echo '==== SIP / DevToolsSecurity / task_for_pid policy (attach-mode gate) ====' ; csrutil status 2>&1 ; DevToolsSecurity -status 2>&1 ; sysctl kern.tfp.policy 2>&1 ; true
// RUN: echo '==== active developer dir / SDK ====' ; xcode-select -p 2>&1 ; xcrun --show-sdk-path 2>&1 ; true
// RUN: echo '==== network-dSYM hook (a set DBGShellCommands => hang risk) ====' ; defaults read com.apple.DebugSymbols 2>&1 ; env | grep -iE 'DBGShellCommands|dsymForUUID' 2>&1 ; true
// RUN: echo '==== built test binary codesign (mode 1c: hardened w/o get-task-allow) ====' ; codesign -dv --verbose=4 %t 2>&1 ; codesign -d --entitlements :- %t 2>&1 | grep -iE 'get-task-allow|runtime|flags' 2>&1 ; true

// --- (A) ATTACH mode: exactly how the runtime uses atos (atos -i -p <pid> over a
//         pipe). This is the REAL gate. `not %run` because asan exits non-zero. ---
// RUN: %env_asan_opts=verbosity=2 ASAN_SYMBOLIZER_PATH=%{readfile:%t.atos_path} not %run %t > %t.attach.log 2>&1 || true
// RUN: echo '==== ATTACH-mode asan report (verbosity=2) ====' ; cat %t.attach.log || true

// --- (B) OFFLINE cross-check: run atos by hand against the same binary. Lets us
//         tell "atos itself is dead" (offline also fails) from "attach/task_for_pid
//         is blocked" (offline works). Never gates. ---
// RUN: echo '==== OFFLINE atos probe (no attach / no task_for_pid) ====' ; atos -o %t -arch %arch -l 0 0x1 </dev/null 2>&1 && echo 'offline-atos: exit 0 (alive)' || echo 'offline-atos: NONZERO exit (atos failed/crashed)' ; true

// --- best-effort syslog tail (task_for_pid / atos / CoreSymbolication denials).
//     `log show` is slow, so it runs ONLY on symbolizer failure (grep the attach
//     log first) -- keeps the passing case fast, still captures denials on failure. ---
// RUN: echo '==== syslog (only on symbolizer failure; log show is slow) ====' ; grep -qE "Can't read from symbolizer at fd|atos failed to symbolize" %t.attach.log && log show --last 2m --style compact --predicate 'process == "atos" OR process == "taskgated" OR eventMessage CONTAINS "task_for_pid" OR eventMessage CONTAINS "Sanitizer"' 2>&1 ; true

// --- On symbolizer failure, dump any atos crash report(s). CrashReporter writes
//     the .ips asynchronously so we wait briefly -- but ONLY on the failure path
//     (same grep gate), so a passing run pays nothing. `sh -c` because lit's
//     internal shell has no loops / command substitution. ---
// RUN: grep -qE "Can't read from symbolizer at fd|atos failed to symbolize" %t.attach.log && sh -c 'echo "==== atos crash report(s) (symbolizer failed) ===="; sleep 3; for d in "$HOME/Library/Logs/DiagnosticReports" /Library/Logs/DiagnosticReports; do for f in $(ls -t "$d"/atos*.ips "$d"/atos*.crash 2>/dev/null | head -3); do echo "--- $f ---"; cat "$f" 2>/dev/null; done; done' 2>&1 ; true

// --- PASS/FAIL GATE: the attach-mode report MUST be fully symbolized. FileCheck
//     runs on the SAME captured log so the failing output sits right next to the
//     diagnostics above it in the lit log. ---
// RUN: FileCheck %s --input-file=%t.attach.log

// The runtime must have picked our atos and reported using it (verbosity=2).
// (ASAN_SYMBOLIZER_PATH => "at user-specified path"; a bare PATH search =>
//  "found at". Accept either so the canary is robust to how it resolved.)
// CHECK: {{Using atos (found at|at user-specified path):}}

// The crash header must appear (proves the process ran and asan fired).
// CHECK: AddressSanitizer: heap-use-after-free

// THE GATE: frame #0 must symbolize to boom() at its store line, frame #1 to
// main() at the boom() call. When atos is broken these stay as bare 0x...
// addresses with no `in boom` / no filename => FileCheck fails => canary RED.
// CHECK: #0 0x{{.*}} in boom{{.*}}atos-canary-symbolize.cpp:[[@LINE+11]]
// CHECK: #1 0x{{.*}} in main{{.*}}atos-canary-symbolize.cpp:[[@LINE+16]]

// --- Negative guards: make the RED reason unambiguous in the FileCheck diff.
//     (The positive frame checks already catch it; these name the failure mode.) ---
// CHECK-NOT: Can't read from symbolizer at fd
// CHECK-NOT: atos failed to symbolize address

#include <cstdlib>

__attribute__((noinline)) void boom(int *p) {
  *p = 42; // <-- frame #0 store; the CHECK #0 [[@LINE]] anchor resolves here
}

int main() {
  int *p = new int[4];
  delete[] p;
  boom(
      p); // <-- frame #1 call site; the CHECK #1 [[@LINE]] anchor resolves here
  return 0;
}
