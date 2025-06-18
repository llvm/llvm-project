// REQUIRES: pass-plugins
// UNSUPPORTED: system-windows

// RUN: %clang -fpass-plugin=%pass_plugin_reference \
// RUN:      -S -emit-llvm -Xclang -fdebug-pass-manager %s -o /dev/null 2>&1 | FileCheck --check-prefix=PARAM-FALSE %s
//
// TODO: Can we make this work? (i.e. without the extra -mllvm from below)
// RUN: not %clang -fpass-plugin=%pass_plugin_reference -wave-goodbye \
// RUN:     -S -emit-llvm -Xclang -fdebug-pass-manager %s -o /dev/null 2>&1 | FileCheck --check-prefix=PARAM-ERR-CLANG %s
//
// FIXME: This is supposed to work (i.e. without the extra -load from below)
// RUN: not %clang -fpass-plugin=%pass_plugin_reference -mllvm -wave-goodbye \
// RUN:     -S -emit-llvm -Xclang -fdebug-pass-manager %s -o /dev/null 2>&1 | FileCheck --check-prefix=PARAM-ERR-LLVM %s
//
// RUN: %clang -fpass-plugin=%pass_plugin_reference -Xclang -load -Xclang %pass_plugin_reference -mllvm -wave-goodbye \
// RUN:     -S -emit-llvm -Xclang -fdebug-pass-manager %s -o /dev/null 2>&1 | FileCheck --check-prefix=PARAM-TRUE %s
//
// PARAM-ERR-CLANG: error: unknown argument
// PARAM-ERR-LLVM: Unknown command line argument
// PARAM-TRUE: Plugin parameter value -wave-goodbye=true
// PARAM-FALSE: Plugin parameter value -wave-goodbye=false

int main() { return 0; }
