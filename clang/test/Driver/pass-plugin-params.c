// REQUIRES: pass-plugins
// UNSUPPORTED: system-windows

// FIXME: This is supposed to work, but right now it doesn't. We need the extra -load like below.
// RUN: not %clang -fsyntax-only -fpass-plugin=%pass_plugin_reference \
// RUN:            -mllvm -instrumentor-write-config-file=%t_cfg.json %s 2>&1 | FileCheck %s
//
// RUN: %clang -fsyntax-only -fpass-plugin=%pass_plugin_reference -Xclang -load -Xclang %pass_plugin_reference \
// RUN:        -mllvm -instrumentor-write-config-file=%t_cfg.json %s
//
// CHECK: Unknown command line argument

int main() { return 0; }
