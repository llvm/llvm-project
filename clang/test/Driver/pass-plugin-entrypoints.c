// REQUIRES: pass-plugins
// UNSUPPORTED: system-windows

// Default entry-point is Pipeline-EarlySimplification
//
// RUN: %clang -O0 -fpass-plugin=%pass_plugin_reference \
// RUN:        -S -emit-llvm -Xclang -fdebug-pass-manager %s -o /dev/null 2>&1 | \
// RUN:        FileCheck --check-prefix=EP-EARLY %s
//
// RUN: %clang -O2 -fpass-plugin=%pass_plugin_reference \
// RUN:        -S -emit-llvm -Xclang -fdebug-pass-manager %s -o /dev/null 2>&1 | \
// RUN:        FileCheck --check-prefix=EP-EARLY %s
//
// RUN: %clang -c -flto=full -O2 -fpass-plugin=%pass_plugin_reference \
// RUN:        -Xclang -fdebug-pass-manager %s -o /dev/null 2>&1 | \
// RUN:        FileCheck --check-prefix=EP-EARLY %s
//
// RUN: %clang -c -flto=thin -O2 -fpass-plugin=%pass_plugin_reference \
// RUN:        -Xclang -fdebug-pass-manager %s -o /dev/null 2>&1 | \
// RUN:        FileCheck --check-prefix=EP-EARLY %s
//
// EP-EARLY: Running pass: InstrumentorPass
// EP-EARLY: Running pass: AlwaysInlinerPass

// Pass doesn't run if default entry-point is disabled
// RUN: env registerPipelineEarlySimplificationEPCallback=Off \
// RUN:     %clang -fpass-plugin=%pass_plugin_reference -S -emit-llvm \
// RUN:            -Xclang -fdebug-pass-manager %s -o /dev/null 2>&1 | FileCheck %s
//
// CHECK-NOT: Running pass: InstrumentorPass

// Pass runs twice if we add entry-point Opt-Early
// RUN: env registerOptimizerEarlyEPCallback=On \
// RUN:     %clang -fpass-plugin=%pass_plugin_reference -S -emit-llvm \
// RUN:            -Xclang -fdebug-pass-manager %s -o /dev/null 2>&1 | FileCheck --check-prefix=OPT-EARLY %s
//
// OPT-EARLY: Running pass: InstrumentorPass
// OPT-EARLY: Running pass: AlwaysInlinerPass
// OPT-EARLY: Running pass: InstrumentorPass

int main() { return 0; }
