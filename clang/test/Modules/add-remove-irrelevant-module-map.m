// RUN: rm -rf %t
// RUN: rm -rf %t.mcp
// RUN: mkdir -p %t

// Build without b.modulemap
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t.mcp -fdisable-module-hash -fmodule-map-file=%S/Inputs/AddRemoveIrrelevantModuleMap/a.modulemap %s -verify
// RUN: cp %t.mcp/a.pcm %t/a.pcm

// Build with b.modulemap
// RUN: rm -rf %t.mcp
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t.mcp -fdisable-module-hash -fmodule-map-file=%S/Inputs/AddRemoveIrrelevantModuleMap/a.modulemap -fmodule-map-file=%S/Inputs/AddRemoveIrrelevantModuleMap/b.modulemap %s -verify
// RUN: not diff %t.mcp/a.pcm %t/a.pcm

// expected-no-diagnostics

@import a;
