// RUN: rm -rf %t

// Build Module and set its timestamp
// RUN: echo '@import Module;' | %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash -fsyntax-only -F %S/Inputs -x objective-c -
// RUN: touch -m -a -t 201101010000 %t/Module.pcm
// RUN: cp %t/Module.pcm %t/Module.pcm.saved
// RUN: wc -c %t/Module.pcm > %t/Module.size.saved

// Build DependsOnModule
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash -fsyntax-only -F %S/Inputs %s
// RUN: diff %t/Module.pcm %t/Module.pcm.saved
// RUN: cp %t/DependsOnModule.pcm %t/DependsOnModule.pcm.saved

// Rebuild Module, reset its timestamp, and verify its size hasn't changed
// RUN: rm %t/Module.pcm
// RUN: echo '@import Module;' | %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash -fsyntax-only -F %S/Inputs -x objective-c -
// RUN: touch -m -a -t 201101010000 %t/Module.pcm
// RUN: wc -c %t/Module.pcm > %t/Module.size
// RUN: diff %t/Module.size %t/Module.size.saved
// RUN: cp %t/Module.pcm %t/Module.pcm.saved.2

// But the signature at least is expected to change, so we rebuild DependsOnModule.
// NOTE: if we change how the signature is created, this test may need updating.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash -fsyntax-only -F %S/Inputs %s
// RUN: diff %t/Module.pcm %t/Module.pcm.saved.2
// RUN: not diff %t/DependsOnModule.pcm %t/DependsOnModule.pcm.saved

// Rebuild Module, reset its timestamp, and verify its size hasn't changed
// RUN: rm %t/Module.pcm
// RUN: echo '@import Module;' | %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash -fsyntax-only -F %S/Inputs -x objective-c -
// RUN: touch -m -a -t 201101010000 %t/Module.pcm
// RUN: wc -c %t/Module.pcm > %t/Module.size
// RUN: diff %t/Module.size %t/Module.size.saved
// RUN: cp %t/Module.pcm %t/Module.pcm.saved.2

// Verify again with Module pre-imported.
// NOTE: if we change how the signature is created, this test may need updating.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fdisable-module-hash -fsyntax-only -F %S/Inputs %s
// RUN: diff %t/Module.pcm %t/Module.pcm.saved.2
// RUN: not diff %t/DependsOnModule.pcm %t/DependsOnModule.pcm.saved

#ifdef PREIMPORT
@import Module;
#endif
@import DependsOnModule;
