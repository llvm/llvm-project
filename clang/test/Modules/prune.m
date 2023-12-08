// NetBSD: noatime mounts currently inhibit 'touch -a' updates
// UNSUPPORTED: system-netbsd

// Test the automatic pruning of module cache entries.
#ifdef IMPORT_DEPENDS_ON_MODULE
@import DependsOnModule;
#else
@import Module;
#endif

// Clear out the module cache
// RUN: rm -rf %t
// Run Clang twice so we end up creating the timestamp file (the second time).
// RUN: %clang_cc1 -DIMPORT_DEPENDS_ON_MODULE -Wno-private-module -fmodules-ignore-macro=DIMPORT_DEPENDS_ON_MODULE -fmodules -fimplicit-module-maps -F %S/Inputs -fmodules-cache-path=%t %s -verify
// RUN: %clang_cc1 -DIMPORT_DEPENDS_ON_MODULE -Wno-private-module -fmodules-ignore-macro=DIMPORT_DEPENDS_ON_MODULE -fmodules -fimplicit-module-maps -F %S/Inputs -fmodules-cache-path=%t %s -verify
// RUN: ls %t | grep modules.timestamp
// RUN: ls -R %t | grep ^Module.*pcm
// RUN: ls -R %t | grep DependsOnModule.*pcm

// Set the timestamp back more than two days. We should try to prune,
// but nothing gets pruned because the module files are new enough.
// RUN: touch -m -a -t 201101010000 %t/modules.timestamp 
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -Wno-private-module -F %S/Inputs -fmodules-cache-path=%t -fmodules -fmodules-prune-interval=172800 -fmodules-prune-after=345600 %s -verify
// RUN: ls %t | grep modules.timestamp
// RUN: ls -R %t | grep ^Module.*pcm
// RUN: ls -R %t | grep DependsOnModule.*pcm

// Set the DependsOnModule access time back more than four days.
// This shouldn't prune anything, because the timestamp has been updated, so
// the pruning mechanism won't fire.
// RUN: find %t -name DependsOnModule*.pcm | sed -e 's/\\/\//g' | xargs touch -a -t 201101010000
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -F %S/Inputs -Wno-private-module -fmodules-cache-path=%t -fmodules -fmodules-prune-interval=172800 -fmodules-prune-after=345600 %s -verify
// RUN: ls %t | grep modules.timestamp
// RUN: ls -R %t | grep ^Module.*pcm
// RUN: ls -R %t | grep DependsOnModule.*pcm

// Set both timestamp and DependsOnModule.pcm back beyond the cutoff.
// This should trigger pruning, which will remove DependsOnModule but not Module.
// RUN: touch -m -a -t 201101010000 %t/modules.timestamp 
// RUN: find %t -name DependsOnModule*.pcm | sed -e 's/\\/\//g' | xargs touch -a -t 201101010000
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -F %S/Inputs -Wno-private-module -fmodules-cache-path=%t -fmodules -fmodules-prune-interval=172800 -fmodules-prune-after=345600 %s -verify
// RUN: ls %t | grep modules.timestamp
// RUN: ls -R %t | grep ^Module.*pcm
// RUN: ls -R %t | not grep DependsOnModule.*pcm

// expected-no-diagnostics
