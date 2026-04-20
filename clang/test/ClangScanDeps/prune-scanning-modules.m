// NetBSD: noatime mounts currently inhibit 'touch -a' updates
// UNSUPPORTED: system-netbsd

// Test the automatic pruning of module cache entries.

// RUN: rm -rf %t
// RUN: split-file %s %t

// Check no pruning happens because pcms are new enough.
// RUN: touch -m -a -t 201101010000 %t/modules.timestamp 
// RUN: clang-scan-deps -j 1 -format experimental-full -o /dev/null -- %clang -fmodules -F %t/Frameworks \
// RUN:   -fmodules-cache-path=%t/cache %t/prune.m -fmodules -fmodules-prune-interval=172800 -fmodules-prune-after=345600
// RUN: ls -R %t | grep ^Module.*pcm
// RUN: ls -R %t | grep DependsOnModule.*pcm

// Check no pruning happens because modules.timestamp is new enough.
// RUN: find %t/cache -name DependsOnModule*.pcm | sed -e 's/\\/\//g' | xargs touch -a -t 201101010000
// RUN: clang-scan-deps -j 1 -format experimental-full -o /dev/null -- %clang -fmodules -F %t/Frameworks \
// RUN:   -fmodules-cache-path=%t/cache %t/prune.m -fmodules -fmodules-prune-interval=172800 -fmodules-prune-after=345600
// RUN: ls -R %t/cache | grep ^Module.*pcm
// RUN: ls -R %t/cache | grep DependsOnModule.*pcm

// Check unused & unaccessed modules gets pruned.
// RUN: touch -m -a -t 201101010000 %t/cache/modules.timestamp 
// RUN: find %t/cache -name DependsOnModule*.pcm | sed -e 's/\\/\//g' | xargs touch -a -t 201101010000
// RUN: find %t/cache -name Module*.pcm | sed -e 's/\\/\//g' | xargs touch -a -t 201101010000
// RUN: clang-scan-deps -j 1 -format experimental-full -o /dev/null -- %clang -fmodules -F %t/Frameworks -DSKIP_MODULE \
// RUN:   -fmodules-cache-path=%t/cache %t/prune.m -fmodules -fmodules-prune-interval=172800 -fmodules-prune-after=345600
// RUN: ls -R %t/cache | not grep ^Module.*pcm
// RUN: ls -R %t/cache | not grep DependsOnModule.*pcm

//--- prune.m
#ifndef SKIP_MODULE
@import DependsOnModule;
#endif // SKIP_MODULE

//--- Frameworks/DependsOnModule.framework/Headers/DependsOnModule.h
#include <Module/Module.h> 

//--- Frameworks/DependsOnModule.framework/Modules/module.modulemap
framework module DependsOnModule {
  umbrella header "DependsOnModule.h"
  module * {
    export *
  }
}

//--- Frameworks/Module.framework/Headers/Module.h
#ifndef MODULE_H
#define MODULE_H
const char *getModuleVersion(void);

#endif // MODULE_H


//--- Frameworks/Module.framework/Modules/module.modulemap
framework module Module {
  umbrella header "Module.h"
  module * {
    export *
  }
}
