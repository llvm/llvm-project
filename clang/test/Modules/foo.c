// RUN: rm -rf %t
// RUN: split-file %s %t

// Not to be committed, just here as a demonstration.

//--- frameworks/A.framework/Modules/module.modulemap
framework module A {
  header "A1.h"
  header "A2.h"
  header "A3.h"
}
//--- frameworks/A.framework/Headers/A1.h
//--- frameworks/A.framework/Headers/A2.h
//--- frameworks/A.framework/Headers/A3.h
//--- EndA
// RUN: %clang_cc1 -fmodules -F %t/frameworks -emit-module -fmodule-name=A %t/frameworks/A.framework/Modules/module.modulemap -o %t/A.pcm

//--- frameworks/B.framework/Modules/module.modulemap
framework module B {
  header "B1.h"
  header "B2.h"
  header "B3.h"
}
//--- frameworks/B.framework/Headers/B1.h
//--- frameworks/B.framework/Headers/B2.h
//--- frameworks/B.framework/Headers/B3.h
//--- EndB
// RUN: %clang_cc1 -fmodules -F %t/frameworks -emit-module -fmodule-name=B %t/frameworks/B.framework/Modules/module.modulemap -o %t/B.pcm

//--- frameworks/X.framework/Modules/module.modulemap
framework module X { header "X.h" }
//--- frameworks/X.framework/Headers/X.h
#import <A/A1.h>
#import <B/B1.h>
// RUN: %clang_cc1 -fmodules -F %t/frameworks -emit-module -fmodule-name=X %t/frameworks/X.framework/Modules/module.modulemap -o %t/X.pcm \
// RUN:   -fmodule-map-file=%t/frameworks/A.framework/Modules/module.modulemap -fmodule-file=A=%t/A.pcm \
// RUN:   -fmodule-map-file=%t/frameworks/B.framework/Modules/module.modulemap -fmodule-file=B=%t/B.pcm

// Without this patch, ASTWriter would go looking for:
// * "X.h" in A.pcm and B.pcm and not comparing it to any of their files due to size difference
// * "A2.h" and compare it to "A1.h", "A2.h"                                          in A.pcm
// * "A3.h" and compare it to "A1.h", "A2.h", "A3.h"                                  in A.pcm
// * "B2.h" and compare it to "A1.h", "A2.h", "A3.h" in A.pcm; "B1.h", "B2.h"         in B.pcm
// * "B3.h" and compare it to "A1.h", "A2.h", "A3.h" in A.pcm; "B1.h", "B2.h", "B3.h" in B.pcm

// With this patch, ASTWriter doesn't go looking for anything of the above.
// * Clang already knows that "X.h" belongs to the current module and needs to be serialized,
//   while the other headers belong to A or B and do not need to be serialized.
