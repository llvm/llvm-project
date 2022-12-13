// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -emit-llvm -o %t/test.bc -F%t/Frameworks %t/test.m -DHIDDEN_FIRST=1 \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache
// RUN: %clang_cc1 -emit-llvm -o %t/test.bc -F%t/Frameworks %t/test.m -DHIDDEN_FIRST=0 \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache
// UNSUPPORTED: target={{.*}}-zos{{.*}}, target={{.*}}-aix{{.*}}

// Test a case when Objective-C interface is imported both as hidden and as visible.

//--- Frameworks/Foundation.framework/Headers/Foundation.h
@interface NSObject
@end

//--- Frameworks/Foundation.framework/Modules/module.modulemap
framework module Foundation {
  header "Foundation.h"
  export *
}

//--- Frameworks/Regular.framework/Headers/Regular.h
#import <Foundation/Foundation.h>
@interface Regular : NSObject
@end

//--- Frameworks/Regular.framework/Modules/module.modulemap
framework module Regular {
  header "Regular.h"
  export *
}

//--- Frameworks/RegularHider.framework/Headers/Visible.h
// Empty, file required to create a module.

//--- Frameworks/RegularHider.framework/Headers/Hidden.h
#import <Foundation/Foundation.h>
@interface Regular : NSObject
@end

//--- Frameworks/RegularHider.framework/Modules/module.modulemap
framework module RegularHider {
  header "Visible.h"
  export *

  explicit module Hidden {
    header "Hidden.h"
    export *
  }
}

//--- test.m

#if HIDDEN_FIRST
#import <RegularHider/Visible.h>
#import <Regular/Regular.h>
#else
#import <Regular/Regular.h>
#import <RegularHider/Visible.h>
#endif

@interface SubClass : Regular
@end
