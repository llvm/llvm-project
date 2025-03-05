// UNSUPPORTED: target={{.*}}-aix{{.*}}, target={{.*}}-zos{{.*}}
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -emit-llvm -o %t/test.bc -F%t/Frameworks %t/test.m -Werror=objc-method-access -DHIDDEN_FIRST=1 \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache
// RUN: %clang_cc1 -emit-llvm -o %t/test.bc -F%t/Frameworks %t/test.m -Werror=objc-method-access -DHIDDEN_FIRST=0 \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

// Test a case when Objective-C protocol is imported both as hidden and as visible.

//--- Frameworks/Foundation.framework/Headers/Foundation.h
@interface NSObject
@end

//--- Frameworks/Foundation.framework/Modules/module.modulemap
framework module Foundation {
  header "Foundation.h"
  export *
}

//--- Frameworks/Common.framework/Headers/Common.h
#import <Foundation/Foundation.h>
@protocol Testing;
@interface Common : NSObject
- (id<Testing>)getProtocolObj;
@end

//--- Frameworks/Common.framework/Modules/module.modulemap
framework module Common {
  header "Common.h"
  export *
}

//--- Frameworks/Regular.framework/Headers/Regular.h
@protocol Testing
- (void)protocolMethod;
@end

//--- Frameworks/Regular.framework/Modules/module.modulemap
framework module Regular {
  header "Regular.h"
  export *
}

//--- Frameworks/RegularHider.framework/Headers/Visible.h
// Empty, file required to create a module.

//--- Frameworks/RegularHider.framework/Headers/Hidden.h
@protocol Testing
- (void)protocolMethod;
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
#import <Common/Common.h>

#if HIDDEN_FIRST
#import <RegularHider/Visible.h>
#import <Regular/Regular.h>
#else
#import <Regular/Regular.h>
#import <RegularHider/Visible.h>
#endif

void test(Common *obj) {
  [[obj getProtocolObj] protocolMethod];
}
