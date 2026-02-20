// REQUIRES: system-darwin
// REQUIRES: lld
//
// RUN: rm -rf %t && mkdir -p %t

//--- TestClass.h
#import <Foundation/Foundation.h>
#include <functional>
#include <memory>

struct Request {
  const char* queryLabel;
};

@interface TestClass : NSObject

// Method returning std::function - uses sret, triggers thunk generation
- (std::function<bool(const std::shared_ptr<Request>&)>)shouldScheduleResponse __attribute__((objc_direct));

// Method returning std::shared_ptr - also uses sret
- (std::shared_ptr<Request>)getRequest __attribute__((objc_direct));

@end

//--- TestClass.mm
#include <functional>
#include <memory>
#import "TestClass.h"

@implementation TestClass

- (std::function<bool(const std::shared_ptr<Request>&)>)shouldScheduleResponse {
  return [](const std::shared_ptr<Request>& request) {
    return (request != nullptr);
  };
}

- (std::shared_ptr<Request>)getRequest {
  return std::make_shared<Request>();
}

@end

//--- main.mm
#import "TestClass.h"

// This function calls the direct methods, triggering thunk generation
// in this translation unit. During LTO, the thunk's sret type may differ
// from the implementation's sret type due to type renaming.
void useTestClass(TestClass *obj) {

}

int main() {
  @autoreleasepool {
    TestClass *obj;
    auto callback = [obj shouldScheduleResponse];
    auto request = [obj getRequest];
    (void)callback;
    (void)request;
  }
  return 0;
}

// ============================================================================
// Split the file into individual source files
// ============================================================================
// RUN: split-file %s %t

// ============================================================================
// Compile to bitcode with ThinLTO
// ============================================================================

// Compile implementation to bitcode
// RUN: %clang++ -fobjc-direct-precondition-thunk      \
// RUN:   -target arm64-apple-macos11.0 -fobjc-arc     \
// RUN:   -std=c++17 -O0 -flto=thin                    \
// RUN:   -c %t/TestClass.mm -I%t -o %t/TestClass.bc

// Compile main (with thunk generation) to bitcode
// RUN: %clang++ -fobjc-direct-precondition-thunk      \
// RUN:   -target arm64-apple-macos11.0 -fobjc-arc     \
// RUN:   -std=c++17 -O0 -flto=thin                    \
// RUN:   -c %t/main.mm -I%t -o %t/main.bc

// ============================================================================
// Link with ThinLTO using lld - this should succeed now that sret is properly
// propagated to the thunk function definition.
// ============================================================================
// RUN: %clang++ -fobjc-direct-precondition-thunk      \
// RUN:   -target arm64-apple-macos11.0 -fobjc-arc     \
// RUN:   -std=c++17 -O0 -flto=thin                    \
// RUN:   -fuse-ld=lld                                 \
// RUN:   %t/TestClass.bc %t/main.bc                   \
// RUN:   -framework Foundation -o %t/test_lto
