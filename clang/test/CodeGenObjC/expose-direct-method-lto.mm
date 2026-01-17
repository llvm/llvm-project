// REQUIRES: system-darwin
// REQUIRES: lld
//
// RUN: rm -rf %t && mkdir -p %t

// This test is a regression test for a bug in the LTO.
// When trying to call a direct method from another translation unit,
// the compiler generates a thunk to call the method. But the attribute of
// the arguments may not be set correctly, causing the compiler to generate
// a call that can't be `musttail`-ed, which will later cause a link error.

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
// RUN:   -c %t/Foo.mm -I%t -o %t/Foo.bc

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
// RUN:   %t/Foo.bc %t/main.bc                   \
// RUN:   -framework Foundation -o %t/test_lto

//--- Foo.h
#import <Foundation/Foundation.h>
#include <functional>
#include <memory>

struct Request {
  const char* queryLabel;
};

@interface Foo : NSObject

// Method returning std::function - uses sret, triggers thunk generation
- (std::function<bool(const std::shared_ptr<Request>&)>)shouldScheduleResponse __attribute__((objc_direct));

// Method returning std::shared_ptr - also uses sret
- (std::shared_ptr<Request>)getRequest __attribute__((objc_direct));

@end

//--- Foo.mm
#include <functional>
#include <memory>
#import "Foo.h"

@implementation Foo

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
#import "Foo.h"

// This function calls the direct methods, triggering thunk generation
// in this translation unit. During LTO, the thunk's sret type may differ
// from the implementation's sret type due to type renaming.
void useFoo(Foo *foo) {

}

int main() {
  @autoreleasepool {
    Foo *foo;
    auto callback = [foo shouldScheduleResponse];
    auto request = [foo getRequest];
    (void)callback;
    (void)request;
  }
  return 0;
}
