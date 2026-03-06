// This validates that all expected OSVersions that allow fallbacks 
// from iOS behave as expected against a common version bump.

// RUN: %clang_cc1 "-triple" "arm64-apple-ios26" -fsyntax-only -verify %s
// RUN: %clang_cc1 "-triple" "arm64-apple-watchos26" -fsyntax-only -verify %s
// RUN: %clang_cc1 "-triple" "arm64-apple-tvos26" -fsyntax-only -verify %s

// VisionOS requires SDKSettings support to enable remappings.
// RUN: %clang_cc1 "-triple" "arm64-apple-visionos26" -isysroot %S/Inputs/XROS26.0.sdk -fsyntax-only -verify %s

// expected-no-diagnostics

__attribute__((availability(ios,strict,introduced=19)))
int iOSExistingAPI(void);

__attribute__((availability(ios,strict,introduced=26)))
int iOSExistingAPI2(void);

void testAvailabilityCheck(void) {
  
  if (__builtin_available(iOS 19, *)) {
    iOSExistingAPI();
    iOSExistingAPI2();
  }
  
  if (__builtin_available(iOS 26, *)) {
    iOSExistingAPI();
    iOSExistingAPI2();
  }

  iOSExistingAPI2();
}
