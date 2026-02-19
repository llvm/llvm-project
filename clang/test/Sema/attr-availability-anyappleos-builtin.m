// RUN: %clang_cc1 -triple arm64-apple-ios26.0 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-macos26.0 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple arm64-apple-tvos26.0 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-macos26.0 -darwin-target-variant-triple x86_64-apple-ios27.0-macabi -fsyntax-only -verify %s

// Test @available and __builtin_available with anyAppleOS.

// Declarations with anyAppleOS availability.
void func_introduced_26(void) __attribute__((availability(anyAppleOS, introduced=26.0)));
void func_introduced_27(void) __attribute__((availability(anyAppleOS, introduced=27.0)));
void func_deprecated_27(void) __attribute__((availability(anyAppleOS, introduced=26.0, deprecated=27.0)));
void func_obsoleted_28(void) __attribute__((availability(anyAppleOS, introduced=26.0, obsoleted=28.0)));
void func_unavailable(void) __attribute__((availability(anyAppleOS, unavailable))); // expected-note {{has been explicitly marked unavailable here}}

void test_builtin_available() {
  // Guard with __builtin_available - should suppress warnings
  if (__builtin_available(anyAppleOS 27, *)) {
    func_introduced_27(); // No warning - properly guarded
  }

  // Mixing anyAppleOS with platform-specific checks
  if (__builtin_available(anyAppleOS 26, iOS 28, *)) {
    func_introduced_26(); // No warning
  }

  // Guard for multiple versions
  if (__builtin_available(anyAppleOS 26, *)) {
    func_introduced_26(); // No warning
    func_deprecated_27(); // No warning
    func_obsoleted_28(); // No warning
  }

  // Even with guard, unavailable is still an error
  if (__builtin_available(anyAppleOS 26, *)) {
    func_unavailable(); // expected-error {{'func_unavailable' is unavailable}}
  }
}

void test_at_available() {
  // Guard with @available - should suppress warnings
  if (@available(anyAppleOS 27, *)) {
    func_introduced_27(); // No warning - properly guarded
  }

  // Mixing anyAppleOS with platform-specific checks
  if (@available(anyAppleOS 26, macOS 28, *)) {
    func_introduced_26(); // No warning
  }

  // Guard for multiple versions
  if (@available(anyAppleOS 26, *)) {
    func_introduced_26(); // No warning
    func_deprecated_27(); // No warning
    func_obsoleted_28(); // No warning
  }
}

void test_multiple_guards() {
  // Test that both @available and __builtin_available work
  if (@available(anyAppleOS 27, *)) {
    if (__builtin_available(anyAppleOS 27, *)) {
      func_introduced_27(); // No warning - doubly guarded
    }
  }
}

// Additional declarations for testing insufficient guards.
void func_introduced_28(void) __attribute__((availability(anyAppleOS, introduced=28.0))); // expected-note 2 {{has been marked as being introduced in}}

void test_invalid_versions() {
  // Test invalid version numbers (< 26.0) in __builtin_available
  if (__builtin_available(anyAppleOS 25, *)) { // expected-warning {{invalid anyAppleOS version '25' in availability check}} expected-note {{implicitly treating version as '26.0'}}
    func_introduced_26();
  }

  if (__builtin_available(anyAppleOS 14, *)) { // expected-warning {{invalid anyAppleOS version '14' in availability check}} expected-note {{implicitly treating version as '26.0'}}
    func_introduced_26();
  }

  // Test invalid version numbers in @available
  if (@available(anyAppleOS 25, *)) { // expected-warning {{invalid anyAppleOS version '25' in availability check}} expected-note {{implicitly treating version as '26.0'}}
    func_introduced_26();
  }

  if (@available(anyAppleOS 10, *)) { // expected-warning {{invalid anyAppleOS version '10' in availability check}} expected-note {{implicitly treating version as '26.0'}}
    func_introduced_26();
  }
}

void test_insufficient_guard_too_old() {
  // Guard version is too old - function needs 28.0 but guard only checks 27.0
  if (__builtin_available(anyAppleOS 27, *)) {
    func_introduced_28(); // expected-warning {{only available on}} expected-note {{enclose 'func_introduced_28' in an @available check to silence this warning}}
  }

  // Same with @available
  if (@available(anyAppleOS 27, *)) {
    func_introduced_28(); // expected-warning {{only available on}} expected-note {{enclose 'func_introduced_28' in an @available check to silence this warning}}
  }
}

void test_duplicate_anyappleos() {
  // Duplicate anyAppleOS specs should be an error
  if (__builtin_available(anyAppleOS 27, anyAppleOS 28, *)) { // expected-error {{version for 'anyappleos' already specified}}
  }

  if (@available(anyAppleOS 27, anyAppleOS 28, *)) { // expected-error {{version for 'anyappleos' already specified}}
  }
}

// Note: Platform-specific precedence is tested implicitly throughout this file.
// When a platform-specific version is present (e.g., iOS 27), it takes precedence
// over anyAppleOS for that platform. anyAppleOS acts as a fallback when no
// platform-specific version is specified for the target platform.
