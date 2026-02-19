// RUN: %clang_cc1 "-triple" "x86_64-apple-macosx27.0" -fsyntax-only -verify=macos %s
// RUN: %clang_cc1 "-triple" "arm64-apple-ios27.0" -fsyntax-only -verify=ios %s
// RUN: %clang_cc1 "-triple" "arm64-apple-tvos27.0" -fsyntax-only -verify=tvos %s
// RUN: %clang_cc1 "-triple" "arm64-apple-watchos27.0" -fsyntax-only -verify=watchos %s
// RUN: %clang_cc1 "-triple" "arm64-apple-xros27.0" -fsyntax-only -isysroot %S/Inputs/XROS26.0.sdk -verify=xros %s
// RUN: %clang_cc1 "-triple" "arm64-apple-ios27.0-macabi" -fsyntax-only -verify=maccatalyst %s
// RUN: %clang_cc1 "-triple" "arm64-apple-driverkit27.0" -fsyntax-only -verify=driverkit %s
// RUN: %clang_cc1 "-triple" "aarch64-linux-android27" -fsyntax-only -verify=android %s


void f_introduced_26(void) __attribute__((availability(anyAppleOS, introduced=26.0)));

void f_introduced_25(void) __attribute__((availability(anyAppleOS, introduced=25.0))); // \
  macos-warning{{invalid anyAppleOS version '25.0' in availability attribute}} \
  macos-note{{implicitly treating version as '26.0'}} \
  ios-warning{{invalid anyAppleOS version '25.0' in availability attribute}} \
  ios-note{{implicitly treating version as '26.0'}} \
  tvos-warning{{invalid anyAppleOS version '25.0' in availability attribute}} \
  tvos-note{{implicitly treating version as '26.0'}} \
  watchos-warning{{invalid anyAppleOS version '25.0' in availability attribute}} \
  watchos-note{{implicitly treating version as '26.0'}} \
  xros-warning{{invalid anyAppleOS version '25.0' in availability attribute}} \
  xros-note{{implicitly treating version as '26.0'}} \
  maccatalyst-warning{{invalid anyAppleOS version '25.0' in availability attribute}} \
  maccatalyst-note{{implicitly treating version as '26.0'}} \
  driverkit-warning{{invalid anyAppleOS version '25.0' in availability attribute}} \
  driverkit-note{{implicitly treating version as '26.0'}} \
  android-warning{{invalid anyAppleOS version '25.0' in availability attribute}} \
  android-note{{implicitly treating version as '26.0'}}

void f_introduced_28(void) __attribute__((availability(anyAppleOS, introduced=28.0))); // \
  macos-note{{'f_introduced_28' has been marked as being introduced in macOS 28.0 here, but the deployment target is macOS 27.0}} \
  ios-note{{'f_introduced_28' has been marked as being introduced in iOS 28.0 here, but the deployment target is iOS 27.0}} \
  tvos-note{{'f_introduced_28' has been marked as being introduced in tvOS 28.0 here, but the deployment target is tvOS 27.0}} \
  watchos-note{{'f_introduced_28' has been marked as being introduced in watchOS 28.0 here, but the deployment target is watchOS 27.0}} \
  xros-note{{'f_introduced_28' has been marked as being introduced in visionOS 28.0 here, but the deployment target is visionOS 27.0}} \
  maccatalyst-note{{'f_introduced_28' has been marked as being introduced in macCatalyst 28.0 here, but the deployment target is macCatalyst 27.0}} \
  driverkit-note{{'f_introduced_28' has been marked as being introduced in DriverKit 28.0 here, but the deployment target is DriverKit 27.0}}

void f_deprecated_27(void) __attribute__((availability(anyAppleOS, introduced=26.0, deprecated=27.0))); // \
  macos-note{{'f_deprecated_27' has been explicitly marked deprecated here}} \
  ios-note{{'f_deprecated_27' has been explicitly marked deprecated here}} \
  tvos-note{{'f_deprecated_27' has been explicitly marked deprecated here}} \
  watchos-note{{'f_deprecated_27' has been explicitly marked deprecated here}} \
  xros-note{{'f_deprecated_27' has been explicitly marked deprecated here}} \
  maccatalyst-note{{'f_deprecated_27' has been explicitly marked deprecated here}} \
  driverkit-note{{'f_deprecated_27' has been explicitly marked deprecated here}}

void f_obsoleted_27(void) __attribute__((availability(anyAppleOS, introduced=26.0, obsoleted=27.0))); // \
  macos-note{{'f_obsoleted_27' has been explicitly marked unavailable here}} \
  ios-note{{'f_obsoleted_27' has been explicitly marked unavailable here}} \
  tvos-note{{'f_obsoleted_27' has been explicitly marked unavailable here}} \
  watchos-note{{'f_obsoleted_27' has been explicitly marked unavailable here}} \
  xros-note{{'f_obsoleted_27' has been explicitly marked unavailable here}} \
  maccatalyst-note{{'f_obsoleted_27' has been explicitly marked unavailable here}} \
  driverkit-note{{'f_obsoleted_27' has been explicitly marked unavailable here}}

void f_deprecated_28(void) __attribute__((availability(anyAppleOS, introduced=26.0, deprecated=28.0)));

void f_obsoleted_28(void) __attribute__((availability(anyAppleOS, introduced=26.0, obsoleted=28.0)));

void f_unavailable(void) __attribute__((availability(anyAppleOS, unavailable))); // \
  macos-note{{'f_unavailable' has been explicitly marked unavailable here}} \
  ios-note{{'f_unavailable' has been explicitly marked unavailable here}} \
  tvos-note{{'f_unavailable' has been explicitly marked unavailable here}} \
  watchos-note{{'f_unavailable' has been explicitly marked unavailable here}} \
  xros-note{{'f_unavailable' has been explicitly marked unavailable here}} \
  maccatalyst-note{{'f_unavailable' has been explicitly marked unavailable here}} \
  driverkit-note{{'f_unavailable' has been explicitly marked unavailable here}}

void f_introduced_26_tvos28(void) __attribute__((availability(anyAppleOS, introduced=26.0), availability(tvos, introduced=28.0))); // \
  tvos-note{{'f_introduced_26_tvos28' has been marked as being introduced in tvOS 28.0 here, but the deployment target is tvOS 27.0}}

void f_introduced_tvos28_26(void) __attribute__((availability(tvos, introduced=28.0), availability(anyAppleOS, introduced=26.0))); // \
  tvos-note{{'f_introduced_tvos28_26' has been marked as being introduced in tvOS 28.0 here, but the deployment target is tvOS 27.0}}

#pragma clang attribute push (__attribute__((availability(tvos, introduced=29.0))), apply_to=function)
void f_introduced_26_tvos29(void) __attribute__((availability(anyAppleOS, introduced=26.0))); // \
  tvos-note{{'f_introduced_26_tvos29' has been marked as being introduced in tvOS 29.0 here, but the deployment target is tvOS 27.0}}
#pragma clang attribute pop

#pragma clang attribute push (__attribute__((availability(anyAppleOS, introduced=26.0), availability(tvos, introduced=29.1))), apply_to=function)
void f_introduced_26_tvos29_1(void); // \
  tvos-note{{'f_introduced_26_tvos29_1' has been marked as being introduced in tvOS 29.1 here, but the deployment target is tvOS 27.0}}
#pragma clang attribute pop

#pragma clang attribute push (__attribute__((availability(anyAppleOS, introduced=26.0))), apply_to=function)
void f_introduced_26_tvos29_2(void) __attribute__((availability(anyAppleOS, introduced=29.2))); // \
  macos-note{{'f_introduced_26_tvos29_2' has been marked as being introduced in macOS 29.2 here, but the deployment target is macOS 27.0}} \
  ios-note{{'f_introduced_26_tvos29_2' has been marked as being introduced in iOS 29.2 here, but the deployment target is iOS 27.0}} \
  tvos-note{{'f_introduced_26_tvos29_2' has been marked as being introduced in tvOS 29.2 here, but the deployment target is tvOS 27.0}} \
  watchos-note{{'f_introduced_26_tvos29_2' has been marked as being introduced in watchOS 29.2 here, but the deployment target is watchOS 27.0}} \
  xros-note{{'f_introduced_26_tvos29_2' has been marked as being introduced in visionOS 29.2 here, but the deployment target is visionOS 27.0}} \
  maccatalyst-note{{'f_introduced_26_tvos29_2' has been marked as being introduced in macCatalyst 29.2 here, but the deployment target is macCatalyst 27.0}} \
  driverkit-note{{'f_introduced_26_tvos29_2' has been marked as being introduced in DriverKit 29.2 here, but the deployment target is DriverKit 27.0}}
#pragma clang attribute pop

void f_introduced_ios26_29(void) __attribute__((availability(ios, introduced=26.0), availability(anyAppleOS, introduced=29.0))); // \
  macos-note{{'f_introduced_ios26_29' has been marked as being introduced in macOS 29.0 here, but the deployment target is macOS 27.0}} \
  driverkit-note{{'f_introduced_ios26_29' has been marked as being introduced in DriverKit 29.0 here, but the deployment target is DriverKit 27.0}}

void f_introduced_29_ios26(void) __attribute__((availability(anyAppleOS, introduced=29.0), availability(ios, introduced=26.0))); // \
  macos-note{{'f_introduced_29_ios26' has been marked as being introduced in macOS 29.0 here, but the deployment target is macOS 27.0}} \
  driverkit-note{{'f_introduced_29_ios26' has been marked as being introduced in DriverKit 29.0 here, but the deployment target is DriverKit 27.0}}

void f_ios26_unavailable(void) __attribute__((availability(ios, introduced=26.0), availability(anyAppleOS, unavailable))); // \
  macos-note{{'f_ios26_unavailable' has been explicitly marked unavailable here}} \
  driverkit-note{{'f_ios26_unavailable' has been explicitly marked unavailable here}}

void test(void) {
  f_introduced_26();

  // f_introduced_25 should be treated as if it was introduced in 26.0,
  // so it should be available at deployment target 27.0 with no warning.
  f_introduced_25();

  f_introduced_28(); // \
    macos-warning{{'f_introduced_28' is only available on macOS 28.0 or newer}} \
    macos-note{{enclose 'f_introduced_28' in a __builtin_available check to silence this warning}} \
    ios-warning{{'f_introduced_28' is only available on iOS 28.0 or newer}} \
    ios-note{{enclose 'f_introduced_28' in a __builtin_available check to silence this warning}} \
    tvos-warning{{'f_introduced_28' is only available on tvOS 28.0 or newer}} \
    tvos-note{{enclose 'f_introduced_28' in a __builtin_available check to silence this warning}} \
    watchos-warning{{'f_introduced_28' is only available on watchOS 28.0 or newer}} \
    watchos-note{{enclose 'f_introduced_28' in a __builtin_available check to silence this warning}} \
    xros-warning{{'f_introduced_28' is only available on visionOS 28.0 or newer}} \
    xros-note{{enclose 'f_introduced_28' in a __builtin_available check to silence this warning}} \
    maccatalyst-warning{{'f_introduced_28' is only available on macCatalyst 28.0 or newer}} \
    maccatalyst-note{{enclose 'f_introduced_28' in a __builtin_available check to silence this warning}} \
    driverkit-warning{{'f_introduced_28' is only available on DriverKit 28.0 or newer}} \
    driverkit-note{{enclose 'f_introduced_28' in a __builtin_available check to silence this warning}}

  f_deprecated_27(); // \
    macos-warning{{'f_deprecated_27' is deprecated: first deprecated in macOS 27.0}} \
    ios-warning{{'f_deprecated_27' is deprecated: first deprecated in iOS 27.0}} \
    tvos-warning{{'f_deprecated_27' is deprecated: first deprecated in tvOS 27.0}} \
    watchos-warning{{'f_deprecated_27' is deprecated: first deprecated in watchOS 27.0}} \
    xros-warning{{'f_deprecated_27' is deprecated: first deprecated in visionOS 27.0}} \
    maccatalyst-warning{{'f_deprecated_27' is deprecated: first deprecated in macCatalyst 27.0}} \
    driverkit-warning{{'f_deprecated_27' is deprecated: first deprecated in DriverKit 27.0}}

  f_deprecated_28();

  f_obsoleted_27(); // \
    macos-error{{'f_obsoleted_27' is unavailable: obsoleted in macOS 27.0}} \
    ios-error{{'f_obsoleted_27' is unavailable: obsoleted in iOS 27.0}} \
    tvos-error{{'f_obsoleted_27' is unavailable: obsoleted in tvOS 27.0}} \
    watchos-error{{'f_obsoleted_27' is unavailable: obsoleted in watchOS 27.0}} \
    xros-error{{'f_obsoleted_27' is unavailable: obsoleted in visionOS 27.0}} \
    maccatalyst-error{{'f_obsoleted_27' is unavailable: obsoleted in macCatalyst 27.0}} \
    driverkit-error{{'f_obsoleted_27' is unavailable: obsoleted in DriverKit 27.0}}

  f_obsoleted_28();

  f_unavailable(); // \
    macos-error{{'f_unavailable' is unavailable}} \
    ios-error{{'f_unavailable' is unavailable}} \
    tvos-error{{'f_unavailable' is unavailable}} \
    watchos-error{{'f_unavailable' is unavailable}} \
    xros-error{{'f_unavailable' is unavailable}} \
    maccatalyst-error{{'f_unavailable' is unavailable}} \
    driverkit-error{{'f_unavailable' is unavailable}}

  f_introduced_26_tvos28(); // \
    tvos-warning{{'f_introduced_26_tvos28' is only available on tvOS 28.0 or newer}} \
    tvos-note{{enclose 'f_introduced_26_tvos28' in a __builtin_available check to silence this warning}}

  f_introduced_tvos28_26(); // \
    tvos-warning{{'f_introduced_tvos28_26' is only available on tvOS 28.0 or newer}} \
    tvos-note{{enclose 'f_introduced_tvos28_26' in a __builtin_available check to silence this warning}}

  f_introduced_26_tvos29(); // \
    tvos-warning{{'f_introduced_26_tvos29' is only available on tvOS 29.0 or newer}} \
    tvos-note{{enclose 'f_introduced_26_tvos29' in a __builtin_available check to silence this warning}}

  f_introduced_26_tvos29_1(); // \
    tvos-warning{{'f_introduced_26_tvos29_1' is only available on tvOS 29.1 or newer}} \
    tvos-note{{enclose 'f_introduced_26_tvos29_1' in a __builtin_available check to silence this warning}}

  f_introduced_26_tvos29_2(); // \
    macos-warning{{'f_introduced_26_tvos29_2' is only available on macOS 29.2 or newer}} \
    macos-note{{enclose 'f_introduced_26_tvos29_2' in a __builtin_available check to silence this warning}} \
    ios-warning{{'f_introduced_26_tvos29_2' is only available on iOS 29.2 or newer}} \
    ios-note{{enclose 'f_introduced_26_tvos29_2' in a __builtin_available check to silence this warning}} \
    tvos-warning{{'f_introduced_26_tvos29_2' is only available on tvOS 29.2 or newer}} \
    tvos-note{{enclose 'f_introduced_26_tvos29_2' in a __builtin_available check to silence this warning}} \
    watchos-warning{{'f_introduced_26_tvos29_2' is only available on watchOS 29.2 or newer}} \
    watchos-note{{enclose 'f_introduced_26_tvos29_2' in a __builtin_available check to silence this warning}} \
    xros-warning{{'f_introduced_26_tvos29_2' is only available on visionOS 29.2 or newer}} \
    xros-note{{enclose 'f_introduced_26_tvos29_2' in a __builtin_available check to silence this warning}} \
    maccatalyst-warning{{'f_introduced_26_tvos29_2' is only available on macCatalyst 29.2 or newer}} \
    maccatalyst-note{{enclose 'f_introduced_26_tvos29_2' in a __builtin_available check to silence this warning}} \
    driverkit-warning{{'f_introduced_26_tvos29_2' is only available on DriverKit 29.2 or newer}} \
    driverkit-note{{enclose 'f_introduced_26_tvos29_2' in a __builtin_available check to silence this warning}}

  f_introduced_ios26_29(); // \
    macos-warning{{'f_introduced_ios26_29' is only available on macOS 29.0 or newer}} \
    macos-note{{enclose 'f_introduced_ios26_29' in a __builtin_available check to silence this warning}} \
    driverkit-warning{{'f_introduced_ios26_29' is only available on DriverKit 29.0 or newer}} \
    driverkit-note{{enclose 'f_introduced_ios26_29' in a __builtin_available check to silence this warning}}

  f_introduced_29_ios26(); // \
    macos-warning{{'f_introduced_29_ios26' is only available on macOS 29.0 or newer}} \
    macos-note{{enclose 'f_introduced_29_ios26' in a __builtin_available check to silence this warning}} \
    driverkit-warning{{'f_introduced_29_ios26' is only available on DriverKit 29.0 or newer}} \
    driverkit-note{{enclose 'f_introduced_29_ios26' in a __builtin_available check to silence this warning}}

  f_ios26_unavailable(); // \
    macos-error{{'f_ios26_unavailable' is unavailable: not available on macOS}} \
    driverkit-error{{'f_ios26_unavailable' is unavailable: not available on DriverKit}}
}
