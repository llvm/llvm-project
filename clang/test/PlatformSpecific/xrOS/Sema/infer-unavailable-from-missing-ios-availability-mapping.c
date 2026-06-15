// RUN: %clang_cc1 -triple arm64-apple-xros1 -verify=ios -DNOSDK %s 2>&1
// RUN: %clang_cc1 -triple arm64-apple-xros1 -verify=ios -isysroot %S/Inputs/XROS1.0.sdk %s 2>&1

#ifdef NOSDK
// ios-warning@+2 {{ios availability is ignored without a valid 'SDKSettings.json' in the SDK}}
#endif
__attribute__((availability(ios, introduced=18))) // note the version introduced has to be higher than the versions in SDKSettings
void ios_introduced_10(); // ios-note {{}}

void useIntroduced() {
  ios_introduced_10(); // ios-error {{is unavailable: not available on }}
}
