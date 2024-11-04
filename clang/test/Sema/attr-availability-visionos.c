// RUN: %clang_cc1 -triple arm64-apple-xros1 -fapplication-extension -verify=visionos %s 2>&1

__attribute__((availability(xros, unavailable))) // visionos-warning {{unknown platform 'xros' in availability macro}}
void xros_unavail(); // visionos-note {{}}

__attribute__((availability(xros_app_extension, unavailable))) // visionos-warning {{unknown platform 'xros_app_extension' in availability macro}}
void xros_ext_unavail(); // visionos-note {{}}

__attribute__((availability(visionOSApplicationExtension, unavailable)))
void visionos_ext_unavail(); // visionos-note {{}}

void use() {
  xros_unavail(); // visionos-error {{'xros_unavail' is unavailable: not available on visionOS}}
  xros_ext_unavail(); // visionos-error {{'xros_ext_unavail' is unavailable: not available on visionOS}}
  visionos_ext_unavail(); // visionos-error {{'visionos_ext_unavail' is unavailable: not available on visionOS}}
}

__attribute__((availability(visionOS, introduced=1.0)))
void visionos_introduced_1();

__attribute__((availability(visionos, introduced=1.1)))
void visionos_introduced_1_1(); // visionos-note 4 {{'visionos_introduced_1_1' has been marked as being introduced in visionOS 1.1 here, but the deployment target is visionOS 1}}

void use2() {
  if (__builtin_available(iOS 16.1, *))
    visionos_introduced_1_1(); // visionos-warning {{'visionos_introduced_1_1' is only available on visionOS 1.1 or newer}} visionos-note {{enclose}}
                              
  if (__builtin_available(xrOS 1.1, *)) // visionos-error {{unrecognized platform name xrOS}}
    visionos_introduced_1_1(); // visionos-warning {{'visionos_introduced_1_1' is only available on visionOS 1.1 or newer}} visionos-note {{enclose}}
  
  if (__builtin_available(xros_app_extension 1, *)) // visionos-error {{unrecognized platform name xros_app_extension}}
    visionos_introduced_1_1(); // visionos-warning {{'visionos_introduced_1_1' is only available on visionOS 1.1 or newer}} visionos-note {{enclose}}

  if (__builtin_available(visionOS 1.1, *))
    visionos_introduced_1_1();

  visionos_introduced_1();
  visionos_introduced_1_1(); // visionos-warning {{'visionos_introduced_1_1' is only available on visionOS 1.1 or newer}} visionos-note {{enclose}}
}
