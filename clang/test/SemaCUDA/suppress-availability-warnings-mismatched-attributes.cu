// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify=expected,onhost %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fsyntax-only -fcuda-is-device -verify=expected,ondevice %s

template <bool C, class T = void> struct my_enable_if {};

template <class T> struct my_enable_if<true, T> {
  typedef T type;
};

__attribute__((host, device)) void use(int x);

__attribute__((device)) constexpr int OverloadFunHostDepr(void) { return 1; }
__attribute__((host, deprecated("Host variant"))) constexpr int OverloadFunHostDepr(void) { return 1; } // expected-note 0+ {{has been explicitly marked deprecated here}}


__attribute__((device, deprecated("Device variant"))) constexpr int OverloadFunDeviceDepr(void) { return 1; } // expected-note 0+ {{has been explicitly marked deprecated here}}
__attribute__((host)) constexpr int OverloadFunDeviceDepr(void) { return 1; }


template<typename T>
__attribute__((device)) constexpr T TemplateOverloadFun(void) { return 1; }

template<typename T>
__attribute__((host, deprecated("Host variant"))) constexpr T TemplateOverloadFun(void) { return 1; } // expected-note 0+ {{has been explicitly marked deprecated here}}


__attribute__((device, deprecated)) constexpr int // expected-note 0+ {{has been explicitly marked deprecated here}}
DeviceOnlyFunDeprecated(void) { return 1; }

__attribute__((host, deprecated)) constexpr int // expected-note 0+ {{has been explicitly marked deprecated here}}
HostOnlyFunDeprecated(void) { return 1; }

class FunSelector {
public:
  template<int X> __attribute__((device))
  auto devicefun(void) -> typename my_enable_if<(X == OverloadFunHostDepr()), int>::type {
    return 1;
  }

  template<int X> __attribute__((device))
  auto devicefun(void) -> typename my_enable_if<(X != OverloadFunHostDepr()), int>::type {
      return 0;
  }

  template<int X> __attribute__((device))
  auto devicefun_wrong(void) -> typename my_enable_if<(X == OverloadFunDeviceDepr()), int>::type { // ondevice-warning {{'OverloadFunDeviceDepr' is deprecated: Device variant}}
    return 1;
  }

  template<int X> __attribute__((device))
  auto devicefun_wrong(void) -> typename my_enable_if<(X != OverloadFunDeviceDepr()), int>::type { // ondevice-warning {{'OverloadFunDeviceDepr' is deprecated: Device variant}}
      return 0;
  }

  template<int X> __attribute__((host))
  auto hostfun(void) -> typename my_enable_if<(X == OverloadFunDeviceDepr()), int>::type {
    return 1;
  }

  template<int X> __attribute__((host))
  auto hostfun(void) -> typename my_enable_if<(X != OverloadFunDeviceDepr()), int>::type {
      return 0;
  }

  template<int X> __attribute__((host))
  auto hostfun_wrong(void) -> typename my_enable_if<(X == OverloadFunHostDepr()), int>::type { // onhost-warning {{'OverloadFunHostDepr' is deprecated: Host variant}}
    return 1;
  }

  template<int X> __attribute__((host))
  auto hostfun_wrong(void) -> typename my_enable_if<(X != OverloadFunHostDepr()), int>::type { // onhost-warning {{'OverloadFunHostDepr' is deprecated: Host variant}}
      return 0;
  }
};


// These should not be diagnosed since the device overload of
// OverloadFunHostDepr is not deprecated:
__attribute__((device)) my_enable_if<(OverloadFunHostDepr() > 0), int>::type
DeviceUserOverloadFunHostDepr1(void) { return 2; }

my_enable_if<(OverloadFunHostDepr() > 0), int>::type __attribute__((device))
DeviceUserOverloadFunHostDepr2(void) { return 2; }

__attribute__((device))
my_enable_if<(OverloadFunHostDepr() > 0), int>::type constexpr
DeviceUserOverloadFunHostDeprConstexpr(void) { return 2; }


// Analogously for OverloadFunDeviceDepr:
__attribute__((host)) my_enable_if<(OverloadFunDeviceDepr() > 0), int>::type
DeviceUserOverloadFunDeviceDepr1(void) { return 2; }

my_enable_if<(OverloadFunDeviceDepr() > 0), int>::type __attribute__((host))
DeviceUserOverloadFunDeviceDepr2(void) { return 2; }

__attribute__((host))
my_enable_if<(OverloadFunDeviceDepr() > 0), int>::type constexpr
DeviceUserOverloadFunDeviceDeprConstexpr(void) { return 2; }


// Actual uses of the deprecated overloads should be diagnosed:
__attribute__((host, device)) my_enable_if<(OverloadFunHostDepr() > 0), int>::type // onhost-warning {{'OverloadFunHostDepr' is deprecated: Host variant}}
HostDeviceUserOverloadFunHostDepr(void) { return 3; }

__attribute__((host)) my_enable_if<(OverloadFunHostDepr() > 0), int>::type constexpr // onhost-warning {{'OverloadFunHostDepr' is deprecated: Host variant}}
HostUserOverloadFunHostDeprConstexpr(void) { return 3; }

__attribute__((device)) my_enable_if<(OverloadFunDeviceDepr() > 0), int>::type constexpr // ondevice-warning {{'OverloadFunDeviceDepr' is deprecated: Device variant}}
HostUserOverloadFunDeviceDeprConstexpr(void) { return 3; }


// Making the offending decl a template shouldn't change anything:
__attribute__((host)) my_enable_if<(TemplateOverloadFun<int>() > 0), int>::type // onhost-warning {{'TemplateOverloadFun<int>' is deprecated: Host variant}}
HostUserTemplateOverloadFun(void) { return 3; }

__attribute__((device)) my_enable_if<(TemplateOverloadFun<int>() > 0), int>::type
DeviceUserTemplateOverloadFun(void) { return 3; }


// If the constexpr function is actually called from the mismatched context, diagnostics should be issued:
__attribute__((host))
my_enable_if<(DeviceOnlyFunDeprecated() > 0), int>::type constexpr // onhost-warning {{'DeviceOnlyFunDeprecated' is deprecated}}
HostUserDeviceOnlyFunDeprecated(void) { return 3; }

__attribute__((device))
my_enable_if<(HostOnlyFunDeprecated() > 0), int>::type constexpr // ondevice-warning {{'HostOnlyFunDeprecated' is deprecated}}
DeviceUserHostOnlyFunDeprecated(void) { return 3; }

// Diagnostics for uses in function bodies should work as expected:
__attribute__((device, deprecated)) constexpr int DeviceVarConstDepr = 1; // expected-note 0+ {{has been explicitly marked deprecated here}}

__attribute__((host)) void HostUser(void) {
  use(DeviceVarConstDepr); // expected-warning {{'DeviceVarConstDepr' is deprecated}}
  use(HostOnlyFunDeprecated()); // expected-warning {{'HostOnlyFunDeprecated' is deprecated}}
  use(OverloadFunHostDepr()); // expected-warning {{'OverloadFunHostDepr' is deprecated: Host variant}}
  use(TemplateOverloadFun<int>()); // expected-warning {{'TemplateOverloadFun<int>' is deprecated: Host variant}}

  use(OverloadFunDeviceDepr());
}

__attribute__((device)) void DeviceUser(void) {
  use(DeviceVarConstDepr); // expected-warning {{'DeviceVarConstDepr' is deprecated}}
  use(DeviceOnlyFunDeprecated()); // expected-warning {{'DeviceOnlyFunDeprecated' is deprecated}}
  use(OverloadFunDeviceDepr()); // expected-warning {{'OverloadFunDeviceDepr' is deprecated: Device variant}}

  use(OverloadFunHostDepr());
  use(TemplateOverloadFun<int>());
}
