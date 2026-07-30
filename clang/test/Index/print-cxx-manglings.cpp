// REQUIRES: x86-registered-target

// RUN: c-index-test -write-pch %t.itanium.ast -target i686-pc-linux-gnu -fdeclspec %s
// RUN: c-index-test -test-print-manglings %t.itanium.ast | FileCheck --check-prefix=ITANIUM %s

// RUN: c-index-test -write-pch %t.macho.ast -target i686-apple-darwin -fdeclspec %s
// RUN: c-index-test -test-print-manglings %t.macho.ast | FileCheck --check-prefix=MACHO %s

// RUN: c-index-test -write-pch %t.msvc.ast -target i686-pc-windows %s
// RUN: c-index-test -test-print-manglings %t.msvc.ast | FileCheck --check-prefix=MSVC %s

struct s {
  s(int);
  ~s();
  int m(int);
};

// ITANIUM: CXXConstructor=s{{.*}}[mangled=_ZN1sC2Ei] [mangled=_ZN1sC1Ei]
// ITANIUM: CXXDestructor=~s{{.*}}[mangled=_ZN1sD2Ev] [mangled=_ZN1sD1Ev]

// MACHO: CXXConstructor=s{{.*}}[mangled=__ZN1sC2Ei] [mangled=__ZN1sC1Ei]
// MACHO: CXXDestructor=~s{{.*}}[mangled=__ZN1sD2Ev] [mangled=__ZN1sD1Ev]

// MSVC: CXXConstructor=s{{.*}}[mangled=??0s@@QAE@H@Z]
// MSVC: CXXDestructor=~s{{.*}}[mangled=??1s@@QAE@XZ]

struct t {
  t(int);
  virtual ~t();
  int m(int);
};

// ITANIUM: CXXConstructor=t{{.*}}[mangled=_ZN1tC2Ei] [mangled=_ZN1tC1Ei]
// ITANIUM: CXXDestructor=~t{{.*}}[mangled=_ZN1tD2Ev] [mangled=_ZN1tD1Ev] [mangled=_ZN1tD0Ev]

// MACHO: CXXConstructor=t{{.*}}[mangled=__ZN1tC2Ei] [mangled=__ZN1tC1Ei]
// MACHO: CXXDestructor=~t{{.*}}[mangled=__ZN1tD2Ev] [mangled=__ZN1tD1Ev] [mangled=__ZN1tD0Ev]

// MSVC: CXXConstructor=t{{.*}}[mangled=??0t@@QAE@H@Z]
// MSVC: CXXDestructor=~t{{.*}}[mangled=??1t@@UAE@XZ]

struct u {
  u();
  virtual ~u();
  virtual int m(int) = 0;
};

// ITANIUM: CXXConstructor=u{{.*}}[mangled=_ZN1uC2Ev]
// ITANIUM: CXXDestructor=~u{{.*}}[mangled=_ZN1uD2Ev] [mangled=_ZN1uD1Ev] [mangled=_ZN1uD0Ev]

// MACHO: CXXConstructor=u{{.*}}[mangled=__ZN1uC2Ev]
// MACHO: CXXDestructor=~u{{.*}}[mangled=__ZN1uD2Ev] [mangled=__ZN1uD1Ev] [mangled=__ZN1uD0Ev]

// MSVC: CXXConstructor=u{{.*}}[mangled=??0u@@QAE@XZ]
// MSVC: CXXDestructor=~u{{.*}}[mangled=??1u@@UAE@XZ]

struct v {
  __declspec(dllexport) v(int = 0);
};

// ITANIUM: CXXConstructor=v{{.*}}[mangled=_ZN1vC2Ei] [mangled=_ZN1vC1Ei]

// MACHO: CXXConstructor=v{{.*}}[mangled=__ZN1vC2Ei] [mangled=__ZN1vC1Ei]

// MSVC: CXXConstructor=v{{.*}}[mangled=??0v@@QAE@H@Z] [mangled=??_Fv@@QAEXXZ]

struct w {
  virtual int m(int);
};

// ITANIUM: CXXMethod=m{{.*}} (virtual) [mangled=_ZN1w1mEi]

// MACHO: CXXMethod=m{{.*}} (virtual) [mangled=__ZN1w1mEi]

// MSVC: CXXMethod=m{{.*}} (virtual) [mangled=?m@w@@UAEHH@Z]

struct x {
  virtual int m(int);
};

// ITANIUM: CXXMethod=m{{.*}} (virtual) [mangled=_ZN1x1mEi]

// MACHO: CXXMethod=m{{.*}} (virtual) [mangled=__ZN1x1mEi]

// MSVC: CXXMethod=m{{.*}} (virtual) [mangled=?m@x@@UAEHH@Z]

struct y : w, x {
  virtual int m(int);
};

// ITANIUM: CXXMethod=m{{.*}} (virtual) {{.*}} [mangled=_ZN1y1mEi] [mangled=_ZThn4_N1y1mEi]

// MACHO: CXXMethod=m{{.*}} (virtual) {{.*}} [mangled=__ZN1y1mEi] [mangled=__ZThn4_N1y1mEi]

// MSVC: CXXMethod=m{{.*}} (virtual) {{.*}} [mangled=?m@y@@UAEHH@Z] [mangled=?m@y@@W3AEHH@Z]

