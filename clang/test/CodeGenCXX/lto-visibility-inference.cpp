// RUN: %clang_cc1 -flto -triple x86_64-unknown-linux -std=c++11 -fms-extensions -fvisibility hidden -fwhole-program-vtables -emit-llvm -o - %s | FileCheck --check-prefix=ITANIUM %s
// RUN: %clang_cc1 -flto -triple x86_64-pc-windows-msvc -std=c++11 -fms-extensions -fwhole-program-vtables -emit-llvm -o - %s | FileCheck --check-prefix=MS --check-prefix=MS-STD %s
// RUN: %clang_cc1 -flto -triple x86_64-pc-windows-msvc -std=c++11 -fms-extensions -fwhole-program-vtables -flto-visibility-public-std -emit-llvm -o - %s | FileCheck --check-prefix=MS --check-prefix=MS-NOSTD %s

struct C1 {
  virtual void f();
};

struct __attribute__((visibility("default"))) C2 {
  virtual void f();
};

struct __declspec(dllexport) C3 {
  virtual void f();
};

struct __declspec(dllimport) C4 {
  virtual void f();
};

struct [[clang::lto_visibility_public]] C5 {
  virtual void f();
};

struct __declspec(uuid("00000000-0000-0000-0000-000000000000")) C6 {
  virtual void f();
};

namespace std {

struct C7 {
  virtual void f();
  struct C8 {
    virtual void f();
  };
};

}

extern "C++" {

namespace stdext {

struct C9 {
  virtual void f();
};

}

}

namespace other {

struct C10 {
  virtual void f();
};

}

namespace {

struct C11 {
  virtual void f();
};

}

void f(C1 *c1, C2 *c2, C3 *c3, C4 *c4, C5 *c5, C6 *c6, std::C7 *c7,
       std::C7::C8 *c8, stdext::C9 *c9, other::C10 *c10) {
  // ITANIUM: type.test{{.*}}!"_ZTS2C1"
  // MS: type.test{{.*}}!"?AUC1@@"
  c1->f();
  // ITANIUM: type.test{{.*}}!"_ZTS2C2"
  // MS: type.test{{.*}}!"?AUC2@@"
  c2->f();
  // ITANIUM: type.test{{.*}}!"_ZTS2C3"
  // MS: type.test{{.*}}!"?AUC3@@"
  c3->f();
  // ITANIUM: type.test{{.*}}!"_ZTS2C4"
  // MS: type.test{{.*}}!"?AUC4@@"
  c4->f();
  // ITANIUM: type.test{{.*}}!"_ZTS2C5"
  // MS: type.test{{.*}}!"?AUC5@@"
  c5->f();
  // ITANIUM: type.test{{.*}}!"_ZTS2C6"
  // MS: type.test{{.*}}!"?AUC6@@"
  c6->f();
  // ITANIUM: type.test{{.*}}!"_ZTSSt2C7"
  // MS-STD: type.test{{.*}}!"?AUC7@std@@"
  // MS-NOSTD-NOT: type.test{{.*}}!"?AUC7@std@@"
  c7->f();
  // ITANIUM: type.test{{.*}}!"_ZTSNSt2C72C8E"
  // MS-STD: type.test{{.*}}!"?AUC8@C7@std@@"
  // MS-NOSTD-NOT: type.test{{.*}}!"?AUC8@C7@std@@"
  c8->f();
  // ITANIUM: type.test{{.*}}!"_ZTSN6stdext2C9E"
  // MS-STD: type.test{{.*}}!"?AUC9@stdext@@"
  // MS-NOSTD-NOT: type.test{{.*}}!"?AUC9@stdext@@"
  c9->f();
  // ITANIUM: type.test{{.*}}!"_ZTSN5other3C10E"
  // MS: type.test{{.*}}!"?AUC10@other@@"
  c10->f();
  // ITANIUM: type.test{{.*}}!{{[0-9]}}
  // MS: type.test{{.*}}!{{[0-9]}}
  C11 *c11;
  c11->f();
}
