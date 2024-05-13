// RUN: %clang_cc1 -no-enable-noundef-analysis -triple i686-windows-msvc -emit-llvm -std=c++1y -fno-threadsafe-statics -fms-extensions -O0 -disable-llvm-passes -o - %s | FileCheck --check-prefix=MSC %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-scei-ps4 -emit-llvm -std=c++1y -fno-threadsafe-statics -fms-extensions -O0 -o - %s | FileCheck --check-prefix=PS %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-sie-ps5 -emit-llvm -std=c++1y -fno-threadsafe-statics -fms-extensions -O0 -o - %s | FileCheck --check-prefix=PS %s

template <typename T> struct __declspec(dllimport) ImportedClassTemplate { void func(); };

// Make sure that we do not import classes with unique external linkage.
// Note that MSVC does indeed expect the called function to be defined elsewhere.
void func1() {
  class LocalCRTP : public ImportedClassTemplate<LocalCRTP> {};
  LocalCRTP lc;
  lc.func();
}

namespace {
  class AnonNSCRTP : public ImportedClassTemplate<AnonNSCRTP> {};
  AnonNSCRTP ac;
}

void func2() {
  ac.func();
}

// MSC-NOT: declare {{.*}}dllimport
// MSC:     call {{.*}}@"?func@?$ImportedClassTemplate@VLocalCRTP@?1??func1{{.*}}"
// MSC-NOT: declare {{.*}}dllimport
// MSC:     call {{.*}}@"?func@?$ImportedClassTemplate@VAnonNSCRTP@?{{.*}}"
// MSC-NOT: declare {{.*}}dllimport

// PS-NOT:  declare {{.*}}dllimport
// PS:      call {{.*}}@_ZN21ImportedClassTemplateIZ5func1vE9LocalCRTPE4funcEv
// PS-NOT:  declare {{.*}}dllimport
// PS:      call {{.*}}@_ZN21ImportedClassTemplateIN12_GLOBAL__N_110AnonNSCRTPEE4funcEv
// PS-NOT:  declare {{.*}}dllimport
