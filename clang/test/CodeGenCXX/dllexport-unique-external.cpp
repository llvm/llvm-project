// RUN: %clang_cc1 -no-enable-noundef-analysis -triple i686-windows-msvc -emit-llvm -std=c++1y -fno-threadsafe-statics -fms-extensions -O0 -disable-llvm-passes -o - %s | FileCheck --check-prefix=MSC %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-scei-ps4 -emit-llvm -std=c++1y -fno-threadsafe-statics -fms-extensions -O0 -o - %s | FileCheck --check-prefix=PS %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-sie-ps5 -emit-llvm -std=c++1y -fno-threadsafe-statics -fms-extensions -O0 -o - %s | FileCheck --check-prefix=PS %s

template <typename T> struct __declspec(dllexport) ExportedClassTemplate { void func(); };

// Make sure that we do not export classes with unique external linkage.
// Note that MSVC does indeed export the symbols in the MSC check string.
void func1() {
  class LocalCRTP : public ExportedClassTemplate<LocalCRTP> {};
  LocalCRTP lc;
  lc.func();
}

namespace {
  class AnonNSCRTP : public ExportedClassTemplate<AnonNSCRTP> {};
  AnonNSCRTP ac;
}

void func2() {
  ac.func();
}

// MSC-NOT: declare {{.*}}dllexport
// MSC:     call {{.*}}@"?func@?$ExportedClassTemplate@VLocalCRTP@?1??func1@@{{.*}}"
// MSC-NOT: declare {{.*}}dllexport
// MSC:     call {{.*}}@"?func@?$ExportedClassTemplate@VAnonNSCRTP@?{{.*}}"
// MSC-NOT: declare {{.*}}dllexport

// PS-NOT:  declare {{.*}}dllexport
// PS:      call {{.*}}@_ZN21ExportedClassTemplateIZ5func1vE9LocalCRTPE4funcEv
// PS-NOT:  declare {{.*}}dllexport
// PS:      call {{.*}}@_ZN21ExportedClassTemplateIN12_GLOBAL__N_110AnonNSCRTPEE4funcEv
// PS-NOT:  declare {{.*}}dllexport
