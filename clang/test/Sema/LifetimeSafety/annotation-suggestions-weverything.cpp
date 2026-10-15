// The lifetime annotation suggestions are not enabled by -Weverything alone.
//
// RUN: %clang_cc1 -fsyntax-only -std=c++17 -flifetime-safety-inference \
// RUN:   -fexperimental-lifetime-safety-tu-analysis -Weverything %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=NONE --implicit-check-not="should be marked"
//
// RUN: %clang_cc1 -fsyntax-only -std=c++17 -flifetime-safety-inference \
// RUN:   -fexperimental-lifetime-safety-tu-analysis -Weverything \
// RUN:   -Wlifetime-safety-suggestions %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=PARAM,CTOR \
// RUN:   --implicit-check-not="should be marked"
//
// RUN: %clang_cc1 -fsyntax-only -std=c++17 -flifetime-safety-inference \
// RUN:   -fexperimental-lifetime-safety-tu-analysis \
// RUN:   -Wlifetime-safety-suggestions -Weverything %s 2>&1 | \
// RUN:   FileCheck %s --check-prefixes=PARAM,CTOR \
// RUN:   --implicit-check-not="should be marked"
//
// Enabling only a subgroup must not let -Weverything enable the rest.
// RUN: %clang_cc1 -fsyntax-only -std=c++17 -flifetime-safety-inference \
// RUN:   -fexperimental-lifetime-safety-tu-analysis -Weverything \
// RUN:   -Wlifetime-safety-intra-tu-constructor-suggestions %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=CTOR --implicit-check-not="should be marked"
//
// RUN: not %clang_cc1 -fsyntax-only -std=c++17 -flifetime-safety-inference \
// RUN:   -fexperimental-lifetime-safety-tu-analysis -Weverything \
// RUN:   -Werror=lifetime-safety-intra-tu-suggestions %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERROR --implicit-check-not="should be marked"

// NONE-NOT: should be marked

struct View;

struct [[gsl::Owner]] MyObj {
  int id;
  MyObj();
  ~MyObj();
};

struct [[gsl::Pointer()]] View {
  View(const MyObj &);
  View();
};

View return_view(View a) {
  // PARAM: :[[@LINE-1]]:24: warning: parameter in intra-TU function should be marked {{\[\[}}clang::lifetimebound]] [-Wlifetime-safety-intra-tu-suggestions]
  // ERROR: :[[@LINE-2]]:24: error: parameter in intra-TU function should be marked {{\[\[}}clang::lifetimebound]] [-Werror,-Wlifetime-safety-intra-tu-suggestions]
  return a;
}

struct CaptureRefToView {
  View v;
  CaptureRefToView(const MyObj &obj) : v(obj) {}
  // CTOR: :[[@LINE-1]]:36: warning: parameter in intra-TU constructor should be marked {{\[\[}}clang::lifetimebound]] [-Wlifetime-safety-intra-tu-constructor-suggestions]
  // ERROR: :[[@LINE-2]]:36: error: parameter in intra-TU constructor should be marked {{\[\[}}clang::lifetimebound]] [-Werror,-Wlifetime-safety-intra-tu-constructor-suggestions]
};
