// RUN: rm -rf %t && mkdir %t
// RUN: mkdir -p %t/ctudir
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++17 \
// RUN:   -emit-pch -o %t/ctudir/ctu-cxxdefaultinitexpr-import.cpp.ast %S/Inputs/ctu-cxxdefaultinitexpr-import.cpp
// RUN: cp %S/Inputs/ctu-cxxdefaultinitexpr-import.cpp.externalDefMap.ast-dump.txt %t/ctudir/externalDefMap.txt
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c++17 -analyze \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -verify %s

// Check that importing this code does not cause crash.
// expected-no-diagnostics

namespace QHashPrivate {
template <typename> int b;
struct Data;
} // namespace QHashPrivate

struct QDomNodePrivate {};
template <typename = struct QString> struct QMultiHash {
  QHashPrivate::Data *d = nullptr;
};

struct QDomNamedNodeMapPrivate {
  QMultiHash<> map;
};
struct QDomElementPrivate : QDomNodePrivate {
  QDomElementPrivate();
  void importee();
  QMultiHash<> *m_attr = nullptr;
};
// --------- common part end ---------

void importer(QDomElementPrivate x) { x.importee(); }
