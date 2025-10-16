// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple x86_64-apple-darwin %s -o - | FileCheck %s
// PR11345

class locale {
private:
  void _M_add_reference() const throw() {
  }
};
class ios_base {
  locale _M_ios_locale;
public:
  class Init {
  };
};
static ios_base::Init __ioinit;

// CHECK-NOT: _M_ios_locale
