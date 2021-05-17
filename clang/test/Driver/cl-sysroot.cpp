// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cl /winsysroot %t -### -- %t/foo.cpp 2>&1 | FileCheck %s
// RUN: %clang_cl /vctoolsdir %t/VC/Tools/MSVC/27.1828.18284 \
// RUN:           /winsdkdir "%t/Windows Kits/10" \
// RUN:           -### -- %t/foo.cpp 2>&1 | FileCheck %s

// CHECK: "-internal-isystem" "[[ROOT:[^"]*]]{{/|\\\\}}VC{{/|\\\\}}Tools{{/|\\\\}}MSVC{{/|\\\\}}27.1828.18284{{/|\\\\}}include"
// CHECK: "-internal-isystem" "[[ROOT]]{{/|\\\\}}VC{{/|\\\\}}Tools{{/|\\\\}}MSVC{{/|\\\\}}27.1828.18284{{/|\\\\}}atlmfc{{/|\\\\}}include"
// CHECK: "-internal-isystem" "[[ROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Include{{/|\\\\}}10.0.19041.0{{/|\\\\}}ucrt"
// CHECK: "-internal-isystem" "[[ROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Include{{/|\\\\}}10.0.19041.0{{/|\\\\}}shared"
// CHECK: "-internal-isystem" "[[ROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Include{{/|\\\\}}10.0.19041.0{{/|\\\\}}um"
// CHECK: "-internal-isystem" "[[ROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Include{{/|\\\\}}10.0.19041.0{{/|\\\\}}winrt"

#--- VC/Tools/MSVC/27.1828.18284/include/string
namespace std {
class mystring {
public:
  bool empty();
};
}

#--- Windows Kits/10/Include/10.0.19041.0/ucrt/assert.h
#define myassert(X)

#--- foo.cpp
#include <assert.h>
#include <string>

void f() {
  std::mystring s;
  myassert(s.empty());
}
