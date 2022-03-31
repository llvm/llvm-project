// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cl --target=x86_64-windows \
// RUN:     /winsysroot %t \
// RUN:     -### -- %t/foo.cpp 2>&1 | FileCheck %s
// RUN: %clang_cl --target=x86_64-windows \
// RUN:     /diasdkdir "%t/DIA SDK" \
// RUN:     /vctoolsdir %t/VC/Tools/MSVC/27.1828.18284 \
// RUN:     /winsdkdir "%t/Windows Kits/10" \
// RUN:     -### -- %t/foo.cpp 2>&1 | FileCheck %s

// CHECK: "-internal-isystem" "[[ROOT:[^"]*]]{{/|\\\\}}DIA SDK{{/|\\\\}}include"
// CHECK: "-internal-isystem" "[[ROOT]]{{/|\\\\}}VC{{/|\\\\}}Tools{{/|\\\\}}MSVC{{/|\\\\}}27.1828.18284{{/|\\\\}}include"
// CHECK: "-internal-isystem" "[[ROOT]]{{/|\\\\}}VC{{/|\\\\}}Tools{{/|\\\\}}MSVC{{/|\\\\}}27.1828.18284{{/|\\\\}}atlmfc{{/|\\\\}}include"
// CHECK: "-internal-isystem" "[[ROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Include{{/|\\\\}}10.0.19041.0{{/|\\\\}}ucrt"
// CHECK: "-internal-isystem" "[[ROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Include{{/|\\\\}}10.0.19041.0{{/|\\\\}}shared"
// CHECK: "-internal-isystem" "[[ROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Include{{/|\\\\}}10.0.19041.0{{/|\\\\}}um"
// CHECK: "-internal-isystem" "[[ROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Include{{/|\\\\}}10.0.19041.0{{/|\\\\}}winrt"
// CHECK: "-internal-isystem" "[[ROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Include{{/|\\\\}}10.0.19041.0{{/|\\\\}}cppwinrt"

// CHECK: "-libpath:[[ROOT]]{{/|\\\\}}DIA SDK{{/|\\\\}}lib{{/|\\\\}}amd64"
// CHECK: "-libpath:[[ROOT]]{{/|\\\\}}VC{{/|\\\\}}Tools{{/|\\\\}}MSVC{{/|\\\\}}27.1828.18284{{/|\\\\}}lib{{/|\\\\}}x64"
// CHECK: "-libpath:[[ROOT]]{{/|\\\\}}VC{{/|\\\\}}Tools{{/|\\\\}}MSVC{{/|\\\\}}27.1828.18284{{/|\\\\}}atlmfc{{/|\\\\}}lib{{/|\\\\}}x64"
// CHECK: "-libpath:[[ROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Lib{{/|\\\\}}10.0.19041.0{{/|\\\\}}ucrt{{/|\\\\}}x64"
// CHECK: "-libpath:[[ROOT]]{{/|\\\\}}Windows Kits{{/|\\\\}}10{{/|\\\\}}Lib{{/|\\\\}}10.0.19041.0{{/|\\\\}}um{{/|\\\\}}x64"

#--- VC/Tools/MSVC/27.1828.18284/include/string
namespace std {
class mystring {
public:
  bool empty();
};
}

#--- Windows Kits/10/Include/10.0.19041.0/ucrt/assert.h
#define myassert(X)

#--- DIA SDK/include/cvconst.h
#define myotherassert(X)

#--- foo.cpp
#include <assert.h>
#include <string>

void f() {
  std::mystring s;
  myassert(s.empty());
}
