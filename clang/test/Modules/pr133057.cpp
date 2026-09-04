// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -xc++ -std=c++20 -emit-module -fmodule-name=hf -fno-cxx-modules -fmodules -fno-implicit-modules %t/CMO.cppmap -o %t/WI9.pcm
// RUN: %clang_cc1 -xc++ -std=c++20 -emit-module -fmodule-name=g -fno-cxx-modules -fmodules -fno-implicit-modules -fmodule-file=%t/WI9.pcm %t/E6H.cppmap -o %t/4BK.pcm
// RUN: %clang_cc1 -xc++ -std=c++20 -emit-module -fmodule-name=r -fno-cxx-modules -fmodules -fno-implicit-modules -fmodule-file=%t/WI9.pcm %t/HMT.cppmap -o %t/LUM.pcm
// RUN: %clang_cc1 -xc++ -std=c++20 -emit-module -fmodule-name=q -fno-cxx-modules -fmodules -fno-implicit-modules -fmodule-file=%t/LUM.pcm -fmodule-file=%t/4BK.pcm %t/JOV.cppmap -o %t/9VX.pcm
// RUN: %clang_cc1 -xc++ -std=c++20 -verify -fsyntax-only -fno-cxx-modules -fmodules -fno-implicit-modules -fmodule-file=%t/9VX.pcm %t/XFD.cc

//--- 2OT.h
#include "LQ1.h"

namespace ciy {
namespace xqk {
template <typename>
class vum {
 public:
  using sc = std::C::wmd;
  friend bool operator==(vum, vum);
};
template <typename>
class me {
 public:
  using vbh = vum<me>;
  using sc = std::C::vy<vbh>::sc;
  template <typename db>
  operator db() { return {}; }
};
}  // namespace xqk
template <typename vus>
xqk::me<vus> uvo(std::C::wmd, vus);
}  // namespace ciy

class ua {
  std::C::wmd kij() {
    ciy::uvo(kij(), '-');
    return {};
  }
};

//--- 9KF.h
#include "LQ1.h"
#include "2OT.h"
namespace {
void al(std::C::wmd lou) { std::C::jv<std::C::wmd> yt = ciy::uvo(lou, '/'); }
}  // namespace

//--- CMO.cppmap
module "hf" {
header "LQ1.h"
}


//--- E6H.cppmap
module "g" {
export *
header "2OT.h"
}


//--- HMT.cppmap
module "r" {
header "2OT.h"
}


//--- JOV.cppmap
module "q" {
header "9KF.h"
}


//--- LQ1.h
namespace std {
namespace C {
template <class zd>
struct vy : zd {};
template <class ub>
struct vy<ub*> {
  typedef ub jz;
};
struct wmd {};
template <class uo, class zt>
void sk(uo k, zt gf) {
  (void)(k != gf);
}
template <class uo>
class fm {
 public:
  fm(uo);
};
template <class kj, class kju>
bool operator==(kj, kju);
template <class epn>
void afm(epn) {
  using yp = vy<epn>;
  if (__is_trivially_copyable(yp)) {
    sk(fm(epn()), nullptr);
  }
}
template <class ub>
class jv {
 public:
  constexpr void gq();
  ub *nef;
};
template <class ub>
constexpr void jv<ub>::gq() {
    afm(nef);
}
}  // namespace C
}  // namespace std
namespace ciy {
}  // namespace ciy

//--- XFD.cc
// expected-no-diagnostics
#include "LQ1.h"
#include "2OT.h"
class wiy {
 public:
  std::C::wmd eyb();
};
template <typename wpa>
void i(wpa fg) {
  std::C::jv<std::C::wmd> zs;
  zs = ciy::uvo(fg.eyb(), '\n');
}
namespace ciy {
namespace xqk {
struct sbv;
std::C::jv<sbv> ns() {
  std::C::jv<sbv> ubs;
  ubs.gq();
  return ubs;
}
}  // namespace xqk
}  // namespace ciy
void s() {
  wiy fg;
  i(fg);
}
