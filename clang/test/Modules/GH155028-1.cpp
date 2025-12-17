// RUN: %clang_cc1 -std=c++20 -verify %s
// expected-no-diagnostics

#pragma clang module build M
module "M" {
  module "A" {}
  module "B" {}
}
#pragma clang module contents
#pragma clang module begin M.A
enum E1 {};
#pragma clang module end
#pragma clang module begin M.B
enum E1 {};
using T = __underlying_type(E1);
#pragma clang module end
#pragma clang module endbuild
