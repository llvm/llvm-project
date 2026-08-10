// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

// When looking up a namespace-name in a using-directive or
// namespace-alias-definition, only namespace names are considered.

struct ns1 {};
void ns2();
int ns3 = 0;

namespace ns0 {
  namespace ns1 {
    struct test0 {};
  }
  namespace ns2 {
    struct test1 {};
  }
  namespace ns3 {
    struct test2 {};
  }
}

using namespace ns0;

namespace test3 = ns1;
namespace test4 = ns2;
namespace test5 = ns3;

using namespace ns1;
using namespace ns2;
using namespace ns3;

test0 a;
test1 b;
test2 c;

