// RUN: %check_clang_tidy -std=c++20-or-later %s misc-unused-using-decls %t -- --fix-notes

module;

namespace n {

struct S {};

} // namespace n

using n::S; // n::S
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: using decl 'S' is unused
// CHECK-FIXES: // n::S

export module foo;

struct A {};
export struct B {};
export struct C {};
export struct D {};

namespace ns1 {

using ::A; // ns1::A
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: using decl 'A' is unused
// CHECK-FIXES: // ns1::A

// If the decl isn't exported, it's unused, even if the underlying struct is exported.
using ::B; // ns1::B
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: using decl 'B' is unused
// CHECK-FIXES: // ns1::B

export using ::C;

export {

using ::D;

}

} // namespace n

export namespace ns2 {

using ::B;

} // namespace ns2

export {

namespace ns3 {

using ::B;

} // namespace ns3

}

export namespace ns3 {

struct E {};

} // namespace ns3

module :private;

using ns3::E; // ns3::E
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: using decl 'E' is unused
// CHECK-FIXES: // ns3::E
