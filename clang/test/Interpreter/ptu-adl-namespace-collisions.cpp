// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl | FileCheck %s
//
// ADL / namespace / name-collision PTU rollback regression tests. As in
// the class-focused file, each failing chunk pairs a genuine namespace- or
// lookup-related Sema error (ambiguous lookup, hidden-friend/ADL misuse,
// namespace-alias collision, conflicting using-declaration, ...) with a
// legitimate new namespace member on the same line, so a correct rollback
// must discard both together.

extern "C" int printf(const char *, ...);

// 1. Hidden friend found only via ADL, cf. SemaCXX/adl.cpp. combine() is
// injected into its enclosing namespace but invisible to ordinary
// unqualified lookup; calling it with non-class arguments is a genuine
// "use of undeclared identifier" error, independent of any using-directive.
namespace AdlNs1 { struct Widget1 { Widget1(int) {} friend int combine(Widget1, Widget1) { return 100; } }; }
auto radl1 = printf("combine1=%d\n", combine(AdlNs1::Widget1(1), AdlNs1::Widget1(2)));
// CHECK: combine1=100

namespace AdlNs2 { struct Widget2 { Widget2(int) {} friend int combine(Widget2, Widget2) { return 200; } }; } int ambiguous_call = combine(1, 2);
auto radl2 = printf("combine1_again=%d\n", combine(AdlNs1::Widget1(3), AdlNs1::Widget1(4)));
// CHECK: combine1_again=100
namespace AdlNs2 { struct Widget2 { Widget2(int) {} friend int combine(Widget2, Widget2) { return 200; } }; }
auto radl3 = printf("combine2=%d\n", combine(AdlNs2::Widget2(5), AdlNs2::Widget2(6)));
// CHECK: combine2=200

// 2. Ambiguous unqualified lookup via using-directive collision, cf.
// SemaCXX/using-directive.cpp (ambig_i).
namespace NsX { int shared_val = 111; }
namespace NsY { int other_val = 222; }
using namespace NsX;
auto rns1 = printf("shared_val=%d\n", shared_val);
// CHECK: shared_val=111

namespace NsZ { int shared_val = 333; int extra_z = 1; } using namespace NsZ; int ambiguous_ref = shared_val;
auto rns2 = printf("shared_val=%d other_val=%d\n", shared_val, NsY::other_val);
// CHECK: shared_val=111 other_val=222
namespace NsZ2 { int shared_val = 444; }
auto rns3 = printf("NsZ2_shared_val=%d\n", NsZ2::shared_val);
// CHECK: NsZ2_shared_val=444

// 3. Namespace-alias-vs-other-kind collision, cf. SemaCXX/namespace-alias.cpp.
namespace RealNs { int val = 55; }
namespace AliasA = RealNs;
auto rns4 = printf("AliasA_val=%d\n", AliasA::val);
// CHECK: AliasA_val=55

namespace RealNs2 { int val2 = 66; } namespace AliasB = RealNs2; int CollideName; namespace CollideName = RealNs;
namespace RealNs2 { int val2 = 77; }
namespace AliasB = RealNs2;
auto rns5 = printf("AliasB_val2=%d\n", AliasB::val2);
// CHECK: AliasB_val2=77

// 4. Inline namespace member lookup, cf.
// SemaCXX/warn-inline-namespace-reopened-twice.cpp.
namespace InlineOuter { inline namespace v1 { int inline_val = 9; } }
auto rns6 = printf("inline_val=%d qualified=%d\n", InlineOuter::inline_val, InlineOuter::v1::inline_val);
// CHECK: inline_val=9 qualified=9

namespace InlineOuter { inline namespace v2 { int inline_val2 = 20; struct InlineWidget { int x; }; } } InlineOuter::InlineWidget bad_widget; bad_widget.y = 1;
auto rns7 = printf("inline_val_again=%d\n", InlineOuter::inline_val);
// CHECK: inline_val_again=9
namespace InlineOuter { inline namespace v2 { int inline_val2 = 30; struct InlineWidget { int x; }; } }
InlineOuter::InlineWidget good_widget; good_widget.x = 42;
auto rns8 = printf("inline_val2=%d good_widget_x=%d\n", InlineOuter::inline_val2, good_widget.x);
// CHECK: inline_val2=30 good_widget_x=42

// 5. Conflicting using-declaration merging overload sets across
// namespaces, cf. SemaCXX/using-decl.cpp, SemaCXX/using-decl-1.cpp.
namespace FnNsA { int describe(int x) { return x + 1000; } }
namespace FnNsB { int describe(double x) { return (int)(x + 2000); } }
using FnNsA::describe; using FnNsB::describe;
auto rns9 = printf("describe_int=%d describe_double=%d\n", describe(1), describe(1.0));
// CHECK: describe_int=1001 describe_double=2001

namespace FnNsC { int describe(int x) { return x + 3000; } int helper_c(int x) { return x * 2; } } using FnNsC::describe;
auto rns10 = printf("describe_int2=%d describe_double2=%d\n", describe(2), describe(2.0));
// CHECK: describe_int2=1002 describe_double2=2002
namespace FnNsC { int helper_c(int x) { return x * 2; } }
auto rns11 = printf("helper_c=%d\n", FnNsC::helper_c(5));
// CHECK: helper_c=10

// 6. Namespace reopened across chunks, interrupted by a failed reopening
// that redefines an existing member with a conflicting type -- exercises
// NamespaceDecl redeclaration-chain merging (Ctx.MergedDecls).
namespace ReopenNs { int part1 = 1; }
namespace ReopenNs { int part2 = 2; }
auto rns12 = printf("part1=%d part2=%d\n", ReopenNs::part1, ReopenNs::part2);
// CHECK: part1=1 part2=2

namespace ReopenNs { int part3 = 3; double part1; }
namespace ReopenNs { int part3 = 30; }
auto rns13 = printf("part1=%d part2=%d part3=%d\n", ReopenNs::part1, ReopenNs::part2, ReopenNs::part3);
// CHECK: part1=1 part2=2 part3=30

%quit
