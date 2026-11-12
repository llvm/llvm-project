// RUN: %check_clang_tidy %s bugprone-easily-swappable-parameters %t --

// Test that all of the default entries in the IgnoredParameterTypeSuffixes
// option are indeed ignored.

struct A {};

namespace IgnoredTypes {
using Bool = A;
using _Bool = A;
using it = A;
using It = A;
using iterator = A;
using Iterator = A;
using inputit = A;
using InputIt = A;
using forwardit = A;
using ForwardIt = A;
using bidirit = A;
using BidirIt = A;
using constiterator = A;
using const_iterator = A;
using Const_Iterator = A;
using Constiterator = A;
using ConstIterator = A;
using RandomIt = A;
using randomit = A;
using random_iterator = A;
using ReverseIt = A;
using reverse_iterator = A;
using reverse_const_iterator = A;
using ConstReverseIterator = A;
using Const_Reverse_Iterator = A;
using const_reverse_iterator = A;
using Constreverseiterator = A;
using constreverseiterator = A;
} // namespace IgnoredTypes

// The types used here all have a suffix that is present in the default value of
// IgnoredParameterTypeSuffixes, and should therefore be ignored:
void f1(bool Foo, bool Bar) {}
void f2(IgnoredTypes::Bool Foo, IgnoredTypes::Bool Bar) {}
void f3(IgnoredTypes::_Bool Foo, IgnoredTypes::_Bool Bar) {}
void f4(IgnoredTypes::it Foo, IgnoredTypes::it Bar) {}
void f5(IgnoredTypes::It Foo, IgnoredTypes::It Bar) {}
void f6(IgnoredTypes::iterator Foo, IgnoredTypes::iterator Bar) {}
void f7(IgnoredTypes::Iterator Foo, IgnoredTypes::Iterator Bar) {}
void f8(IgnoredTypes::inputit Foo, IgnoredTypes::inputit Bar) {}
void f9(IgnoredTypes::InputIt Foo, IgnoredTypes::InputIt Bar) {}
void f10(IgnoredTypes::forwardit Foo, IgnoredTypes::forwardit Bar) {}
void f11(IgnoredTypes::ForwardIt Foo, IgnoredTypes::ForwardIt Bar) {}
void f12(IgnoredTypes::bidirit Foo, IgnoredTypes::bidirit Bar) {}
void f13(IgnoredTypes::BidirIt Foo, IgnoredTypes::BidirIt Bar) {}
void f14(IgnoredTypes::constiterator Foo, IgnoredTypes::constiterator Bar) {}
void f15(IgnoredTypes::const_iterator Foo, IgnoredTypes::const_iterator Bar) {}
void f16(IgnoredTypes::Const_Iterator Foo, IgnoredTypes::Const_Iterator Bar) {}
void f17(IgnoredTypes::Constiterator Foo, IgnoredTypes::Constiterator Bar) {}
void f18(IgnoredTypes::ConstIterator Foo, IgnoredTypes::ConstIterator Bar) {}
void f19(IgnoredTypes::RandomIt Foo, IgnoredTypes::RandomIt Bar) {}
void f20(IgnoredTypes::randomit Foo, IgnoredTypes::randomit Bar) {}
void f21(IgnoredTypes::random_iterator Foo, IgnoredTypes::random_iterator Bar) {}
void f22(IgnoredTypes::ReverseIt Foo, IgnoredTypes::ReverseIt Bar) {}
void f23(IgnoredTypes::reverse_iterator Foo, IgnoredTypes::reverse_iterator Bar) {}
void f24(IgnoredTypes::reverse_const_iterator Foo, IgnoredTypes::reverse_const_iterator Bar) {}
void f25(IgnoredTypes::ConstReverseIterator Foo, IgnoredTypes::ConstReverseIterator Bar) {}
void f26(IgnoredTypes::Const_Reverse_Iterator Foo, IgnoredTypes::Const_Reverse_Iterator Bar) {}
void f27(IgnoredTypes::const_reverse_iterator Foo, IgnoredTypes::const_reverse_iterator Bar) {}
void f28(IgnoredTypes::Constreverseiterator Foo, IgnoredTypes::Constreverseiterator Bar) {}
void f29(IgnoredTypes::constreverseiterator Foo, IgnoredTypes::constreverseiterator Bar) {}

// This suffix of this type is not present in IgnoredParameterTypeSuffixes'
// default value, therefore, a warning _should_ be generated.
using ShouldNotBeIgnored = A;
void f30(ShouldNotBeIgnored Foo, ShouldNotBeIgnored Bar) {}
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: 2 adjacent parameters of 'f30' of similar type ('ShouldNotBeIgnored') are easily swapped by mistake [bugprone-easily-swappable-parameters]
// CHECK-MESSAGES: :[[@LINE-2]]:29: note: the first parameter in the range is 'Foo'
// CHECK-MESSAGES: :[[@LINE-3]]:53: note: the last parameter in the range is 'Bar'
