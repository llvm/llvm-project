// RUN: not clang-tidy %s --checks='-*,google-explicit-constructor,google-readability-casting' 2>&1 | FileCheck %s

// NOLINTBEGIN(google-explicit-constructor,google-readability-casting)
class B { B(int i); };
// NOLINTEND(google-explicit-constructor)
auto Num2 = (unsigned int)(-1);
// NOLINTEND(google-readability-casting)

// Note: the expected output has been split over several lines so that clang-tidy
//       does not see the "no lint" suppression comment and mistakenly assume it
//       is meant for itself.
// CHECK: :[[@LINE-9]]:4: error: unmatched 'NOLIN
// CHECK: TBEGIN' comment without a subsequent 'NOLIN
// CHECK: TEND' comment [clang-tidy-nolint]
// CHECK: :[[@LINE-11]]:11: warning: single-argument constructors must be marked explicit
// CHECK: :[[@LINE-11]]:4: error: unmatched 'NOLIN
// CHECK: TEND' comment without a previous 'NOLIN
// CHECK: TBEGIN' comment [clang-tidy-nolint]
// CHECK: :[[@LINE-13]]:13: warning: C-style casts are discouraged; use static_cast
// CHECK: :[[@LINE-13]]:4: error: unmatched 'NOLIN
// CHECK: TEND' comment without a previous 'NOLIN
// CHECK: TBEGIN' comment [clang-tidy-nolint]
