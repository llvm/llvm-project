// RUN: not clang-tidy %s --checks="-*,google-explicit-constructor" 2>&1 | FileCheck %s

// NOLINTBEGIN(google-explicit-constructor)
class A { A(int i); };
// NOLINTEND

// Note: the expected output has been split over several lines so that clang-tidy
//       does not see the "no lint" suppression comment and mistakenly assume it
//       is meant for itself.
// CHECK: :[[@LINE-6]]:11: warning: single-argument constructors must be marked explicit
// CHECK: :[[@LINE-6]]:4: error: unmatched 'NOLIN
// CHECK: TEND' comment without a previous 'NOLIN
// CHECK: TBEGIN' comment [clang-tidy-nolint]
