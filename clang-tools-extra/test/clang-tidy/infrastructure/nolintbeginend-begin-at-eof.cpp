// RUN: not clang-tidy %s --checks='-*,google-explicit-constructor' 2>&1 | FileCheck %s

// CHECK: :[[@LINE+8]]:11: warning: single-argument constructors must be marked explicit
// Note: the expected output has been split over several lines so that clang-tidy
//       does not see the "no lint" suppression comment and mistakenly assume it
//       is meant for itself.
// CHECK: :[[@LINE+5]]:4: error: unmatched 'NOLIN
// CHECK: TBEGIN' comment without a subsequent 'NOLIN
// CHECK: TEND' comment [clang-tidy-nolint]

class A { A(int i); };
// NOLINTBEGIN