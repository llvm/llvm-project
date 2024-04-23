// RUN: %clang_cc1 -O0 -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -target-cpu gfx90a \
// RUN:   -verify -S -o - %s

void test_sched_group_barrier_rule()
{
  __builtin_amdgcn_sched_group_barrier(0, 1, 2, -1);  // expected-error {{__builtin_amdgcn_sched_group_barrier RuleID must be within [0,63].}}
  __builtin_amdgcn_sched_group_barrier(1, 2, 4, 64);  // expected-error {{__builtin_amdgcn_sched_group_barrier RuleID must be within [0,63].}}
  __builtin_amdgcn_sched_group_barrier(1, 2, 4, 101);  // expected-error {{__builtin_amdgcn_sched_group_barrier RuleID must be within [0,63].}}
  __builtin_amdgcn_sched_group_barrier(1, 2, 4, -2147483648); // expected-error {{__builtin_amdgcn_sched_group_barrier RuleID must be within [0,63].}}
}
