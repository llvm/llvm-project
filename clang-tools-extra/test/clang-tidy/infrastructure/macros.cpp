// RUN: clang-tidy -checks='-*,google-explicit-constructor' %s -- | FileCheck %s

#define Q(name) class name { name(int i); }

Q(A);
// CHECK: :[[@LINE-1]]:3: warning: single-argument constructors must be marked explicit
// CHECK: :3:30: note: expanded from macro 'Q'
