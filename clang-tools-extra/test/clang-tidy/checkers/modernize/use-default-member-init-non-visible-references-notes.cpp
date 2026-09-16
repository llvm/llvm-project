// RUN: %check_clang_tidy -std=c++11-or-later -check-suffix=ALLOW \
// RUN:   %s modernize-use-default-member-init %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-default-member-init.IgnoreNonVisibleReferences: false}}"

struct NonVisibleNote {
  NonVisibleNote();
  int member;
  // CHECK-NOTES-ALLOW: :[[@LINE-1]]:7: warning: use default member initializer for 'member' [modernize-use-default-member-init]
};

constexpr int LocalConstant = 1;

NonVisibleNote::NonVisibleNote() : member(LocalConstant) {}
// CHECK-NOTES-ALLOW: :[[@LINE-1]]:42: note: move the referenced declaration or definition before the field declaration to use a default member initializer
