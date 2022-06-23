// RUN: %check_clang_tidy -std=c++14-or-later %s modernize-make-unique %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             [{key: modernize-make-unique.IgnoreDefaultInitialization, \
// RUN:               value: 'false'}] \
// RUN:             }" \
// RUN:   -- -I %S/Inputs/smart-ptr

#include "initializer_list.h"
#include "unique_ptr.h"
// CHECK-FIXES: #include <memory>

void basic() {
  std::unique_ptr<int> P1 = std::unique_ptr<int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: std::unique_ptr<int> P1 = std::make_unique<int>();
  std::unique_ptr<int> P2 = std::unique_ptr<int>(new int);
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: std::unique_ptr<int> P2 = std::make_unique<int>();

  P1.reset(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: P1 = std::make_unique<int>();
  P2.reset(new int);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: P2 = std::make_unique<int>();

  P1 = std::unique_ptr<int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: P1 = std::make_unique<int>();
  P2 = std::unique_ptr<int>(new int);
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: P2 = std::make_unique<int>();

  // With auto.
  auto P3 = std::unique_ptr<int>(new int());
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use std::make_unique instead
  // CHECK-FIXES: auto P3 = std::make_unique<int>();
  auto P4 = std::unique_ptr<int>(new int);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use std::make_unique instead
  // CHECK-FIXES: auto P4 = std::make_unique<int>();

  std::unique_ptr<int> P5 = std::unique_ptr<int>((new int()));
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: std::unique_ptr<int> P5 = std::make_unique<int>();
  std::unique_ptr<int> P6 = std::unique_ptr<int>((new int));
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: std::unique_ptr<int> P6 = std::make_unique<int>();

  P5.reset((new int()));
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: P5 = std::make_unique<int>();
  P6.reset((new int));
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: P6 = std::make_unique<int>();

  std::unique_ptr<int[]> P7, P8;
  P7.reset(new int[5]());
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: P7 = std::make_unique<int[]>(5);

  P8.reset(new int[5]);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: P8 = std::make_unique<int[]>(5);

  int Num = 3;
  P7.reset(new int[Num]);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: P7 = std::make_unique<int[]>(Num);

  P8.reset(new int[Num]);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use std::make_unique instead [modernize-make-unique]
  // CHECK-FIXES: P8 = std::make_unique<int[]>(Num);
}
