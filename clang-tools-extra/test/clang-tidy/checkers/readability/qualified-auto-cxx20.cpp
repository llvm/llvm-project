// RUN: %check_clang_tidy %s readability-qualified-auto %t -- -- -std=c++20 -isystem %clang_tidy_headers
#include <vector>

std::vector<int> *getVec();
const std::vector<int> *getCVec();
void foo() {
  if (auto X = getVec(); X->size() > 0) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'auto X' can be declared as 'auto *X'
    // CHECK-FIXES: if (auto *X = getVec(); X->size() > 0) {
  }
  switch (auto X = getVec(); X->size()) {
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: 'auto X' can be declared as 'auto *X'
    // CHECK-FIXES: switch (auto *X = getVec(); X->size()) {
  default:
    break;
  }
  for (auto X = getVec(); auto Xi : *X) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'auto X' can be declared as 'auto *X'
    // CHECK-FIXES: for (auto *X = getVec(); auto Xi : *X) {
  }
}
void bar() {
  if (auto X = getCVec(); X->size() > 0) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'auto X' can be declared as 'const auto *X'
    // CHECK-FIXES: if (const auto *X = getCVec(); X->size() > 0) {
  }
  switch (auto X = getCVec(); X->size()) {
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: 'auto X' can be declared as 'const auto *X'
    // CHECK-FIXES: switch (const auto *X = getCVec(); X->size()) {
  default:
    break;
  }
  for (auto X = getCVec(); auto Xi : *X) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'auto X' can be declared as 'const auto *X'
    // CHECK-FIXES: for (const auto *X = getCVec(); auto Xi : *X) {
  }
}
