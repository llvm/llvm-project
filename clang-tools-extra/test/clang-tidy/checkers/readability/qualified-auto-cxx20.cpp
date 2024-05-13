// RUN: %check_clang_tidy %s readability-qualified-auto %t -- -- -std=c++20
namespace std {
template <typename T>
class vector { // dummy impl
  T _data[1];

public:
  T *begin() { return _data; }
  const T *begin() const { return _data; }
  T *end() { return &_data[1]; }
  const T *end() const { return &_data[1]; }
  unsigned size() const { return 0; }
};
} // namespace std

std::vector<int> *getVec();
const std::vector<int> *getCVec();
void foo() {
  if (auto X = getVec(); X->size() > 0) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'auto X' can be declared as 'auto *X'
    // CHECK-FIXES: {{^}}  if (auto *X = getVec(); X->size() > 0) {
  }
  switch (auto X = getVec(); X->size()) {
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: 'auto X' can be declared as 'auto *X'
    // CHECK-FIXES: {{^}}  switch (auto *X = getVec(); X->size()) {
  default:
    break;
  }
  for (auto X = getVec(); auto Xi : *X) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'auto X' can be declared as 'auto *X'
    // CHECK-FIXES: {{^}}  for (auto *X = getVec(); auto Xi : *X) {
  }
}
void bar() {
  if (auto X = getCVec(); X->size() > 0) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: 'auto X' can be declared as 'const auto *X'
    // CHECK-FIXES: {{^}}  if (const auto *X = getCVec(); X->size() > 0) {
  }
  switch (auto X = getCVec(); X->size()) {
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: 'auto X' can be declared as 'const auto *X'
    // CHECK-FIXES: {{^}}  switch (const auto *X = getCVec(); X->size()) {
  default:
    break;
  }
  for (auto X = getCVec(); auto Xi : *X) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: 'auto X' can be declared as 'const auto *X'
    // CHECK-FIXES: {{^}}  for (const auto *X = getCVec(); auto Xi : *X) {
  }
}
