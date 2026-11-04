// RUN: %check_clang_tidy %s bugprone-unused-return-value %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-unused-return-value.CheckedReturnTypes: '::CustomError;::ns::Result', \
// RUN:     bugprone-unused-return-value.CheckedFunctions: '' \
// RUN:   }}"

struct CustomError {
  int code;
};

namespace ns {
struct Result {
  bool ok;
};
} // namespace ns

struct NotChecked {
  int value;
};

CustomError makeError();
ns::Result doWork();
NotChecked other();

void TestCheckedReturnTypes() {
  makeError();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors [bugprone-unused-return-value]

  doWork();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors [bugprone-unused-return-value]

  other();

  CustomError e = makeError();
  ns::Result r = doWork();
}
