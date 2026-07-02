// RUN: %check_clang_tidy %s cert-err33-c %t

namespace std {
class error_code {
  int val_;
public:
  error_code() : val_(0) {}
  error_code(int v) : val_(v) {}
  int value() const { return val_; }
};
} // namespace std

std::error_code doSomething(int x);

void negativeReturnType() {
  doSomething(42);
}

typedef struct FILE FILE;
extern "C" int fclose(FILE *);

void positiveCheckedFunction() {
  fclose(nullptr);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:3: note: cast the expression to void to silence this warning
}
