// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=both,expected %s
// RUN: %clang_cc1                                         -verify=both,ref      %s

// both-no-diagnostics

extern const double Num;
extern const double Num = 12;

extern const int E;
constexpr int getE() {
  return E;
}
const int E = 10;
static_assert(getE() == 10);

