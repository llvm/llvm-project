// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=both,expected %s
// RUN: %clang_cc1                                         -verify=both,ref      %s

extern const double Num;
extern const double Num = 12;

extern const int E;
constexpr int getE() {
  return E;
}
const int E = 10;
static_assert(getE() == 10);


extern const int carr[]; // both-note {{declared here}}
constexpr int n = carr[0]; // both-error {{must be initialized by a constant expression}} \
                           // both-note {{read of non-constexpr variable}}

