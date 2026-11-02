// Enabling only one of these warnings must be enough to run the analysis.
//
// RUN: %clang_cc1 -fsyntax-only -Wno-dangling -Wlifetime-safety-dangling-global-in-main -verify=main %s
// RUN: %clang_cc1 -fsyntax-only -Wno-dangling -Wlifetime-safety-dangling-field-lambda-capture -verify=lambda %s

int *global; // main-note {{this global dangles}}

int main() {
  int local = 0;
  global = &local; // main-warning {{stack memory associated with local variable 'local' escapes to the global variable 'global' which will dangle after main returns}}
  return 0;
}

struct HasField {
  int *ptr; // lambda-note {{this field dangles}}

  void capture_this() {
    int local = 0;
    ptr = &local; // lambda-warning-re {{stack memory associated with local variable 'local' may escape to the field 'ptr' which will dangle. {{.*}} captured by a lambda}}
    auto cleanup = [this]() { ptr = nullptr; };
  }
};
