#pragma clang system_header

struct S {
  ~S(){}
};

void foo() {
  S s;
}
