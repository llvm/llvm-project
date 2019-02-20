int FuncCCall() { return 0; }
auto FuncCCallPtr = &FuncCCall;

int __stdcall FuncStdCall() { return 0; }
auto FuncStdCallPtr = &FuncStdCall;

int __fastcall FuncFastCall() { return 0; }
auto FuncFastCallPtr = &FuncFastCall;

int __vectorcall FuncVectorCall() { return 0; }
auto FuncVectorCallPtr = &FuncVectorCall;

struct S {
  int FuncThisCall() { return 0; }
};
auto FuncThisCallPtr = &S::FuncThisCall;

int main() {
  return 0;
}
