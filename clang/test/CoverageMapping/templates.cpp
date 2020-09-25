// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name templates.cpp %s | FileCheck %s

template<typename T>
void unused(T x) {
  return;
}

template<typename T>
int func(T x) {  // CHECK: func
  if(x)          // CHECK: func
    return 0;
  else
    return 1;
  int j = 1;
}

int main() {
  func<int>(0);
  func<bool>(true);
  return 0;
}
