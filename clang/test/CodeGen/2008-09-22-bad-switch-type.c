// RUN: %clang_cc1 -emit-llvm -o %t %s
// PR2817

void f0(void) {
  switch (0) {
  case (unsigned long long) 0 < 0: 
    break;
  }

  switch (0) {
  case (unsigned long long) 0 > 0: 
    break;
  }

  switch (0) {
  case (unsigned long long) 0 <= 0: 
    break;
  }

  switch (0) {
  case (unsigned long long) 0 >= 0: 
    break;
  }

  switch (0) {
  case (unsigned long long) 0 == 0: 
    break;
  }

  switch (0) {
  case (unsigned long long) 0 != 0: 
    break;
  }
}
