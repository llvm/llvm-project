/// Test we close file handle on flush, so the .gcda file can be deleted on
/// Windows while the process is still running. In addition, test we create
/// a new .gcda on flush, so there is a file when the process exists.
// RUN: mkdir -p %t.d && cd %t.d
// RUN: %clang --coverage -o %t %s
// RUN: test -f gcov-dump-and-remove.gcno

// RUN: rm -f gcov-dump-and-remove.gcda && %run %t
// RUN: llvm-cov gcov -t gcov-dump-and-remove.gcda | FileCheck %s

extern void __gcov_flush(void);
extern int remove(const char *);   // CHECK:          -: [[#@LINE]]:extern int remove
int main(void) {                   // CHECK-NEXT: #####: [[#@LINE]]:
  __gcov_flush();                  // CHECK-NEXT: #####: [[#@LINE]]:
  if (remove("gcov-dump-and-remove.gcda") != 0) // CHECK-NEXT: #####: [[#@LINE]]:
    return 1;                      // CHECK-NEXT: #####: [[#@LINE]]: return 1;
                                   // CHECK-NEXT:     -: [[#@LINE]]:
  __gcov_flush();                  // CHECK-NEXT: #####: [[#@LINE]]:
  __gcov_flush();                  // CHECK-NEXT: #####: [[#@LINE]]:
  if (remove("gcov-dump-and-remove.gcda") != 0) // CHECK-NEXT: #####: [[#@LINE]]:
    return 1;                      // CHECK-NEXT: #####: [[#@LINE]]: return 1;

  return 0;
}
