// RUN: %check_clang_tidy %s bugprone-exception-escape %t -- -extra-arg=-fopenmp=libomp -extra-arg=-fexceptions --

int thrower() {
  throw 42;
}

void ok_parallel() {
#pragma omp parallel
  thrower();
}

void bad_for_header_XFAIL(const int a) noexcept {
#pragma omp for
  for (int i = 0; i < thrower(); i++)
    ;
  // FIXME: this really should be caught by bugprone-exception-escape.
  // https://bugs.llvm.org/show_bug.cgi?id=41102
}

void ok_forloop(const int a) {
#pragma omp for
  for (int i = 0; i < a; i++)
    thrower();
}

void some_exception_just_so_that_check_clang_tidy_shuts_up() noexcept {
  thrower();
}
// CHECK-MESSAGES: :[[@LINE-3]]:6: warning: an exception may be thrown in function 'some_exception_just_so_that_check_clang_tidy_shuts_up' which should not throw exceptions
