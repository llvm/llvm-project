// RUN: %check_clang_tidy %s openmp-exception-escape %t -- -extra-arg=-fopenmp=libomp -extra-arg=-fexceptions -config="{CheckOptions: {openmp-exception-escape.IgnoredExceptions: 'ignored, ignored2'}}" --

int thrower() {
  throw 42;
}

class ignored {};
class ignored2 {};
namespace std {
class bad_alloc {};
} // namespace std

void parallel() {
#pragma omp parallel
  thrower();
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: an exception thrown inside of the OpenMP 'parallel' region is not caught in that same region
}

void ignore() {
#pragma omp parallel
  throw ignored();
}

void ignore2() {
#pragma omp parallel
  throw ignored2();
}

void standalone_directive() {
#pragma omp taskwait
  throw ignored(); // not structured block
}

void ignore_alloc() {
#pragma omp parallel
  throw std::bad_alloc();
}

void parallel_caught() {
#pragma omp parallel
  {
    try {
      thrower();
    } catch (...) {
    }
  }
}

void for_header(const int a) {
  // Only the body of the loop counts.
#pragma omp for
  for (int i = 0; i < thrower(); i++)
    ;
}

void forloop(const int a) {
#pragma omp for
  for (int i = 0; i < a; i++)
    thrower();
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: an exception thrown inside of the OpenMP 'for' region is not caught in that same region
}

void parallel_forloop(const int a) {
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < a; i++)
      thrower();
    thrower();
    // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: an exception thrown inside of the OpenMP 'parallel' region is not caught in that same region
    // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: an exception thrown inside of the OpenMP 'for' region is not caught in that same region
  }
}

void parallel_forloop_caught(const int a) {
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < a; i++) {
      try {
        thrower();
      } catch (...) {
      }
    }
    thrower();
    // CHECK-MESSAGES: :[[@LINE-9]]:3: warning: an exception thrown inside of the OpenMP 'parallel' region is not caught in that same region
  }
}

void parallel_caught_forloop(const int a) {
#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < a; i++)
      thrower();
    try {
      thrower();
    } catch (...) {
    }
    // CHECK-MESSAGES: :[[@LINE-5]]:7: warning: an exception thrown inside of the OpenMP 'for' region is not caught in that same region
  }
}

void parallel_outercaught_forloop(const int a) {
#pragma omp parallel
  {
    try {
#pragma omp for
      for (int i = 0; i < a; i++)
        thrower();
      thrower();
    } catch (...) {
    }
    // CHECK-MESSAGES: :[[@LINE-4]]:9: warning: an exception thrown inside of the OpenMP 'for' region is not caught in that same region
  }
}

void parallel_outercaught_forloop_caught(const int a) {
#pragma omp parallel
  {
    try {
#pragma omp for
      for (int i = 0; i < a; i++) {
        try {
          thrower();
        } catch (...) {
        }
      }
    } catch (...) {
    }
  }
}
