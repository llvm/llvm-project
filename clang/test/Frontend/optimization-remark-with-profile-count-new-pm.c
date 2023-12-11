// Testing the remark output of the `-fdiagnostics-show-profile-count`.

// Generate instrumentation and sampling profile data.
// RUN: llvm-profdata merge \
// RUN:     %S/Inputs/optimization-remark-with-profile-count.proftext \
// RUN:     -o %t.profdata
//
// RUN: %clang  -fprofile-instr-use=%t.profdata \
// RUN:     -O2 -Rpass=loop-vectorize -Rpass-analysis=loop-vectorize \
// RUN:     -Rpass-missed=loop-vecotrize \
// RUN:     -fdiagnostics-show-profile-count \
// RUN:     2>&1 %s\
// RUN:     | FileCheck -check-prefix=SHOW_PROFILE_COUNT %s
// RUN: %clang  -fprofile-instr-use=%t.profdata \
// RUN:     -O2 -Rpass=loop-vectorize -Rpass-analysis=loop-vectorize \
// RUN:     -Rpass-missed=loop-vecotrize \
// RUN:     -fdiagnostics-show-profile-count -fdiagnostics-show-hotness \
// RUN:     2>&1 %s\
// RUN:     | FileCheck -check-prefix=SHOW_PROFILE_COUNT_AND_HOTNESS %s
// RUN: %clang  \
// RUN:     -O2 -Rpass=loop-vectorize -Rpass-analysis=loop-vectorize \
// RUN:     -Rpass-missed=loop-vecotrize \
// RUN:     -fdiagnostics-show-profile-count  \
// RUN:     2>&1 %s\
// RUN:     | FileCheck -check-prefix=NO_PGO %s

int sum = 0;
int x[20] = {0, 112, 32, 11, 99, 88, 99, 88,34, 342, 85,99, 43, 75, 71, 871, 84, 65, 37, 98};

// SHOW_PROFILE_COUNT_AND_HOTNESS: hotness: {{[0-9]+}}
// SHOW_PROFILE_COUNT_AND_HOTNESS: ProfileCount: {{[0-9]+}}
// SHOW_PROFILE_COUNT: ProfileCount: {{[0-9]+}}
// NO_PGO: argument '-fdiagnostics-show-profile-count' requires profile-guided optimization information
int main(int argc, const char *argv[]) {
#pragma clang loop vectorize(enable)
  for(int i = 0;  i < argc % 20; i++){
    sum += x[i];
    sum += argc;
  }
  return sum;
}
