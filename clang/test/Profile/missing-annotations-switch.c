// RUN: rm -rf %t && mkdir -p %t
// RUN: split-file %s %t

/// Test that missing-annotations detects switch conditions that are hot, but not annotated.
// RUN: llvm-profdata merge %t/a.proftext -o %t/profdata
// RUN: %clang_cc1 %t/a.c -O2 -o - -emit-llvm -fprofile-instrument-use-path=%t/profdata -verify -mllvm -pgo-missing-annotations -Rpass=missing-annotations -fdiagnostics-misexpect-tolerance=10

/// Test that we don't report any diagnostics, if the threshold isn't exceeded.
// RUN: %clang_cc1 %t/a.c -O2 -o - -emit-llvm -fprofile-instrument-use-path=%t/profdata -mllvm -pgo-missing-annotations -Rpass=missing-annotations  2>&1 | FileCheck -implicit-check-not=remark %s

//--- a.c
#define inner_loop 1000
#define outer_loop 20
#define arry_size 25

int arry[arry_size] = {0};

int rand(void);
int sum(int *buff, int size);
int random_sample(int *buff, int size);

int main(void) {
  int val = 0;

  int j, k;
  for (j = 0; j < outer_loop; ++j) {
    for (k = 0; k < inner_loop; ++k) {
      unsigned condition = rand() % 10000;
      switch (condition) { // expected-remark {{Extremely hot condition. Consider adding llvm.expect intrinsic}}

      case 0:
        val += sum(arry, arry_size);
        break;
      case 1:
      case 2:
      case 3:
        break;
      default:
        val += random_sample(arry, arry_size);
        break;
      } // end switch
    }   // end inner_loop
  }     // end outer_loop

  return val;
}

//--- a.proftext
main
# Func Hash:
872687477373597607
# Num Counters:
9
# Counter Values:
2
9
2
2
3
3
1
999
18001

