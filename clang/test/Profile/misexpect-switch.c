// Test that misexpect detects mis-annotated switch statements

// RUN: llvm-profdata merge %S/Inputs/misexpect-switch.proftext -o %t.profdata
// RUN: %clang_cc1 %s -O2 -o - -emit-llvm -fprofile-instrument-use-path=%t.profdata -verify -Wmisexpect -debug-info-kind=line-tables-only

#define inner_loop 1000
#define outer_loop 20
#define arry_size 25

int arry[arry_size] = {0};

int rand();
int sum(int *buff, int size);
int random_sample(int *buff, int size);

int main() {
  int val = 0;

  int j, k;
  for (j = 0; j < outer_loop; ++j) {
    for (k = 0; k < inner_loop; ++k) {
      unsigned condition = rand() % 10000;
      switch (__builtin_expect(condition, 0)) { // expected-warning-re {{potential performance regression from use of __builtin_expect(): annotation was correct on {{.+}}% ({{[0-9]+ / [0-9]+}}) of profiled executions}}
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

  return 0;
}
