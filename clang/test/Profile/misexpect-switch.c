// Test that misexpect detects mis-annotated switch statements

// RUN: llvm-profdata merge %S/Inputs/misexpect-switch.proftext -o %t.profdata
// RUN: %clang_cc1 %s -O2 -o - -emit-llvm -fprofile-instrument-use-path=%t.profdata -verify -Wmisexpect -debug-info-kind=line-tables-only

int sum(int *buff, int size);
int random_sample(int *buff, int size);
int rand();
void init_arry();

const int inner_loop = 1000;
const int outer_loop = 20;
const int arry_size = 25;

int arry[arry_size] = {0};

int main() {
  init_arry();
  int val = 0;

  int j, k;
  for (j = 0; j < outer_loop; ++j) {
    for (k = 0; k < inner_loop; ++k) {
      unsigned condition = rand() % 10000;
      switch (__builtin_expect(condition, 0)) { // expected-warning-re {{Potential performance regression from use of __builtin_expect(): Annotation was correct on {{.+}}% ({{[0-9]+ / [0-9]+}}) of profiled executions.}}
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
