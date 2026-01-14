// Test that misexpect detects mis-annotated switch statements for default case

// RUN: llvm-profdata merge %S/Inputs/misexpect-switch-default.proftext -o %t.profdata
// RUN: %clang_cc1 %s -O2 -o - -emit-llvm -fprofile-instrument-use=clang -fprofile-instrument-use-path=%t.profdata -verify -Wmisexpect -debug-info-kind=line-tables-only

#define inner_loop 1000
#define outer_loop 20
#define arry_size 25

int sum(int *buff, int size);
int random_sample(int *buff, int size);
int rand();
void init_arry();

int arry[arry_size] = {0};

int main() {
  init_arry();
  int val = 0;
  int j;
  for (j = 0; j < outer_loop * inner_loop; ++j) {
    unsigned condition = rand() % 5;
    switch (__builtin_expect(condition, 6)) { // expected-warning-re {{potential performance regression from use of __builtin_expect(): annotation was correct on {{.+}}% ({{[0-9]+ / [0-9]+}}) of profiled executions}}
    case 0:
      val += sum(arry, arry_size);
      break;
    case 1:
    case 2:
    case 3:
      break;
    case 4:
      val += random_sample(arry, arry_size);
      break;
    default:
      __builtin_unreachable();
    } // end switch
  }   // end outer_loop

  return 0;
}
