// Test that misexpect emits no warning when switch condition is non-const

// RUN: llvm-profdata merge %S/Inputs/misexpect-switch-nonconst.proftext -o %t.profdata
// RUN: %clang_cc1 %s -O2 -o - -disable-llvm-passes -emit-llvm -fprofile-instrument-use=clang -fprofile-instrument-use-path=%t.profdata -verify

// expected-no-diagnostics

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

  int j, k;
  for (j = 0; j < outer_loop; ++j) {
    for (k = 0; k < inner_loop; ++k) {
      unsigned condition = rand() % 10000;
      switch (__builtin_expect(condition, rand())) {
      case 0:
        val += sum(arry, arry_size);
        break;
      case 1:
      case 2:
      case 3:
      case 4:
        val += random_sample(arry, arry_size);
        break;
      default:
        __builtin_unreachable();
      } // end switch
    }   // end inner_loop
  }     // end outer_loop

  return 0;
}
