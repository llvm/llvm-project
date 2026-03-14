// RUN: %clang_cc1 %s -emit-llvm -o %t
typedef struct { unsigned int i: 1; } c;
const c d = { 1 };

// PR2310
struct Token {
  unsigned n : 31;
};
void sqlite3CodeSubselect(void){
  struct Token one = { 1 };
}

typedef union T0 { char field0 : 2; } T0;
T0 T0_values = { 0 };
