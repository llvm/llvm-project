// RUN: %check_clang_tidy -std=c99 %s bugprone-sizeof-expression %t

typedef struct V { const struct V *child; } V;
int main(void) {
  V value = (V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = 0}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}};
  (void)value;
  return sizeof(sizeof(int));
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: suspicious usage of 'sizeof(sizeof(...))' [bugprone-sizeof-expression]
}
