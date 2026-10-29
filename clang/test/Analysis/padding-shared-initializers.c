// RUN: %clang_analyze_cc1 -std=c99 -analyzer-checker=optin.performance.Padding -verify %s
// expected-no-diagnostics

typedef struct V { const struct V *child; } V;
int main(void) {
  V value = (V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = (V[]){(V){.child = 0}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}};
  (void)value;
  return sizeof(sizeof(int));
}
