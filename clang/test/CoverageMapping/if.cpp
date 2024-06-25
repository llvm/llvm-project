// RUN: %clang_cc1 -fms-extensions -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -std=c++23 -triple %itanium_abi_triple -main-file-name if.cpp %s | FileCheck %s
// RUN: %clang_cc1 -fms-extensions -mllvm -emptyline-comment-coverage=false -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -std=c++2b -triple %itanium_abi_triple -main-file-name if.cpp %s | FileCheck %s

int nop() { return 0; }
struct S {
   int i;
   int putprop(int j) {
      i = j;
      return i;
   }
   __declspec(property(put = putprop)) int the_prop;
};

// CHECK-LABEL: _Z3foov:
                                // CHECK-NEXT: [[@LINE+3]]:12 -> [[@LINE+8]]:2 = #0
                                // CHECK-NEXT: [[@LINE+3]]:15 -> [[@LINE+3]]:19 = #0
                                // CHECK-NEXT: Branch,File 0, [[@LINE+2]]:15 -> [[@LINE+2]]:19 = 0, 0
void foo() {                    // CHECK-NEXT: Gap,File 0, [[@LINE+1]]:21 -> [[@LINE+1]]:22 = #2
  if (int j = true ? nop()      // CHECK-NEXT: [[@LINE]]:22 -> [[@LINE]]:27 = #2
                   : nop();     // CHECK-NEXT: [[@LINE]]:22 -> [[@LINE]]:27 = (#0 - #2)
      j)                        // CHECK-NEXT: [[@LINE]]:7 -> [[@LINE]]:8 = #0
    ++j;                        // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:7 -> [[@LINE-1]]:8 = #1, (#0 - #1)
}                               // CHECK-NEXT: [[@LINE-2]]:9 -> [[@LINE-1]]:5 = #1
                                // CHECK-NEXT: [[@LINE-2]]:5 -> [[@LINE-2]]:8 = #1

// GH-57377
// CHECK-LABEL: _Z30check_constexpr_true_with_elsei:
int check_constexpr_true_with_else(int i) {   // CHECK-NEXT: [[@LINE]]:{{[0-9]+}} -> {{[0-9]+}}:2 = #0
  if constexpr(true) {          // CHECK-NEXT: Skipped,File 0, [[@LINE]]:3 -> [[@LINE]]:22 = 0
    i *= 3;                     // CHECK-NEXT: File 0, [[@LINE-1]]:22 -> [[@LINE+1]]:4 = #0
  } else {                      // CHECK-NEXT: Skipped,File 0, [[@LINE]]:4 -> [[@LINE+2]]:4 = 0
    i *= 5;
  }
  return i;
}

// GH-57377
// CHECK-LABEL: _Z33check_constexpr_true_without_elsei:
int check_constexpr_true_without_else(int i) {   // CHECK-NEXT: [[@LINE]]:{{[0-9]+}} -> {{[0-9]+}}:2 = #0
  if constexpr(true) {          // CHECK-NEXT: Skipped,File 0, [[@LINE]]:3 -> [[@LINE]]:22 = 0
    i *= 3;                     // CHECK-NEXT: File 0, [[@LINE-1]]:22 -> [[@LINE+1]]:4 = #0
  }
  return i;
}

// GH-57377
// CHECK-LABEL: _Z31check_constexpr_false_with_elsei:
int check_constexpr_false_with_else(int i) {   // CHECK-NEXT: [[@LINE]]:{{[0-9]+}} -> {{[0-9]+}}:2 = #0
  if constexpr(false) {         // CHECK-NEXT: Skipped,File 0, [[@LINE]]:3 -> [[@LINE+2]]:10 = 0
    i *= 3;
  } else {                      // CHECK-NEXT: File 0, [[@LINE]]:10 -> [[@LINE+2]]:4 = #0
    i *= 5;
  }
  return i;
}

// GH-57377
// CHECK-LABEL: _Z34check_constexpr_false_without_elsei:
int check_constexpr_false_without_else(int i) {   // CHECK-NEXT: [[@LINE]]:{{[0-9]+}} -> {{[0-9]+}}:2 = #0
  if constexpr(false) {         // CHECK-NEXT: Skipped,File 0, [[@LINE]]:3 -> [[@LINE+2]]:4 = 0
    i *= 3;
  }
  return i;
}

// GH-57377
// CHECK-LABEL: _Z35check_constexpr_init_true_with_elsei:
int check_constexpr_init_true_with_else(int i) {   // CHECK-NEXT: [[@LINE]]:{{[0-9]+}} -> {{[0-9]+}}:2 = #0
  if constexpr(int j = i; true) {          // CHECK-NEXT: Skipped,File 0, [[@LINE]]:3 -> [[@LINE]]:16 = 0
                                           // CHECK-NEXT: File 0, [[@LINE-1]]:16 -> [[@LINE-1]]:26 = #0
                                           // CHECK-NEXT: Skipped,File 0, [[@LINE-2]]:26 -> [[@LINE-2]]:33 = 0
                                           // CHECK-NEXT: File 0, [[@LINE-3]]:33 -> [[@LINE+2]]:4 = #0
    i *= j;
  } else {                                 // CHECK-NEXT: Skipped,File 0, [[@LINE]]:4 -> [[@LINE+2]]:4 = 0
    i *= j;
  }
  return i;
}

// GH-57377
// CHECK-LABEL: _Z38check_constexpr_init_true_without_elsei:
int check_constexpr_init_true_without_else(int i) {   // CHECK-NEXT: [[@LINE]]:{{[0-9]+}} -> {{[0-9]+}}:2 = #0
  if constexpr(int j = i; true) {          // CHECK-NEXT: Skipped,File 0, [[@LINE]]:3 -> [[@LINE]]:16 = 0
                                           // CHECK-NEXT: File 0, [[@LINE-1]]:16 -> [[@LINE-1]]:26 = #0
                                           // CHECK-NEXT: Skipped,File 0, [[@LINE-2]]:26 -> [[@LINE-2]]:33 = 0
                                           // CHECK-NEXT: File 0, [[@LINE-3]]:33 -> [[@LINE+2]]:4 = #0
    i *= j;
  }
  return i;
}

// GH-57377
// CHECK-LABEL: _Z36check_constexpr_init_false_with_elsei:
int check_constexpr_init_false_with_else(int i) {   // CHECK-NEXT: [[@LINE]]:{{[0-9]+}} -> {{[0-9]+}}:2 = #0
  if constexpr(int j = i; false) {         // CHECK-NEXT: Skipped,File 0, [[@LINE]]:3 -> [[@LINE]]:16 = 0
                                           // CHECK-NEXT: File 0, [[@LINE-1]]:16 -> [[@LINE-1]]:26 = #0
    i *= j;                                // CHECK-NEXT: Skipped,File 0, [[@LINE-2]]:26 -> [[@LINE+1]]:10 = 0
  } else {                                 // CHECK-NEXT: File 0, [[@LINE]]:10 -> [[@LINE+2]]:4 = #0
    i *= j;
  }
  return i;
}

// GH-57377
// CHECK-LABEL: _Z39check_constexpr_init_false_without_elsei:
int check_constexpr_init_false_without_else(int i) {   // CHECK-NEXT: [[@LINE]]:{{[0-9]+}} -> {{[0-9]+}}:2 = #0
  if constexpr(int j = i; false) {         // CHECK-NEXT: Skipped,File 0, [[@LINE]]:3 -> [[@LINE]]:16 = 0
                                           // CHECK-NEXT: File 0, [[@LINE-1]]:16 -> [[@LINE-1]]:26 = #0
    i *= j;                                // CHECK-NEXT: Skipped,File 0, [[@LINE-2]]:26 -> [[@LINE+1]]:4 = 0
  }
  return i;
}

// CHECK-LABEL: _Z32check_constexpr_init_with_if_defi:
int check_constexpr_init_with_if_def(int i) {   // CHECK-NEXT: [[@LINE]]:{{[0-9]+}} -> {{[0-9]+}}:2 = #0
  if constexpr(using foo = int; true) {         // CHECK-NEXT: Skipped,File 0, [[@LINE]]:3 -> [[@LINE]]:39 = 0
    i *= foo(42);                               // CHECK-NEXT: File 0, [[@LINE-1]]:39 -> [[@LINE+1]]:4 = #0
  }
  return i;
}

// CHECK-LABEL: _Z32check_macro_constexpr_if_skippedi:
int check_macro_constexpr_if_skipped(int i) {   // CHECK-NEXT: [[@LINE]]:{{[0-9]+}} -> {{[0-9]+}}:2 = #0
#define IF_CONSTEXPR if constexpr               // CHECK-NEXT: Expansion,File 0, [[@LINE+1]]:3 -> [[@LINE+1]]:15 = #0 (Expanded file = 1)
  IF_CONSTEXPR(false) {                         // CHECK-NEXT: Skipped,File 0, [[@LINE]]:3 -> [[@LINE+2]]:4 = 0
    i *= 2;                                     // CHECK-NEXT: File 1, [[@LINE-2]]:22 -> [[@LINE-2]]:34 = #0
  }
  return i;
}

// CHECK-LABEL: main:
int main() {                    // CHECK: File 0, [[@LINE]]:12 -> {{[0-9]+}}:2 = #0
  int i = 0;
                                // CHECK-NEXT: File 0, [[@LINE+3]]:6 -> [[@LINE+3]]:12 = #0
                                // CHECK-NEXT: Branch,File 0, [[@LINE+2]]:6 -> [[@LINE+2]]:12 = #1, (#0 - #1)
                                // CHECK-NEXT: Gap,File 0, [[@LINE+1]]:13 -> [[@LINE+1]]:14 = #1
  if(i == 0) i = 1;             // CHECK-NEXT: File 0, [[@LINE]]:14 -> [[@LINE]]:19 = #1

                                // CHECK-NEXT: File 0, [[@LINE+2]]:6 -> [[@LINE+2]]:12 = #0
                                // CHECK-NEXT: Branch,File 0, [[@LINE+1]]:6 -> [[@LINE+1]]:12 = #2, (#0 - #2)
  if(i == 1)                    // CHECK-NEXT: Gap,File 0, [[@LINE]]:13 -> [[@LINE+1]]:5 = #2
    i = 2;                      // CHECK-NEXT: File 0, [[@LINE]]:5 -> [[@LINE]]:10 = #2

                                // CHECK-NEXT: File 0, [[@LINE+2]]:6 -> [[@LINE+2]]:12 = #0
                                // CHECK-NEXT: Branch,File 0, [[@LINE+1]]:6 -> [[@LINE+1]]:12 = #3, (#0 - #3)
  if(i == 0) { i = 1;           // CHECK-NEXT: Gap,File 0, [[@LINE]]:13 -> [[@LINE]]:14 = #3
    i = 2;                      // CHECK-NEXT: File 0, [[@LINE-1]]:14 -> [[@LINE+1]]:4 = #3
  }
                                // CHECK-NEXT: File 0, [[@LINE+2]]:6 -> [[@LINE+2]]:12 = #0
                                // CHECK-NEXT: Branch,File 0, [[@LINE+1]]:6 -> [[@LINE+1]]:12 = #4, (#0 - #4)
  if(i != 0) {                  // CHECK-NEXT: Gap,File 0, [[@LINE]]:13 -> [[@LINE]]:14 = #4
    i = 1;                      // CHECK-NEXT: File 0, [[@LINE-1]]:14 -> [[@LINE+1]]:4 = #4
  } else {                      // CHECK-NEXT: Gap,File 0, [[@LINE]]:4 -> [[@LINE]]:10 = (#0 - #4)
    i = 3;                      // CHECK-NEXT: File 0, [[@LINE-1]]:10 -> [[@LINE+1]]:4 = (#0 - #4)
  }

                                // CHECK-NEXT: File 0, [[@LINE+2]]:7 -> [[@LINE+2]]:13 = #0
                                // CHECK-NEXT: Branch,File 0, [[@LINE+1]]:7 -> [[@LINE+1]]:13 = #5, (#0 - #5)
  i = i == 0?                   // CHECK-NEXT: Gap,File 0, [[@LINE]]:14 -> [[@LINE+1]]:9 = #5
        i + 1 :                 // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE]]:14 = #5
        i + 2;                  // CHECK-NEXT: File 0, [[@LINE]]:9 -> [[@LINE]]:14 = (#0 - #5)

                                // CHECK-NEXT: File 0, [[@LINE+3]]:7 -> [[@LINE+3]]:13 = #0
                                // CHECK-NEXT: Branch,File 0, [[@LINE+2]]:7 -> [[@LINE+2]]:13 = #6, (#0 - #6)
                                // CHECK-NEXT: File 0, [[@LINE+1]]:14 -> [[@LINE+1]]:20 = #6
  i = i == 0?i + 12:i + 10;     // CHECK-NEXT: File 0, [[@LINE]]:21 -> [[@LINE]]:27 = (#0 - #6)

  // GH-45481
  S s;                    
  s.the_prop = 0? 1 : 2;        // CHECK-NEXT: File 0, [[@LINE]]:16 -> [[@LINE]]:17 = #0
                                // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:16 -> [[@LINE-1]]:17 = 0, 0
                                // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:18 -> [[@LINE-2]]:19 = #7
                                // CHECK-NEXT: File 0, [[@LINE-3]]:19 -> [[@LINE-3]]:20 = #7
                                // CHECK-NEXT: File 0, [[@LINE-4]]:23 -> [[@LINE-4]]:24 = (#0 - #7)
  if (s.the_prop = 1) {         // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:21 = #0
     return 1;                  // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:7 -> [[@LINE-1]]:21 = #8, (#0 - #8)
  }                             // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:22 -> [[@LINE-2]]:23 = #8
                                // CHECK-NEXT: File 0, [[@LINE-3]]:23 -> [[@LINE-1]]:4 = #8
}

#define FOO true

// CHECK-LABEL: _Z7ternaryv:
void ternary() {
  true ? FOO : FOO; // CHECK-NOT: Gap,{{.*}}, [[@LINE]]:8 ->
}

// GH-57377
// CHECK-LABEL: _Z40check_consteval_with_else_discarded_theni:
constexpr int check_consteval_with_else_discarded_then(int i) { // CHECK-NEXT: [[@LINE]]:{{[0-9]+}} -> {{[0-9]+}}:2 = #0
  if consteval {                        // CHECK-NEXT: Skipped,File 0, [[@LINE]]:3 -> [[@LINE+2]]:10 = 0
    i *= 3;
  } else {                              // CHECK-NEXT: File 0, [[@LINE]]:10 -> [[@LINE+2]]:4 = #0
    i *= 5;
  }
  return i;
}

// GH-57377
// CHECK-LABEL: _Z43check_notconsteval_with_else_discarded_elsei:
constexpr int check_notconsteval_with_else_discarded_else(int i) { // CHECK-NEXT: [[@LINE]]:{{[0-9]+}} -> {{[0-9]+}}:2 = #0
  if !consteval {                       // CHECK-NEXT: Skipped,File 0, [[@LINE]]:3 -> [[@LINE]]:17 = 0
    i *= 3;                             
  } else {                              // CHECK-NEXT: File 0, [[@LINE-2]]:17 -> [[@LINE]]:4 = #0
    i *= 5;                             // CHECK-NEXT: Skipped,File 0, [[@LINE-1]]:4 -> [[@LINE+1]]:4 = 0
  }
  return i;
}

// GH-57377
// CHECK-LABEL: _Z32check_consteval_branch_discardedi:
constexpr int check_consteval_branch_discarded(int i) { // CHECK-NEXT: [[@LINE]]:{{[0-9]+}} -> {{[0-9]+}}:2 = #0
  if consteval {                        // CHECK-NEXT: Skipped,File 0, [[@LINE]]:3 -> [[@LINE+2]]:4 = 0 
    i *= 3;      
  }
  return i;      
}

// GH-57377
// CHECK-LABEL: _Z30check_notconsteval_branch_kepti:
constexpr int check_notconsteval_branch_kept(int i) { // CHECK-NEXT: [[@LINE]]:{{[0-9]+}} -> {{[0-9]+}}:2 = #0
  if !consteval {                       // CHECK-NEXT: Skipped,File 0, [[@LINE]]:3 -> [[@LINE]]:17 = 0
    i *= 3;                             // CHECK-NEXT: File 0, [[@LINE-1]]:17 -> [[@LINE+1]]:4 = #0
  }
  return i;
}

// CHECK-LABEL: _Z32check_macro_consteval_if_skippedi:
constexpr int check_macro_consteval_if_skipped(int i) {   // CHECK-NEXT: [[@LINE]]:{{[0-9]+}} -> {{[0-9]+}}:2 = #0
#define IF_RUNTIME if !consteval               // CHECK-NEXT: Expansion,File 0, [[@LINE+1]]:3 -> [[@LINE+1]]:13 = #0 (Expanded file = 1)
  IF_RUNTIME {                                 // CHECK-NEXT: Skipped,File 0, [[@LINE]]:3 -> [[@LINE]]:14 = 0
    i *= 2;                                    // CHECK-NEXT: File 0, [[@LINE-1]]:14 -> [[@LINE+1]]:4 = #0
  }                                            // CHECK-NEXT: File 1, [[@LINE-3]]:20 -> [[@LINE-3]]:33 = #0
  return i;
}

struct false_value {
  constexpr operator bool() {
    return false;
  }
};

template <typename> struct dependable_false_value {
  constexpr operator bool() {
    return false;
  }
};

// GH-80285
void should_not_crash() {
  if constexpr (false_value{}) { };
}

template <typename> void should_not_crash_dependable() {
  if constexpr (dependable_false_value<int>{}) { };
}

void should_not_crash_with_template_instance() {
  should_not_crash_dependable<int>();
}

void should_not_crash_with_requires_expr() {
  if constexpr (requires {42;}) { };
}

int instantiate_consteval(int i) {
  i *= check_consteval_with_else_discarded_then(i);
  i *= check_notconsteval_with_else_discarded_else(i);
  i *= check_consteval_branch_discarded(i);
  i *= check_notconsteval_branch_kept(i);
  i *= check_macro_consteval_if_skipped(i);
  return i;
}
