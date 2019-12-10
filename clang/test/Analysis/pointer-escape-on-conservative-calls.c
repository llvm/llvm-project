// RUN: %clang_cc1 -analyze -analyzer-checker=debug.AnalysisOrder -analyzer-config debug.AnalysisOrder:PointerEscape=true -analyzer-config debug.AnalysisOrder:PostCall=true %s 2>&1 | FileCheck %s


void f(int *);
int *getMem();

int main() {
    f(getMem());
    return 0;
}

// CHECK: PostCall (f)
// CHECK-NEXT: PointerEscape
