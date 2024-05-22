// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -main-file-name continue.c %s | FileCheck %s

class A {
public:
    int a;
    A(int a): a(a) {}
};
class B: public A {
public:
    int b;
    int c;
    B(int x, int y)
    : A(x),                 // CHECK:      File 0, [[@LINE]]:7 -> [[@LINE]]:11 = #0
                            // CHECK-NEXT: File 0, [[@LINE+1]]:7 -> [[@LINE+1]]:13 = #0
    b(x == 0? 1: 2),        // CHECK-NEXT: File 0, [[@LINE]]:7 -> [[@LINE]]:19 = #0
                            // CHECK-NEXT: Branch,File 0, [[@LINE-1]]:7 -> [[@LINE-1]]:13 = #1, (#0 - #1)
                            // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:14 -> [[@LINE-2]]:15 = #1
                            // CHECK-NEXT: File 0, [[@LINE-3]]:15 -> [[@LINE-3]]:16 = #1
                            // CHECK-NEXT: File 0, [[@LINE-4]]:18 -> [[@LINE-4]]:19 = (#0 - #1)
                            // CHECK-NEXT: File 0, [[@LINE+2]]:7 -> [[@LINE+8]]:8 = #0
                            // CHECK-NEXT: File 0, [[@LINE+7]]:10 -> [[@LINE+7]]:12 = #0
    c([&]() {
                            // CHECK:      File 0, [[@LINE-1]]:13 -> [[@LINE+5]]:6 = #0
                            // CHECK-NEXT: File 0, [[@LINE+1]]:13 -> [[@LINE+1]]:19 = #0
        if (y == 0)         // CHECK-NEXT: Branch,File 0, [[@LINE]]:13 -> [[@LINE]]:19 = #1, (#0 - #1)
            return 1;       // CHECK-NEXT: Gap,File 0, [[@LINE-1]]:20 -> [[@LINE]]:13 = #1
        return 2;           // CHECK-NEXT: File 0, [[@LINE-1]]:13 -> [[@LINE-1]]:21 = #1
    }()) {}                 // CHECK-NEXT: Gap,File 0, [[@LINE-2]]:22 -> [[@LINE-1]]:9 = (#0 - #1)
};                          // CHECK-NEXT: File 0, [[@LINE-2]]:9 -> [[@LINE-2]]:17 = (#0 - #1)

int main() {
    B b(1,2);
    return 0;
}
