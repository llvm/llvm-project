// RUN: c-index-test -test-load-source all %s | FileCheck %s

void func(void) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"
    int a = 0;
    int *b = &a;
    *b;
    a++;
    a--;
    ++a;
    --a;
    +a;
    -a;
    !a;
    ~a;
    
    float _Complex c = 0;
    __real c;
    __imag c;

    __extension__ a;
#pragma clang diagnostic pop
}

// CHECK: unaryop.cpp:7:14: UnaryOperator=& Extent=[7:14 - 7:16]
// CHECK: unaryop.cpp:8:5: UnaryOperator=* Extent=[8:5 - 8:7]
// CHECK: unaryop.cpp:9:5: UnaryOperator=++ Extent=[9:5 - 9:8]
// CHECK: unaryop.cpp:10:5: UnaryOperator=-- Extent=[10:5 - 10:8]
// CHECK: unaryop.cpp:11:5: UnaryOperator=++ Extent=[11:5 - 11:8]
// CHECK: unaryop.cpp:12:5: UnaryOperator=-- Extent=[12:5 - 12:8]
// CHECK: unaryop.cpp:13:5: UnaryOperator=+ Extent=[13:5 - 13:7]
// CHECK: unaryop.cpp:14:5: UnaryOperator=- Extent=[14:5 - 14:7]
// CHECK: unaryop.cpp:15:5: UnaryOperator=! Extent=[15:5 - 15:7]
// CHECK: unaryop.cpp:16:5: UnaryOperator=~ Extent=[16:5 - 16:7]
// CHECK: unaryop.cpp:19:5: UnaryOperator=__real Extent=[19:5 - 19:13]
// CHECK: unaryop.cpp:20:5: UnaryOperator=__imag Extent=[20:5 - 20:13]
// CHECK: unaryop.cpp:22:5: UnaryOperator=__extension__ Extent=[22:5 - 22:20]
