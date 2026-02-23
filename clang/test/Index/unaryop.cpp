// RUN: c-index-test -test-print-unops %s | FileCheck %s

void func(){
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"
    int i;

    i++;
    ++i;
    i--;
    --i;
    int *p = &i;
    *p;
    int c= +i;
    int d= -i;

    ~i;
    !i;

#pragma clang diagnostic pop
}

// CHECK: UnaryOperator= UnOp=++ 1
// CHECK: UnaryOperator= UnOp=++ 3
// CHECK: UnaryOperator= UnOp=-- 2
// CHECK: UnaryOperator= UnOp=-- 4
// CHECK: UnaryOperator= UnOp=& 5
// CHECK: UnaryOperator= UnOp=* 6
// CHECK: UnaryOperator= UnOp=+ 7
// CHECK: UnaryOperator= UnOp=- 8
// CHECK: UnaryOperator= UnOp=~ 9
// CHECK: UnaryOperator= UnOp=! 10

struct C{
    C() = default;
    C& operator++();
    C& operator++(int);
    C& operator--();
    C& operator--(int);
    C& operator*();
    C* operator&();
    C& operator+();
    C& operator-();
    C& operator!();
    C& operator~();
};

void func2(){
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value"
    C i;
    i++;
    ++i;
    i--;
    --i;
    C *p = &i;
    *i;
    +i;
    C n = +i;
    -i;
    C m = -i;

    ~i;
    !i;

#pragma clang diagnostic pop
}

// CHECK: CallExpr=C:35:5 UnOp= 0
// CHECK: CallExpr=operator++:37:8 UnOp=++ 1
// CHECK: CallExpr=operator++:36:8 UnOp=++ 3
// CHECK: CallExpr=operator--:39:8 UnOp=-- 2
// CHECK: CallExpr=operator--:38:8 UnOp=-- 4
// CHECK: CallExpr=operator&:41:8 UnOp=& 5
// CHECK: CallExpr=operator*:40:8 UnOp=* 6
// CHECK: CallExpr=operator+:42:8 UnOp=+ 7
// CHECK: CallExpr=C:34:8 UnOp= 0
// CHECK: CallExpr=operator+:42:8 UnOp=+ 7
// CHECK: CallExpr=operator-:43:8 UnOp=- 8
// CHECK: CallExpr=C:34:8 UnOp= 0
// CHECK: CallExpr=operator-:43:8 UnOp=- 8
// CHECK: CallExpr=operator~:45:8 UnOp=~ 9
// CHECK: CallExpr=operator!:44:8 UnOp=! 10
