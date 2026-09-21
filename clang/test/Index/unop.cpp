// RUN: c-index-test -test-print-unops -std=c++20 %s | FileCheck %s
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
    C& operator++(int);
    C& operator++();
    C& operator--(int);
    C& operator--();
    C& operator*();
    C* operator&();
    C& operator+();
    C& operator-();
    C& operator!();
    C& operator~();
    void operator co_await();
    void foo();
    C& operator+(const C&);
    C& operator-(const C&);
};

// CHECK: CXXMethod=operator++:31:8 UnOp=++ 1
// CHECK: CXXMethod=operator++:32:8 UnOp=++ 3
// CHECK: CXXMethod=operator--:33:8 UnOp=-- 2
// CHECK: CXXMethod=operator--:34:8 UnOp=-- 4
// CHECK: CXXMethod=operator*:35:8 UnOp=* 6
// CHECK: CXXMethod=operator&:36:8 UnOp=& 5
// CHECK: CXXMethod=operator+:37:8 UnOp=+ 7
// CHECK: CXXMethod=operator-:38:8 UnOp=- 8
// CHECK: CXXMethod=operator!:39:8 UnOp=! 10
// CHECK: CXXMethod=operator~:40:8 UnOp=~ 9
// CHECK: CXXMethod=operator co_await:41:10 UnOp=co_await 14
// CHECK: CXXMethod=foo:42:10 UnOp= 0
// CHECK: CXXMethod=operator+:43:8 UnOp= 0
// CHECK: CXXMethod=operator-:44:8 UnOp= 0

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
    
    i + i;
    i - i;
    #pragma clang diagnostic pop
}

// CHECK: CallExpr=C:30:5 UnOp= 0
// CHECK: CallExpr=operator++:31:8 UnOp=++ 1
// CHECK: CallExpr=operator++:32:8 UnOp=++ 3
// CHECK: CallExpr=operator--:33:8 UnOp=-- 2
// CHECK: CallExpr=operator--:34:8 UnOp=-- 4
// CHECK: CallExpr=operator&:36:8 UnOp=& 5
// CHECK: CallExpr=operator*:35:8 UnOp=* 6
// CHECK: CallExpr=operator+:37:8 UnOp=+ 7
// CHECK: CallExpr=C:29:8 UnOp= 0
// CHECK: CallExpr=operator+:37:8 UnOp=+ 7
// CHECK: CallExpr=operator-:38:8 UnOp=- 8
// CHECK: CallExpr=C:29:8 UnOp= 0
// CHECK: CallExpr=operator-:38:8 UnOp=- 8
// CHECK: CallExpr=operator~:40:8 UnOp=~ 9
// CHECK: CallExpr=operator!:39:8 UnOp=! 10
// CHECK: CallExpr=operator+:43:8 UnOp= 0
// CHECK: CallExpr=operator-:44:8 UnOp= 0


struct D {
    
};

D operator++(const D&a, int);
D operator++(const D&a );
D operator--(const D&a, int);
D operator--(const D&a);
D operator*(const D&a);
D operator&(const D&a);
D operator+(const D&a);
D operator-(const D&a);
D operator!(const D&a);
D operator~(const D&a);
void operator co_await(const D &d);
void foo(const D&);
D operator+(const D&a, const D&);
D operator-(const D&a, const D&);

// CHECK: FunctionDecl=operator++:108:3 UnOp=++ 1
// CHECK: FunctionDecl=operator++:109:3 UnOp=++ 3
// CHECK: FunctionDecl=operator--:110:3 UnOp=-- 2
// CHECK: FunctionDecl=operator--:111:3 UnOp=-- 4
// CHECK: FunctionDecl=operator*:112:3 UnOp=* 6
// CHECK: FunctionDecl=operator&:113:3 UnOp=& 5
// CHECK: FunctionDecl=operator+:114:3 UnOp=+ 7
// CHECK: FunctionDecl=operator-:115:3 UnOp=- 8
// CHECK: FunctionDecl=operator!:116:3 UnOp=! 10
// CHECK: FunctionDecl=operator~:117:3 UnOp=~ 9
// CHECK: FunctionDecl=operator co_await:118:6 UnOp=co_await 14
// CHECK: FunctionDecl=foo:119:6 UnOp= 0
// CHECK: FunctionDecl=operator+:120:3 UnOp= 0
// CHECK: FunctionDecl=operator-:121:3 UnOp= 0

void func3(){
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wunused-value"
    D i;
    i++;
    ++i;
    i--;
    --i;
     &i;
    *i;
    +i;
    D n = +i;
    -i;
    D m = -i;
    
    ~i;
    !i;
    i + i;
    i - i;
    
    #pragma clang diagnostic pop
}

// CHECK: CallExpr=D:104:8 UnOp= 0
// CHECK: CallExpr=operator++:108:3 UnOp=++ 1
// CHECK: CallExpr=operator++:109:3 UnOp=++ 3
// CHECK: CallExpr=operator--:110:3 UnOp=-- 2
// CHECK: CallExpr=operator--:111:3 UnOp=-- 4
// CHECK: CallExpr=operator&:113:3 UnOp=& 5
// CHECK: CallExpr=operator*:112:3 UnOp=* 6
// CHECK: CallExpr=operator+:114:3 UnOp=+ 7
// CHECK: CallExpr=operator+:114:3 UnOp=+ 7
// CHECK: CallExpr=operator-:115:3 UnOp=- 8
// CHECK: CallExpr=operator-:115:3 UnOp=- 8
// CHECK: CallExpr=operator~:117:3 UnOp=~ 9
// CHECK: CallExpr=operator!:116:3 UnOp=! 10
// CHECK: CallExpr=operator+:120:3 UnOp= 0
// CHECK: CallExpr=operator-:121:3 UnOp= 0
