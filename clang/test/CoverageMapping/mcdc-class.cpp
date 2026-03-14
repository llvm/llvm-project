// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++11 -fcoverage-mcdc -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s | FileCheck %s

extern void foo();
extern void bar();
class Value {
  public:
    void setValue( int len );
    int getValue( void );
    Value();   // This is the constructor declaration
    ~Value();  // This is the destructor declaration

  private:
    int value;
};

// Member functions definitions including constructor
Value::Value(void) {
  if (value == 2 || value == 6)
    foo();
}
Value::~Value(void) {
  if (value == 2 || value == 3)
    bar();
}

// CHECK-LABEL:  Decision,File 0, 18:7 -> 18:31 = M:3, C:2
// CHECK-NEXT:  Branch,File 0, 18:7 -> 18:17 = (#0 - #2), #2 [1,0,2]
// CHECK:  Branch,File 0, 18:21 -> 18:31 = (#2 - #3), #3 [2,0,0]
// CHECK-LABEL:  Decision,File 0, 22:7 -> 22:31 = M:3, C:2
// CHECK-NEXT:  Branch,File 0, 22:7 -> 22:17 = (#0 - #2), #2 [1,0,2]
// CHECK:  Branch,File 0, 22:21 -> 22:31 = (#2 - #3), #3 [2,0,0]
