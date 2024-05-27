// RUN: %clang_cc1 -triple %itanium_abi_triple %s -o - -emit-llvm -fprofile-instrument=clang -fcoverage-mapping -fcoverage-mcdc | FileCheck %s -check-prefix=MCDCCTOR
// RUN: %clang_cc1 -triple %itanium_abi_triple %s -o - -emit-llvm -fprofile-instrument=clang -fcoverage-mapping -fcoverage-mcdc | FileCheck %s -check-prefix=MCDCDTOR

extern void foo();
extern void bar();
class Value {
  public:
    void setValue(int len);
    int getValue(void);
    Value();   // This is the constructor declaration
    ~Value();  // This is the destructor declaration

  private:
    int value;
};

// Member functions definitions including constructor
Value::Value(void) {
  if (value != 2 || value != 6)
    foo();
}
Value::~Value(void) {
  if (value != 2 || value != 3)
    bar();
}

// MCDC BOOKKEEPING.
// MCDCCTOR: @__profbm__ZN5ValueC2Ev = private global [1 x i8] zeroinitializer
// MCDCCTOR: @__profc__ZN5ValueC2Ev = private global [4 x i64] zeroinitializer

// ALLOCATE MCDC TEMP AND ZERO IT.
// MCDCCTOR-LABEL: @_ZN5ValueC2Ev(
// MCDCCTOR: %mcdc.addr = alloca i32, align 4
// MCDCCTOR: store i32 0, ptr %mcdc.addr, align 4

// SHIFT FIRST CONDITION WITH ID = 0.
// MCDCCTOR:  %[[LAB1:[0-9]+]] = load i32, ptr %value, align 4
// MCDCCTOR-DAG:  %[[BOOL:cmp[0-9]*]] = icmp ne i32 %[[LAB1]], 2
// MCDCCTOR-DAG:  %[[TEMP:mcdc.temp[0-9]*]] = load i32, ptr %mcdc.addr, align 4
// MCDCCTOR-DAG:  %[[LAB2:[0-9]+]] = zext i1 %[[BOOL]] to i32
// MCDCCTOR-DAG:  %[[LAB3:[0-9]+]] = shl i32 %[[LAB2]], 0
// MCDCCTOR-DAG:  %[[LAB4:[0-9]+]] = or i32 %[[TEMP]], %[[LAB3]]
// MCDCCTOR-DAG:  store i32 %[[LAB4]], ptr %mcdc.addr, align 4

// SHIFT SECOND CONDITION WITH ID = 1.
// MCDCCTOR:  %[[LAB1:[0-9]+]] = load i32, ptr %value2, align 4
// MCDCCTOR-DAG:  %[[BOOL:cmp[0-9]*]] = icmp ne i32 %[[LAB1]], 6
// MCDCCTOR-DAG:  %[[TEMP:mcdc.temp[0-9]*]] = load i32, ptr %mcdc.addr, align 4
// MCDCCTOR-DAG:  %[[LAB2:[0-9]+]] = zext i1 %[[BOOL]] to i32
// MCDCCTOR-DAG:  %[[LAB3:[0-9]+]] = shl i32 %[[LAB2]], 1
// MCDCCTOR-DAG:  %[[LAB4:[0-9]+]] = or i32 %[[TEMP]], %[[LAB3]]
// MCDCCTOR-DAG:  store i32 %[[LAB4]], ptr %mcdc.addr, align 4

// UPDATE FINAL BITMASK WITH RESULT.
// MCDCCTOR-DAG:  %[[TEMP:mcdc.temp[0-9]*]] = load i32, ptr %mcdc.addr, align 4
// MCDCCTOR:  %[[LAB1:[0-9]+]] = lshr i32 %[[TEMP]], 3
// MCDCCTOR:  %[[LAB4:[0-9]+]] = getelementptr inbounds i8, ptr @__profbm__ZN5ValueC2Ev, i32 %[[LAB1]]
// MCDCCTOR:  %[[LAB5:[0-9]+]] = and i32 %[[TEMP]], 7
// MCDCCTOR:  %[[LAB6:[0-9]+]] = trunc i32 %[[LAB5]] to i8
// MCDCCTOR:  %[[LAB7:[0-9]+]] = shl i8 1, %[[LAB6]]
// MCDCCTOR:  %mcdc.bits = load i8, ptr %[[LAB4]], align 1
// MCDCCTOR:  %[[LAB8:[0-9]+]] = or i8 %mcdc.bits, %[[LAB7]]
// MCDCCTOR:  store i8 %[[LAB8]], ptr %[[LAB4]], align 1

// MCDCDTOR: @__profbm__ZN5ValueD2Ev = private global [1 x i8] zeroinitializer
// MCDCDTOR: @__profc__ZN5ValueD2Ev = private global [4 x i64] zeroinitializer

// ALLOCATE MCDC TEMP AND ZERO IT.
// MCDCDTOR-LABEL: @_ZN5ValueD2Ev(
// MCDCDTOR: %mcdc.addr = alloca i32, align 4
// MCDCDTOR: store i32 0, ptr %mcdc.addr, align 4

// SHIFT FIRST CONDITION WITH ID = 0.
// MCDCDTOR:  %[[LAB1:[0-9]+]] = load i32, ptr %value, align 4
// MCDCDTOR-DAG:  %[[BOOL:cmp[0-9]*]] = icmp ne i32 %[[LAB1]], 2
// MCDCDTOR-DAG:  %[[TEMP:mcdc.temp[0-9]*]] = load i32, ptr %mcdc.addr, align 4
// MCDCDTOR-DAG:  %[[LAB2:[0-9]+]] = zext i1 %[[BOOL]] to i32
// MCDCDTOR-DAG:  %[[LAB3:[0-9]+]] = shl i32 %[[LAB2]], 0
// MCDCDTOR-DAG:  %[[LAB4:[0-9]+]] = or i32 %[[TEMP]], %[[LAB3]]
// MCDCDTOR-DAG:  store i32 %[[LAB4]], ptr %mcdc.addr, align 4

// SHIFT SECOND CONDITION WITH ID = 1.
// MCDCDTOR:  %[[LAB1:[0-9]+]] = load i32, ptr %value2, align 4
// MCDCDTOR-DAG:  %[[BOOL:cmp[0-9]*]] = icmp ne i32 %[[LAB1]], 3
// MCDCDTOR-DAG:  %[[TEMP:mcdc.temp[0-9]*]] = load i32, ptr %mcdc.addr, align 4
// MCDCDTOR-DAG:  %[[LAB2:[0-9]+]] = zext i1 %[[BOOL]] to i32
// MCDCDTOR-DAG:  %[[LAB3:[0-9]+]] = shl i32 %[[LAB2]], 1
// MCDCDTOR-DAG:  %[[LAB4:[0-9]+]] = or i32 %[[TEMP]], %[[LAB3]]
// MCDCDTOR-DAG:  store i32 %[[LAB4]], ptr %mcdc.addr, align 4

// UPDATE FINAL BITMASK WITH RESULT.
// MCDCDTOR-DAG:  %[[TEMP:mcdc.temp[0-9]*]] = load i32, ptr %mcdc.addr, align 4
// MCDCDTOR:  %[[LAB1:[0-9]+]] = lshr i32 %[[TEMP]], 3
// MCDCDTOR:  %[[LAB4:[0-9]+]] = getelementptr inbounds i8, ptr @__profbm__ZN5ValueD2Ev, i32 %[[LAB1]]
// MCDCDTOR:  %[[LAB5:[0-9]+]] = and i32 %[[TEMP]], 7
// MCDCDTOR:  %[[LAB6:[0-9]+]] = trunc i32 %[[LAB5]] to i8
// MCDCDTOR:  %[[LAB7:[0-9]+]] = shl i8 1, %[[LAB6]]
// MCDCDTOR:  %mcdc.bits = load i8, ptr %[[LAB4]], align 1
// MCDCDTOR:  %[[LAB8:[0-9]+]] = or i8 %mcdc.bits, %[[LAB7]]
// MCDCDTOR:  store i8 %[[LAB8]], ptr %[[LAB4]], align 1
