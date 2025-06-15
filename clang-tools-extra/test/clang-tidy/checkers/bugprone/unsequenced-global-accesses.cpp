// RUN: %check_clang_tidy -std=c++98 -check-suffixes=,PRE-CPP11,PRE-CPP17 %s bugprone-unsequenced-global-accesses %t
// RUN: %check_clang_tidy -std=c++11,c++14 -check-suffixes=,POST-CPP11,PRE-CPP17 %s bugprone-unsequenced-global-accesses %t
// RUN: %check_clang_tidy -std=c++17-or-later -check-suffixes=,POST-CPP11 %s bugprone-unsequenced-global-accesses %t
// RUN: %check_clang_tidy -std=c++17-or-later -check-suffixes=,POST-CPP11,PARAM %s bugprone-unsequenced-global-accesses %t -config="{CheckOptions: {bugprone-unsequenced-global-accesses.HandleMutableFunctionParametersAsWrites: true}}"

#if __cplusplus > 199711L
    // Used to exclude code that would give compiler errors on older standards.
    #define POST_CPP11
#endif

int GlobalVarA;

int incGlobalVarA(void) {
    GlobalVarA++;
    return 0;
}

int getGlobalVarA(void) {
    return GlobalVarA;
}

int undefinedFunc1(int);

int testFunc1(void) {

    int B = getGlobalVarA() + incGlobalVarA();
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: read/write conflict on global variable GlobalVarA
    (void)B;

    return GlobalVarA + incGlobalVarA();
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: read/write conflict on global variable GlobalVarA

    return GlobalVarA + undefinedFunc1(incGlobalVarA());
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: read/write conflict on global variable GlobalVarA

}

int addAll(int A, int B, int C, int D) {
    return A + B + C + D;
}

int testFunc2(void) {
    int B;
    (void)B;
    // Make sure the order does not affect the outcome

    B = getGlobalVarA() + (GlobalVarA++);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: read/write conflict on global variable GlobalVarA

    B = (GlobalVarA++) + getGlobalVarA();
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: read/write conflict on global variable GlobalVarA

    B = incGlobalVarA() + GlobalVarA;
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: read/write conflict on global variable GlobalVarA

    B = addAll(GlobalVarA++, getGlobalVarA(), 0, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: read/write conflict on global variable GlobalVarA

    B = addAll(getGlobalVarA(), GlobalVarA++, 0, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: read/write conflict on global variable GlobalVarA

    // This is already checked by the unsequenced clang warning, so we don't
    // want to warn about this.
    return GlobalVarA + (++GlobalVarA);
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: read/write conflict on global variable GlobalVarA
}

int testFunc3(void) {

    // Make sure double reads are not flagged
    int B = GlobalVarA + GlobalVarA; (void)B;
    B = GlobalVarA + getGlobalVarA();

    return GlobalVarA - GlobalVarA;
}

bool testFunc4(void) {

    // Make sure || and && operators are not flagged
    bool B = GlobalVarA || (GlobalVarA++);
    if(GlobalVarA && (GlobalVarA--)) {

        B = GlobalVarA || (GlobalVarA++) + getGlobalVarA();
        // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: read/write conflict on global variable GlobalVarA

        return (++GlobalVarA) || B || getGlobalVarA();
    }

    int C = GlobalVarA << incGlobalVarA(); (void)C;
    // CHECK-MESSAGES-PRE-CPP17: :[[@LINE-1]]:13: warning: read/write conflict on global variable GlobalVarA

    return false;
}

int incArg(int& P) {
    P++;
    return 0;
}

int incArgPtr(int* P) {
    (*P)++;
    return 0;
}

int incAndAddFn(int* A, int B) {
    return (*A)++ + B;
}

int incAndAddBothPtrFn(int *A, int* B) {
    return (*A)++ + (*B)++;
}

int testFunc5() {

    // Also check if statements

    if(GlobalVarA > incGlobalVarA()) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: read/write conflict on global variable GlobalVarA

        return 1;
    }

    if(addAll(GlobalVarA, incArg(GlobalVarA), 0, 0)) {
    // CHECK-MESSAGES-PARAM: :[[@LINE-1]]:8: warning: read/write conflict on global variable GlobalVarA
        return 1;
    }

    if(addAll(GlobalVarA, incArgPtr(&GlobalVarA), 0, 0)) {
    // CHECK-MESSAGES-PARAM: :[[@LINE-1]]:8: warning: read/write conflict on global variable GlobalVarA
        return 2;
    }

    if(addAll(GlobalVarA, 0, incGlobalVarA(), 0)) {
    // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: read/write conflict on global variable GlobalVarA
        return 2;
    }

    // Shouldn't warn here, as the value gets copied before the
    // addition/increment happens.
    int C = incAndAddFn(&GlobalVarA, GlobalVarA); (void)C;

    // -Wunsequenced doesn't warn here. Neither do we. Not sure if we should
    // cover this case.
    incAndAddBothPtrFn(&GlobalVarA, &GlobalVarA);

    return 0;
}

void *memset(void* S, int C, unsigned int N);
int* GlobalPtrA;

int* incGlobalPtrA() {
    GlobalPtrA++;
    return GlobalPtrA;
}

typedef struct TwoStringStruct {
    char* A;
    char* B;
} TwoStringStruct;

TwoStringStruct* TwoStringPtr;

struct TwoStringStruct* incTwoStringPtr() {
    TwoStringPtr++;
    return TwoStringPtr;
}

int testFunc6() {

    // Shouldn't warn here as the write takes place after the expression is
    // evaluated.
    GlobalVarA = GlobalVarA + 1;
    GlobalVarA = incGlobalVarA();

    // Also check the assignment expression, array element assignment, and
    // pointer dereference lvalues.
    int A = (GlobalVarA = 1) + incGlobalVarA(); (void)A;
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: read/write conflict on global variable GlobalVarA
    
    int Array[] = {1, 2, 3};
    Array[GlobalVarA] = incGlobalVarA();
    // CHECK-MESSAGES-PRE-CPP17: :[[@LINE-1]]:5: warning: read/write conflict on global variable GlobalVarA
    
    *(Array + GlobalVarA) = incGlobalVarA();
    // CHECK-MESSAGES-PRE-CPP17: :[[@LINE-1]]:5: warning: read/write conflict on global variable GlobalVarA

    *(Array + GlobalVarA) = getGlobalVarA();
    // This is fine
    
    // Should also check the array subscript operator

    int B = (Array + GlobalVarA)[incGlobalVarA()]; (void)B;
    // CHECK-MESSAGES-PRE-CPP17: :[[@LINE-1]]:13: warning: read/write conflict on global variable GlobalVarA

    int C = (Array + GlobalVarA)[getGlobalVarA()]; (void)C;
    // This is also fine
    
    // Shouldn't warn here as the clang warning takes care of it.
    return addAll(GlobalVarA, getGlobalVarA(), GlobalVarA++, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: read/write conflict on global variable GlobalVarA

    // Shouldn't warn here as the ampersand operator doesn't read the variable.
    return addAll(&GlobalVarA == &A ? 1 : 0, 1, incGlobalVarA(), 0);

    memset(incGlobalPtrA(), 0, *GlobalPtrA);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on global variable GlobalPtrA

    // Shouldn't warn here as sizeof doesn't read the value.
    memset(incGlobalPtrA(), 0, sizeof(*GlobalPtrA));

    memset(incTwoStringPtr(), 0, (int)TwoStringPtr->A[0]);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on global variable TwoStringPtr
}

class TestClass1 {
public:
    static int StaticVar1;

    int incStaticVar1() {
        StaticVar1++;
        return 0;
    }

    int getStaticVar1() {
        return StaticVar1;
    }

    int testClass1MemberFunc1() {
        
        return incStaticVar1() + getStaticVar1();
        // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: read/write conflict on global variable StaticVar1

    }

    TestClass1 operator++() {
        incStaticVar1();
        return *this;
    }

    operator int() {
        return StaticVar1;
    }

    int operator[](int N) {
        return N;
    }
};

void testFunc7() {
    TestClass1 Obj;
    addAll(TestClass1::StaticVar1, Obj.incStaticVar1(), 0, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on global variable StaticVar1
    addAll(TestClass1::StaticVar1, (Obj.incStaticVar1()), 0, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on global variable StaticVar1
    addAll(TestClass1::StaticVar1, (Obj.incStaticVar1(), 0), 0, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on global variable StaticVar1
    addAll(TestClass1::StaticVar1, ++Obj, 0, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on global variable StaticVar1
    
    TestClass1 Objects[3];
    int A = (Objects + Objects[0].getStaticVar1())[TestClass1::StaticVar1++]; (void)A;
    // CHECK-MESSAGES-PRE-CPP17: :[[@LINE-1]]:13: warning: read/write conflict on global variable StaticVar1
}

struct {
    int VarA;
    int VarB;
    struct {
        int VarC;
        int VarD;
    } StructA;
} GlobalStruct;

struct {
    int VarA;
    union {
        struct {
            int VarB;
            int VarC;
        } StructA;
        int VarD;
    } UnionA;
    int VarE;
} ComplexGlobalStruct;

struct QuiteComplexStruct {
    int VarA;
    union {
        union {
            int VarB;
            int VarC;
            struct QuiteComplexStruct* PtrA;
        } UnionB;
        int VarD;
    } UnionA;
    int VarE;
} QuiteComplexGlobalStruct;

union {
    int VarA;
    struct {
        int VarB, VarC;
    } StructA;
} GlobalUnion;


void testFunc8() {

    // Check if unions and structs are handled properly

    addAll(GlobalStruct.VarA, GlobalStruct.VarB++, 0, 0);
    addAll(GlobalStruct.StructA.VarD, GlobalStruct.VarA++, 0, 0);
    addAll(GlobalStruct.StructA.VarC, GlobalStruct.StructA.VarD++, GlobalStruct.VarB++, GlobalStruct.VarA++);

    addAll(GlobalStruct.VarA, (GlobalStruct.VarA++, 0), 0, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object GlobalStruct
    addAll(ComplexGlobalStruct.UnionA.VarD, ComplexGlobalStruct.UnionA.StructA.VarC++, 0, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object ComplexGlobalStruct
    addAll(ComplexGlobalStruct.UnionA.StructA.VarB, ComplexGlobalStruct.UnionA.VarD++, 0, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object ComplexGlobalStruct

    addAll(ComplexGlobalStruct.UnionA.StructA.VarB, ComplexGlobalStruct.UnionA.StructA.VarC++, 0, 0);

    addAll(QuiteComplexGlobalStruct.UnionA.UnionB.VarC, QuiteComplexGlobalStruct.UnionA.VarD++, 0, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object QuiteComplexGlobalStruct
    addAll(QuiteComplexGlobalStruct.UnionA.UnionB.VarC, QuiteComplexGlobalStruct.UnionA.UnionB.VarB++, 0, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object QuiteComplexGlobalStruct

    addAll(QuiteComplexGlobalStruct.UnionA.UnionB.VarC, QuiteComplexGlobalStruct.UnionA.UnionB.VarB++, 0, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object QuiteComplexGlobalStruct

    addAll(QuiteComplexGlobalStruct.UnionA.UnionB.PtrA->VarA, QuiteComplexGlobalStruct.UnionA.UnionB.VarB++, 0, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object QuiteComplexGlobalStruct
    addAll(QuiteComplexGlobalStruct.UnionA.UnionB.PtrA->VarA, QuiteComplexGlobalStruct.VarA++, 0, 0);

    addAll(QuiteComplexGlobalStruct.UnionA.UnionB.PtrA->VarA, (long)QuiteComplexGlobalStruct.UnionA.UnionB.PtrA++, 0, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object QuiteComplexGlobalStruct

    addAll(GlobalUnion.VarA, 0, GlobalUnion.StructA.VarB++, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object GlobalUnion
    
#ifdef POST_CPP11
    addAll(GlobalStruct.StructA.VarD, (GlobalStruct.StructA = {}, 0), 0, 0);
    // CHECK-MESSAGES-POST-CPP11: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object GlobalStruct
    addAll(GlobalStruct.StructA.VarC, (GlobalStruct = {}, 0), 0, 0);
    // CHECK-MESSAGES-POST-CPP11: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object GlobalStruct

    addAll(GlobalStruct.VarA, (GlobalStruct.StructA = {}, 0), 0, 0);

    addAll((GlobalStruct.StructA = {}, 1), (GlobalStruct = {}, 0), 0, 0);
    // CHECK-MESSAGES-POST-CPP11: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object GlobalStruct
    addAll((GlobalStruct.StructA = {}, 1), (GlobalStruct.VarA++, 0), GlobalStruct.StructA.VarD, 0);
    // CHECK-MESSAGES-POST-CPP11: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object GlobalStruct
 
    addAll((ComplexGlobalStruct.UnionA = {}, 0), ComplexGlobalStruct.UnionA.VarD++, 0, 0);
    // CHECK-MESSAGES-POST-CPP11: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object ComplexGlobalStruct
    
    addAll(ComplexGlobalStruct.UnionA.StructA.VarB, (ComplexGlobalStruct.UnionA.StructA = {}, 0), 0, 0);
    // CHECK-MESSAGES-POST-CPP11: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object ComplexGlobalStruct
#endif
}


int GlobalVarB;

int incGlobalVarB() {
    ++GlobalVarB;
    return GlobalVarB;
}

struct TwoValue {
    int A, B;
};

// Check initializers
void testFunc9() {
    int Arr[] = { incGlobalVarB(), incGlobalVarB() }; (void)Arr;
    // CHECK-MESSAGES-PRE-CPP11: :[[@LINE-1]]:17: warning: read/write conflict on global variable GlobalVarB
    
    TwoValue Ts1 = { incGlobalVarB(), GlobalVarB }; (void)Ts1;
    // CHECK-MESSAGES-PRE-CPP11: :[[@LINE-1]]:20: warning: read/write conflict on global variable GlobalVarB
    
    Ts1 = (TwoValue){ GlobalVarB, incGlobalVarB() };
    // CHECK-MESSAGES-PRE-CPP11: :[[@LINE-1]]:21: warning: read/write conflict on global variable GlobalVarB
    
    TwoValue TsArr[] = { { incGlobalVarB(), 0 }, { 0, GlobalVarB } }; (void)TsArr;
    // CHECK-MESSAGES-PRE-CPP11: :[[@LINE-1]]:24: warning: read/write conflict on global variable GlobalVarB
    
    TwoValue TsArr2[4] = { [1].A = incGlobalVarB(), [3] = { .B = 0, .A = GlobalVarB } }; (void)TsArr2;
    // CHECK-MESSAGES-PRE-CPP11: :[[@LINE-1]]:26: warning: read/write conflict on global variable GlobalVarB

    TwoValue Ts2 = { .A = incGlobalVarB(), .B = GlobalVarB }; (void)Ts2;
    // CHECK-MESSAGES-PRE-CPP11: :[[@LINE-1]]:20: warning: read/write conflict on global variable GlobalVarB
}

class InstanceCountedClass {
public:
    static int InstanceCount;
    InstanceCountedClass() {
        InstanceCount++;
    }

    ~InstanceCountedClass() {
        InstanceCount--;
    }
};

int InstanceCountedClass::InstanceCount = 0;

class GlobalDefaultFieldClass {
public:
    GlobalDefaultFieldClass(int PA, int PB) : A(PA), B(PB) { (void)A; (void)B; }
    GlobalDefaultFieldClass() : A(GlobalVarB), B(0) {}
private:
    int A, B;
};

class NestedInstanceCountedClass {
    InstanceCountedClass C2;
public:
    NestedInstanceCountedClass();
};

class InstanceCountedBaseClass : public InstanceCountedClass {
public:
    InstanceCountedBaseClass();
};

class NestedGlobalDefaultFieldClass {
    GlobalDefaultFieldClass G;
};

class GlobalDefaultFieldBaseClass : public GlobalDefaultFieldClass {
};

class DestructCountedClass {
public:
    static int DestructCount;
    ~DestructCountedClass() {
        DestructCount++;
    }

    operator int() {
        return 1;
    }
};

int DestructCountedClass::DestructCount = 0;

int createAndDestructTestClass4() {
    NestedInstanceCountedClass Test; (void)Test;
    return InstanceCountedClass::InstanceCount;
}

template <class T>
int destructParam(T Param) {
    (void)Param;
    return 0;
}

int temporaryDestroy() {
    int K = 42 + DestructCountedClass(); 
    // Destructor gets called here.
    return K;
}

// Check constructors/destructors
void testFunc10() {
    InstanceCountedClass* TestArr[] = { new InstanceCountedClass(), new InstanceCountedClass() };
    // CHECK-MESSAGES-PRE-CPP11: :[[@LINE-1]]:39: warning: read/write conflict on global variable InstanceCount
    (void)TestArr;
    
    InstanceCountedClass TestArr2[2]; // I'm not sure about this. Is this sequenced?

    InstanceCountedClass* NewTestArr = new InstanceCountedClass[2]; // Is this sequenced?
    
    GlobalDefaultFieldClass Simple1(GlobalVarB, incGlobalVarB());
    // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: read/write conflict on global variable GlobalVarB

    GlobalDefaultFieldClass Simple2(GlobalVarB, GlobalVarB);
    // This is fine
    
    int A = InstanceCountedClass::InstanceCount + (delete TestArr[0], 1); (void)A;
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: read/write conflict on global variable InstanceCount

    int B = InstanceCountedClass::InstanceCount + (TestArr[1]->~InstanceCountedClass(), 1);
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: read/write conflict on global variable InstanceCount
    (void)B;
    
    NestedInstanceCountedClass* TestArr3 = new NestedInstanceCountedClass[2];

    int C = InstanceCountedClass::InstanceCount + (delete &TestArr3[0], 1); (void)C;
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: read/write conflict on global variable InstanceCount

    int D = InstanceCountedClass::InstanceCount + (TestArr3[1].~NestedInstanceCountedClass(), 1); (void)D;
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: read/write conflict on global variable InstanceCount

    delete[] NewTestArr; // Same thing. Is this sequenced?
    
    bool E = InstanceCountedClass::InstanceCount == createAndDestructTestClass4(); (void)E;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: read/write conflict on global variable InstanceCount

    InstanceCountedClass Test4;
    int F  = InstanceCountedClass::InstanceCount + destructParam(Test4); (void)F;
    // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: read/write conflict on global variable InstanceCount

    int G = incGlobalVarB() + (GlobalDefaultFieldClass(), 1); (void)G;
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: read/write conflict on global variable GlobalVarB
    
    InstanceCountedBaseClass* BadDestructor = new InstanceCountedBaseClass();
    int H = InstanceCountedClass::InstanceCount + (delete BadDestructor, 1); (void)H;
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: read/write conflict on global variable InstanceCount
    int I = incGlobalVarB() + (NestedGlobalDefaultFieldClass(), 1); (void)I;
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: read/write conflict on global variable GlobalVarB
    
    int J = incGlobalVarB() + (GlobalDefaultFieldBaseClass(), 1); (void)J;
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: read/write conflict on global variable GlobalVarB
    
    int K = DestructCountedClass::DestructCount + DestructCountedClass(); (void)K;
    // The temporary should be destroyed at the end of the full-expression, so
    // this should be fine.

    int L = DestructCountedClass::DestructCount + temporaryDestroy(); (void)L;
    // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: read/write conflict on global variable DestructCount

}

void functionWithDefaultParam(int A, int B = GlobalVarB) {
    (void)(A + B);
}

void testFunc11() {
    functionWithDefaultParam(GlobalVarB);
    // This is fine

    functionWithDefaultParam(incGlobalVarB());
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on global variable GlobalVarB

    functionWithDefaultParam(incGlobalVarB(), GlobalVarB);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on global variable GlobalVarB

    functionWithDefaultParam(incGlobalVarB(), 0);
    // This is fine
}

// Check default parameters
class DefaultConstructorArgClass {
public:
    DefaultConstructorArgClass(void* A, int B, int C = GlobalVarB++);
    DefaultConstructorArgClass(int A, int B, int C = incGlobalVarB());
};

void testFunc12() {
    DefaultConstructorArgClass TestObj1(0, 0, 0);
    // This is fine
    
    DefaultConstructorArgClass TestObj2((void*)0, GlobalVarB, incGlobalVarB());
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: read/write conflict on global variable GlobalVarB

    DefaultConstructorArgClass TestObj3((void*)0, GlobalVarB);
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: read/write conflict on global variable GlobalVarB
    
    DefaultConstructorArgClass TestObj4(0, GlobalVarB);
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: read/write conflict on global variable GlobalVarB
}

class ConstructorArgWritingClass {
public:
    ConstructorArgWritingClass(int& A) {
        A++;
    }
    ConstructorArgWritingClass(int* A) {
        (*A)++;
    }

    operator int() {
        return 1;
    }
};

// Check constructor argument writes
int functionThatPassesReferenceToCtor() {
    ConstructorArgWritingClass Ref(GlobalVarB);

    return Ref;
}

int functionThatPassesPtrToCtor() {
    ConstructorArgWritingClass Ptr(&GlobalVarB);

    return Ptr;
}

void testFunc13() {
    int A = functionThatPassesReferenceToCtor() + GlobalVarB; (void)A;
    // CHECK-MESSAGES-PARAM: :[[@LINE-1]]:13: warning: read/write conflict on global variable GlobalVarB
    
    int B = functionThatPassesPtrToCtor() + GlobalVarB; (void)B;
    // CHECK-MESSAGES-PARAM: :[[@LINE-1]]:13: warning: read/write conflict on global variable GlobalVarB
}
