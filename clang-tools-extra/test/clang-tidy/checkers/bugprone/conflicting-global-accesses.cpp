// RUN: %check_clang_tidy -std=c++11 -check-suffixes=,CPP11 %s bugprone-conflicting-global-accesses %t
// RUN: %check_clang_tidy -std=c++17 %s bugprone-conflicting-global-accesses %t
// RUN: %check_clang_tidy -std=c++17 -check-suffixes=,PARAM %s bugprone-conflicting-global-accesses %t -config="{CheckOptions: {bugprone-conflicting-global-accesses.HandleMutableFunctionParametersAsWrites: true}}"

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
    // CHECK-MESSAGES-CPP11: :[[@LINE-1]]:13: warning: read/write conflict on global variable GlobalVarA

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

    return 0;
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
    // CHECK-MESSAGES-CPP11: :[[@LINE-1]]:5: warning: read/write conflict on global variable GlobalVarA
    
    *(Array + GlobalVarA) = incGlobalVarA();
    // CHECK-MESSAGES-CPP11: :[[@LINE-1]]:5: warning: read/write conflict on global variable GlobalVarA
    
    // Shouldn't warn here as the clang warning takes care of it.
    return addAll(GlobalVarA, getGlobalVarA(), GlobalVarA++, 0);
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
    addAll(GlobalStruct.VarA, (GlobalStruct.StructA = {}, 0), 0, 0);

    // Actually, I'm not sure about this. In the next check, the thing that
    // looks like an assignment is actually an overload operator call, which is
    // a call expression.
    // 
    // In my opinion, this should be checked by default, and it is. But by
    // default, the non-const pointer and reference call expression arguments
    // shouldn't be considered as writes. They aren't unless they are overloaded
    // operator call expressions.
    addAll(GlobalStruct.StructA.VarD, (GlobalStruct.StructA = {}, 0), 0, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object GlobalStruct

    addAll(GlobalStruct.VarA, (GlobalStruct.VarA++, 0), 0, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object GlobalStruct
    addAll(GlobalStruct.StructA.VarC, (GlobalStruct = {}, 0), 0, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object GlobalStruct
    addAll((GlobalStruct.StructA = {}, 1), (GlobalStruct = {}, 0), 0, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object GlobalStruct
    addAll((GlobalStruct.StructA = {}, 1), (GlobalStruct.VarA++, 0), GlobalStruct.StructA.VarD, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object GlobalStruct
    
    addAll(ComplexGlobalStruct.UnionA.VarD, ComplexGlobalStruct.UnionA.StructA.VarC++, 0, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object ComplexGlobalStruct
    addAll(ComplexGlobalStruct.UnionA.StructA.VarB, (ComplexGlobalStruct.UnionA.StructA = {}, 0), 0, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object ComplexGlobalStruct
    addAll(ComplexGlobalStruct.UnionA.StructA.VarB, ComplexGlobalStruct.UnionA.VarD++, 0, 0);
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: read/write conflict on the field of the global object ComplexGlobalStruct
    addAll((ComplexGlobalStruct.UnionA = {}, 0), ComplexGlobalStruct.UnionA.VarD++, 0, 0);
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
}
