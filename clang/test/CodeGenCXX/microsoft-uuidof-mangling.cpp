// RUN: %clang_cc1 -emit-llvm %s -o - -triple x86_64-unknown-unknown -fms-extensions | FileCheck %s
// rdar://17784718

typedef struct _GUID
{
    unsigned int  Data1;
    unsigned short Data2;
    unsigned short Data3;
    unsigned char  Data4[ 8 ];
} GUID;


template < typename T, const GUID & T_iid = __uuidof(T)>
class UUIDTest
{
public:
	UUIDTest() { }
};

struct __declspec(uuid("EAFA1952-66F8-438B-8FBA-AF1BBAE42191")) TestStruct
{
	int foo;
};

template <class T> void test_uuidofType(void *arg[sizeof(__uuidof(T))] = 0) {}

template <class T> void test_uuidofExpr(void *arg[sizeof(__uuidof(typename T::member))] = 0) {}

struct HasMember { typedef TestStruct member; };

int main(int argc, const char * argv[])
{

    UUIDTest<TestStruct> uuidof_test;
    test_uuidofType<TestStruct>();
    test_uuidofExpr<HasMember>();
    return 0;
}

// CHECK: define i32 @main
// CHECK: call void @_ZN8UUIDTestI10TestStructXu8__uuidoftS0_EEC1Ev
// CHECK: call void @_Z15test_uuidofTypeI10TestStructEvPPv(i8** null)
// CHECK: call void @_Z15test_uuidofExprI9HasMemberEvPPv(i8** null)

// CHECK: define linkonce_odr void @_ZN8UUIDTestI10TestStructXu8__uuidoftS0_EEC1Ev
// CHECK: define linkonce_odr void @_Z15test_uuidofTypeI10TestStructEvPPv
// CHECK: define linkonce_odr void @_Z15test_uuidofExprI9HasMemberEvPPv
// CHECK: define linkonce_odr void @_ZN8UUIDTestI10TestStructXu8__uuidoftS0_EEC2Ev
