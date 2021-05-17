// RUN: %clang_cc1 -emit-llvm %s -o - -triple x86_64-unknown-unknown -fms-extensions | FileCheck %s --check-prefixes=CHECK,CHECK-V12
// RUN: %clang_cc1 -emit-llvm %s -o - -triple x86_64-unknown-unknown -fms-extensions -fclang-abi-compat=11 | FileCheck %s --check-prefixes=CHECK,CHECK-V11
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

struct __declspec(uuid("EAFA1952-66F8-438B-8FBA-AF1BBAE42191")) OtherStruct {};

template <class T> void test_uuidofType(decltype(__uuidof(T)) arg) {}

template <class T> void test_uuidofExpr(decltype(__uuidof(T::member)) arg) {}

struct HasMember {
  TestStruct member;
};

// Ensure that mangling an "expr-primary" argument is handled properly.
template <class T> void test_uuidofExpr2(decltype(T{}, __uuidof(HasMember::member)) arg) {}

template<const GUID&> struct UUIDTestTwo { UUIDTestTwo(); };

int main(int argc, const char * argv[])
{
    UUIDTest<TestStruct> uuidof_test;
    // Note that these variables have the same type, so the mangling of that
    // type had better not mention TestStruct or OtherStruct!
    UUIDTestTwo<__uuidof(TestStruct)> uuidof_test2;
    UUIDTestTwo<__uuidof(OtherStruct)> uuidof_test3;
    test_uuidofType<TestStruct>(GUID{});
    test_uuidofExpr<HasMember>(GUID{});
    test_uuidofExpr2<TestStruct>(GUID{});
    return 0;
}

// CHECK: define{{.*}} i32 @main
// CHECK: call void @_ZN8UUIDTestI10TestStructL_Z42_GUID_eafa1952_66f8_438b_8fba_af1bbae42191EEC1Ev(
// CHECK: call void @_ZN11UUIDTestTwoIL_Z42_GUID_eafa1952_66f8_438b_8fba_af1bbae42191EEC1Ev(
// CHECK: call void @_ZN11UUIDTestTwoIL_Z42_GUID_eafa1952_66f8_438b_8fba_af1bbae42191EEC1Ev(
// CHECK-V11: call void @_Z15test_uuidofTypeI10TestStructEvDTu8__uuidoftT_E(
// CHECK-V12: call void @_Z15test_uuidofTypeI10TestStructEvDTu8__uuidofT_EE(
// CHECK-V11: call void @_Z15test_uuidofExprI9HasMemberEvDTu8__uuidofzsrT_6memberE(
// CHECK-V12: call void @_Z15test_uuidofExprI9HasMemberEvDTu8__uuidofXsrT_6memberEEE(
// CHECK-V11: call void @_Z16test_uuidofExpr2I10TestStructEvDTcmtlT_Eu8__uuidofzL_ZN9HasMember6memberEEE(
// CHECK-V12: call void @_Z16test_uuidofExpr2I10TestStructEvDTcmtlT_Eu8__uuidofL_ZN9HasMember6memberEEEE(
// CHECK: define linkonce_odr void @_ZN8UUIDTestI10TestStructL_Z42_GUID_eafa1952_66f8_438b_8fba_af1bbae42191EEC1Ev
// CHECK-V11: define linkonce_odr void @_Z15test_uuidofTypeI10TestStructEvDTu8__uuidoftT_E(
// CHECK-V12: define linkonce_odr void @_Z15test_uuidofTypeI10TestStructEvDTu8__uuidofT_EE(
// CHECK-V11: define linkonce_odr void @_Z15test_uuidofExprI9HasMemberEvDTu8__uuidofzsrT_6memberE(
// CHECK-V12: define linkonce_odr void @_Z15test_uuidofExprI9HasMemberEvDTu8__uuidofXsrT_6memberEEE(
// CHECK-V11: define linkonce_odr void @_Z16test_uuidofExpr2I10TestStructEvDTcmtlT_Eu8__uuidofzL_ZN9HasMember6memberEEE(
// CHECK-V12: define linkonce_odr void @_Z16test_uuidofExpr2I10TestStructEvDTcmtlT_Eu8__uuidofL_ZN9HasMember6memberEEEE(
// CHECK: define linkonce_odr void @_ZN8UUIDTestI10TestStructL_Z42_GUID_eafa1952_66f8_438b_8fba_af1bbae42191EEC2Ev
