// RUN: %check_clang_tidy %s readability-identifier-naming %t -- \
// RUN:   --config-file=%S/Inputs/identifier-naming/hungarian-notation3/.clang-tidy -- -I %S

#include "identifier-naming-standard-types.h"

// clang-format off
//===----------------------------------------------------------------------===//
// Cases to CheckOptions
//===----------------------------------------------------------------------===//
class C_MyClass1 {
public:
  static int ClassMemberCase;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for class member 'ClassMemberCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  static int i_ClassMemberCase;

  char const ConstantMemberCase = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for constant member 'ConstantMemberCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  char const c_ConstantMemberCase = 0;

  void MyFunc1(const int ConstantParameterCase);
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: invalid case style for constant parameter 'ConstantParameterCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  void MyFunc1(const int i_ConstantParameterCase);

  void MyFunc2(const int* ConstantPointerParameterCase);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: invalid case style for pointer parameter 'ConstantPointerParameterCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  void MyFunc2(const int* pi_ConstantPointerParameterCase);

  static constexpr int ConstexprVariableCase = 123;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: invalid case style for constexpr variable 'ConstexprVariableCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  static constexpr int i_ConstexprVariableCase = 123;
};

const int GlobalConstantCase = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for global constant 'GlobalConstantCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const int i_GlobalConstantCase = 0;

const int* GlobalConstantPointerCase = nullptr;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global pointer 'GlobalConstantPointerCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const int* pi_GlobalConstantPointerCase = nullptr;

int* GlobalPointerCase = nullptr;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global pointer 'GlobalPointerCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int* pi_GlobalPointerCase = nullptr;

int GlobalVariableCase = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'GlobalVariableCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int i_GlobalVariableCase = 0;

void Func1(){
  int const LocalConstantCase = 3;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for local constant 'LocalConstantCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  int const i_LocalConstantCase = 3;

  unsigned const ConstantCase = 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for local constant 'ConstantCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  unsigned const u_ConstantCase = 1;

  int* const LocalConstantPointerCase = nullptr;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for local constant pointer 'LocalConstantPointerCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  int* const pi_LocalConstantPointerCase = nullptr;

  int *LocalPointerCase = nullptr;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for local pointer 'LocalPointerCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  int *pi_LocalPointerCase = nullptr;

  int LocalVariableCase = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for local variable 'LocalVariableCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  int i_LocalVariableCase = 0;
}

class C_MyClass2 {
  char MemberCase;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for private member 'MemberCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  char c_MemberCase;

  void Func1(int ParameterCase);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for parameter 'ParameterCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  void Func1(int i_ParameterCase);

  void Func2(const int ParameterCase);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: invalid case style for constant parameter 'ParameterCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  void Func2(const int i_ParameterCase);

  void Func3(const int *PointerParameterCase);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: invalid case style for pointer parameter 'PointerParameterCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  void Func3(const int *pi_PointerParameterCase);
};

class C_MyClass3 {
private:
  char PrivateMemberCase;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for private member 'PrivateMemberCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  char c_PrivateMemberCase;

protected:
  char ProtectedMemberCase;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for protected member 'ProtectedMemberCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  char c_ProtectedMemberCase;

public:
  char PublicMemberCase;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for public member 'PublicMemberCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  char c_PublicMemberCase;
};

static const int StaticConstantCase = 3;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for global constant 'StaticConstantCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}static const int i_StaticConstantCase = 3;

static int StaticVariableCase = 3;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global variable 'StaticVariableCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}static int i_StaticVariableCase = 3;

struct MyStruct { int StructCase; };
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: invalid case style for public member 'StructCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}struct MyStruct { int i_StructCase; };

struct shouldBeCamelCaseStruct { int i_Field; };
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for struct 'shouldBeCamelCaseStruct' [readability-identifier-naming]
// CHECK-FIXES: {{^}}struct ShouldBeCamelCaseStruct { int i_Field; };

union MyUnion { int UnionCase; long l_UnionCase; };
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for union 'MyUnion' [readability-identifier-naming]
// CHECK-MESSAGES: :[[@LINE-2]]:21: warning: invalid case style for public member 'UnionCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}union myUnion { int i_UnionCase; long l_UnionCase; };

//===----------------------------------------------------------------------===//
// C string
//===----------------------------------------------------------------------===//
const char *NamePtr = "Name";
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for global pointer 'NamePtr' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const char *sz_NamePtr = "Name";

const char NameArray[] = "Name";
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global constant 'NameArray' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const char sz_NameArray[] = "Name";

const char *NamePtrArray[] = {"AA", "BB"};
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for global variable 'NamePtrArray' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const char *psz_NamePtrArray[] = {"AA", "BB"};

const wchar_t *WideNamePtr = L"Name";
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for global pointer 'WideNamePtr' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const wchar_t *wsz_WideNamePtr = L"Name";

const wchar_t WideNameArray[] = L"Name";
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for global constant 'WideNameArray' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const wchar_t wsz_WideNameArray[] = L"Name";

const wchar_t *WideNamePtrArray[] = {L"AA", L"BB"};
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for global variable 'WideNamePtrArray' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const wchar_t *pwsz_WideNamePtrArray[] = {L"AA", L"BB"};

class C_MyClass4 {
private:
  char *Name = "Text";
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for private member 'Name' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  char *sz_Name = "Text";

  const char *ConstName = "Text";
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for private member 'ConstName' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  const char *sz_ConstName = "Text";

public:
  const char* DuplicateString(const char* Input, size_t n_RequiredSize);
  // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: invalid case style for pointer parameter 'Input' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  const char* DuplicateString(const char* sz_Input, size_t n_RequiredSize);

  size_t UpdateText(const char* Buffer, size_t n_BufferSize);
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: invalid case style for pointer parameter 'Buffer' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  size_t UpdateText(const char* sz_Buffer, size_t n_BufferSize);
};


//===----------------------------------------------------------------------===//
// Microsoft Windows data types
//===----------------------------------------------------------------------===//
DWORD MsDword = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsDword' [readability-identifier-naming]
// CHECK-FIXES: {{^}}DWORD dw_MsDword = 0;

BYTE MsByte = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsByte' [readability-identifier-naming]
// CHECK-FIXES: {{^}}BYTE by_MsByte = 0;

WORD MsWord = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsWord' [readability-identifier-naming]
// CHECK-FIXES: {{^}}WORD w_MsWord = 0;

BOOL MsBool = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsBool' [readability-identifier-naming]
// CHECK-FIXES: {{^}}BOOL b_MsBool = 0;

BOOLEAN MsBoolean = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'MsBoolean' [readability-identifier-naming]
// CHECK-FIXES: {{^}}BOOLEAN b_MsBoolean = 0;

CHAR MsValueChar = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsValueChar' [readability-identifier-naming]
// CHECK-FIXES: {{^}}CHAR c_MsValueChar = 0;

UCHAR MsValueUchar = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueUchar' [readability-identifier-naming]
// CHECK-FIXES: {{^}}UCHAR uc_MsValueUchar = 0;

SHORT MsValueShort = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueShort' [readability-identifier-naming]
// CHECK-FIXES: {{^}}SHORT s_MsValueShort = 0;

USHORT MsValueUshort = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'MsValueUshort' [readability-identifier-naming]
// CHECK-FIXES: {{^}}USHORT us_MsValueUshort = 0;

WORD MsValueWord = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsValueWord' [readability-identifier-naming]
// CHECK-FIXES: {{^}}WORD w_MsValueWord = 0;

DWORD MsValueDword = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueDword' [readability-identifier-naming]
// CHECK-FIXES: {{^}}DWORD dw_MsValueDword = 0;

DWORD32 MsValueDword32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'MsValueDword32' [readability-identifier-naming]
// CHECK-FIXES: {{^}}DWORD32 dw32_MsValueDword32 = 0;

DWORD64 MsValueDword64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'MsValueDword64' [readability-identifier-naming]
// CHECK-FIXES: {{^}}DWORD64 dw64_MsValueDword64 = 0;

LONG MsValueLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsValueLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}LONG l_MsValueLong = 0;

ULONG MsValueUlong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueUlong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}ULONG ul_MsValueUlong = 0;

ULONG32 MsValueUlong32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'MsValueUlong32' [readability-identifier-naming]
// CHECK-FIXES: {{^}}ULONG32 ul32_MsValueUlong32 = 0;

ULONG64 MsValueUlong64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'MsValueUlong64' [readability-identifier-naming]
// CHECK-FIXES: {{^}}ULONG64 ul64_MsValueUlong64 = 0;

ULONGLONG MsValueUlongLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for global variable 'MsValueUlongLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}ULONGLONG ull_MsValueUlongLong = 0;

HANDLE MsValueHandle = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global pointer 'MsValueHandle' [readability-identifier-naming]
// CHECK-FIXES: {{^}}HANDLE h_MsValueHandle = 0;

INT MsValueInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'MsValueInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}INT i_MsValueInt = 0;

INT8 MsValueInt8 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsValueInt8' [readability-identifier-naming]
// CHECK-FIXES: {{^}}INT8 i8_MsValueInt8 = 0;

INT16 MsValueInt16 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueInt16' [readability-identifier-naming]
// CHECK-FIXES: {{^}}INT16 i16_MsValueInt16 = 0;

INT32 MsValueInt32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueInt32' [readability-identifier-naming]
// CHECK-FIXES: {{^}}INT32 i32_MsValueInt32 = 0;

INT64 MsValueINt64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueINt64' [readability-identifier-naming]
// CHECK-FIXES: {{^}}INT64 i64_MsValueINt64 = 0;

UINT MsValueUint = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsValueUint' [readability-identifier-naming]
// CHECK-FIXES: {{^}}UINT ui_MsValueUint = 0;

UINT8 MsValueUint8 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueUint8' [readability-identifier-naming]
// CHECK-FIXES: {{^}}UINT8 u8_MsValueUint8 = 0;

UINT16 MsValueUint16 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'MsValueUint16' [readability-identifier-naming]
// CHECK-FIXES: {{^}}UINT16 u16_MsValueUint16 = 0;

UINT32 MsValueUint32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'MsValueUint32' [readability-identifier-naming]
// CHECK-FIXES: {{^}}UINT32 u32_MsValueUint32 = 0;

UINT64 MsValueUint64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'MsValueUint64' [readability-identifier-naming]
// CHECK-FIXES: {{^}}UINT64 u64_MsValueUint64 = 0;

PVOID MsValuePvoid = NULL;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global pointer 'MsValuePvoid' [readability-identifier-naming]
// CHECK-FIXES: {{^}}PVOID p_MsValuePvoid = NULL;


//===----------------------------------------------------------------------===//
// Array
//===----------------------------------------------------------------------===//
unsigned GlobalUnsignedArray[] = {1, 2, 3};
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'GlobalUnsignedArray' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned a_GlobalUnsignedArray[] = {1, 2, 3};

int GlobalIntArray[] = {1, 2, 3};
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'GlobalIntArray' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int a_GlobalIntArray[] = {1, 2, 3};

int DataInt[1] = {0};
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'DataInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int a_DataInt[1] = {0};

int DataArray[2] = {0};
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'DataArray' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int a_DataArray[2] = {0};


//===----------------------------------------------------------------------===//
// Pointer
//===----------------------------------------------------------------------===//
int *DataIntPtr[1] = {0};
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'DataIntPtr' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int *pa_DataIntPtr[1] = {0};

void *BufferPtr1;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global pointer 'BufferPtr1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}void *p_BufferPtr1;

void **BufferPtr2;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global pointer 'BufferPtr2' [readability-identifier-naming]
// CHECK-FIXES: {{^}}void **pp_BufferPtr2;

void **pBufferPtr3;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global pointer 'pBufferPtr3' [readability-identifier-naming]
// CHECK-FIXES: {{^}}void **pp_BufferPtr3;

int *pBufferPtr4;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global pointer 'pBufferPtr4' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int *pi_BufferPtr4;

typedef void (*FUNC_PTR_HELLO)();
FUNC_PTR_HELLO Hello = NULL;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for global pointer 'Hello' [readability-identifier-naming]
// CHECK-FIXES: {{^}}FUNC_PTR_HELLO fn_Hello = NULL;

void *ValueVoidPtr = NULL;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global pointer 'ValueVoidPtr' [readability-identifier-naming]
// CHECK-FIXES: {{^}}void *p_ValueVoidPtr = NULL;

ptrdiff_t PtrDiff = NULL;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for global variable 'PtrDiff' [readability-identifier-naming]
// CHECK-FIXES: {{^}}ptrdiff_t p_PtrDiff = NULL;

int8_t *ValueI8Ptr;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global pointer 'ValueI8Ptr' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int8_t *pi8_ValueI8Ptr;

uint8_t *ValueU8Ptr;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global pointer 'ValueU8Ptr' [readability-identifier-naming]
// CHECK-FIXES: {{^}}uint8_t *pu8_ValueU8Ptr;

unsigned char *ValueUcPtr;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for global pointer 'ValueUcPtr' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned char *puc_ValueUcPtr;

unsigned char **ValueUcPtr2;
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: invalid case style for global pointer 'ValueUcPtr2' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned char **ppuc_ValueUcPtr2;

void MyFunc2(void* Val){}
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: invalid case style for pointer parameter 'Val' [readability-identifier-naming]
// CHECK-FIXES: {{^}}void MyFunc2(void* p_Val){}


//===----------------------------------------------------------------------===//
// Reference
//===----------------------------------------------------------------------===//
int i_ValueIndex = 1;
int &RefValueIndex = i_ValueIndex;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'RefValueIndex' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int &i_RefValueIndex = i_ValueIndex;

const int &ConstRefValue = i_ValueIndex;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global variable 'ConstRefValue' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const int &i_ConstRefValue = i_ValueIndex;

long long ll_ValueLongLong = 2;
long long &RefValueLongLong = ll_ValueLongLong;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global variable 'RefValueLongLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}long long &ll_RefValueLongLong = ll_ValueLongLong;


//===----------------------------------------------------------------------===//
// Various types
//===----------------------------------------------------------------------===//
int8_t ValueI8;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'ValueI8' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int8_t i8_ValueI8;

int16_t ValueI16 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'ValueI16' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int16_t i16_ValueI16 = 0;

int32_t ValueI32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'ValueI32' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int32_t i32_ValueI32 = 0;

int64_t ValueI64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'ValueI64' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int64_t i64_ValueI64 = 0;

uint8_t ValueU8 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'ValueU8' [readability-identifier-naming]
// CHECK-FIXES: {{^}}uint8_t u8_ValueU8 = 0;

uint16_t ValueU16 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'ValueU16' [readability-identifier-naming]
// CHECK-FIXES: {{^}}uint16_t u16_ValueU16 = 0;

uint32_t ValueU32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'ValueU32' [readability-identifier-naming]
// CHECK-FIXES: {{^}}uint32_t u32_ValueU32 = 0;

uint64_t ValueU64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'ValueU64' [readability-identifier-naming]
// CHECK-FIXES: {{^}}uint64_t u64_ValueU64 = 0;

float ValueFloat = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'ValueFloat' [readability-identifier-naming]
// CHECK-FIXES: {{^}}float f_ValueFloat = 0;

double ValueDouble = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'ValueDouble' [readability-identifier-naming]
// CHECK-FIXES: {{^}}double d_ValueDouble = 0;

char ValueChar = 'c';
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'ValueChar' [readability-identifier-naming]
// CHECK-FIXES: {{^}}char c_ValueChar = 'c';

bool ValueBool = true;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'ValueBool' [readability-identifier-naming]
// CHECK-FIXES: {{^}}bool b_ValueBool = true;

int ValueInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'ValueInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int i_ValueInt = 0;

size_t ValueSize = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'ValueSize' [readability-identifier-naming]
// CHECK-FIXES: {{^}}size_t n_ValueSize = 0;

wchar_t ValueWchar = 'w';
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'ValueWchar' [readability-identifier-naming]
// CHECK-FIXES: {{^}}wchar_t wc_ValueWchar = 'w';

short ValueShort = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'ValueShort' [readability-identifier-naming]
// CHECK-FIXES: {{^}}short s_ValueShort = 0;

unsigned ValueUnsigned = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'ValueUnsigned' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned u_ValueUnsigned = 0;

signed ValueSigned = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'ValueSigned' [readability-identifier-naming]
// CHECK-FIXES: {{^}}signed s_ValueSigned = 0;

long ValueLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'ValueLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}long l_ValueLong = 0;

long long ValueLongLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for global variable 'ValueLongLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}long long ll_ValueLongLong = 0;

long long int ValueLongLongInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for global variable 'ValueLongLongInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}long long int lli_ValueLongLongInt = 0;

long double ValueLongDouble = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for global variable 'ValueLongDouble' [readability-identifier-naming]
// CHECK-FIXES: {{^}}long double ld_ValueLongDouble = 0;

signed int ValueSignedInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global variable 'ValueSignedInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}signed int si_ValueSignedInt = 0;

signed short ValueSignedShort = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for global variable 'ValueSignedShort' [readability-identifier-naming]
// CHECK-FIXES: {{^}}signed short ss_ValueSignedShort = 0;

signed short int ValueSignedShortInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for global variable 'ValueSignedShortInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}signed short int ssi_ValueSignedShortInt = 0;

signed long long ValueSignedLongLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for global variable 'ValueSignedLongLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}signed long long sll_ValueSignedLongLong = 0;

signed long int ValueSignedLongInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: invalid case style for global variable 'ValueSignedLongInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}signed long int sli_ValueSignedLongInt = 0;

signed long ValueSignedLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for global variable 'ValueSignedLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}signed long sl_ValueSignedLong = 0;

unsigned long long int ValueUnsignedLongLongInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: invalid case style for global variable 'ValueUnsignedLongLongInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned long long int ulli_ValueUnsignedLongLongInt = 0;

unsigned long long ValueUnsignedLongLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: invalid case style for global variable 'ValueUnsignedLongLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned long long ull_ValueUnsignedLongLong = 0;

unsigned long int ValueUnsignedLongInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: invalid case style for global variable 'ValueUnsignedLongInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned long int uli_ValueUnsignedLongInt = 0;

unsigned long ValueUnsignedLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for global variable 'ValueUnsignedLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned long ul_ValueUnsignedLong = 0;

unsigned short int ValueUnsignedShortInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: invalid case style for global variable 'ValueUnsignedShortInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned short int usi_ValueUnsignedShortInt = 0;

unsigned short ValueUnsignedShort = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for global variable 'ValueUnsignedShort' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned short us_ValueUnsignedShort = 0;

unsigned int ValueUnsignedInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for global variable 'ValueUnsignedInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned int ui_ValueUnsignedInt = 0;

unsigned char ValueUnsignedChar = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for global variable 'ValueUnsignedChar' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned char uc_ValueUnsignedChar = 0;

long int ValueLongInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'ValueLongInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}long int li_ValueLongInt = 0;


//===----------------------------------------------------------------------===//
// Specifier, Qualifier, Other keywords
//===----------------------------------------------------------------------===//
volatile int VolatileInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for global variable 'VolatileInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}volatile int i_VolatileInt = 0;

thread_local int ThreadLocalValueInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for global variable 'ThreadLocalValueInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}thread_local int i_ThreadLocalValueInt = 0;

extern int ExternValueInt;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global variable 'ExternValueInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}extern int i_ExternValueInt;

struct DataBuffer {
    mutable size_t Size;
};
// CHECK-MESSAGES: :[[@LINE-2]]:20: warning: invalid case style for public member 'Size' [readability-identifier-naming]
// CHECK-FIXES: {{^}}    mutable size_t n_Size;

static constexpr int const &ConstExprInt = 42;
// CHECK-MESSAGES: :[[@LINE-1]]:29: warning: invalid case style for constexpr variable 'ConstExprInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}static constexpr int const &i_ConstExprInt = 42;


//===----------------------------------------------------------------------===//
// Redefined types
//===----------------------------------------------------------------------===//
typedef int INDEX;
INDEX iIndex = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'iIndex' [readability-identifier-naming]
// CHECK-FIXES: {{^}}INDEX Index = 0;


//===----------------------------------------------------------------------===//
// Class
//===----------------------------------------------------------------------===//
class ClassCase { int Func(); };
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 'ClassCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}class C_ClassCase { int Func(); };

class AbstractClassCase { virtual int Func() = 0; };
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for abstract class 'AbstractClassCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}class I_AbstractClassCase { virtual int Func() = 0; };

class AbstractClassCase1 { virtual int Func1() = 0; int Func2(); };
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for abstract class 'AbstractClassCase1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}class I_AbstractClassCase1 { virtual int Func1() = 0; int Func2(); };

class ClassConstantCase { public: static const int i_ConstantCase; };
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 'ClassConstantCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}class C_ClassConstantCase { public: static const int i_ConstantCase; };

//===----------------------------------------------------------------------===//
// Other Cases
//===----------------------------------------------------------------------===//
int lower_case = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'lower_case' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int i_LowerCase = 0;

int lower_case1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'lower_case1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int i_LowerCase1 = 0;

int lower_case_2 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'lower_case_2' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int i_LowerCase2 = 0;

int UPPER_CASE = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'UPPER_CASE' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int i_UpperCase = 0;

int UPPER_CASE_1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'UPPER_CASE_1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int i_UpperCase1 = 0;

int camelBack = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'camelBack' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int i_CamelBack = 0;

int camelBack_1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'camelBack_1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int i_CamelBack1 = 0;

int camelBack2 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'camelBack2' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int i_CamelBack2 = 0;

int CamelCase = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'CamelCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int i_CamelCase = 0;

int CamelCase_1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'CamelCase_1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int i_CamelCase1 = 0;

int CamelCase2 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'CamelCase2' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int i_CamelCase2 = 0;

int camel_Snake_Back = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'camel_Snake_Back' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int i_CamelSnakeBack = 0;

int camel_Snake_Back_1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'camel_Snake_Back_1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int i_CamelSnakeBack1 = 0;

int Camel_Snake_Case = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'Camel_Snake_Case' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int i_CamelSnakeCase = 0;

int Camel_Snake_Case_1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'Camel_Snake_Case_1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int i_CamelSnakeCase1 = 0;

//===----------------------------------------------------------------------===//
// Enum
//===----------------------------------------------------------------------===//
enum REV_TYPE { RevValid };
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: invalid case style for enum constant 'RevValid' [readability-identifier-naming]
// CHECK-FIXES: {{^}}enum REV_TYPE { rt_RevValid };

enum EnumConstantCase { OneByte, TwoByte };
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: invalid case style for enum constant 'OneByte' [readability-identifier-naming]
// CHECK-MESSAGES: :[[@LINE-2]]:34: warning: invalid case style for enum constant 'TwoByte' [readability-identifier-naming]
// CHECK-FIXES: {{^}}enum EnumConstantCase { ecc_OneByte, ecc_TwoByte };

enum class ScopedEnumConstantCase { Case1 };
// CHECK-MESSAGES: :[[@LINE-1]]:37: warning: invalid case style for scoped enum constant 'Case1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}enum class ScopedEnumConstantCase { secc_Case1 };
// clang-format on
