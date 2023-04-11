// RUN: %check_clang_tidy %s readability-identifier-naming %t -- \
// RUN:   --config-file=%S/Inputs/identifier-naming/hungarian-notation2/.clang-tidy -- -I %S

#include "identifier-naming-standard-types.h"

// clang-format off
//===----------------------------------------------------------------------===//
// Cases to CheckOptions
//===----------------------------------------------------------------------===//
class CMyClass1 {
public:
  static int ClassMemberCase;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for class member 'ClassMemberCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  static int custiClassMemberCase;

  char const ConstantMemberCase = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for constant member 'ConstantMemberCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  char const custcConstantMemberCase = 0;

  void MyFunc1(const int ConstantParameterCase);
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: invalid case style for constant parameter 'ConstantParameterCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  void MyFunc1(const int custiConstantParameterCase);

  void MyFunc2(const int* ConstantPointerParameterCase);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: invalid case style for pointer parameter 'ConstantPointerParameterCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  void MyFunc2(const int* custpcustiConstantPointerParameterCase);

  static constexpr int ConstexprVariableCase = 123;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: invalid case style for constexpr variable 'ConstexprVariableCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  static constexpr int custiConstexprVariableCase = 123;
};

const int GlobalConstantCase = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for global constant 'GlobalConstantCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const int custiGlobalConstantCase = 0;

const int* GlobalConstantPointerCase = nullptr;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global pointer 'GlobalConstantPointerCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const int* custpcustiGlobalConstantPointerCase = nullptr;

int* GlobalPointerCase = nullptr;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global pointer 'GlobalPointerCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int* custpcustiGlobalPointerCase = nullptr;

int GlobalVariableCase = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'GlobalVariableCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int custiGlobalVariableCase = 0;

void Func1(){
  int const LocalConstantCase = 3;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for local constant 'LocalConstantCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  int const custiLocalConstantCase = 3;

  unsigned const ConstantCase = 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for local constant 'ConstantCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  unsigned const custuConstantCase = 1;

  int* const LocalConstantPointerCase = nullptr;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for local constant pointer 'LocalConstantPointerCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  int* const custpcustiLocalConstantPointerCase = nullptr;

  int *LocalPointerCase = nullptr;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for local pointer 'LocalPointerCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  int *custpcustiLocalPointerCase = nullptr;

  int LocalVariableCase = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for local variable 'LocalVariableCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  int custiLocalVariableCase = 0;
}

class CMyClass2 {
  char MemberCase;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for private member 'MemberCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  char custcMemberCase;

  void Func1(int ParameterCase);
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for parameter 'ParameterCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  void Func1(int custiParameterCase);

  void Func2(const int ParameterCase);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: invalid case style for constant parameter 'ParameterCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  void Func2(const int custiParameterCase);

  void Func3(const int *PointerParameterCase);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: invalid case style for pointer parameter 'PointerParameterCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  void Func3(const int *custpcustiPointerParameterCase);
};

class CMyClass3 {
private:
  char PrivateMemberCase;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for private member 'PrivateMemberCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  char custcPrivateMemberCase;

protected:
  char ProtectedMemberCase;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for protected member 'ProtectedMemberCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  char custcProtectedMemberCase;

public:
  char PublicMemberCase;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for public member 'PublicMemberCase' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  char custcPublicMemberCase;
};

static const int StaticConstantCase = 3;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for global constant 'StaticConstantCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}static const int custiStaticConstantCase = 3;

static int StaticVariableCase = 3;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global variable 'StaticVariableCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}static int custiStaticVariableCase = 3;

struct CMyStruct { int StructCase; };
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: invalid case style for public member 'StructCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}struct CMyStruct { int custiStructCase; };

union MyUnion { int UnionCase; long custlUnionCase; };
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: invalid case style for public member 'UnionCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}union MyUnion { int custiUnionCase; long custlUnionCase; };

//===----------------------------------------------------------------------===//
// C string
//===----------------------------------------------------------------------===//
const char *NamePtr = "Name";
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for global pointer 'NamePtr' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const char *custszNamePtr = "Name";

const char NameArray[] = "Name";
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global constant 'NameArray' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const char custszNameArray[] = "Name";

const char *NamePtrArray[] = {"AA", "BB"};
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for global variable 'NamePtrArray' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const char *custpcustszNamePtrArray[] = {"AA", "BB"};

const wchar_t *WideNamePtr = L"Name";
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for global pointer 'WideNamePtr' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const wchar_t *custwszWideNamePtr = L"Name";

const wchar_t WideNameArray[] = L"Name";
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for global constant 'WideNameArray' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const wchar_t custwszWideNameArray[] = L"Name";

const wchar_t *WideNamePtrArray[] = {L"AA", L"BB"};
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for global variable 'WideNamePtrArray' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const wchar_t *custpcustwszWideNamePtrArray[] = {L"AA", L"BB"};

class CMyClass4 {
private:
  char *Name = "Text";
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for private member 'Name' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  char *custszName = "Text";

  const char *ConstName = "Text";
  // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for private member 'ConstName' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  const char *custszConstName = "Text";

public:
  const char* DuplicateString(const char* Input, size_t custnRequiredSize);
  // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: invalid case style for pointer parameter 'Input' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  const char* DuplicateString(const char* custszInput, size_t custnRequiredSize);

  size_t UpdateText(const char* Buffer, size_t custnBufferSize);
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: invalid case style for pointer parameter 'Buffer' [readability-identifier-naming]
  // CHECK-FIXES: {{^}}  size_t UpdateText(const char* custszBuffer, size_t custnBufferSize);
};


//===----------------------------------------------------------------------===//
// Microsoft Windows data types
//===----------------------------------------------------------------------===//
DWORD MsDword = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsDword' [readability-identifier-naming]
// CHECK-FIXES: {{^}}DWORD custdwMsDword = 0;

BYTE MsByte = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsByte' [readability-identifier-naming]
// CHECK-FIXES: {{^}}BYTE custbyMsByte = 0;

WORD MsWord = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsWord' [readability-identifier-naming]
// CHECK-FIXES: {{^}}WORD custwMsWord = 0;

BOOL MsBool = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsBool' [readability-identifier-naming]
// CHECK-FIXES: {{^}}BOOL custbMsBool = 0;

BOOLEAN MsBoolean = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'MsBoolean' [readability-identifier-naming]
// CHECK-FIXES: {{^}}BOOLEAN custbMsBoolean = 0;

CHAR MsValueChar = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsValueChar' [readability-identifier-naming]
// CHECK-FIXES: {{^}}CHAR custcMsValueChar = 0;

UCHAR MsValueUchar = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueUchar' [readability-identifier-naming]
// CHECK-FIXES: {{^}}UCHAR custucMsValueUchar = 0;

SHORT MsValueShort = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueShort' [readability-identifier-naming]
// CHECK-FIXES: {{^}}SHORT custsMsValueShort = 0;

USHORT MsValueUshort = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'MsValueUshort' [readability-identifier-naming]
// CHECK-FIXES: {{^}}USHORT custusMsValueUshort = 0;

WORD MsValueWord = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsValueWord' [readability-identifier-naming]
// CHECK-FIXES: {{^}}WORD custwMsValueWord = 0;

DWORD MsValueDword = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueDword' [readability-identifier-naming]
// CHECK-FIXES: {{^}}DWORD custdwMsValueDword = 0;

DWORD32 MsValueDword32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'MsValueDword32' [readability-identifier-naming]
// CHECK-FIXES: {{^}}DWORD32 custdw32MsValueDword32 = 0;

DWORD64 MsValueDword64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'MsValueDword64' [readability-identifier-naming]
// CHECK-FIXES: {{^}}DWORD64 custdw64MsValueDword64 = 0;

LONG MsValueLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsValueLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}LONG custlMsValueLong = 0;

ULONG MsValueUlong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueUlong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}ULONG custulMsValueUlong = 0;

ULONG32 MsValueUlong32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'MsValueUlong32' [readability-identifier-naming]
// CHECK-FIXES: {{^}}ULONG32 custul32MsValueUlong32 = 0;

ULONG64 MsValueUlong64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'MsValueUlong64' [readability-identifier-naming]
// CHECK-FIXES: {{^}}ULONG64 custul64MsValueUlong64 = 0;

ULONGLONG MsValueUlongLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for global variable 'MsValueUlongLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}ULONGLONG custullMsValueUlongLong = 0;

HANDLE MsValueHandle = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global pointer 'MsValueHandle' [readability-identifier-naming]
// CHECK-FIXES: {{^}}HANDLE custhMsValueHandle = 0;

INT MsValueInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'MsValueInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}INT custiMsValueInt = 0;

INT8 MsValueInt8 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsValueInt8' [readability-identifier-naming]
// CHECK-FIXES: {{^}}INT8 custi8MsValueInt8 = 0;

INT16 MsValueInt16 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueInt16' [readability-identifier-naming]
// CHECK-FIXES: {{^}}INT16 custi16MsValueInt16 = 0;

INT32 MsValueInt32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueInt32' [readability-identifier-naming]
// CHECK-FIXES: {{^}}INT32 custi32MsValueInt32 = 0;

INT64 MsValueINt64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueINt64' [readability-identifier-naming]
// CHECK-FIXES: {{^}}INT64 custi64MsValueINt64 = 0;

UINT MsValueUint = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsValueUint' [readability-identifier-naming]
// CHECK-FIXES: {{^}}UINT custuiMsValueUint = 0;

UINT8 MsValueUint8 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueUint8' [readability-identifier-naming]
// CHECK-FIXES: {{^}}UINT8 custu8MsValueUint8 = 0;

UINT16 MsValueUint16 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'MsValueUint16' [readability-identifier-naming]
// CHECK-FIXES: {{^}}UINT16 custu16MsValueUint16 = 0;

UINT32 MsValueUint32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'MsValueUint32' [readability-identifier-naming]
// CHECK-FIXES: {{^}}UINT32 custu32MsValueUint32 = 0;

UINT64 MsValueUint64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'MsValueUint64' [readability-identifier-naming]
// CHECK-FIXES: {{^}}UINT64 custu64MsValueUint64 = 0;

PVOID MsValuePvoid = NULL;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global pointer 'MsValuePvoid' [readability-identifier-naming]
// CHECK-FIXES: {{^}}PVOID custpMsValuePvoid = NULL;


//===----------------------------------------------------------------------===//
// Array
//===----------------------------------------------------------------------===//
unsigned GlobalUnsignedArray[] = {1, 2, 3};
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'GlobalUnsignedArray' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned custaGlobalUnsignedArray[] = {1, 2, 3};

int GlobalIntArray[] = {1, 2, 3};
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'GlobalIntArray' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int custaGlobalIntArray[] = {1, 2, 3};

int DataInt[1] = {0};
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'DataInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int custaDataInt[1] = {0};

int DataArray[2] = {0};
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'DataArray' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int custaDataArray[2] = {0};


//===----------------------------------------------------------------------===//
// Pointer
//===----------------------------------------------------------------------===//
int *DataIntPtr[1] = {0};
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'DataIntPtr' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int *custpcustaDataIntPtr[1] = {0};

void *BufferPtr1;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global pointer 'BufferPtr1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}void *custpcustvBufferPtr1;

void **BufferPtr2;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global pointer 'BufferPtr2' [readability-identifier-naming]
// CHECK-FIXES: {{^}}void **custpcustpcustvBufferPtr2;

void **custpBufferPtr3;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global pointer 'custpBufferPtr3' [readability-identifier-naming]
// CHECK-FIXES: {{^}}void **custpcustpcustvBufferPtr3;

int *custpBufferPtr4;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global pointer 'custpBufferPtr4' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int *custpcustiBufferPtr4;

typedef void (*FUNC_PTR_HELLO)();
FUNC_PTR_HELLO Hello = NULL;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for global pointer 'Hello' [readability-identifier-naming]
// CHECK-FIXES: {{^}}FUNC_PTR_HELLO custfnHello = NULL;

void *ValueVoidPtr = NULL;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global pointer 'ValueVoidPtr' [readability-identifier-naming]
// CHECK-FIXES: {{^}}void *custpcustvValueVoidPtr = NULL;

ptrdiff_t PtrDiff = NULL;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for global variable 'PtrDiff' [readability-identifier-naming]
// CHECK-FIXES: {{^}}ptrdiff_t custpPtrDiff = NULL;

int8_t *ValueI8Ptr;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global pointer 'ValueI8Ptr' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int8_t *custpcusti8ValueI8Ptr;

uint8_t *ValueU8Ptr;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global pointer 'ValueU8Ptr' [readability-identifier-naming]
// CHECK-FIXES: {{^}}uint8_t *custpcustu8ValueU8Ptr;

unsigned char *ValueUcPtr;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for global pointer 'ValueUcPtr' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned char *custpcustucValueUcPtr;

unsigned char **ValueUcPtr2;
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: invalid case style for global pointer 'ValueUcPtr2' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned char **custpcustpcustucValueUcPtr2;

void MyFunc2(void* Val){}
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: invalid case style for pointer parameter 'Val' [readability-identifier-naming]
// CHECK-FIXES: {{^}}void MyFunc2(void* custpcustvVal){}


//===----------------------------------------------------------------------===//
// Reference
//===----------------------------------------------------------------------===//
int custiValueIndex = 1;
int &RefValueIndex = custiValueIndex;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'RefValueIndex' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int &custiRefValueIndex = custiValueIndex;

const int &ConstRefValue = custiValueIndex;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global variable 'ConstRefValue' [readability-identifier-naming]
// CHECK-FIXES: {{^}}const int &custiConstRefValue = custiValueIndex;

long long custllValueLongLong = 2;
long long &RefValueLongLong = custllValueLongLong;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global variable 'RefValueLongLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}long long &custllRefValueLongLong = custllValueLongLong;


//===----------------------------------------------------------------------===//
// Various types
//===----------------------------------------------------------------------===//
int8_t ValueI8;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'ValueI8' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int8_t custi8ValueI8;

int16_t ValueI16 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'ValueI16' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int16_t custi16ValueI16 = 0;

int32_t ValueI32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'ValueI32' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int32_t custi32ValueI32 = 0;

int64_t ValueI64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'ValueI64' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int64_t custi64ValueI64 = 0;

uint8_t ValueU8 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'ValueU8' [readability-identifier-naming]
// CHECK-FIXES: {{^}}uint8_t custu8ValueU8 = 0;

uint16_t ValueU16 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'ValueU16' [readability-identifier-naming]
// CHECK-FIXES: {{^}}uint16_t custu16ValueU16 = 0;

uint32_t ValueU32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'ValueU32' [readability-identifier-naming]
// CHECK-FIXES: {{^}}uint32_t custu32ValueU32 = 0;

uint64_t ValueU64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'ValueU64' [readability-identifier-naming]
// CHECK-FIXES: {{^}}uint64_t custu64ValueU64 = 0;

float ValueFloat = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'ValueFloat' [readability-identifier-naming]
// CHECK-FIXES: {{^}}float custfValueFloat = 0;

double ValueDouble = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'ValueDouble' [readability-identifier-naming]
// CHECK-FIXES: {{^}}double custdValueDouble = 0;

char ValueChar = 'c';
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'ValueChar' [readability-identifier-naming]
// CHECK-FIXES: {{^}}char custcValueChar = 'c';

bool ValueBool = true;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'ValueBool' [readability-identifier-naming]
// CHECK-FIXES: {{^}}bool custbValueBool = true;

int ValueInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'ValueInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int custiValueInt = 0;

size_t ValueSize = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'ValueSize' [readability-identifier-naming]
// CHECK-FIXES: {{^}}size_t custnValueSize = 0;

wchar_t ValueWchar = 'w';
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'ValueWchar' [readability-identifier-naming]
// CHECK-FIXES: {{^}}wchar_t custwcValueWchar = 'w';

short ValueShort = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'ValueShort' [readability-identifier-naming]
// CHECK-FIXES: {{^}}short custsValueShort = 0;

unsigned ValueUnsigned = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'ValueUnsigned' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned custuValueUnsigned = 0;

signed ValueSigned = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'ValueSigned' [readability-identifier-naming]
// CHECK-FIXES: {{^}}signed custsValueSigned = 0;

long ValueLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'ValueLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}long custlValueLong = 0;

long long ValueLongLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for global variable 'ValueLongLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}long long custllValueLongLong = 0;

long long int ValueLongLongInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for global variable 'ValueLongLongInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}long long int custlliValueLongLongInt = 0;

long double ValueLongDouble = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for global variable 'ValueLongDouble' [readability-identifier-naming]
// CHECK-FIXES: {{^}}long double custldValueLongDouble = 0;

signed int ValueSignedInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global variable 'ValueSignedInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}signed int custsiValueSignedInt = 0;

signed short ValueSignedShort = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for global variable 'ValueSignedShort' [readability-identifier-naming]
// CHECK-FIXES: {{^}}signed short custssValueSignedShort = 0;

signed short int ValueSignedShortInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for global variable 'ValueSignedShortInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}signed short int custssiValueSignedShortInt = 0;

signed long long ValueSignedLongLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for global variable 'ValueSignedLongLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}signed long long custsllValueSignedLongLong = 0;

signed long int ValueSignedLongInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: invalid case style for global variable 'ValueSignedLongInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}signed long int custsliValueSignedLongInt = 0;

signed long ValueSignedLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for global variable 'ValueSignedLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}signed long custslValueSignedLong = 0;

unsigned long long int ValueUnsignedLongLongInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: invalid case style for global variable 'ValueUnsignedLongLongInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned long long int custulliValueUnsignedLongLongInt = 0;

unsigned long long ValueUnsignedLongLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: invalid case style for global variable 'ValueUnsignedLongLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned long long custullValueUnsignedLongLong = 0;

unsigned long int ValueUnsignedLongInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: invalid case style for global variable 'ValueUnsignedLongInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned long int custuliValueUnsignedLongInt = 0;

unsigned long ValueUnsignedLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for global variable 'ValueUnsignedLong' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned long custulValueUnsignedLong = 0;

unsigned short int ValueUnsignedShortInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: invalid case style for global variable 'ValueUnsignedShortInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned short int custusiValueUnsignedShortInt = 0;

unsigned short ValueUnsignedShort = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for global variable 'ValueUnsignedShort' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned short custusValueUnsignedShort = 0;

unsigned int ValueUnsignedInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for global variable 'ValueUnsignedInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned int custuiValueUnsignedInt = 0;

unsigned char ValueUnsignedChar = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for global variable 'ValueUnsignedChar' [readability-identifier-naming]
// CHECK-FIXES: {{^}}unsigned char custucValueUnsignedChar = 0;

long int ValueLongInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'ValueLongInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}long int custliValueLongInt = 0;


//===----------------------------------------------------------------------===//
// Specifier, Qualifier, Other keywords
//===----------------------------------------------------------------------===//
volatile int VolatileInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for global variable 'VolatileInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}volatile int custiVolatileInt = 0;

thread_local int ThreadLocalValueInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for global variable 'ThreadLocalValueInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}thread_local int custiThreadLocalValueInt = 0;

extern int ExternValueInt;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global variable 'ExternValueInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}extern int custiExternValueInt;

struct CDataBuffer {
    mutable size_t Size;
};
// CHECK-MESSAGES: :[[@LINE-2]]:20: warning: invalid case style for public member 'Size' [readability-identifier-naming]
// CHECK-FIXES: {{^}}    mutable size_t custnSize;

static constexpr int const &ConstExprInt = 42;
// CHECK-MESSAGES: :[[@LINE-1]]:29: warning: invalid case style for constexpr variable 'ConstExprInt' [readability-identifier-naming]
// CHECK-FIXES: {{^}}static constexpr int const &custiConstExprInt = 42;


//===----------------------------------------------------------------------===//
// Redefined types
//===----------------------------------------------------------------------===//
typedef int INDEX;
INDEX custiIndex = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'custiIndex' [readability-identifier-naming]
// CHECK-FIXES: {{^}}INDEX Index = 0;


//===----------------------------------------------------------------------===//
// Class and struct
//===----------------------------------------------------------------------===//
class ClassCase { int Func(); };
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 'ClassCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}class CClassCase { int Func(); };

class AbstractClassCase { virtual int Func() = 0; };
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for abstract class 'AbstractClassCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}class IAbstractClassCase { virtual int Func() = 0; };

class AbstractClassCase1 { virtual int Func1() = 0; int Func2(); };
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for abstract class 'AbstractClassCase1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}class IAbstractClassCase1 { virtual int Func1() = 0; int Func2(); };

class ClassConstantCase { public: static const int custiConstantCase; };
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for class 'ClassConstantCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}class CClassConstantCase { public: static const int custiConstantCase; };

struct StructCase { int Func(); };
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for class 'StructCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}struct CStructCase { int Func(); };

//===----------------------------------------------------------------------===//
// Other Cases
//===----------------------------------------------------------------------===//
int lower_case = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'lower_case' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int custiLowerCase = 0;

int lower_case1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'lower_case1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int custiLowerCase1 = 0;

int lower_case_2 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'lower_case_2' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int custiLowerCase2 = 0;

int UPPER_CASE = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'UPPER_CASE' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int custiUpperCase = 0;

int UPPER_CASE_1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'UPPER_CASE_1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int custiUpperCase1 = 0;

int camelBack = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'camelBack' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int custiCamelBack = 0;

int camelBack_1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'camelBack_1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int custiCamelBack1 = 0;

int camelBack2 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'camelBack2' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int custiCamelBack2 = 0;

int CamelCase = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'CamelCase' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int custiCamelCase = 0;

int CamelCase_1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'CamelCase_1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int custiCamelCase1 = 0;

int CamelCase2 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'CamelCase2' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int custiCamelCase2 = 0;

int camel_Snake_Back = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'camel_Snake_Back' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int custiCamelSnakeBack = 0;

int camel_Snake_Back_1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'camel_Snake_Back_1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int custiCamelSnakeBack1 = 0;

int Camel_Snake_Case = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'Camel_Snake_Case' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int custiCamelSnakeCase = 0;

int Camel_Snake_Case_1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'Camel_Snake_Case_1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}int custiCamelSnakeCase1 = 0;

//===----------------------------------------------------------------------===//
// Enum
//===----------------------------------------------------------------------===//
enum REV_TYPE { RevValid };
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: invalid case style for enum constant 'RevValid' [readability-identifier-naming]
// CHECK-FIXES: {{^}}enum REV_TYPE { rtRevValid };

enum EnumConstantCase { OneByte, TwoByte };
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: invalid case style for enum constant 'OneByte' [readability-identifier-naming]
// CHECK-MESSAGES: :[[@LINE-2]]:34: warning: invalid case style for enum constant 'TwoByte' [readability-identifier-naming]
// CHECK-FIXES: {{^}}enum EnumConstantCase { eccOneByte, eccTwoByte };

enum class ScopedEnumConstantCase { Case1 };
// CHECK-MESSAGES: :[[@LINE-1]]:37: warning: invalid case style for scoped enum constant 'Case1' [readability-identifier-naming]
// CHECK-FIXES: {{^}}enum class ScopedEnumConstantCase { seccCase1 };
// clang-format on
