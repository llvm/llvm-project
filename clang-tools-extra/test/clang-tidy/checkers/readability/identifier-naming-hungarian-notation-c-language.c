// RUN: %check_clang_tidy %s readability-identifier-naming %t -- \
// RUN:   --config-file=%S/Inputs/identifier-naming/hungarian-notation1/.clang-tidy -- -I %S

#include "identifier-naming-standard-types.h"

// clang-format off
//===----------------------------------------------------------------------===//
// Cases to CheckOptions
//===----------------------------------------------------------------------===//
const int GlobalConstantCase = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for global constant 'GlobalConstantCase' [readability-identifier-naming]
// CHECK-FIXES: const int iGlobalConstantCase = 0;

const int* GlobalConstantPointerCase = NULL;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global pointer 'GlobalConstantPointerCase' [readability-identifier-naming]
// CHECK-FIXES: const int* piGlobalConstantPointerCase = NULL;

int* GlobalPointerCase = NULL;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global pointer 'GlobalPointerCase' [readability-identifier-naming]
// CHECK-FIXES: int* piGlobalPointerCase = NULL;

int GlobalVariableCase = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'GlobalVariableCase' [readability-identifier-naming]
// CHECK-FIXES: int iGlobalVariableCase = 0;

void Func1(){
  int const LocalConstantCase = 3;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for local constant 'LocalConstantCase' [readability-identifier-naming]
  // CHECK-FIXES: int const iLocalConstantCase = 3;

  unsigned const ConstantCase = 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for local constant 'ConstantCase' [readability-identifier-naming]
  // CHECK-FIXES: unsigned const uConstantCase = 1;

  int* const LocalConstantPointerCase = NULL;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for local constant pointer 'LocalConstantPointerCase' [readability-identifier-naming]
  // CHECK-FIXES: int* const piLocalConstantPointerCase = NULL;

  int *LocalPointerCase = NULL;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for local pointer 'LocalPointerCase' [readability-identifier-naming]
  // CHECK-FIXES: int *piLocalPointerCase = NULL;

  int LocalVariableCase = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for local variable 'LocalVariableCase' [readability-identifier-naming]
  // CHECK-FIXES: int iLocalVariableCase = 0;
}

struct CMyClass2 {
  char MemberCase;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for public member 'MemberCase' [readability-identifier-naming]
  // CHECK-FIXES: char cMemberCase;
};

static const int StaticConstantCase = 3;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for global constant 'StaticConstantCase' [readability-identifier-naming]
// CHECK-FIXES: static const int iStaticConstantCase = 3;

static int StaticVariableCase = 3;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global variable 'StaticVariableCase' [readability-identifier-naming]
// CHECK-FIXES: static int iStaticVariableCase = 3;

struct MyStruct { int StructCase; };
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: invalid case style for public member 'StructCase' [readability-identifier-naming]
// CHECK-FIXES: struct MyStruct { int iStructCase; };

struct shouldBeCamelCaseStruct { int iField; };
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for struct 'shouldBeCamelCaseStruct' [readability-identifier-naming]
// CHECK-FIXES: struct ShouldBeCamelCaseStruct { int iField; };

union MyUnion { int UnionCase; long lUnionCase; };
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for union 'MyUnion' [readability-identifier-naming]
// CHECK-MESSAGES: :[[@LINE-2]]:21: warning: invalid case style for public member 'UnionCase' [readability-identifier-naming]
// CHECK-FIXES: union myUnion { int iUnionCase; long lUnionCase; };

//===----------------------------------------------------------------------===//
// C string
//===----------------------------------------------------------------------===//
const char *NamePtr = "Name";
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for global pointer 'NamePtr' [readability-identifier-naming]
// CHECK-FIXES: const char *szNamePtr = "Name";

const char NameArray[] = "Name";
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global constant 'NameArray' [readability-identifier-naming]
// CHECK-FIXES: const char szNameArray[] = "Name";

const char *NamePtrArray[] = {"AA", "BB"};
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for global variable 'NamePtrArray' [readability-identifier-naming]
// CHECK-FIXES: const char *pszNamePtrArray[] = {"AA", "BB"};

const wchar_t *WideNamePtr = L"Name";
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for global pointer 'WideNamePtr' [readability-identifier-naming]
// CHECK-FIXES: const wchar_t *wszWideNamePtr = L"Name";

const wchar_t WideNameArray[] = L"Name";
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for global constant 'WideNameArray' [readability-identifier-naming]
// CHECK-FIXES: const wchar_t wszWideNameArray[] = L"Name";

const wchar_t *WideNamePtrArray[] = {L"AA", L"BB"};
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for global variable 'WideNamePtrArray' [readability-identifier-naming]
// CHECK-FIXES: const wchar_t *pwszWideNamePtrArray[] = {L"AA", L"BB"};


//===----------------------------------------------------------------------===//
// Microsoft Windows data types
//===----------------------------------------------------------------------===//
DWORD MsDword = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsDword' [readability-identifier-naming]
// CHECK-FIXES: DWORD dwMsDword = 0;

BYTE MsByte = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsByte' [readability-identifier-naming]
// CHECK-FIXES: BYTE byMsByte = 0;

WORD MsWord = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsWord' [readability-identifier-naming]
// CHECK-FIXES: WORD wMsWord = 0;

BOOL MsBool = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsBool' [readability-identifier-naming]
// CHECK-FIXES: BOOL bMsBool = 0;

BOOLEAN MsBoolean = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'MsBoolean' [readability-identifier-naming]
// CHECK-FIXES: BOOLEAN bMsBoolean = 0;

CHAR MsValueChar = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsValueChar' [readability-identifier-naming]
// CHECK-FIXES: CHAR cMsValueChar = 0;

UCHAR MsValueUchar = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueUchar' [readability-identifier-naming]
// CHECK-FIXES: UCHAR ucMsValueUchar = 0;

SHORT MsValueShort = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueShort' [readability-identifier-naming]
// CHECK-FIXES: SHORT sMsValueShort = 0;

USHORT MsValueUshort = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'MsValueUshort' [readability-identifier-naming]
// CHECK-FIXES: USHORT usMsValueUshort = 0;

WORD MsValueWord = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsValueWord' [readability-identifier-naming]
// CHECK-FIXES: WORD wMsValueWord = 0;

DWORD MsValueDword = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueDword' [readability-identifier-naming]
// CHECK-FIXES: DWORD dwMsValueDword = 0;

DWORD32 MsValueDword32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'MsValueDword32' [readability-identifier-naming]
// CHECK-FIXES: DWORD32 dw32MsValueDword32 = 0;

DWORD64 MsValueDword64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'MsValueDword64' [readability-identifier-naming]
// CHECK-FIXES: DWORD64 dw64MsValueDword64 = 0;

LONG MsValueLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsValueLong' [readability-identifier-naming]
// CHECK-FIXES: LONG lMsValueLong = 0;

ULONG MsValueUlong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueUlong' [readability-identifier-naming]
// CHECK-FIXES: ULONG ulMsValueUlong = 0;

ULONG32 MsValueUlong32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'MsValueUlong32' [readability-identifier-naming]
// CHECK-FIXES: ULONG32 ul32MsValueUlong32 = 0;

ULONG64 MsValueUlong64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'MsValueUlong64' [readability-identifier-naming]
// CHECK-FIXES: ULONG64 ul64MsValueUlong64 = 0;

ULONGLONG MsValueUlongLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for global variable 'MsValueUlongLong' [readability-identifier-naming]
// CHECK-FIXES: ULONGLONG ullMsValueUlongLong = 0;

HANDLE MsValueHandle = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global pointer 'MsValueHandle' [readability-identifier-naming]
// CHECK-FIXES: HANDLE hMsValueHandle = 0;

INT MsValueInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'MsValueInt' [readability-identifier-naming]
// CHECK-FIXES: INT iMsValueInt = 0;

INT8 MsValueInt8 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsValueInt8' [readability-identifier-naming]
// CHECK-FIXES: INT8 i8MsValueInt8 = 0;

INT16 MsValueInt16 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueInt16' [readability-identifier-naming]
// CHECK-FIXES: INT16 i16MsValueInt16 = 0;

INT32 MsValueInt32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueInt32' [readability-identifier-naming]
// CHECK-FIXES: INT32 i32MsValueInt32 = 0;

INT64 MsValueINt64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueINt64' [readability-identifier-naming]
// CHECK-FIXES: INT64 i64MsValueINt64 = 0;

UINT MsValueUint = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'MsValueUint' [readability-identifier-naming]
// CHECK-FIXES: UINT uiMsValueUint = 0;

UINT8 MsValueUint8 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'MsValueUint8' [readability-identifier-naming]
// CHECK-FIXES: UINT8 u8MsValueUint8 = 0;

UINT16 MsValueUint16 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'MsValueUint16' [readability-identifier-naming]
// CHECK-FIXES: UINT16 u16MsValueUint16 = 0;

UINT32 MsValueUint32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'MsValueUint32' [readability-identifier-naming]
// CHECK-FIXES: UINT32 u32MsValueUint32 = 0;

UINT64 MsValueUint64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'MsValueUint64' [readability-identifier-naming]
// CHECK-FIXES: UINT64 u64MsValueUint64 = 0;

PVOID MsValuePvoid = NULL;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global pointer 'MsValuePvoid' [readability-identifier-naming]
// CHECK-FIXES: PVOID pMsValuePvoid = NULL;


//===----------------------------------------------------------------------===//
// Array
//===----------------------------------------------------------------------===//
unsigned GlobalUnsignedArray[] = {1, 2, 3};
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'GlobalUnsignedArray' [readability-identifier-naming]
// CHECK-FIXES: unsigned aGlobalUnsignedArray[] = {1, 2, 3};

int GlobalIntArray[] = {1, 2, 3};
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'GlobalIntArray' [readability-identifier-naming]
// CHECK-FIXES: int aGlobalIntArray[] = {1, 2, 3};

int DataInt[1] = {0};
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'DataInt' [readability-identifier-naming]
// CHECK-FIXES: int aDataInt[1] = {0};

int DataArray[2] = {0};
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'DataArray' [readability-identifier-naming]
// CHECK-FIXES: int aDataArray[2] = {0};


//===----------------------------------------------------------------------===//
// Pointer
//===----------------------------------------------------------------------===//
int *DataIntPtr[1] = {0};
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'DataIntPtr' [readability-identifier-naming]
// CHECK-FIXES: int *paDataIntPtr[1] = {0};

void *BufferPtr1;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global pointer 'BufferPtr1' [readability-identifier-naming]
// CHECK-FIXES: void *pBufferPtr1;

void **BufferPtr2;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global pointer 'BufferPtr2' [readability-identifier-naming]
// CHECK-FIXES: void **ppBufferPtr2;

void **pBufferPtr3;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global pointer 'pBufferPtr3' [readability-identifier-naming]
// CHECK-FIXES: void **ppBufferPtr3;

int *pBufferPtr4;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global pointer 'pBufferPtr4' [readability-identifier-naming]
// CHECK-FIXES: int *piBufferPtr4;

typedef void (*FUNC_PTR_HELLO)();
FUNC_PTR_HELLO Hello = NULL;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for global pointer 'Hello' [readability-identifier-naming]
// CHECK-FIXES: FUNC_PTR_HELLO fnHello = NULL;

void *ValueVoidPtr = NULL;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global pointer 'ValueVoidPtr' [readability-identifier-naming]
// CHECK-FIXES: void *pValueVoidPtr = NULL;

ptrdiff_t PtrDiff = NULL;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for global variable 'PtrDiff' [readability-identifier-naming]
// CHECK-FIXES: ptrdiff_t pPtrDiff = NULL;

int8_t *ValueI8Ptr;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global pointer 'ValueI8Ptr' [readability-identifier-naming]
// CHECK-FIXES: int8_t *pi8ValueI8Ptr;

uint8_t *ValueU8Ptr;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global pointer 'ValueU8Ptr' [readability-identifier-naming]
// CHECK-FIXES: uint8_t *pu8ValueU8Ptr;

unsigned char *ValueUcPtr;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for global pointer 'ValueUcPtr' [readability-identifier-naming]
// CHECK-FIXES: unsigned char *pucValueUcPtr;

unsigned char **ValueUcPtr2;
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: invalid case style for global pointer 'ValueUcPtr2' [readability-identifier-naming]
// CHECK-FIXES: unsigned char **ppucValueUcPtr2;

void MyFunc2(void* Val){}
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: invalid case style for pointer parameter 'Val' [readability-identifier-naming]
// CHECK-FIXES: void MyFunc2(void* pVal){}


//===----------------------------------------------------------------------===//
// Various types
//===----------------------------------------------------------------------===//
int8_t ValueI8;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'ValueI8' [readability-identifier-naming]
// CHECK-FIXES: int8_t i8ValueI8;

int16_t ValueI16 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'ValueI16' [readability-identifier-naming]
// CHECK-FIXES: int16_t i16ValueI16 = 0;

int32_t ValueI32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'ValueI32' [readability-identifier-naming]
// CHECK-FIXES: int32_t i32ValueI32 = 0;

int64_t ValueI64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'ValueI64' [readability-identifier-naming]
// CHECK-FIXES: int64_t i64ValueI64 = 0;

uint8_t ValueU8 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'ValueU8' [readability-identifier-naming]
// CHECK-FIXES: uint8_t u8ValueU8 = 0;

uint16_t ValueU16 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'ValueU16' [readability-identifier-naming]
// CHECK-FIXES: uint16_t u16ValueU16 = 0;

uint32_t ValueU32 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'ValueU32' [readability-identifier-naming]
// CHECK-FIXES: uint32_t u32ValueU32 = 0;

uint64_t ValueU64 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'ValueU64' [readability-identifier-naming]
// CHECK-FIXES: uint64_t u64ValueU64 = 0;

float ValueFloat = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'ValueFloat' [readability-identifier-naming]
// CHECK-FIXES: float fValueFloat = 0;

double ValueDouble = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'ValueDouble' [readability-identifier-naming]
// CHECK-FIXES: double dValueDouble = 0;

char ValueChar = 'c';
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'ValueChar' [readability-identifier-naming]
// CHECK-FIXES: char cValueChar = 'c';

bool ValueBool = true;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'ValueBool' [readability-identifier-naming]
// CHECK-FIXES: bool bValueBool = true;

int ValueInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'ValueInt' [readability-identifier-naming]
// CHECK-FIXES: int iValueInt = 0;

size_t ValueSize = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'ValueSize' [readability-identifier-naming]
// CHECK-FIXES: size_t nValueSize = 0;

wchar_t ValueWchar = 'w';
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: invalid case style for global variable 'ValueWchar' [readability-identifier-naming]
// CHECK-FIXES: wchar_t wcValueWchar = 'w';

short ValueShort = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'ValueShort' [readability-identifier-naming]
// CHECK-FIXES: short sValueShort = 0;

unsigned ValueUnsigned = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'ValueUnsigned' [readability-identifier-naming]
// CHECK-FIXES: unsigned uValueUnsigned = 0;

signed ValueSigned = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: invalid case style for global variable 'ValueSigned' [readability-identifier-naming]
// CHECK-FIXES: signed sValueSigned = 0;

long ValueLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for global variable 'ValueLong' [readability-identifier-naming]
// CHECK-FIXES: long lValueLong = 0;

long long ValueLongLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: invalid case style for global variable 'ValueLongLong' [readability-identifier-naming]
// CHECK-FIXES: long long llValueLongLong = 0;

long long int ValueLongLongInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for global variable 'ValueLongLongInt' [readability-identifier-naming]
// CHECK-FIXES: long long int lliValueLongLongInt = 0;

long double ValueLongDouble = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for global variable 'ValueLongDouble' [readability-identifier-naming]
// CHECK-FIXES: long double ldValueLongDouble = 0;

signed int ValueSignedInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global variable 'ValueSignedInt' [readability-identifier-naming]
// CHECK-FIXES: signed int siValueSignedInt = 0;

signed short ValueSignedShort = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for global variable 'ValueSignedShort' [readability-identifier-naming]
// CHECK-FIXES: signed short ssValueSignedShort = 0;

signed short int ValueSignedShortInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for global variable 'ValueSignedShortInt' [readability-identifier-naming]
// CHECK-FIXES: signed short int ssiValueSignedShortInt = 0;

signed long long ValueSignedLongLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for global variable 'ValueSignedLongLong' [readability-identifier-naming]
// CHECK-FIXES: signed long long sllValueSignedLongLong = 0;

signed long int ValueSignedLongInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: invalid case style for global variable 'ValueSignedLongInt' [readability-identifier-naming]
// CHECK-FIXES: signed long int sliValueSignedLongInt = 0;

signed long ValueSignedLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: invalid case style for global variable 'ValueSignedLong' [readability-identifier-naming]
// CHECK-FIXES: signed long slValueSignedLong = 0;

unsigned long long int ValueUnsignedLongLongInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: invalid case style for global variable 'ValueUnsignedLongLongInt' [readability-identifier-naming]
// CHECK-FIXES: unsigned long long int ulliValueUnsignedLongLongInt = 0;

unsigned long long ValueUnsignedLongLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: invalid case style for global variable 'ValueUnsignedLongLong' [readability-identifier-naming]
// CHECK-FIXES: unsigned long long ullValueUnsignedLongLong = 0;

unsigned long int ValueUnsignedLongInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:19: warning: invalid case style for global variable 'ValueUnsignedLongInt' [readability-identifier-naming]
// CHECK-FIXES: unsigned long int uliValueUnsignedLongInt = 0;

unsigned long ValueUnsignedLong = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for global variable 'ValueUnsignedLong' [readability-identifier-naming]
// CHECK-FIXES: unsigned long ulValueUnsignedLong = 0;

unsigned short int ValueUnsignedShortInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:20: warning: invalid case style for global variable 'ValueUnsignedShortInt' [readability-identifier-naming]
// CHECK-FIXES: unsigned short int usiValueUnsignedShortInt = 0;

unsigned short ValueUnsignedShort = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:16: warning: invalid case style for global variable 'ValueUnsignedShort' [readability-identifier-naming]
// CHECK-FIXES: unsigned short usValueUnsignedShort = 0;

unsigned int ValueUnsignedInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for global variable 'ValueUnsignedInt' [readability-identifier-naming]
// CHECK-FIXES: unsigned int uiValueUnsignedInt = 0;

unsigned char ValueUnsignedChar = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: invalid case style for global variable 'ValueUnsignedChar' [readability-identifier-naming]
// CHECK-FIXES: unsigned char ucValueUnsignedChar = 0;

long int ValueLongInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:10: warning: invalid case style for global variable 'ValueLongInt' [readability-identifier-naming]
// CHECK-FIXES: long int liValueLongInt = 0;


//===----------------------------------------------------------------------===//
// Specifier, Qualifier, Other keywords
//===----------------------------------------------------------------------===//
volatile int VolatileInt = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning: invalid case style for global variable 'VolatileInt' [readability-identifier-naming]
// CHECK-FIXES: volatile int iVolatileInt = 0;

extern int ExternValueInt;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: invalid case style for global variable 'ExternValueInt' [readability-identifier-naming]
// CHECK-FIXES: extern int iExternValueInt;

//===----------------------------------------------------------------------===//
// Redefined types
//===----------------------------------------------------------------------===//
typedef int INDEX;
INDEX iIndex = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for global variable 'iIndex' [readability-identifier-naming]
// CHECK-FIXES: INDEX Index = 0;


//===----------------------------------------------------------------------===//
// Other Cases
//===----------------------------------------------------------------------===//
int lower_case = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'lower_case' [readability-identifier-naming]
// CHECK-FIXES: int iLowerCase = 0;

int lower_case1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'lower_case1' [readability-identifier-naming]
// CHECK-FIXES: int iLowerCase1 = 0;

int lower_case_2 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'lower_case_2' [readability-identifier-naming]
// CHECK-FIXES: int iLowerCase2 = 0;

int UPPER_CASE = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'UPPER_CASE' [readability-identifier-naming]
// CHECK-FIXES: int iUpperCase = 0;

int UPPER_CASE_1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'UPPER_CASE_1' [readability-identifier-naming]
// CHECK-FIXES: int iUpperCase1 = 0;

int camelBack = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'camelBack' [readability-identifier-naming]
// CHECK-FIXES: int iCamelBack = 0;

int camelBack_1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'camelBack_1' [readability-identifier-naming]
// CHECK-FIXES: int iCamelBack1 = 0;

int camelBack2 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'camelBack2' [readability-identifier-naming]
// CHECK-FIXES: int iCamelBack2 = 0;

int CamelCase = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'CamelCase' [readability-identifier-naming]
// CHECK-FIXES: int iCamelCase = 0;

int CamelCase_1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'CamelCase_1' [readability-identifier-naming]
// CHECK-FIXES: int iCamelCase1 = 0;

int CamelCase2 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'CamelCase2' [readability-identifier-naming]
// CHECK-FIXES: int iCamelCase2 = 0;

int camel_Snake_Back = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'camel_Snake_Back' [readability-identifier-naming]
// CHECK-FIXES: int iCamelSnakeBack = 0;

int camel_Snake_Back_1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'camel_Snake_Back_1' [readability-identifier-naming]
// CHECK-FIXES: int iCamelSnakeBack1 = 0;

int Camel_Snake_Case = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'Camel_Snake_Case' [readability-identifier-naming]
// CHECK-FIXES: int iCamelSnakeCase = 0;

int Camel_Snake_Case_1 = 0;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: invalid case style for global variable 'Camel_Snake_Case_1' [readability-identifier-naming]
// CHECK-FIXES: int iCamelSnakeCase1 = 0;

//===----------------------------------------------------------------------===//
// Enum
//===----------------------------------------------------------------------===//
enum REV_TYPE { RevValid };
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: invalid case style for enum constant 'RevValid' [readability-identifier-naming]
// CHECK-FIXES: enum REV_TYPE { rtRevValid };

enum EnumConstantCase { OneByte, TwoByte };
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: invalid case style for enum constant 'OneByte' [readability-identifier-naming]
// CHECK-MESSAGES: :[[@LINE-2]]:34: warning: invalid case style for enum constant 'TwoByte' [readability-identifier-naming]
// CHECK-FIXES: enum EnumConstantCase { eccOneByte, eccTwoByte };
