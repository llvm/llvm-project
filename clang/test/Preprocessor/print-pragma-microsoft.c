// RUN: %clang_cc1 %s -fms-extensions -E -o - | FileCheck %s

#define BAR "2"
#pragma comment(linker, "bar=" BAR)
// CHECK: #pragma comment(linker, "bar=" "2")
#pragma comment(user, "Compiled on " __DATE__ " at " __TIME__)
// CHECK: #pragma comment(user, "Compiled on " "{{[^"]*}}" " at " "{{[^"]*}}")

#define KEY1 "KEY1"
#define KEY2 "KEY2"
#define VAL1 "VAL1\""
#define VAL2 "VAL2"

#pragma detect_mismatch(KEY1 KEY2, VAL1 VAL2)
// CHECK: #pragma detect_mismatch("KEY1" "KEY2", "VAL1\"" "VAL2")

#define _CRT_PACKING 8
#pragma pack(push, _CRT_PACKING)
// CHECK: #pragma pack(push, 8)
#pragma pack(pop)
