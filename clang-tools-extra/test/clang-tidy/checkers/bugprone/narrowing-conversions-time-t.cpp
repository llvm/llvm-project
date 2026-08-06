// RUN: %check_clang_tidy -check-suffix=DEFAULT %s \
// RUN: bugprone-narrowing-conversions %t -- \
// RUN: -config='{CheckOptions: {bugprone-narrowing-conversions.WarnOnTimeTNarrowingConversion: true}}'

typedef long long time_t;
typedef time_t mytime;
typedef mytime opaque_time;

namespace std {
    using ::time_t;
}

using int32_t = int;
using uint32_t = unsigned int;
using int64_t = long;
using uint64_t = unsigned long;

time_t time(time_t *);

void takes_int(int);
int returns_int(float f) {
    return f;
}

void ignore(int i, long l, double d) {
    short s1 = i;
    short s2 = (short)i;
    int i1 = l, i2 = (int)l;
    float f = d;
    takes_int(l);
}

void implicit(time_t t) {

    int32_t i32 = t;
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:19: warning: conversion from 'time_t' (aka 'long long') to 'int32_t' (aka 'int') may not preserve the full range of time_t

    uint32_t u32 = t;
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:20: warning: conversion from 'time_t' (aka 'long long') to 'uint32_t' (aka 'unsigned int') may not preserve the full range of time_t

    takes_int(t);
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:15: warning: conversion from 'time_t' (aka 'long long') to 'int' may not preserve the full range of time_t

    bool b = t;
}

void explicit_test(time_t t) {
    int i = (int)t;
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:13: warning: conversion from 'time_t' (aka 'long long') to 'int' may not preserve the full range of time_t

    int32_t i32 = (int32_t)t;
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:19: warning: conversion from 'time_t' (aka 'long long') to 'int32_t' (aka 'int') may not preserve the full range of time_t

    short s = (short)t;
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:15: warning: conversion from 'time_t' (aka 'long long') to 'short' may not preserve the full range of time_t

    int c = static_cast<int>(t);
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:13: warning: conversion from 'time_t' (aka 'long long') to 'int' may not preserve the full range of time_t

    int j = int(t);
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:13: warning: conversion from 'time_t' (aka 'long long') to 'int' may not preserve the full range of time_t
}

void misc(std::time_t t, int offset) {
    int i = t;
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:13: warning: conversion from 'std::time_t' (aka 'long long') to 'int' may not preserve the full range of time_t

    opaque_time op = time(nullptr);
    uint32_t u32 = (uint32_t)op;
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:20: warning: conversion from 'opaque_time' (aka 'long long') to 'uint32_t' (aka 'unsigned int') may not preserve the full range of time_t

    int oper = t + offset;
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:16: warning: conversion from 'std::time_t' (aka 'long long') to 'int' may not preserve the full range of time_t

#define TO_UINT32(arg) ((uint32_t)(arg))
    uint32_t trunc = TO_UINT32(t);
    // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:22: warning: conversion from 'std::time_t' (aka 'long long') to 'uint32_t' (aka 'unsigned int') may not preserve the full range of time_t
}
