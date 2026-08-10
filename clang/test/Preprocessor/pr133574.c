// RUN: %clang_cc1 -E -verify %s
// expected-no-diagnostics

#define DATE_LBR __D\
ATE__

const char* test1(void) {
    return __DATE\
__;
}
const char* test2(void) {
    return DATE_LBR;
}

#define TIME_LBR __TIME_\
_

const char* test3(void) {
    return __TIM\
E__;
}

const char* test4(void) {
    return TIME_LBR;
}

#define LINE_LBR __LI\
NE__

int test5(void) {
    return _\
_LINE__;
}

int test6(void) {
    return LINE_LBR;
}

#define FILE_LBR __FI\
LE__

const char* test7(void) {
    return __\
FILE__;
}

const char* test8(void) {
    return FILE_LBR;
}

#define FILE_NAME_LBR __FILE_NA\
ME__

const char* test9(void) {
    return __FILE_NAM\
E__;
}

const char* test10(void) {
    return FILE_NAME_LBR;
}

#define BASE_FILE_LBR __BASE_FIL\
E__

const char* test11(void) {
    return __BASE_\
FILE__;
}

const char* test12(void) {
    return BASE_FILE_LBR;
}

#define INCLUDE_LEVEL_LBR __INCLUDE\
_LEVEL__

int test13(void) {
    return __IN\
CLUDE_LEVEL__;
}

int test14(void) {
    return INCLUDE_LEVEL_LBR;
}

#define TIMESTAMP_LBR __TIMESTA\
MP__

const char* test15(void) {
    return __TIMESTA\
MP__;
}

const char* test16(void) {
    return TIMESTAMP_LBR;
}

#define FLT_EVAL_METHOD_LBR __FLT_EVAL_METH\
OD__

int test17(void) {
    return __FL\
T_EVAL_METHOD__;
}

int test18(void) {
    return FLT_EVAL_METHOD_LBR;
}

#define COUNTER_LBR __COUNTE\
R__

int test19(void) {
    return _\
_COUNTER__;
}

int test20(void) {
    return COUNTER_LBR;
}
