// RUN: %clang_cc1 -triple amdgcn -std=c++11 %s

typedef __WINT_TYPE__ wint_t;

#if _WIN32
static_assert(sizeof(wchar_t)==2, "fail");
static_assert(sizeof(wint_t)==2, "fail");
#else
static_assert(sizeof(wchar_t)==4, "fail");
static_assert(sizeof(wint_t)==4, "fail");
#endif
