// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits  -fsafe-buffer-usage-suggestions -DCMD_UNSAFE_ATTR=[[clang::unsafe_buffer_usage]] %s 2>&1 | FileCheck %s


// no need to check fix-its for definition in this test ...
void foo(int *p) {
  int tmp = p[5];
}

// Will use the macro defined from the command line:
void foo(int *);
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:1-[[@LINE-1]]:1}:"CMD_UNSAFE_ATTR "
// CHECK: fix-it:{{.*}}:{[[@LINE-2]]:16-[[@LINE-2]]:16}:";\nvoid foo(std::span<int>)"


#undef CMD_UNSAFE_ATTR
void foo(int *);
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:1-[[@LINE-1]]:1}:"{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} "
// CHECK: fix-it:{{.*}}:{[[@LINE-2]]:16-[[@LINE-2]]:16}:";\nvoid foo(std::span<int>)"


#define UNSAFE_ATTR [[clang::unsafe_buffer_usage]]

void foo(int *);
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:1-[[@LINE-1]]:1}:"UNSAFE_ATTR "
// CHECK: fix-it:{{.*}}:{[[@LINE-2]]:16-[[@LINE-2]]:16}:";\nvoid foo(std::span<int>)"

#undef UNSAFE_ATTR

#if __has_cpp_attribute(clang::unsafe_buffer_usage)
#define UNSAFE_ATTR [[clang::unsafe_buffer_usage]]
#endif

void foo(int *);
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:1-[[@LINE-1]]:1}:"UNSAFE_ATTR "
// CHECK: fix-it:{{.*}}:{[[@LINE-2]]:16-[[@LINE-2]]:16}:";\nvoid foo(std::span<int>)"

#undef UNSAFE_ATTR

#if __has_cpp_attribute(clang::unsafe_buffer_usage)
// we don't know how to use this macro
#define UNSAFE_ATTR(x) [[clang::unsafe_buffer_usage]]
#endif

void foo(int *);
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:1-[[@LINE-1]]:1}:"{{\[}}{{\[}}clang::unsafe_buffer_usage{{\]}}{{\]}} "
// CHECK: fix-it:{{.*}}:{[[@LINE-2]]:16-[[@LINE-2]]:16}:";\nvoid foo(std::span<int>)"

#undef UNSAFE_ATTR

#define UNSAFE_ATTR_1 [[clang::unsafe_buffer_usage]]
#define UNSAFE_ATTR_2 [[clang::unsafe_buffer_usage]]
#define UNSAFE_ATTR_3 [[clang::unsafe_buffer_usage]]

// Should use the last defined macro (i.e., UNSAFE_ATTR_3) for
// `[[clang::unsafe_buffer_usage]]`
void foo(int *p);
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:1-[[@LINE-1]]:1}:"UNSAFE_ATTR_3 "
// CHECK: fix-it:{{.*}}:{[[@LINE-2]]:17-[[@LINE-2]]:17}:";\nvoid foo(std::span<int> p)"


#define WRONG_ATTR_1 [clang::unsafe_buffer_usage]]
#define WRONG_ATTR_2 [[clang::unsafe_buffer_usage]
#define WRONG_ATTR_3 [[clang::unsafe_buffer_usag]]

// The last defined macro for
// `[[clang::unsafe_buffer_usage]]` is still UNSAFE_ATTR_3
void foo(int *p);
// CHECK: fix-it:{{.*}}:{[[@LINE-1]]:1-[[@LINE-1]]:1}:"UNSAFE_ATTR_3 "
// CHECK: fix-it:{{.*}}:{[[@LINE-2]]:17-[[@LINE-2]]:17}:";\nvoid foo(std::span<int> p)"
