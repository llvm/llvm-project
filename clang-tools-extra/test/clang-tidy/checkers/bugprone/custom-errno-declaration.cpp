// RUN: %check_clang_tidy %s bugprone-custom-errno-declaration %t

namespace errno_test_0 {
    extern int errno;
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: errno declaration detected, include cerrno instead [bugprone-custom-errno-declaration]
    // CHECK-FIXES: {{^}}{{$}}
} // namespace errno_test_0

namespace errno_test_1 {
    extern "C" int errno;
    // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: errno declaration detected, include cerrno instead [bugprone-custom-errno-declaration]
    // CHECK-FIXES: {{^}}{{$}}
} // namespace errno_test_1

namespace errno_test_2 {
    extern int errno, preserved;
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: errno declaration detected, include cerrno instead [bugprone-custom-errno-declaration]
    // CHECK-FIXES: {{^}}{{$}}
} // namespace errno_test_2

namespace errno_test_3 { // all cases should be ignored in this namespace
    extern bool errno;

    void foo(int errno) {}

    int fooo()
    {
        int errno = 0;
        return errno;
    }
} // namespace errno_test_3
