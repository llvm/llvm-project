// RUN: %clang_cl_asan /Od %s -Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <Windows.h>
#include <iostream>

// Small sanity test to make sure ASAN does not stomp on
// GetLastError values. This is motivated by __asan::ShouldReplaceIntrinsic,
// which remedied infinite recursion due to ntdll exception handling paths calling
// instrumented functions on startup before shadow memory can be committed.
int TestNTStatusMaintained() {
  ::SetLastError(ERROR_SUCCESS);
  constexpr unsigned long c_initialSizeGuess = 1;
  wchar_t szBuffer[c_initialSizeGuess];
  unsigned long size = ::GetDllDirectoryW(c_initialSizeGuess, szBuffer);
  auto le = ::GetLastError();
  if (size == 0 && le != ERROR_SUCCESS) {
    std::cerr << "Last error is different.\n";
    return -1;
  }
  std::cerr << "Success.\n";
  return 0;
  // CHECK: Success.
}

int main() { return TestNTStatusMaintained(); }