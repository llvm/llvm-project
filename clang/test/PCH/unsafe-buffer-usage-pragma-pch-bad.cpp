// Test that a serialized safe buffer opt-out region begins and ends at different files.
// RUN: %clang_cc1 -Wno-unused-value -std=c++20 -emit-pch -o %t %s -include %s
// RUN: not %clang_cc1 -Wno-unused-value -Wunsafe-buffer-usage -std=c++20 -include-pch %t %s 2>&1 | FileCheck %s

// CHECK: fatal error:{{.*}}'Cannot de-serialize a safe buffer opt-out region that begins and ends at different files'
#ifndef A_H
#define A_H

int a() {
#pragma clang unsafe_buffer_usage begin
  return 0;
}

#elif (defined(A_H) && !defined(B_H))
#define B_H

#pragma clang unsafe_buffer_usage end

#else

int main() {
  return a();
}

#endif
