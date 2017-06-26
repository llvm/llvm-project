//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// REQUIRES: availability=macosx10.12

// test availability of new/delete operators introduced in c++17.

#include <new>

int main () {
  int *p0 = new ((std::align_val_t)16) int(1);
  (void)p0;
  int *p1 = new ((std::align_val_t)16) int[1];
  (void)p1;
  // expected-error@-4 {{call to unavailable function 'operator new': introduced in macOS 10.13}}
  // expected-note@new:* {{candidate function has been explicitly made unavailable}}
  // expected-note@new:* {{candidate function not viable: no known conversion from 'std::align_val_t' to 'const std::nothrow_t' for 2nd argument}}
  // expected-note@new:* {{candidate function not viable: no known conversion from 'std::align_val_t' to 'void *' for 2nd argument}}
  // expected-note@new:* {{candidate function not viable: requires single argument '__sz', but 2 arguments were provided}}
  // expected-note@new:* {{candidate function not viable: requires 3 arguments, but 2 were provided}}

  // expected-error@-9 {{call to unavailable function 'operator new[]': introduced in macOS 10.13}}
  // expected-note@new:* {{candidate function has been explicitly made unavailable}}
  // expected-note@new:* {{candidate function not viable: no known conversion from 'std::align_val_t' to 'const std::nothrow_t' for 2nd argument}}
  // expected-note@new:* {{candidate function not viable: no known conversion from 'std::align_val_t' to 'void *' for 2nd argument}}
  // expected-note@new:* {{candidate function not viable: requires single argument '__sz', but 2 arguments were provided}}
  // expected-note@new:* {{candidate function not viable: requires 3 arguments, but 2 were provided}}
  return 0;
}
