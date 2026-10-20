//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test ensures that we can catch an Objective-C++ exception by type when
// throwing an exception created via `std::make_exception_ptr`.
// See http://llvm.org/PR135089.

// UNSUPPORTED: no-exceptions
// UNSUPPORTED: c++03

// This test requires the Objective-C ARC, which is (only?) available on Darwin
// out-of-the-box.
// REQUIRES: has-fobjc-arc && darwin

// FIXME: including <Foundation/Foundation.h> seems to be currently broken with modules enabled
// ADDITIONAL_COMPILE_FLAGS: -fobjc-arc -fno-modules

#include <cassert>
#include <exception>

#import <Foundation/Foundation.h>

NSError* RecoverException(const std::exception_ptr& exc) {
  try {
    std::rethrow_exception(exc);
  } catch (NSError* error) {
    return error;
  } catch (...) {
  }
  return nullptr;
}

int main(int, char**) {
  NSError* error         = [NSError errorWithDomain:NSPOSIXErrorDomain code:EPERM userInfo:nil];
  std::exception_ptr exc = std::make_exception_ptr(error);
  NSError* recov         = RecoverException(exc);
  assert(recov != nullptr);

  return 0;
}
