//===-- Unittests for utimes --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// temp file related stuff
#include "src/fcntl/open.h"    // to open
#include "src/unistd/close.h"  // to close
#include "src/sys/stat/stat.h" // for info
#include "src/unistd/unlink.h" // to delete
// testing error handling
#include "test/UnitTest/Test.h"
#include "src/errno/libc_errno.h"           
#include "test/UnitTest/ErrnoSetterMatcher.h"
// dependencies for the tests themselves
#include "hdr/types/struct_timeval.h"
#include <cerrno>
#include <fcntl.h>
#include "hdr/fcntl_macros.h"
// the utimes function
#include "src/sys/time/utimes.h" 
constexpr const char* TEST_FILE = "testdata/utimes.test"; 

// SUCCESS: Takes a file and successfully updates 
// its last access and modified times.
TEST(LlvmLibcUtimesTest, ChangeTimesSpecific){
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Succeeds;
  
  // make a dummy timeval struct
  struct timeval times[2];
  times[0].tv_sec = 54321;
  times[0].tv_usec = 12345;
  times[1].tv_sec = 43210;
  times[1].tv_usec = 23456;
  
  // ensure utimes succeeds
  ASSERT_THAT(LIBC_NAMESPACE::utimes(TEST_FILE, times), Succeeds(0));

  // verify the times values against stat of the TEST_FILE
  struct stat statbuf;
  ASSERT_EQ(stat(TEST_FILE, &statbuf), 0);
  
  // seconds
  ASSERT_EQ(statbuf.st_atim.tv_sec, times[0].tv_sec);
  ASSERT_EQ(statbuf.st_mtim.tv_sec, times[1].tv_sec);

  //microseconds
  ASSERT_EQ(statbuf.st_atim.tv_nsec, times[0].tv_usec * 1000);
  ASSERT_EQ(statbuf.st_mtim.tv_nsec, times[1].tv_usec * 1000); 
}

// FAILURE: Invalid values in the timeval struct
// to check that utimes rejects it.
TEST(LlvmLibcUtimesTest, InvalidMicroseconds){
  using LIBC_NAMESPACE::testing::ErrnoSetterMatcher::Fails; 
  
  // make a dummy timeval struct 
  // populated with bad usec values
  struct timeval times[2];
  times[0].tv_sec = 54321;
  times[0].tv_usec = 4567;
  times[1].tv_sec = 43210;
  times[1].tv_usec = 1000000; //invalid
  
  // ensure utimes fails
  ASSERT_THAT(LIBC_NAMESPACE::utimes(TEST_FILE, times), Fails(EINVAL));
  
  // check for failure on 
  // the other possible bad values

  times[0].tv_sec = 54321;
  times[0].tv_usec = -4567; //invalid
  times[1].tv_sec = 43210;
  times[1].tv_usec = 1000;
  
  // ensure utimes fails once more
  ASSERT_THAT(LIBC_NAMESPACE::utimes(TEST_FILE, times), Fails(EINVAL));  
}