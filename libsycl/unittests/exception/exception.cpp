//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <sycl/sycl.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

using namespace sycl;

TEST(ExceptionTest, ConstructorsWithoutContext) {
  std::string Msg = "custom error message";
  const char *MsgCStr = "c-string error message";

  exception Ex1(make_error_code(errc::runtime), Msg);
  EXPECT_EQ(Ex1.code(), make_error_code(errc::runtime));
  EXPECT_EQ(Ex1.category(), sycl_category());
  EXPECT_STREQ(Ex1.what(), Msg.c_str());
  EXPECT_FALSE(Ex1.has_context());
  EXPECT_THROW(
      {
        try {
          Ex1.get_context();
        } catch (const sycl::exception &E) {
          EXPECT_EQ(E.code(), make_error_code(errc::invalid));
          throw;
        }
      },
      sycl::exception);

  exception Ex2(make_error_code(errc::kernel), MsgCStr);
  EXPECT_EQ(Ex2.code(), make_error_code(errc::kernel));
  EXPECT_STREQ(Ex2.what(), MsgCStr);
  EXPECT_FALSE(Ex2.has_context());

  exception Ex3(make_error_code(errc::accessor));
  EXPECT_EQ(Ex3.code(), make_error_code(errc::accessor));
  EXPECT_STREQ(Ex3.what(), "");
  EXPECT_FALSE(Ex3.has_context());

  exception Ex4(static_cast<int>(errc::nd_range), sycl_category(), Msg);
  EXPECT_EQ(Ex4.code(), make_error_code(errc::nd_range));
  EXPECT_STREQ(Ex4.what(), Msg.c_str());
  EXPECT_FALSE(Ex4.has_context());

  exception Ex5(static_cast<int>(errc::event), sycl_category(), MsgCStr);
  EXPECT_EQ(Ex5.code(), make_error_code(errc::event));
  EXPECT_STREQ(Ex5.what(), MsgCStr);
  EXPECT_FALSE(Ex5.has_context());

  exception Ex6(static_cast<int>(errc::kernel_argument), sycl_category());
  EXPECT_EQ(Ex6.code(), make_error_code(errc::kernel_argument));
  EXPECT_STREQ(Ex6.what(), "");
  EXPECT_FALSE(Ex6.has_context());
}

TEST(ExceptionTest, ConstructorsWithContext) {
  auto Platforms = sycl::platform::get_platforms();
  ASSERT_FALSE(Platforms.empty());
  context Ctx = Platforms[0].khr_get_default_context();

  std::string Msg = "context error message";
  const char *MsgCStr = "context c-string error message";

  exception Ex1(Ctx, make_error_code(errc::build), Msg);
  EXPECT_EQ(Ex1.code(), make_error_code(errc::build));
  EXPECT_EQ(Ex1.category(), sycl_category());
  EXPECT_STREQ(Ex1.what(), Msg.c_str());
  EXPECT_TRUE(Ex1.has_context());
  EXPECT_NO_THROW({ EXPECT_EQ(Ex1.get_context(), Ctx); });

  exception Ex2(Ctx, make_error_code(errc::invalid), MsgCStr);
  EXPECT_EQ(Ex2.code(), make_error_code(errc::invalid));
  EXPECT_STREQ(Ex2.what(), MsgCStr);
  EXPECT_TRUE(Ex2.has_context());
  EXPECT_EQ(Ex2.get_context(), Ctx);

  exception Ex3(Ctx, make_error_code(errc::memory_allocation));
  EXPECT_EQ(Ex3.code(), make_error_code(errc::memory_allocation));
  EXPECT_STREQ(Ex3.what(), "");
  EXPECT_TRUE(Ex3.has_context());
  EXPECT_EQ(Ex3.get_context(), Ctx);

  exception Ex4(Ctx, static_cast<int>(errc::platform), sycl_category(), Msg);
  EXPECT_EQ(Ex4.code(), make_error_code(errc::platform));
  EXPECT_STREQ(Ex4.what(), Msg.c_str());
  EXPECT_TRUE(Ex4.has_context());
  EXPECT_EQ(Ex4.get_context(), Ctx);

  exception Ex5(Ctx, static_cast<int>(errc::profiling), sycl_category(),
                MsgCStr);
  EXPECT_EQ(Ex5.code(), make_error_code(errc::profiling));
  EXPECT_STREQ(Ex5.what(), MsgCStr);
  EXPECT_TRUE(Ex5.has_context());
  EXPECT_EQ(Ex5.get_context(), Ctx);

  exception Ex6(Ctx, static_cast<int>(errc::feature_not_supported),
                sycl_category());
  EXPECT_EQ(Ex6.code(), make_error_code(errc::feature_not_supported));
  EXPECT_STREQ(Ex6.what(), "");
  EXPECT_TRUE(Ex6.has_context());
  EXPECT_EQ(Ex6.get_context(), Ctx);
}

TEST(ExceptionTest, CopyAndMoveSemantics) {
  static_assert(std::is_nothrow_copy_constructible_v<exception>,
                "sycl::exception must be nothrow copy constructible");

  constexpr char Message[] = "original exception";

  auto Platforms = sycl::platform::get_platforms();
  ASSERT_FALSE(Platforms.empty());
  context Ctx = Platforms[0].khr_get_default_context();

  exception Orig(Ctx, make_error_code(errc::runtime), Message);
  EXPECT_TRUE(Orig.has_context());

  exception Copy = Orig;
  EXPECT_TRUE(Copy.has_context());
  EXPECT_EQ(Copy.get_context(), Ctx);
  EXPECT_EQ(Copy.code(), make_error_code(errc::runtime));
  EXPECT_STREQ(Copy.what(), Message);

  exception Assigned(make_error_code(errc::invalid));
  EXPECT_FALSE(Assigned.has_context());
  Assigned = Copy;
  EXPECT_TRUE(Assigned.has_context());
  EXPECT_EQ(Assigned.get_context(), Ctx);
  EXPECT_EQ(Assigned.code(), make_error_code(errc::runtime));
  EXPECT_STREQ(Assigned.what(), Message);
}
