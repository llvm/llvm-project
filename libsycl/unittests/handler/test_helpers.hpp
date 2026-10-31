//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_UNITTESTS_HANDLER_TEST_HELPERS_HPP
#define _LIBSYCL_UNITTESTS_HANDLER_TEST_HELPERS_HPP

#include <mock/helpers.hpp>

#include <sycl/__impl/detail/config.hpp>

#include <gmock/gmock.h>

#include <algorithm>
#include <vector>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace unittests {

// Expects Count olGetMemInfo(OL_MEM_INFO_DEVICE) queries, each for one of
// ExpectedPtrs, and reports Device as the owning device.
inline void
expectDeviceMemoryInfo(mock::MockWrapper &Mock,
                       const std::vector<const void *> &ExpectedPtrs,
                       ol_device_handle_t Device, int Count) {
  EXPECT_CALL(Mock.get(),
              olGetMemInfo(::testing::_, ::testing::_, OL_MEM_INFO_DEVICE,
                           sizeof(ol_device_handle_t), ::testing::_))
      .Times(Count)
      .WillRepeatedly([ExpectedPtrs, Device](
                          ol_context_handle_t, const void *Ptr, ol_mem_info_t,
                          size_t, void *PropValue) -> ol_result_t {
        EXPECT_NE(std::find(ExpectedPtrs.begin(), ExpectedPtrs.end(), Ptr),
                  ExpectedPtrs.end());
        *(static_cast<ol_device_handle_t *>(PropValue)) = Device;
        return OL_SUCCESS;
      });
}

} // namespace unittests
_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_UNITTESTS_HANDLER_TEST_HELPERS_HPP
