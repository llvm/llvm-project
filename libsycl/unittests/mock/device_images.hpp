//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains common fake device-image data used by libsycl unit
/// tests.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_UNITTESTS_MOCK_DEVICE_IMAGES_HPP
#define _LIBSYCL_UNITTESTS_MOCK_DEVICE_IMAGES_HPP

#include <detail/program_manager.hpp>

namespace sycl::unittest {

inline constexpr llvm::offloading::EntryTy GenericEntry = {
    /// Reserved bytes used to detect an older version of the struct, always
    /// zero.
    0,
    /// The current version of the struct for runtime forward compatibility.
    1,
    /// The expected consumer of this entry, e.g. CUDA or OpenMP.
    llvm::object::OFK_SYCL,
    /// Flags associated with the global.
    0,
    /// The address of the global to be registered by the runtime.
    nullptr,
    /// The name of the symbol in the device image.
    nullptr,
    /// The number of bytes the symbol takes.
    0,
    /// Extra generic data used to register this entry.
    0,
    /// An extra pointer, usually null.
    nullptr};

inline constexpr detail::__sycl_tgt_device_image GenericDeviceImage = {
    // Version
    3,
    // OffloadKind
    llvm::object::OFK_SYCL,
    // ImageFormat
    llvm::object::IMG_SPIRV,
    // TripleString
    "spirv64-unknown-unknown",
    // CompileOptions
    "",
    // LinkOptions
    "",
    // ImageStart
    nullptr,
    // ImageEnd
    nullptr,
    // EntriesBegin
    nullptr,
    // EntriesEnd
    nullptr,
    // PropertiesBegin
    nullptr,
    // PropertiesEnd
    nullptr};

inline constexpr detail::__sycl_tgt_bin_desc GenericDeviceImages = {
    // Version.
    1,
    // Num binaries
    0,
    /// Device binaries data.
    nullptr,
    // HostEntriesBegin.
    nullptr,
    // HostEntriesEnd.
    nullptr};

} // namespace sycl::unittest

#endif // _LIBSYCL_UNITTESTS_MOCK_DEVICE_IMAGES_HPP