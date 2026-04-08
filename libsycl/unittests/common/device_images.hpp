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

#ifndef _LIBSYCL_UNITTESTS_COMMON_DEVICE_IMAGES_HPP
#define _LIBSYCL_UNITTESTS_COMMON_DEVICE_IMAGES_HPP

#include <detail/device_binary_structures.hpp>

#include <llvm/Frontend/Offloading/Utility.h>
#include <llvm/Object/OffloadBinary.h>

namespace sycl::unittest {

inline llvm::object::OffloadBinary::OffloadingImage createSYCLImage(
    llvm::StringRef SymbolsBlob,
    llvm::object::ImageKind ImageKind = llvm::object::IMG_SPIRV,
    llvm::object::OffloadKind OffloadKind = llvm::object::OFK_SYCL) {
  llvm::object::OffloadBinary::OffloadingImage Image;
  Image.TheImageKind = ImageKind;
  Image.TheOffloadKind = OffloadKind;
  Image.Flags = 0;
  Image.StringData["triple"] = sycl::detail::DeviceBinaryTripleSPIRV64;
  Image.StringData["compile-opts"] = "";
  Image.StringData["link-opts"] = "";
  Image.StringData["symbols"] = SymbolsBlob;
  static constexpr char DummyImageData[] = "dummy image data";
  Image.Image = llvm::MemoryBuffer::getMemBufferCopy(
      llvm::StringRef(DummyImageData, sizeof(DummyImageData)));
  return Image;
}

inline llvm::SmallString<0> createSYCLDeviceBinary(
    llvm::ArrayRef<llvm::StringRef> KernelNames,
    llvm::object::ImageKind ImageKind = llvm::object::IMG_SPIRV,
    llvm::object::OffloadKind OffloadKind = llvm::object::OFK_SYCL) {
  llvm::SmallString<0> SymbolsBlob;
  llvm::offloading::sycl::writeSymbolTable(KernelNames, SymbolsBlob);

  llvm::object::OffloadBinary::OffloadingImage Image =
      createSYCLImage(SymbolsBlob, ImageKind, OffloadKind);
  return llvm::object::OffloadBinary::write(Image);
}

} // namespace sycl::unittest

#endif // _LIBSYCL_UNITTESTS_COMMON_DEVICE_IMAGES_HPP
