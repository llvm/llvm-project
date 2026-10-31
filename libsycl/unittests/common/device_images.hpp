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

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace unittests {

/// The triple clang-sycl-linker records in SYCL offload images.
inline constexpr llvm::StringRef DefaultImageTriple = "spirv64-unknown-unknown";

inline llvm::object::OffloadBinary::OffloadingImage
createSYCLImage(llvm::StringRef SymbolsBlob,
                llvm::object::ImageKind ImageKind = llvm::object::IMG_SPIRV,
                llvm::object::OffloadKind OffloadKind = llvm::object::OFK_SYCL,
                llvm::StringRef Triple = DefaultImageTriple) {
  llvm::object::OffloadBinary::OffloadingImage Image;
  Image.TheImageKind = ImageKind;
  Image.TheOffloadKind = OffloadKind;
  Image.StringData["triple"] = Triple;
  Image.StringData["symbols"] = SymbolsBlob;
  static constexpr char DummyImageData[] = "dummy image data";
  Image.Image = llvm::MemoryBuffer::getMemBufferCopy(
      llvm::StringRef(DummyImageData, sizeof(DummyImageData)));
  return Image;
}

inline llvm::SmallString<0> createSYCLDeviceBinary(
    llvm::ArrayRef<llvm::StringRef> KernelNames,
    llvm::object::ImageKind ImageKind = llvm::object::IMG_SPIRV,
    llvm::object::OffloadKind OffloadKind = llvm::object::OFK_SYCL,
    llvm::StringRef Triple = DefaultImageTriple) {
  llvm::SmallString<0> SymbolsBlob;
  llvm::offloading::sycl::writeSymbolTable(KernelNames, SymbolsBlob);

  llvm::object::OffloadBinary::OffloadingImage Image =
      createSYCLImage(SymbolsBlob, ImageKind, OffloadKind, Triple);
  return llvm::object::OffloadBinary::write(Image);
}

} // namespace unittests
_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_UNITTESTS_COMMON_DEVICE_IMAGES_HPP
