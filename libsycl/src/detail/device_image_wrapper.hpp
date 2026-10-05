//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the helpers for device images and
/// programs.
///
//===----------------------------------------------------------------------===//

#ifndef _LIBSYCL_SRC_DETAIL_DEVICE_IMAGE_WRAPPER_HPP
#define _LIBSYCL_SRC_DETAIL_DEVICE_IMAGE_WRAPPER_HPP

#include <sycl/__impl/detail/config.hpp>

#include <detail/suppress_extra_warnings.hpp>

_LIBSYCL_SUPPRESS_EXTRA_WARNINGS_BEGIN
#include <llvm/Object/OffloadBinary.h>
_LIBSYCL_SUPPRESS_EXTRA_WARNINGS_END

#include <OffloadAPI.h>

#include <memory>
#include <string_view>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

class ContextImpl;
class DeviceImageManager;

/// A wrapper of liboffload program handle to manage its lifetime.
class ProgramWrapper {
public:
  /// Constructs ProgramWrapper by creating a liboffload program with the
  /// provided arguments.
  ///
  /// \param Context is the context to use for program creation.
  /// \param Device is the device to use for program creation.
  /// \param DevImage is the device image to use for program creation.
  /// \throw sycl::exception with sycl::errc::runtime when failed to create the
  /// program.
  ProgramWrapper(ContextImpl &Context, ol_device_handle_t Device,
                 const DeviceImageManager &DevImage);

  /// Releases the corresponding liboffload program handle by calling
  /// olDestroyProgram.
  ~ProgramWrapper();

  ProgramWrapper(const ProgramWrapper &) = delete;
  ProgramWrapper &operator=(const ProgramWrapper &) = delete;
  ProgramWrapper(ProgramWrapper &&) = delete;
  ProgramWrapper &operator=(ProgramWrapper &&) = delete;

  /// \return the corresponding liboffload program handle.
  ol_program_handle_t getOLHandle() { return MProgram; }

  /// Returns the liboffload kernel symbol for the specified kernel, looking it
  /// up in this program.
  ///
  /// liboffload caches symbols per program, so repeated and concurrent lookups
  /// of the same kernel return the same handle. Symbols belong to the program
  /// they were retrieved from: liboffload has no olDestroySymbol, so they are
  /// released together with this program.
  ///
  /// \param KernelName the name of the kernel to look up.
  /// \throw sycl::exception with sycl::errc::runtime when the symbol lookup
  /// fails.
  /// \return the liboffload symbol handle of the kernel.
  ol_symbol_handle_t getOrCreateKernel(std::string_view KernelName);

private:
  // Programs are owned by their context, so the context outlives them.
  ContextImpl &MContext;
  ol_program_handle_t MProgram{};
};

/// This class manages data parsing of device images.
class DeviceImageManager {
public:
  explicit DeviceImageManager(std::unique_ptr<llvm::object::OffloadBinary> Bin)
      : MBin(std::move(Bin)) {}
  // Explicitly delete copy constructor/operator= to avoid unintentional copies.
  DeviceImageManager(const DeviceImageManager &) = delete;
  DeviceImageManager &operator=(const DeviceImageManager &) = delete;

  DeviceImageManager(DeviceImageManager &&) = default;
  DeviceImageManager &operator=(DeviceImageManager &&) = default;

  ~DeviceImageManager() = default;

  /// \return a reference to the corresponding parsed OffloadBinary object.
  const llvm::object::OffloadBinary &getOffloadBinary() const { return *MBin; }

private:
  std::unique_ptr<llvm::object::OffloadBinary> MBin;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_SRC_DETAIL_DEVICE_IMAGE_WRAPPER_HPP
