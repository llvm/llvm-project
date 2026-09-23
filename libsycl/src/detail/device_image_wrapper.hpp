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

#ifndef _LIBSYCL_DEVICE_IMAGE_WRAPPER
#define _LIBSYCL_DEVICE_IMAGE_WRAPPER

#include <sycl/__impl/detail/config.hpp>

#include <llvm/Object/OffloadBinary.h>

#include <OffloadAPI.h>

#include <memory>
#include <string_view>
#include <unordered_map>

_LIBSYCL_BEGIN_NAMESPACE_SYCL
namespace detail {

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
  ProgramWrapper(ol_context_handle_t Context, ol_device_handle_t Device,
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
  /// up in this program on first use.
  ///
  /// Symbols belong to the program they were retrieved from: liboffload has no
  /// olDestroySymbol, so they are released together with this program. Caching
  /// them here rather than per device keeps a symbol from ever being handed out
  /// for a program it does not belong to.
  ///
  /// \param KernelName the name of the kernel to look up.
  /// \throw sycl::exception with sycl::errc::runtime when the symbol lookup
  /// fails.
  /// \return the liboffload symbol handle of the kernel.
  ol_symbol_handle_t getOrCreateKernel(std::string_view KernelName);

private:
  ol_program_handle_t MProgram{};

  // Kernel names are backed by the "symbols" string of the device image this
  // program was created from, so entries stay valid only while that image is
  // registered. ContextImpl::releaseProgramsForImage() destroys this program
  // before the image goes away.
  std::unordered_map<std::string_view, ol_symbol_handle_t> MKernels;
};

/// This class manages data parsing of device images.
class DeviceImageManager {
public:
  DeviceImageManager(std::unique_ptr<llvm::object::OffloadBinary> Bin)
      : MBin(std::move(Bin)) {}
  // Explicitly delete copy constructor/operator= to avoid unintentional copies.
  DeviceImageManager(const DeviceImageManager &) = delete;
  DeviceImageManager &operator=(const DeviceImageManager &) = delete;

  DeviceImageManager(DeviceImageManager &&) = default;
  DeviceImageManager &operator=(DeviceImageManager &&) = default;

  ~DeviceImageManager() = default;

  /// \return a reference to the corresponding parsed OffloadBinary object.
  const llvm::object::OffloadBinary &getOffloadBinary() const { return *MBin; }

protected:
  std::unique_ptr<llvm::object::OffloadBinary> MBin;
};

} // namespace detail

_LIBSYCL_END_NAMESPACE_SYCL

#endif // _LIBSYCL_DEVICE_IMAGE_WRAPPER
