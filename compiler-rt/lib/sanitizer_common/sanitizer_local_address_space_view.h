//===-- sanitizer_local_address_space_view.h --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// `LocalAddressSpaceView` provides the local (i.e. target and current address
// space are the same) implementation of the `AddressSpaveView` interface which
// provides a simple interface to load memory from another process (i.e.
// out-of-process)
//
// The `AddressSpaveView` interface requires that the type can be used as a
// template parameter to objects that wish to be able to operate in an
// out-of-process manner. In normal usage, objects are in-process and are thus
// instantiated with the `LocalAddressSpaceView` type. This type is used to
// load any pointers in instance methods. This implementation is effectively
// a no-op. When an object is to be used in an out-of-process manner it is
// instansiated with the `RemoteAddressSpaceView` type.
//
// By making `AddressSpaceView` a template parameter of an object, it can
// change its implementation at compile time which has no run time overhead.
// This also allows unifying in-process and out-of-process code which avoids
// code duplication.
//
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_LOCAL_ADDRES_SPACE_VIEW_H
#define SANITIZER_LOCAL_ADDRES_SPACE_VIEW_H

namespace __sanitizer {
struct LocalAddressSpaceView {
  // Load memory `sizeof(T) * num_elements` bytes of memory
  // from the target process (always local for this implementation)
  // starting at address `target_address`. The local copy of
  // this memory is returned as a pointer. It is guaranteed that
  //
  // * That the function will always return the same value
  //   for a given set of arguments.
  // * That the memory returned is writable and that writes will persist.
  //
  // The lifetime of loaded memory is implementation defined.
  template <typename T>
  static T *Load(T *target_address, uptr num_elements = 1) {
    // The target address space is the local address space so
    // nothing needs to be copied. Just return the pointer.
    return target_address;
  }
};
}  // namespace __sanitizer

#endif
