//===--- AddressSpaces.h - Language-specific address spaces -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Provides definitions for the various language-specific address
/// spaces.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_ADDRESSSPACES_H
#define LLVM_CLANG_BASIC_ADDRESSSPACES_H

namespace clang {

namespace LangAS {

/// \brief Defines the address space values used by the address space qualifier
/// of QualType.
///
enum ID {
  // The default value 0 is the value used in QualType for the the situation
  // where there is no address space qualifier. For most languages, this also
  // corresponds to the situation where there is no address space qualifier in
  // the source code, except for OpenCL, where the address space value 0 in
  // QualType represents private address space in OpenCL source code.
  Default = 0,

  // OpenCL specific address spaces.
  opencl_global,
  opencl_local,
  opencl_constant,
  opencl_generic,

  // CUDA specific address spaces.
  cuda_device,
  cuda_constant,
  cuda_shared,

  // This denotes the count of language-specific address spaces and also
  // the offset added to the target-specific address spaces, which are usually
  // specified by address space attributes __attribute__(address_space(n))).
  FirstTargetAddressSpace
};

/// The type of a lookup table which maps from language-specific address spaces
/// to target-specific ones.
typedef unsigned Map[FirstTargetAddressSpace];
}

}

#endif
