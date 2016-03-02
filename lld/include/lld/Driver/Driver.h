//===- lld/Driver/Driver.h - Linker Driver Emulator -----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// Interface for Drivers which convert command line arguments into
/// LinkingContext objects, then perform the link.
///
//===----------------------------------------------------------------------===//

#ifndef LLD_DRIVER_DRIVER_H
#define LLD_DRIVER_DRIVER_H

#include "lld/Core/LLVM.h"
#include "lld/Core/Node.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <set>
#include <vector>

namespace lld {
class LinkingContext;
class MachOLinkingContext;

/// Base class for all Drivers.
class Driver {
protected:

  /// Performs link using specified options
  static bool link(LinkingContext &context,
                   raw_ostream &diag = llvm::errs());

  /// Parses the LLVM options from the context.
  static void parseLLVMOptions(const LinkingContext &context);

private:
  Driver() = delete;
};

/// Driver for darwin/ld64 'ld' command line options.
class DarwinLdDriver : public Driver {
public:
  /// Parses command line arguments same as darwin's ld and performs link.
  /// Returns true iff there was an error.
  static bool linkMachO(llvm::ArrayRef<const char *> args,
                        raw_ostream &diag = llvm::errs());

  /// Uses darwin style ld command line options to update LinkingContext object.
  /// Returns true iff there was an error.
  static bool parse(llvm::ArrayRef<const char *> args,
                    MachOLinkingContext &info,
                    raw_ostream &diag = llvm::errs());

private:
  DarwinLdDriver() = delete;
};

/// Driver for Windows 'link.exe' command line options
namespace coff {
bool link(llvm::ArrayRef<const char *> args);
}

namespace elf {
bool link(llvm::ArrayRef<const char *> args, raw_ostream &diag = llvm::errs());
}

} // end namespace lld

#endif
