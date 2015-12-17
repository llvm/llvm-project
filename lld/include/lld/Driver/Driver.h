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
class CoreLinkingContext;
class MachOLinkingContext;
class PECOFFLinkingContext;
class ELFLinkingContext;

typedef std::vector<std::unique_ptr<File>> FileVector;

FileVector makeErrorFile(StringRef path, std::error_code ec);
FileVector parseMemberFiles(std::unique_ptr<File> File);
FileVector loadFile(LinkingContext &ctx, StringRef path, bool wholeArchive);

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

/// Driver for "universal" lld tool which can mimic any linker command line
/// parsing once it figures out which command line flavor to use.
class UniversalDriver : public Driver {
public:
  /// Determine flavor and pass control to Driver for that flavor.
  static bool link(llvm::MutableArrayRef<const char *> args,
                   raw_ostream &diag = llvm::errs());

private:
  UniversalDriver() = delete;
};

/// Driver for gnu/binutil 'ld' command line options.
class GnuLdDriver : public Driver {
public:
  /// Parses command line arguments same as gnu/binutils ld and performs link.
  /// Returns true iff an error occurred.
  static bool linkELF(llvm::ArrayRef<const char *> args,
                      raw_ostream &diag = llvm::errs());

  /// Uses gnu/binutils style ld command line options to fill in options struct.
  /// Returns true iff there was an error.
  static bool parse(llvm::ArrayRef<const char *> args,
                    std::unique_ptr<ELFLinkingContext> &context,
                    raw_ostream &diag = llvm::errs());

  /// Parses a given memory buffer as a linker script and evaluate that.
  /// Public function for testing.
  static std::error_code evalLinkerScript(ELFLinkingContext &ctx,
                                          std::unique_ptr<MemoryBuffer> mb,
                                          raw_ostream &diag, bool nostdlib);

  /// A factory method to create an instance of ELFLinkingContext.
  static std::unique_ptr<ELFLinkingContext>
  createELFLinkingContext(llvm::Triple triple);

private:
  static llvm::Triple getDefaultTarget(const char *progName);
  static bool applyEmulation(llvm::Triple &triple,
                             llvm::opt::InputArgList &args,
                             raw_ostream &diag);
  static void addPlatformSearchDirs(ELFLinkingContext &ctx,
                                    llvm::Triple &triple,
                                    llvm::Triple &baseTriple);

  GnuLdDriver() = delete;
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
void link(llvm::ArrayRef<const char *> args);
}

namespace elf2 {
void link(llvm::ArrayRef<const char *> args);
}

/// Driver for lld unit tests
class CoreDriver : public Driver {
public:
  /// Parses command line arguments same as lld-core and performs link.
  /// Returns true iff there was an error.
  static bool link(llvm::ArrayRef<const char *> args,
                   raw_ostream &diag = llvm::errs());

  /// Uses lld-core command line options to fill in options struct.
  /// Returns true iff there was an error.
  static bool parse(llvm::ArrayRef<const char *> args, CoreLinkingContext &info,
                    raw_ostream &diag = llvm::errs());

private:
  CoreDriver() = delete;
};

} // end namespace lld

#endif
