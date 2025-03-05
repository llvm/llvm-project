//===- Timing.cpp - Execution time measurement facilities -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Facilities to measure and provide statistics on execution time.
//
//===----------------------------------------------------------------------===//

#include "flang/Support/Timing.h"
#include "llvm/Support/Format.h"

class OutputStrategyText : public mlir::OutputStrategy {
protected:
  static constexpr llvm::StringLiteral header = "Flang execution timing report";

public:
  OutputStrategyText(llvm::raw_ostream &os) : mlir::OutputStrategy(os) {}

  void printHeader(const mlir::TimeRecord &total) override {
    // Figure out how many spaces to description name.
    unsigned padding = (80 - header.size()) / 2;
    os << "===" << std::string(73, '-') << "===\n";
    os.indent(padding) << header << '\n';
    os << "===" << std::string(73, '-') << "===\n";

    // Print the total time followed by the section headers.
    os << llvm::format("  Total Execution Time: %.4f seconds\n\n", total.wall);
    os << "  ----User Time----  ----Wall Time----  ----Name----\n";
  }

  void printFooter() override { os.flush(); }

  void printTime(
      const mlir::TimeRecord &time, const mlir::TimeRecord &total) override {
    os << llvm::format(
        "  %8.4f (%5.1f%%)", time.user, 100.0 * time.user / total.user);
    os << llvm::format(
        "  %8.4f (%5.1f%%)  ", time.wall, 100.0 * time.wall / total.wall);
  }

  void printListEntry(llvm::StringRef name, const mlir::TimeRecord &time,
      const mlir::TimeRecord &total, bool lastEntry) override {
    printTime(time, total);
    os << name << "\n";
  }

  void printTreeEntry(unsigned indent, llvm::StringRef name,
      const mlir::TimeRecord &time, const mlir::TimeRecord &total) override {
    printTime(time, total);
    os.indent(indent) << name << "\n";
  }

  void printTreeEntryEnd(unsigned indent, bool lastEntry) override {}
};

namespace Fortran::support {

std::unique_ptr<mlir::OutputStrategy> createTimingFormatterText(
    llvm::raw_ostream &os) {
  return std::make_unique<OutputStrategyText>(os);
}

} // namespace Fortran::support
