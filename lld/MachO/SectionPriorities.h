//===- SectionPriorities.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_SECTION_PRIORITIES_H
#define LLD_MACHO_SECTION_PRIORITIES_H

#include "InputSection.h"
#include "llvm/ADT/DenseMap.h"

namespace lld {
namespace macho {

// Reads every input section's call graph profile, and combines them into
// config->callGraphProfile. If an order file is present, any edges where one
// or both of the vertices are specified in the order file are discarded.
void extractCallGraphProfile();

// Reads the order file at `path` into config->priorities.
//
// An order file has one entry per line, in the following format:
//
//   <cpu>:<object file>:<symbol name>
//
// <cpu> and <object file> are optional. If not specified, then that entry
// matches any symbol of that name. Parsing this format is not quite
// straightforward because the symbol name itself can contain colons, so when
// encountering a colon, we consider the preceding characters to decide if it
// can be a valid CPU type or file path.
//
// If a symbol is matched by multiple entries, then it takes the lowest-ordered
// entry (the one nearest to the front of the list.)
//
// The file can also have line comments that start with '#'.
void parseOrderFile(StringRef path);

// Returns layout priorities for some or all input sections. Sections are laid
// out in decreasing order; that is, a higher priority section will be closer
// to the beginning of its output section.
//
// If either an order file or a call graph profile are present, this is used
// as the source of priorities. If both are present, the order file takes
// precedence, but the call graph profile is still used for symbols that don't
// appear in the order file. If neither is present, an empty map is returned.
//
// Each section gets assigned the priority of the highest-priority symbol it
// contains.
llvm::DenseMap<const InputSection *, size_t> buildInputSectionPriorities();
} // namespace macho
} // namespace lld

#endif
