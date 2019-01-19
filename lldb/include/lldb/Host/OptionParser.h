//===-- OptionParser.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OptionParser_h_
#define liblldb_OptionParser_h_

#include <mutex>
#include <string>

#include "llvm/ADT/StringRef.h"

struct option;

namespace lldb_private {

struct OptionDefinition;

struct Option {
  // The definition of the option that this refers to.
  const OptionDefinition *definition;
  // if not NULL, set *flag to val when option found
  int *flag;
  // if flag not NULL, value to set *flag to; else return value
  int val;
};

class OptionParser {
public:
  enum OptionArgument { eNoArgument = 0, eRequiredArgument, eOptionalArgument };

  static void Prepare(std::unique_lock<std::mutex> &lock);

  static void EnableError(bool error);

  static int Parse(int argc, char *const argv[], llvm::StringRef optstring,
                   const Option *longopts, int *longindex);

  static char *GetOptionArgument();
  static int GetOptionIndex();
  static int GetOptionErrorCause();
  static std::string GetShortOptionString(struct option *long_options);
};
}

#endif // liblldb_OptionParser_h_
