//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_TOOLS_H
#define _LIBCPP_STACKTRACE_TOOLS_H

#include <__config>
#include <__config_site>
#include <cassert>
#include <cstddef>
#include <list>
#include <string>

#include <__stacktrace/context.h>
#include <__stacktrace/entry.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

struct tool {
  char const* progName_;
  explicit tool(char const* progName) : progName_(progName) {}
  constexpr virtual ~tool() = default;

  /** Construct complete `argv` for the spawned process.
  Includes the program name at argv[0], followed by flags */
  virtual std::pmr::list<std::pmr::string> buildArgs(context& trace) const = 0;

  /** Parse line(s) output by the tool, and modify `entry`. */
  virtual void parseOutput(context& trace, entry& e, std::istream& output) const = 0;
};

struct llvm_symbolizer : tool {
  constexpr virtual ~llvm_symbolizer() = default;
  llvm_symbolizer() : llvm_symbolizer("llvm-symbolizer") {}
  explicit llvm_symbolizer(char const* progName) : tool{progName} {}
  std::pmr::list<std::pmr::string> buildArgs(context& trace) const override;
  void parseOutput(context& trace, entry& entry, std::istream& output) const override;
};

struct addr2line : tool {
  constexpr virtual ~addr2line() = default;
  addr2line() : addr2line("addr2line") {}
  explicit addr2line(char const* progName) : tool{progName} {}
  std::pmr::list<std::pmr::string> buildArgs(context& trace) const override;
  void parseOutput(context& trace, entry& e, std::istream& stream) const override;
};

struct atos : tool {
  constexpr virtual ~atos() = default;
  atos() : atos("atos") {}
  explicit atos(char const* progName) : tool{progName} {}
  std::pmr::list<std::pmr::string> buildArgs(context& trace) const override;
  void parseOutput(context& trace, entry& entry, std::istream& output) const override;
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_TOOLS_H
