//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_HOST_PYTHONPATHSETUP_H
#define LLDB_SOURCE_HOST_PYTHONPATHSETUP_H

#include "llvm/Support/Error.h"

#ifdef LLDB_PYTHON_DLL_RELATIVE_PATH
/// Resolve the full path of the directory defined by
/// LLDB_PYTHON_DLL_RELATIVE_PATH. If it exists, add it to the list of DLL
/// search directories.
///
/// \return `true` if the library was added to the search path.
/// `false` otherwise.
bool AddPythonDLLToSearchPath();
#endif

/// Attempts to setup the DLL search path for the Python runtime library.
///
/// In the following paragraphs, python3xx.dll refers to the Python runtime
/// library name which is defined by LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME, e.g.
/// python311.dll for Python 3.11.
///
/// The setup flow depends on which macros are defined:
///
/// - If only LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME is defined, checks whether
///   python3xx.dll can be loaded. Returns an error if it cannot.
///
/// - If only LLDB_PYTHON_DLL_RELATIVE_PATH is defined, attempts to resolve the
///   relative path and add it to the DLL search path. Returns an error if this
///   fails. Note that this may succeed even if python3xx.dll is not present in
///   the added search path.
///
/// - If both LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME and
///   LLDB_PYTHON_DLL_RELATIVE_PATH are defined, first checks if python3xx.dll
///   can be loaded. If successful, returns immediately. Otherwise, attempts to
///   resolve the relative path and add it to the DLL search path, then checks
///   again if python3xx.dll can be loaded.
///
/// \return If LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME is defined, return the
/// absolute path of the Python shared library which was resolved or an error if
/// it could not be found. If LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME and
/// LLDB_PYTHON_DLL_RELATIVE_PATH are not defined, return an empty string.
llvm::Expected<std::string> SetupPythonRuntimeLibrary();

#endif // LLDB_SOURCE_HOST_PYTHONPATHSETUP_H
