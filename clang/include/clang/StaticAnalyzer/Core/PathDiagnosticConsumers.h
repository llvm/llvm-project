//===--- PathDiagnosticClients.h - Path Diagnostic Clients ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the interface to create different path diagostic clients.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_PATH_DIAGNOSTIC_CLIENTS_H
#define LLVM_CLANG_GR_PATH_DIAGNOSTIC_CLIENTS_H

#include <string>
#include <vector>

namespace clang {

class Preprocessor;

namespace ento {

class PathDiagnosticConsumer;
typedef std::vector<PathDiagnosticConsumer*> PathDiagnosticConsumers;

void createHTMLDiagnosticConsumer(PathDiagnosticConsumers &C,
                                  const std::string& prefix,
                                  const Preprocessor &PP);

void createPlistDiagnosticConsumer(PathDiagnosticConsumers &C,
                                   const std::string& prefix,
                                   const Preprocessor &PP);

void createPlistMultiFileDiagnosticConsumer(PathDiagnosticConsumers &C,
                                            const std::string& prefix,
                                            const Preprocessor &PP);

void createTextPathDiagnosticConsumer(PathDiagnosticConsumers &C,
                                      const std::string& prefix,
                                      const Preprocessor &PP);

} // end 'ento' namespace
} // end 'clang' namespace

#endif
