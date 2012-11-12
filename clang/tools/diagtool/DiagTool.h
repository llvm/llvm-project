//===- DiagTool.h - Classes for defining diagtool tools -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the boilerplate for defining diagtool tools.
//
//===----------------------------------------------------------------------===//

#ifndef DIAGTOOL_DIAGTOOL_H
#define DIAGTOOL_DIAGTOOL_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ManagedStatic.h"
#include <string>


namespace diagtool {

class DiagTool {
  const std::string cmd;
  const std::string description;
public:
  DiagTool(llvm::StringRef toolCmd, llvm::StringRef toolDesc);
  virtual ~DiagTool();
  
  llvm::StringRef getName() const { return cmd; }  
  llvm::StringRef getDescription() const { return description; }  

  virtual int run(unsigned argc, char *argv[], llvm::raw_ostream &out) = 0;
};
  
class DiagTools {
  void *tools;
public:
  DiagTools();
  ~DiagTools();
  
  DiagTool *getTool(llvm::StringRef toolCmd);
  void registerTool(DiagTool *tool);  
  void printCommands(llvm::raw_ostream &out);  
};

extern llvm::ManagedStatic<DiagTools> diagTools;

template <typename DIAGTOOL>
class RegisterDiagTool {
public:
  RegisterDiagTool() { diagTools->registerTool(new DIAGTOOL()); }
};

} // end diagtool namespace

#define DEF_DIAGTOOL(NAME, DESC, CLSNAME)\
namespace {\
class CLSNAME : public diagtool::DiagTool {\
public:\
  CLSNAME() : DiagTool(NAME, DESC) {}\
  virtual ~CLSNAME() {}\
  virtual int run(unsigned argc, char *argv[], llvm::raw_ostream &out);\
};\
diagtool::RegisterDiagTool<CLSNAME> Register##CLSNAME;\
}

#endif
