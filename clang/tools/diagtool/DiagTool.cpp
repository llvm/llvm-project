//===- DiagTool.cpp - Classes for defining diagtool tools -------------------===//
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

#include "DiagTool.h"
#include "llvm/ADT/StringMap.h"
#include <vector>

using namespace diagtool;

DiagTool::DiagTool(llvm::StringRef toolCmd,
                   llvm::StringRef toolDesc)
  : cmd(toolCmd), description(toolDesc) {}

DiagTool::~DiagTool() {}

typedef llvm::StringMap<DiagTool *> ToolMap;
static inline ToolMap *getTools(void *v) { return static_cast<ToolMap*>(v); }

DiagTools::DiagTools() : tools(new ToolMap()) {}
DiagTools::~DiagTools() { delete getTools(tools); }

DiagTool *DiagTools::getTool(llvm::StringRef toolCmd) {
  ToolMap::iterator it = getTools(tools)->find(toolCmd);
  return (it == getTools(tools)->end()) ? nullptr : it->getValue();
}

void DiagTools::registerTool(DiagTool *tool) {
  (*getTools(tools))[tool->getName()] = tool;
}

void DiagTools::printCommands(llvm::raw_ostream &out) {
  std::vector<llvm::StringRef> toolNames;
  unsigned maxName = 0;
  for (ToolMap::iterator it = getTools(tools)->begin(),
       ei = getTools(tools)->end(); it != ei; ++it) {
    toolNames.push_back(it->getKey());
    unsigned len = it->getKey().size();
    if (len > maxName)
      maxName = len;    
  }
  std::sort(toolNames.begin(), toolNames.end());

  for (std::vector<llvm::StringRef>::iterator it = toolNames.begin(),
       ei = toolNames.end(); it != ei; ++it) {

    out << "  " << (*it);
    unsigned spaces = (maxName + 3) - (it->size());
    for (unsigned i = 0; i < spaces; ++i)
      out << ' ';
    
    out << getTool(*it)->getDescription() << '\n';    
  }
}

namespace diagtool {
  llvm::ManagedStatic<DiagTools> diagTools;
}
