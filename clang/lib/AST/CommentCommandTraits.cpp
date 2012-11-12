//===--- CommentCommandTraits.cpp - Comment command properties --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/CommentCommandTraits.h"
#include "llvm/ADT/STLExtras.h"

namespace clang {
namespace comments {

#include "clang/AST/CommentCommandInfo.inc"

CommandTraits::CommandTraits(llvm::BumpPtrAllocator &Allocator) :
    NextID(llvm::array_lengthof(Commands)), Allocator(Allocator)
{ }

const CommandInfo *CommandTraits::getCommandInfoOrNULL(StringRef Name) const {
  if (const CommandInfo *Info = getBuiltinCommandInfo(Name))
    return Info;
  return getRegisteredCommandInfo(Name);
}

const CommandInfo *CommandTraits::getCommandInfo(unsigned CommandID) const {
  if (const CommandInfo *Info = getBuiltinCommandInfo(CommandID))
    return Info;
  return getRegisteredCommandInfo(CommandID);
}

const CommandInfo *CommandTraits::registerUnknownCommand(StringRef CommandName) {
  char *Name = Allocator.Allocate<char>(CommandName.size() + 1);
  memcpy(Name, CommandName.data(), CommandName.size());
  Name[CommandName.size()] = '\0';

  // Value-initialize (=zero-initialize in this case) a new CommandInfo.
  CommandInfo *Info = new (Allocator) CommandInfo();
  Info->Name = Name;
  Info->ID = NextID++;
  Info->IsUnknownCommand = true;

  RegisteredCommands.push_back(Info);

  return Info;
}

const CommandInfo *CommandTraits::getBuiltinCommandInfo(
                                                  unsigned CommandID) {
  if (CommandID < llvm::array_lengthof(Commands))
    return &Commands[CommandID];
  return NULL;
}

const CommandInfo *CommandTraits::getRegisteredCommandInfo(
                                                  StringRef Name) const {
  for (unsigned i = 0, e = RegisteredCommands.size(); i != e; ++i) {
    if (RegisteredCommands[i]->Name == Name)
      return RegisteredCommands[i];
  }
  return NULL;
}

const CommandInfo *CommandTraits::getRegisteredCommandInfo(
                                                  unsigned CommandID) const {
  return RegisteredCommands[CommandID - llvm::array_lengthof(Commands)];
}

} // end namespace comments
} // end namespace clang

