#include "clang-c/CXCompilationDatabase.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "CXString.h"

using namespace clang;
using namespace clang::tooling;
using namespace clang::cxstring;

extern "C" {

// FIXME: do something more usefull with the error message
CXCompilationDatabase
clang_CompilationDatabase_fromDirectory(const char *BuildDir,
                                        CXCompilationDatabase_Error *ErrorCode)
{
  std::string ErrorMsg;
  CXCompilationDatabase_Error Err = CXCompilationDatabase_NoError;

  CompilationDatabase *db = CompilationDatabase::loadFromDirectory(BuildDir,
                                                                   ErrorMsg);

  if (!db) {
    fprintf(stderr, "LIBCLANG TOOLING ERROR: %s\n", ErrorMsg.c_str());
    Err = CXCompilationDatabase_CanNotLoadDatabase;
  }

  if (ErrorCode)
    *ErrorCode = Err;

  return db;
}

void
clang_CompilationDatabase_dispose(CXCompilationDatabase CDb)
{
  delete static_cast<CompilationDatabase *>(CDb);
}

struct AllocatedCXCompileCommands
{
  std::vector<CompileCommand> CCmd;

  AllocatedCXCompileCommands(const std::vector<CompileCommand>& Cmd)
    : CCmd(Cmd)
  { }
};

CXCompileCommands
clang_CompilationDatabase_getCompileCommands(CXCompilationDatabase CDb,
                                             const char *CompleteFileName)
{
  if (CompilationDatabase *db = static_cast<CompilationDatabase *>(CDb)) {
    const std::vector<CompileCommand>
      CCmd(db->getCompileCommands(CompleteFileName));
    if (!CCmd.empty())
      return new AllocatedCXCompileCommands( CCmd );
  }

  return 0;
}

void
clang_CompileCommands_dispose(CXCompileCommands Cmds)
{
  delete static_cast<AllocatedCXCompileCommands *>(Cmds);
}

unsigned
clang_CompileCommands_getSize(CXCompileCommands Cmds)
{
  if (!Cmds)
    return 0;

  AllocatedCXCompileCommands *ACC =
    static_cast<AllocatedCXCompileCommands *>(Cmds);

  return ACC->CCmd.size();
}

CXCompileCommand
clang_CompileCommands_getCommand(CXCompileCommands Cmds, unsigned I)
{
  if (!Cmds)
    return 0;

  AllocatedCXCompileCommands *ACC =
    static_cast<AllocatedCXCompileCommands *>(Cmds);

  if (I >= ACC->CCmd.size())
    return 0;

  return &ACC->CCmd[I];
}

CXString
clang_CompileCommand_getDirectory(CXCompileCommand CCmd)
{
  if (!CCmd)
    return createCXString((const char*)NULL);

  CompileCommand *cmd = static_cast<CompileCommand *>(CCmd);
  return createCXString(cmd->Directory);
}

unsigned
clang_CompileCommand_getNumArgs(CXCompileCommand CCmd)
{
  if (!CCmd)
    return 0;

  return static_cast<CompileCommand *>(CCmd)->CommandLine.size();
}

CXString
clang_CompileCommand_getArg(CXCompileCommand CCmd, unsigned Arg)
{
  if (!CCmd)
    return createCXString((const char*)NULL);

  CompileCommand *Cmd = static_cast<CompileCommand *>(CCmd);

  if (Arg >= Cmd->CommandLine.size())
    return createCXString((const char*)NULL);

  return createCXString(Cmd->CommandLine[Arg]);
}


} // end: extern "C"
