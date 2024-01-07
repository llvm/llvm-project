#include "clang-symbolgraph-merger/SymbolGraphMerger.h"
#include "clang-symbolgraph-merger/SymbolGraphVisitor.h"
#include "clang/ExtractAPI/Serialization/SymbolGraphSerializer.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Execution.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <system_error>

using namespace clang::tooling;
using namespace llvm;
using namespace sgmerger;

namespace {

bool collectSymbolGraphs(StringRef SGDirectory,
                         SmallVector<SymbolGraph> &SymbolGraphs) {
  std::error_code Error;
  for (sys::fs::directory_iterator I(SGDirectory, Error), End; I != End;
       I.increment(Error)) {
    if (Error)
      return false;
    std::string File(I->path());
    llvm::ErrorOr<sys::fs::basic_file_status> Status = I->status();
    if (!Status)
      return false;
    sys::fs::file_type Type = Status->type();
    // If the file is a directory, ignore the name and recurse.
    if (Type == sys::fs::file_type::directory_file) {
      if (!collectSymbolGraphs(File, SymbolGraphs))
        return false;
      continue;
    }

    // Ignore all the non json files
    if (!sys::path::extension(File).equals(".json"))
      continue;

    // Read the Symbol Graph from the file
    int FileFD;
    if (auto OpenError = sys::fs::openFileForRead(File, FileFD))
      return false;

    llvm::SmallString<256> Payload;
    if (auto ReadError = sys::fs::readNativeFileToEOF(FileFD, Payload))
      return false;

    SymbolGraphs.emplace_back(SymbolGraph(Payload));
    llvm::sys::fs::closeFile(FileFD);
  }
  return true;
}

} // namespace

static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
// TODO: add more help text
// static cl::extrahelp MoreHelp("\nMore help text...\n");

static cl::OptionCategory
    SymbolGraphMergerCategory("clang-symbolgraph-merger options");

static cl::opt<std::string> ProjectName("project-name",
                                        cl::desc("Name of project."),
                                        cl::cat(SymbolGraphMergerCategory),
                                        cl::init(""));

static cl::opt<std::string>
    OutFile("o", cl::desc("File for outputing generated Symbol Graph."),
            cl::cat(SymbolGraphMergerCategory), cl::value_desc("filepath"),
            cl::init("output"));

static cl::opt<std::string>
    InputDir(cl::Positional, cl::Required, cl::cat(SymbolGraphMergerCategory),
             cl::value_desc("filepath"),
             cl::desc("Input directory containing all the SymbolGraphs"));

int main(int argc, const char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

  cl::ParseCommandLineOptions(argc, argv);

  // collect symbol graphs from input dir
  SmallVector<SymbolGraph> SymbolGraphs;
  if (collectSymbolGraphs(InputDir, SymbolGraphs)) {
    llvm::outs() << "found " << SymbolGraphs.size() << " Symbol-graphs in "
                 << InputDir << "\n";

    // merge them together to form unified APIset
    llvm::outs() << "merging ...\n";
    SymbolGraphMerger Merger(SymbolGraphs);
    if (Merger.merge()) {
      // serialize the unified symbol graph
      std::error_code Error;
      if (!sys::path::extension(OutFile).equals(".json"))
        OutFile.append(".json");

      llvm::outs() << "serializing...\n";
      auto OS = std::make_unique<raw_fd_ostream>(OutFile, Error);
      const auto APISet = Merger.getAPISet();
      if (APISet) {
        clang::extractapi::SymbolGraphSerializer SGSerializer(
            *APISet, clang::extractapi::APIIgnoresList());
        SGSerializer.serialize(*OS);
        OS.reset();
        llvm::outs() << "successfully serialized resultant SymbolGraph to "
                     << OutFile << "\n";
      }
    } else {
      llvm::errs() << "merge faliure\n";
      return 1;
    }
  } else {
    llvm::errs() << "some error occured while accessing " << InputDir << "\n";
    return 1;
  }
  return 0;
}
