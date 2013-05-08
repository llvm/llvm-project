//===-- Core/IncludeExcludeInfo.cpp - IncludeExclude class impl -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the implemention of the IncludeExcludeInfo class
/// to handle the include and exclude command line options.
///
//===----------------------------------------------------------------------===//

#include "IncludeExcludeInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
/// \brief Helper function to determine whether a file has the same path
/// prefix as \a Path.
///
/// \a Path must be an absolute path.
bool fileHasPathPrefix(StringRef File, StringRef Path) {
  // Converts File to its absolute path.
  SmallString<64> AbsoluteFile = File;
  sys::fs::make_absolute(AbsoluteFile);

  // Convert path strings to sys::path to iterate over each of its directories.
  sys::path::const_iterator FileI = sys::path::begin(AbsoluteFile),
                            FileE = sys::path::end(AbsoluteFile),
                            PathI = sys::path::begin(Path),
                            PathE = sys::path::end(Path);
  while (FileI != FileE && PathI != PathE) {
    // If the strings aren't equal then the two paths aren't contained within
    // each other.
    if (!FileI->equals(*PathI))
      return false;
    ++FileI;
    ++PathI;
  }
  return true;
}

/// \brief Helper function to tokenize a string of paths and populate
/// the vector.
error_code parseCLInput(StringRef Line, std::vector<std::string> &List,
                        StringRef Separator) {
  SmallVector<StringRef, 32> Tokens;
  Line.split(Tokens, Separator, /*MaxSplit=*/ -1, /*KeepEmpty=*/ false);
  for (SmallVectorImpl<StringRef>::iterator I = Tokens.begin(),
                                            E = Tokens.end();
       I != E; ++I) {
    // Convert each path to its absolute path.
    SmallString<64> AbsolutePath = I->rtrim();
    if (error_code Err = sys::fs::make_absolute(AbsolutePath))
      return Err;
    List.push_back(std::string(AbsolutePath.str()));
  }
  return error_code::success();
}
} // end anonymous namespace

error_code IncludeExcludeInfo::readListFromString(StringRef IncludeString,
                                                  StringRef ExcludeString) {
  if (error_code Err = parseCLInput(IncludeString, IncludeList,
                                    /*Separator=*/ ","))
    return Err;
  if (error_code Err = parseCLInput(ExcludeString, ExcludeList,
                                    /*Separator=*/ ","))
    return Err;
  return error_code::success();
}

error_code IncludeExcludeInfo::readListFromFile(StringRef IncludeListFile,
                                                StringRef ExcludeListFile) {
  if (!IncludeListFile.empty()) {
    OwningPtr<MemoryBuffer> FileBuf;
    if (error_code Err = MemoryBuffer::getFile(IncludeListFile, FileBuf)) {
      errs() << "Unable to read from include file.\n";
      return Err;
    }
    if (error_code Err = parseCLInput(FileBuf->getBuffer(), IncludeList,
                                      /*Separator=*/ "\n"))
      return Err;
  }
  if (!ExcludeListFile.empty()) {
    OwningPtr<MemoryBuffer> FileBuf;
    if (error_code Err = MemoryBuffer::getFile(ExcludeListFile, FileBuf)) {
      errs() << "Unable to read from exclude file.\n";
      return Err;
    }
    if (error_code Err = parseCLInput(FileBuf->getBuffer(), ExcludeList,
                                      /*Separator=*/ "\n"))
      return Err;
  }
  return error_code::success();
}

bool IncludeExcludeInfo::isFileIncluded(StringRef FilePath) {
  bool InIncludeList = false;

  for (std::vector<std::string>::iterator I = IncludeList.begin(),
                                          E = IncludeList.end();
       I != E; ++I)
    if ((InIncludeList = fileHasPathPrefix(FilePath, *I)))
      break;
  // If file is not in the list of included paths then it is not necessary
  // to check the excluded path list.
  if (!InIncludeList)
    return false;

  for (std::vector<std::string>::iterator I = ExcludeList.begin(),
                                          E = ExcludeList.end();
       I != E; ++I)
    if (fileHasPathPrefix(FilePath, *I))
      return false;

  // If the file is in the included list but not in the excluded list, then
  // it is safe to transform.
  return true;
}
