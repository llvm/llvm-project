//===- lib/Core/LinkingContext.cpp - Linker Context Object Interface ------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/Alias.h"
#include "lld/Core/LinkingContext.h"
#include "lld/Core/Resolver.h"
#include "lld/Core/Simple.h"
#include "lld/Core/Writer.h"
#include "llvm/ADT/Triple.h"

namespace lld {

LinkingContext::LinkingContext()
    : _deadStrip(false), _allowDuplicates(false),
      _globalsAreDeadStripRoots(false),
      _searchArchivesToOverrideTentativeDefinitions(false),
      _searchSharedLibrariesToOverrideTentativeDefinitions(false),
      _warnIfCoalesableAtomsHaveDifferentCanBeNull(false),
      _warnIfCoalesableAtomsHaveDifferentLoadName(false),
      _printRemainingUndefines(true), _allowRemainingUndefines(false),
      _logInputFiles(false), _allowShlibUndefines(true),
      _outputFileType(OutputFileType::Default), _nextOrdinal(0) {}

LinkingContext::~LinkingContext() {}

bool LinkingContext::validate(raw_ostream &diagnostics) {
  return validateImpl(diagnostics);
}

std::error_code LinkingContext::writeFile(const File &linkedFile) const {
  return this->writer().writeFile(linkedFile, _outputPath);
}

void LinkingContext::createImplicitFiles(
    std::vector<std::unique_ptr<File>> &result) {
  this->writer().createImplicitFiles(result);
}

std::unique_ptr<File> LinkingContext::createEntrySymbolFile() const {
  return createEntrySymbolFile("<command line option -e>");
}

std::unique_ptr<File>
LinkingContext::createEntrySymbolFile(StringRef filename) const {
  if (entrySymbolName().empty())
    return nullptr;
  std::unique_ptr<SimpleFile> entryFile(new SimpleFile(filename,
                                                       File::kindEntryObject));
  entryFile->addAtom(
      *(new (_allocator) SimpleUndefinedAtom(*entryFile, entrySymbolName())));
  return std::move(entryFile);
}

std::unique_ptr<File> LinkingContext::createUndefinedSymbolFile() const {
  return createUndefinedSymbolFile("<command line option -u or --defsym>");
}

std::unique_ptr<File>
LinkingContext::createUndefinedSymbolFile(StringRef filename) const {
  if (_initialUndefinedSymbols.empty())
    return nullptr;
  std::unique_ptr<SimpleFile> undefinedSymFile(
    new SimpleFile(filename, File::kindUndefinedSymsObject));
  for (StringRef undefSym : _initialUndefinedSymbols)
    undefinedSymFile->addAtom(*(new (_allocator) SimpleUndefinedAtom(
                                   *undefinedSymFile, undefSym)));
  return std::move(undefinedSymFile);
}

std::unique_ptr<File> LinkingContext::createAliasSymbolFile() const {
  if (getAliases().empty())
    return nullptr;
  std::unique_ptr<SimpleFile> file(
    new SimpleFile("<alias>", File::kindUndefinedSymsObject));
  for (const auto &i : getAliases()) {
    StringRef from = i.first;
    StringRef to = i.second;
    SimpleDefinedAtom *fromAtom = new (_allocator) AliasAtom(*file, from);
    UndefinedAtom *toAtom = new (_allocator) SimpleUndefinedAtom(*file, to);
    fromAtom->addReference(Reference::KindNamespace::all,
                           Reference::KindArch::all, Reference::kindLayoutAfter,
                           0, toAtom, 0);
    file->addAtom(*fromAtom);
    file->addAtom(*toAtom);
  }
  return std::move(file);
}

void LinkingContext::createInternalFiles(
    std::vector<std::unique_ptr<File> > &result) const {
  if (std::unique_ptr<File> file = createEntrySymbolFile())
    result.push_back(std::move(file));
  if (std::unique_ptr<File> file = createUndefinedSymbolFile())
    result.push_back(std::move(file));
  if (std::unique_ptr<File> file = createAliasSymbolFile())
    result.push_back(std::move(file));
}

void LinkingContext::addPasses(PassManager &pm) {}

} // end namespace lld
