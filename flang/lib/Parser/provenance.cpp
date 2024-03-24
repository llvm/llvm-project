//===-- lib/Parser/provenance.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Parser/provenance.h"
#include "flang/Common/idioms.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <set>
#include <utility>

namespace Fortran::parser {

ProvenanceRangeToOffsetMappings::ProvenanceRangeToOffsetMappings() {}
ProvenanceRangeToOffsetMappings::~ProvenanceRangeToOffsetMappings() {}

void ProvenanceRangeToOffsetMappings::Put(
    ProvenanceRange range, std::size_t offset) {
  auto fromTo{map_.equal_range(range)};
  for (auto iter{fromTo.first}; iter != fromTo.second; ++iter) {
    if (range == iter->first) {
      iter->second = std::min(offset, iter->second);
      return;
    }
  }
  if (fromTo.second != map_.end()) {
    map_.emplace_hint(fromTo.second, range, offset);
  } else {
    map_.emplace(range, offset);
  }
}

std::optional<std::size_t> ProvenanceRangeToOffsetMappings::Map(
    ProvenanceRange range) const {
  auto fromTo{map_.equal_range(range)};
  std::optional<std::size_t> result;
  for (auto iter{fromTo.first}; iter != fromTo.second; ++iter) {
    ProvenanceRange that{iter->first};
    if (that.Contains(range)) {
      std::size_t offset{iter->second + that.MemberOffset(range.start())};
      if (!result || offset < *result) {
        result = offset;
      }
    }
  }
  return result;
}

bool ProvenanceRangeToOffsetMappings::WhollyPrecedes::operator()(
    ProvenanceRange before, ProvenanceRange after) const {
  return before.start() + before.size() <= after.start();
}

void OffsetToProvenanceMappings::clear() { provenanceMap_.clear(); }

void OffsetToProvenanceMappings::swap(OffsetToProvenanceMappings &that) {
  provenanceMap_.swap(that.provenanceMap_);
}

void OffsetToProvenanceMappings::shrink_to_fit() {
  provenanceMap_.shrink_to_fit();
}

std::size_t OffsetToProvenanceMappings::SizeInBytes() const {
  if (provenanceMap_.empty()) {
    return 0;
  } else {
    const ContiguousProvenanceMapping &last{provenanceMap_.back()};
    return last.start + last.range.size();
  }
}

void OffsetToProvenanceMappings::Put(ProvenanceRange range) {
  if (provenanceMap_.empty()) {
    provenanceMap_.push_back({0, range});
  } else {
    ContiguousProvenanceMapping &last{provenanceMap_.back()};
    if (!last.range.AnnexIfPredecessor(range)) {
      provenanceMap_.push_back({last.start + last.range.size(), range});
    }
  }
}

void OffsetToProvenanceMappings::Put(const OffsetToProvenanceMappings &that) {
  for (const auto &map : that.provenanceMap_) {
    Put(map.range);
  }
}

ProvenanceRange OffsetToProvenanceMappings::Map(std::size_t at) const {
  if (provenanceMap_.empty()) {
    CHECK(at == 0);
    return {};
  }
  std::size_t low{0}, count{provenanceMap_.size()};
  while (count > 1) {
    std::size_t mid{low + (count >> 1)};
    if (provenanceMap_[mid].start > at) {
      count = mid - low;
    } else {
      count -= mid - low;
      low = mid;
    }
  }
  std::size_t offset{at - provenanceMap_[low].start};
  return provenanceMap_[low].range.Suffix(offset);
}

void OffsetToProvenanceMappings::RemoveLastBytes(std::size_t bytes) {
  for (; bytes > 0; provenanceMap_.pop_back()) {
    CHECK(!provenanceMap_.empty());
    ContiguousProvenanceMapping &last{provenanceMap_.back()};
    std::size_t chunk{last.range.size()};
    if (bytes < chunk) {
      last.range = last.range.Prefix(chunk - bytes);
      break;
    }
    bytes -= chunk;
  }
}

ProvenanceRangeToOffsetMappings OffsetToProvenanceMappings::Invert(
    const AllSources &allSources) const {
  ProvenanceRangeToOffsetMappings result;
  for (const auto &contig : provenanceMap_) {
    ProvenanceRange range{contig.range};
    while (!range.empty()) {
      ProvenanceRange source{allSources.IntersectionWithSourceFiles(range)};
      if (source.empty()) {
        break;
      }
      result.Put(
          source, contig.start + contig.range.MemberOffset(source.start()));
      Provenance after{source.NextAfter()};
      if (range.Contains(after)) {
        range = range.Suffix(range.MemberOffset(after));
      } else {
        break;
      }
    }
  }
  return result;
}

AllSources::AllSources() : range_{1, 1} {
  // Start the origin_ array with a dummy entry that has a forced provenance,
  // so that provenance offset 0 remains reserved as an uninitialized
  // value.
  origin_.emplace_back(range_, std::string{'?'});
}

AllSources::~AllSources() {}

const char &AllSources::operator[](Provenance at) const {
  const Origin &origin{MapToOrigin(at)};
  return origin[origin.covers.MemberOffset(at)];
}

void AllSources::ClearSearchPath() { searchPath_.clear(); }

void AllSources::AppendSearchPathDirectory(std::string directory) {
  // gfortran and ifort append to current path, PGI prepends
  searchPath_.push_back(directory);
}

const SourceFile *AllSources::OpenPath(
    std::string path, llvm::raw_ostream &error) {
  std::unique_ptr<SourceFile> source{std::make_unique<SourceFile>(encoding_)};
  if (source->Open(path, error)) {
    return ownedSourceFiles_.emplace_back(std::move(source)).get();
  } else {
    return nullptr;
  }
}

const SourceFile *AllSources::Open(std::string path, llvm::raw_ostream &error,
    std::optional<std::string> &&prependPath) {
  std::unique_ptr<SourceFile> source{std::make_unique<SourceFile>(encoding_)};
  if (prependPath) {
    // Set to "." for the initial source file; set to the directory name
    // of the including file for #include "quoted-file" directives &
    // INCLUDE statements.
    searchPath_.emplace_front(std::move(*prependPath));
  }
  std::optional<std::string> found{LocateSourceFile(path, searchPath_)};
  if (prependPath) {
    searchPath_.pop_front();
  }
  if (found) {
    return OpenPath(*found, error);
  } else {
    error << "Source file '" << path << "' was not found";
    return nullptr;
  }
}

const SourceFile *AllSources::ReadStandardInput(llvm::raw_ostream &error) {
  std::unique_ptr<SourceFile> source{std::make_unique<SourceFile>(encoding_)};
  if (source->ReadStandardInput(error)) {
    return ownedSourceFiles_.emplace_back(std::move(source)).get();
  }
  return nullptr;
}

ProvenanceRange AllSources::AddIncludedFile(
    const SourceFile &source, ProvenanceRange from, bool isModule) {
  ProvenanceRange covers{range_.NextAfter(), source.bytes()};
  CHECK(range_.AnnexIfPredecessor(covers));
  CHECK(origin_.back().covers.ImmediatelyPrecedes(covers));
  origin_.emplace_back(covers, source, from, isModule);
  return covers;
}

ProvenanceRange AllSources::AddMacroCall(
    ProvenanceRange def, ProvenanceRange use, const std::string &expansion) {
  ProvenanceRange covers{range_.NextAfter(), expansion.size()};
  CHECK(range_.AnnexIfPredecessor(covers));
  CHECK(origin_.back().covers.ImmediatelyPrecedes(covers));
  origin_.emplace_back(covers, def, use, expansion);
  return covers;
}

ProvenanceRange AllSources::AddCompilerInsertion(std::string text) {
  ProvenanceRange covers{range_.NextAfter(), text.size()};
  CHECK(range_.AnnexIfPredecessor(covers));
  CHECK(origin_.back().covers.ImmediatelyPrecedes(covers));
  origin_.emplace_back(covers, text);
  return covers;
}

static void EmitPrefix(llvm::raw_ostream &o, llvm::raw_ostream::Colors color,
    const std::string &prefix, bool showColors) {
  if (prefix.empty()) {
    return;
  }
  if (showColors) {
    o.changeColor(color, true);
  }
  o << prefix;
  if (showColors) {
    o.resetColor();
  }
}

void AllSources::EmitMessage(llvm::raw_ostream &o,
    const std::optional<ProvenanceRange> &range, const std::string &message,
    const std::string &prefix, llvm::raw_ostream::Colors color,
    bool echoSourceLine) const {
  if (!range) {
    EmitPrefix(o, color, prefix, this->getShowColors());
    o << message << '\n';
    return;
  }
  CHECK(IsValid(*range));
  const Origin &origin{MapToOrigin(range->start())};
  common::visit(
      common::visitors{
          [&](const Inclusion &inc) {
            std::size_t offset{origin.covers.MemberOffset(range->start())};
            SourcePosition pos{inc.source.GetSourcePosition(offset)};
            o << pos.path << ':' << pos.line << ':' << pos.column << ": ";
            EmitPrefix(o, color, prefix, this->getShowColors());
            o << message << '\n';
            if (echoSourceLine) {
              const char *text{inc.source.content().data() +
                  inc.source.GetLineStartOffset(pos.trueLineNumber)};
              o << "  ";
              for (const char *p{text}; *p != '\n'; ++p) {
                o << *p;
              }
              o << "\n  ";
              for (int j{1}; j < pos.column; ++j) {
                char ch{text[j - 1]};
                o << (ch == '\t' ? '\t' : ' ');
              }
              o << '^';
              if (range->size() > 1) {
                auto last{range->start() + range->size() - 1};
                if (&MapToOrigin(last) == &origin) {
                  auto endOffset{origin.covers.MemberOffset(last)};
                  auto endPos{inc.source.GetSourcePosition(endOffset)};
                  if (pos.line == endPos.line) {
                    for (int j{pos.column}; j < endPos.column; ++j) {
                      o << '^';
                    }
                  }
                }
              }
              o << '\n';
            }
            if (IsValid(origin.replaces)) {
              EmitMessage(o, origin.replaces,
                  inc.isModule ? "used here"s : "included here"s, prefix, color,
                  echoSourceLine);
            }
          },
          [&](const Macro &mac) {
            EmitMessage(
                o, origin.replaces, message, prefix, color, echoSourceLine);
            EmitMessage(o, mac.definition, "in a macro defined here", ""s,
                color, echoSourceLine);
            if (echoSourceLine) {
              o << "that expanded to:\n  " << mac.expansion << "\n  ";
              for (std::size_t j{0};
                   origin.covers.OffsetMember(j) < range->start(); ++j) {
                o << (mac.expansion[j] == '\t' ? '\t' : ' ');
              }
              o << "^\n";
            }
          },
          [&](const CompilerInsertion &) {
            EmitPrefix(o, color, prefix, this->getShowColors());
            o << message << '\n';
          },
      },
      origin.u);
}

const SourceFile *AllSources::GetSourceFile(
    Provenance at, std::size_t *offset) const {
  const Origin &origin{MapToOrigin(at)};
  return common::visit(common::visitors{
                           [&](const Inclusion &inc) {
                             if (offset) {
                               *offset = origin.covers.MemberOffset(at);
                             }
                             return &inc.source;
                           },
                           [&](const Macro &) {
                             return GetSourceFile(
                                 origin.replaces.start(), offset);
                           },
                           [offset](const CompilerInsertion &) {
                             if (offset) {
                               *offset = 0;
                             }
                             return static_cast<const SourceFile *>(nullptr);
                           },
                       },
      origin.u);
}

const char *AllSources::GetSource(ProvenanceRange range) const {
  Provenance start{range.start()};
  const Origin &origin{MapToOrigin(start)};
  return origin.covers.Contains(range)
      ? &origin[origin.covers.MemberOffset(start)]
      : nullptr;
}

std::optional<SourcePosition> AllSources::GetSourcePosition(
    Provenance prov) const {
  const Origin &origin{MapToOrigin(prov)};
  return common::visit(
      common::visitors{
          [&](const Inclusion &inc) -> std::optional<SourcePosition> {
            std::size_t offset{origin.covers.MemberOffset(prov)};
            return inc.source.GetSourcePosition(offset);
          },
          [&](const Macro &) {
            return GetSourcePosition(origin.replaces.start());
          },
          [](const CompilerInsertion &) -> std::optional<SourcePosition> {
            return std::nullopt;
          },
      },
      origin.u);
}

std::optional<ProvenanceRange> AllSources::GetFirstFileProvenance() const {
  for (const auto &origin : origin_) {
    if (std::holds_alternative<Inclusion>(origin.u)) {
      return origin.covers;
    }
  }
  return std::nullopt;
}

std::string AllSources::GetPath(Provenance at) const {
  std::size_t offset{0};
  const SourceFile *source{GetSourceFile(at, &offset)};
  return source ? *source->GetSourcePosition(offset).path : ""s;
}

int AllSources::GetLineNumber(Provenance at) const {
  std::size_t offset{0};
  const SourceFile *source{GetSourceFile(at, &offset)};
  return source ? source->GetSourcePosition(offset).line : 0;
}

Provenance AllSources::CompilerInsertionProvenance(char ch) {
  auto iter{compilerInsertionProvenance_.find(ch)};
  if (iter != compilerInsertionProvenance_.end()) {
    return iter->second;
  }
  ProvenanceRange newCharRange{AddCompilerInsertion(std::string{ch})};
  Provenance newCharProvenance{newCharRange.start()};
  compilerInsertionProvenance_.insert(std::make_pair(ch, newCharProvenance));
  return newCharProvenance;
}

ProvenanceRange AllSources::IntersectionWithSourceFiles(
    ProvenanceRange range) const {
  if (range.empty()) {
    return {};
  } else {
    const Origin &origin{MapToOrigin(range.start())};
    if (std::holds_alternative<Inclusion>(origin.u)) {
      return range.Intersection(origin.covers);
    } else {
      auto skip{
          origin.covers.size() - origin.covers.MemberOffset(range.start())};
      return IntersectionWithSourceFiles(range.Suffix(skip));
    }
  }
}

AllSources::Origin::Origin(ProvenanceRange r, const SourceFile &source)
    : u{Inclusion{source}}, covers{r} {}
AllSources::Origin::Origin(ProvenanceRange r, const SourceFile &included,
    ProvenanceRange from, bool isModule)
    : u{Inclusion{included, isModule}}, covers{r}, replaces{from} {}
AllSources::Origin::Origin(ProvenanceRange r, ProvenanceRange def,
    ProvenanceRange use, const std::string &expansion)
    : u{Macro{def, expansion}}, covers{r}, replaces{use} {}
AllSources::Origin::Origin(ProvenanceRange r, const std::string &text)
    : u{CompilerInsertion{text}}, covers{r} {}

const char &AllSources::Origin::operator[](std::size_t n) const {
  return common::visit(
      common::visitors{
          [n](const Inclusion &inc) -> const char & {
            return inc.source.content()[n];
          },
          [n](const Macro &mac) -> const char & { return mac.expansion[n]; },
          [n](const CompilerInsertion &ins) -> const char & {
            return ins.text[n];
          },
      },
      u);
}

const AllSources::Origin &AllSources::MapToOrigin(Provenance at) const {
  CHECK(range_.Contains(at));
  std::size_t low{0}, count{origin_.size()};
  while (count > 1) {
    std::size_t mid{low + (count >> 1)};
    if (at < origin_[mid].covers.start()) {
      count = mid - low;
    } else {
      count -= mid - low;
      low = mid;
    }
  }
  CHECK(origin_[low].covers.Contains(at));
  return origin_[low];
}

Provenance AllSources::GetReplacedProvenance(Provenance provenance) const {
  const Origin &origin{MapToOrigin(provenance)};
  if (std::holds_alternative<Macro>(origin.u)) {
    return origin.replaces.start();
  }
  return provenance;
}

std::optional<ProvenanceRange> CookedSource::GetProvenanceRange(
    CharBlock cookedRange) const {
  if (!AsCharBlock().Contains(cookedRange)) {
    return std::nullopt;
  }
  ProvenanceRange first{provenanceMap_.Map(cookedRange.begin() - &data_[0])};
  if (cookedRange.size() <= first.size()) { // always true when empty
    return first.Prefix(cookedRange.size());
  }
  ProvenanceRange last{provenanceMap_.Map(cookedRange.end() - 1 - &data_[0])};
  if (first.start() <= last.start()) {
    return {ProvenanceRange{first.start(), last.start() - first.start() + 1}};
  } else {
    // cookedRange may start (resp. end) in a macro expansion while it does not
    // end (resp. start) in this macro expansion. Attempt to build a range
    // over the replaced source.
    Provenance firstStart{allSources_.GetReplacedProvenance(first.start())};
    Provenance lastStart{allSources_.GetReplacedProvenance(last.start())};
    if (firstStart <= lastStart) {
      return {ProvenanceRange{firstStart, lastStart - firstStart + 1}};
    } else {
      return std::nullopt;
    }
  }
}

std::optional<CharBlock> CookedSource::GetCharBlock(
    ProvenanceRange range) const {
  CHECK(!invertedMap_.empty() &&
      "CompileProvenanceRangeToOffsetMappings not called");
  if (auto to{invertedMap_.Map(range)}) {
    return CharBlock{data_.c_str() + *to, range.size()};
  } else {
    return std::nullopt;
  }
}

std::size_t CookedSource::BufferedBytes() const { return buffer_.bytes(); }

void CookedSource::Marshal(AllCookedSources &allCookedSources) {
  CHECK(provenanceMap_.SizeInBytes() == buffer_.bytes());
  provenanceMap_.Put(allCookedSources.allSources().AddCompilerInsertion(
      "(after end of source)"));
  data_ = buffer_.Marshal();
  buffer_.clear();
  allCookedSources.Register(*this);
}

void CookedSource::CompileProvenanceRangeToOffsetMappings(
    AllSources &allSources) {
  if (invertedMap_.empty()) {
    invertedMap_ = provenanceMap_.Invert(allSources);
  }
}

static void DumpRange(llvm::raw_ostream &o, const ProvenanceRange &r) {
  o << "[" << r.start().offset() << ".." << r.Last().offset() << "] ("
    << r.size() << " bytes)";
}

llvm::raw_ostream &ProvenanceRangeToOffsetMappings::Dump(
    llvm::raw_ostream &o) const {
  for (const auto &m : map_) {
    o << "provenances ";
    DumpRange(o, m.first);
    o << " -> offsets [" << m.second << ".." << (m.second + m.first.size() - 1)
      << "]\n";
  }
  return o;
}

llvm::raw_ostream &OffsetToProvenanceMappings::Dump(
    llvm::raw_ostream &o) const {
  for (const ContiguousProvenanceMapping &m : provenanceMap_) {
    std::size_t n{m.range.size()};
    o << "offsets [" << m.start << ".." << (m.start + n - 1)
      << "] -> provenances ";
    DumpRange(o, m.range);
    o << '\n';
  }
  return o;
}

llvm::raw_ostream &AllSources::Dump(llvm::raw_ostream &o) const {
  o << "AllSources range_ ";
  DumpRange(o, range_);
  o << '\n';
  std::set<const SourceFile *> sources;
  for (const Origin &m : origin_) {
    o << "   ";
    DumpRange(o, m.covers);
    o << " -> ";
    common::visit(common::visitors{
                      [&](const Inclusion &inc) {
                        if (inc.isModule) {
                          o << "module ";
                        }
                        o << "file " << inc.source.path();
                        sources.emplace(&inc.source);
                      },
                      [&](const Macro &mac) { o << "macro " << mac.expansion; },
                      [&](const CompilerInsertion &ins) {
                        o << "compiler '" << ins.text << '\'';
                        if (ins.text.length() == 1) {
                          int ch = ins.text[0];
                          o << "(0x";
                          o.write_hex(ch & 0xff) << ")";
                        }
                      },
                  },
        m.u);
    if (IsValid(m.replaces)) {
      o << " replaces ";
      DumpRange(o, m.replaces);
    }
    o << '\n';
  }
  for (const SourceFile *sf : sources) {
    sf->Dump(o);
  }
  return o;
}

llvm::raw_ostream &CookedSource::Dump(llvm::raw_ostream &o) const {
  o << "CookedSource::provenanceMap_:\n";
  provenanceMap_.Dump(o);
  o << "CookedSource::invertedMap_:\n";
  invertedMap_.Dump(o);
  return o;
}

AllCookedSources::AllCookedSources(AllSources &s) : allSources_{s} {}
AllCookedSources::~AllCookedSources() {}

CookedSource &AllCookedSources::NewCookedSource() {
  return cooked_.emplace_back(allSources_);
}

const CookedSource *AllCookedSources::Find(CharBlock x) const {
  auto pair{index_.equal_range(x)};
  for (auto iter{pair.first}; iter != pair.second; ++iter) {
    if (iter->second.AsCharBlock().Contains(x)) {
      return &iter->second;
    }
  }
  return nullptr;
}

std::optional<ProvenanceRange> AllCookedSources::GetProvenanceRange(
    CharBlock cb) const {
  if (const CookedSource * c{Find(cb)}) {
    return c->GetProvenanceRange(cb);
  } else {
    return std::nullopt;
  }
}

std::optional<CharBlock> AllCookedSources::GetCharBlockFromLineAndColumns(
    int line, int startColumn, int endColumn) const {
  // 2nd column is exclusive, meaning it is target column + 1.
  CHECK(line > 0 && startColumn > 0 && endColumn > 0);
  CHECK(startColumn < endColumn);
  auto provenanceStart{allSources_.GetFirstFileProvenance().value().start()};
  if (auto sourceFile{allSources_.GetSourceFile(provenanceStart)}) {
    CHECK(line <= static_cast<int>(sourceFile->lines()));
    return GetCharBlock(ProvenanceRange(sourceFile->GetLineStartOffset(line) +
            provenanceStart.offset() + startColumn - 1,
        endColumn - startColumn));
  }
  return std::nullopt;
}

std::optional<std::pair<SourcePosition, SourcePosition>>
AllCookedSources::GetSourcePositionRange(CharBlock cookedRange) const {
  if (auto range{GetProvenanceRange(cookedRange)}) {
    if (auto firstOffset{allSources_.GetSourcePosition(range->start())}) {
      if (auto secondOffset{
              allSources_.GetSourcePosition(range->start() + range->size())}) {
        return std::pair{*firstOffset, *secondOffset};
      }
    }
  }
  return std::nullopt;
}

std::optional<CharBlock> AllCookedSources::GetCharBlock(
    ProvenanceRange range) const {
  for (const auto &c : cooked_) {
    if (auto result{c.GetCharBlock(range)}) {
      return result;
    }
  }
  return std::nullopt;
}

void AllCookedSources::Dump(llvm::raw_ostream &o) const {
  o << "AllSources:\n";
  allSources_.Dump(o);
  for (const auto &c : cooked_) {
    c.Dump(o);
  }
}

bool AllCookedSources::Precedes(CharBlock x, CharBlock y) const {
  if (const CookedSource * xSource{Find(x)}) {
    if (xSource->AsCharBlock().Contains(y)) {
      return x.begin() < y.begin();
    } else if (const CookedSource * ySource{Find(y)}) {
      return xSource->number() < ySource->number();
    } else {
      return true; // by fiat, all cooked source < anything outside
    }
  } else if (Find(y)) {
    return false;
  } else {
    // Both names are compiler-created (SaveTempName).
    return x < y;
  }
}

void AllCookedSources::Register(CookedSource &cooked) {
  index_.emplace(cooked.AsCharBlock(), cooked);
  cooked.set_number(static_cast<int>(index_.size()));
}

} // namespace Fortran::parser
