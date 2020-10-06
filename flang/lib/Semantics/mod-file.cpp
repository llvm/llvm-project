//===-- lib/Semantics/mod-file.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mod-file.h"
#include "resolve-names.h"
#include "flang/Common/restorer.h"
#include "flang/Evaluate/tools.h"
#include "flang/Parser/message.h"
#include "flang/Parser/parsing.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <fstream>
#include <set>
#include <string_view>
#include <vector>

namespace Fortran::semantics {

using namespace parser::literals;

// The first line of a file that identifies it as a .mod file.
// The first three bytes are a Unicode byte order mark that ensures
// that the module file is decoded as UTF-8 even if source files
// are using another encoding.
struct ModHeader {
  static constexpr const char bom[3 + 1]{"\xef\xbb\xbf"};
  static constexpr int magicLen{13};
  static constexpr int sumLen{16};
  static constexpr const char magic[magicLen + 1]{"!mod$ v1 sum:"};
  static constexpr char terminator{'\n'};
  static constexpr int len{magicLen + 1 + sumLen};
};

static std::optional<SourceName> GetSubmoduleParent(const parser::Program &);
static SymbolVector CollectSymbols(const Scope &);
static void PutEntity(llvm::raw_ostream &, const Symbol &);
static void PutObjectEntity(llvm::raw_ostream &, const Symbol &);
static void PutProcEntity(llvm::raw_ostream &, const Symbol &);
static void PutPassName(llvm::raw_ostream &, const std::optional<SourceName> &);
static void PutTypeParam(llvm::raw_ostream &, const Symbol &);
static void PutEntity(
    llvm::raw_ostream &, const Symbol &, std::function<void()>, Attrs);
static void PutInit(llvm::raw_ostream &, const Symbol &, const MaybeExpr &);
static void PutInit(llvm::raw_ostream &, const MaybeIntExpr &);
static void PutBound(llvm::raw_ostream &, const Bound &);
static llvm::raw_ostream &PutAttrs(llvm::raw_ostream &, Attrs,
    const MaybeExpr & = std::nullopt, std::string before = ","s,
    std::string after = ""s);

static llvm::raw_ostream &PutAttr(llvm::raw_ostream &, Attr);
static llvm::raw_ostream &PutType(llvm::raw_ostream &, const DeclTypeSpec &);
static llvm::raw_ostream &PutLower(llvm::raw_ostream &, const std::string &);
static std::error_code WriteFile(
    const std::string &, const std::string &, bool = true);
static bool FileContentsMatch(
    const std::string &, const std::string &, const std::string &);
static std::string CheckSum(const std::string_view &);

// Collect symbols needed for a subprogram interface
class SubprogramSymbolCollector {
public:
  SubprogramSymbolCollector(const Symbol &symbol, const Scope &scope)
      : symbol_{symbol}, scope_{scope} {}
  const SymbolVector &symbols() const { return need_; }
  const std::set<SourceName> &imports() const { return imports_; }
  void Collect();

private:
  const Symbol &symbol_;
  const Scope &scope_;
  bool isInterface_{false};
  SymbolVector need_; // symbols that are needed
  SymbolSet needSet_; // symbols already in need_
  SymbolSet useSet_; // use-associations that might be needed
  std::set<SourceName> imports_; // imports from host that are needed

  void DoSymbol(const Symbol &);
  void DoSymbol(const SourceName &, const Symbol &);
  void DoType(const DeclTypeSpec *);
  void DoBound(const Bound &);
  void DoParamValue(const ParamValue &);
  bool NeedImport(const SourceName &, const Symbol &);

  template <typename T> void DoExpr(evaluate::Expr<T> expr) {
    for (const Symbol &symbol : evaluate::CollectSymbols(expr)) {
      DoSymbol(symbol);
    }
  }
};

bool ModFileWriter::WriteAll() {
  // this flag affects character literals: force it to be consistent
  auto restorer{
      common::ScopedSet(parser::useHexadecimalEscapeSequences, false)};
  WriteAll(context_.globalScope());
  return !context_.AnyFatalError();
}

void ModFileWriter::WriteAll(const Scope &scope) {
  for (const auto &child : scope.children()) {
    WriteOne(child);
  }
}

void ModFileWriter::WriteOne(const Scope &scope) {
  if (scope.kind() == Scope::Kind::Module) {
    auto *symbol{scope.symbol()};
    if (!symbol->test(Symbol::Flag::ModFile)) {
      Write(*symbol);
    }
    WriteAll(scope); // write out submodules
  }
}

// Construct the name of a module file. Non-empty ancestorName means submodule.
static std::string ModFileName(const SourceName &name,
    const std::string &ancestorName, const std::string &suffix) {
  std::string result{name.ToString() + suffix};
  return ancestorName.empty() ? result : ancestorName + '-' + result;
}

// Write the module file for symbol, which must be a module or submodule.
void ModFileWriter::Write(const Symbol &symbol) {
  auto *ancestor{symbol.get<ModuleDetails>().ancestor()};
  auto ancestorName{ancestor ? ancestor->GetName().value().ToString() : ""s};
  auto path{context_.moduleDirectory() + '/' +
      ModFileName(symbol.name(), ancestorName, context_.moduleFileSuffix())};
  PutSymbols(DEREF(symbol.scope()));
  if (std::error_code error{
          WriteFile(path, GetAsString(symbol), context_.debugModuleWriter())}) {
    context_.Say(
        symbol.name(), "Error writing %s: %s"_err_en_US, path, error.message());
  }
}

// Return the entire body of the module file
// and clear saved uses, decls, and contains.
std::string ModFileWriter::GetAsString(const Symbol &symbol) {
  std::string buf;
  llvm::raw_string_ostream all{buf};
  auto &details{symbol.get<ModuleDetails>()};
  if (!details.isSubmodule()) {
    all << "module " << symbol.name();
  } else {
    auto *parent{details.parent()->symbol()};
    auto *ancestor{details.ancestor()->symbol()};
    all << "submodule(" << ancestor->name();
    if (parent != ancestor) {
      all << ':' << parent->name();
    }
    all << ") " << symbol.name();
  }
  all << '\n' << uses_.str();
  uses_.str().clear();
  all << useExtraAttrs_.str();
  useExtraAttrs_.str().clear();
  all << decls_.str();
  decls_.str().clear();
  auto str{contains_.str()};
  contains_.str().clear();
  if (!str.empty()) {
    all << "contains\n" << str;
  }
  all << "end\n";
  return all.str();
}

// Put out the visible symbols from scope.
bool ModFileWriter::PutSymbols(const Scope &scope) {
  std::string buf;
  llvm::raw_string_ostream typeBindings{
      buf}; // stuff after CONTAINS in derived type
  for (const Symbol &symbol : CollectSymbols(scope)) {
    PutSymbol(typeBindings, symbol);
  }
  if (auto str{typeBindings.str()}; !str.empty()) {
    CHECK(scope.IsDerivedType());
    decls_ << "contains\n" << str;
    return true;
  } else {
    return false;
  }
}

// Emit a symbol to decls_, except for bindings in a derived type (type-bound
// procedures, type-bound generics, final procedures) which go to typeBindings.
void ModFileWriter::PutSymbol(
    llvm::raw_ostream &typeBindings, const Symbol &symbol) {
  std::visit(common::visitors{
                 [&](const ModuleDetails &) { /* should be current module */ },
                 [&](const DerivedTypeDetails &) { PutDerivedType(symbol); },
                 [&](const SubprogramDetails &) { PutSubprogram(symbol); },
                 [&](const GenericDetails &x) {
                   if (symbol.owner().IsDerivedType()) {
                     // generic binding
                     for (const Symbol &proc : x.specificProcs()) {
                       typeBindings << "generic::" << symbol.name() << "=>"
                                    << proc.name() << '\n';
                     }
                   } else {
                     PutGeneric(symbol);
                     if (x.specific()) {
                       PutSymbol(typeBindings, *x.specific());
                     }
                     if (x.derivedType()) {
                       PutSymbol(typeBindings, *x.derivedType());
                     }
                   }
                 },
                 [&](const UseDetails &) { PutUse(symbol); },
                 [](const UseErrorDetails &) {},
                 [&](const ProcBindingDetails &x) {
                   bool deferred{symbol.attrs().test(Attr::DEFERRED)};
                   typeBindings << "procedure";
                   if (deferred) {
                     typeBindings << '(' << x.symbol().name() << ')';
                   }
                   PutPassName(typeBindings, x.passName());
                   auto attrs{symbol.attrs()};
                   if (x.passName()) {
                     attrs.reset(Attr::PASS);
                   }
                   PutAttrs(typeBindings, attrs);
                   typeBindings << "::" << symbol.name();
                   if (!deferred && x.symbol().name() != symbol.name()) {
                     typeBindings << "=>" << x.symbol().name();
                   }
                   typeBindings << '\n';
                 },
                 [&](const NamelistDetails &x) {
                   decls_ << "namelist/" << symbol.name();
                   char sep{'/'};
                   for (const Symbol &object : x.objects()) {
                     decls_ << sep << object.name();
                     sep = ',';
                   }
                   decls_ << '\n';
                 },
                 [&](const CommonBlockDetails &x) {
                   decls_ << "common/" << symbol.name();
                   char sep = '/';
                   for (const auto &object : x.objects()) {
                     decls_ << sep << object->name();
                     sep = ',';
                   }
                   decls_ << '\n';
                   if (symbol.attrs().test(Attr::BIND_C)) {
                     PutAttrs(decls_, symbol.attrs(), x.bindName(), ""s);
                     decls_ << "::/" << symbol.name() << "/\n";
                   }
                 },
                 [](const HostAssocDetails &) {},
                 [](const MiscDetails &) {},
                 [&](const auto &) { PutEntity(decls_, symbol); },
             },
      symbol.details());
}

void ModFileWriter::PutDerivedType(const Symbol &typeSymbol) {
  auto &details{typeSymbol.get<DerivedTypeDetails>()};
  PutAttrs(decls_ << "type", typeSymbol.attrs());
  if (const DerivedTypeSpec * extends{typeSymbol.GetParentTypeSpec()}) {
    decls_ << ",extends(" << extends->name() << ')';
  }
  decls_ << "::" << typeSymbol.name();
  auto &typeScope{*typeSymbol.scope()};
  if (!details.paramNames().empty()) {
    char sep{'('};
    for (const auto &name : details.paramNames()) {
      decls_ << sep << name;
      sep = ',';
    }
    decls_ << ')';
  }
  decls_ << '\n';
  if (details.sequence()) {
    decls_ << "sequence\n";
  }
  bool contains{PutSymbols(typeScope)};
  if (!details.finals().empty()) {
    const char *sep{contains ? "final::" : "contains\nfinal::"};
    for (const auto &pair : details.finals()) {
      decls_ << sep << pair.second->name();
      sep = ",";
    }
    if (*sep == ',') {
      decls_ << '\n';
    }
  }
  decls_ << "end type\n";
}

// Attributes that may be in a subprogram prefix
static const Attrs subprogramPrefixAttrs{Attr::ELEMENTAL, Attr::IMPURE,
    Attr::MODULE, Attr::NON_RECURSIVE, Attr::PURE, Attr::RECURSIVE};

void ModFileWriter::PutSubprogram(const Symbol &symbol) {
  auto attrs{symbol.attrs()};
  auto &details{symbol.get<SubprogramDetails>()};
  Attrs bindAttrs{};
  if (attrs.test(Attr::BIND_C)) {
    // bind(c) is a suffix, not prefix
    bindAttrs.set(Attr::BIND_C, true);
    attrs.set(Attr::BIND_C, false);
  }
  Attrs prefixAttrs{subprogramPrefixAttrs & attrs};
  // emit any non-prefix attributes in an attribute statement
  attrs &= ~subprogramPrefixAttrs;
  std::string ssBuf;
  llvm::raw_string_ostream ss{ssBuf};
  PutAttrs(ss, attrs);
  if (!ss.str().empty()) {
    decls_ << ss.str().substr(1) << "::" << symbol.name() << '\n';
  }
  bool isInterface{details.isInterface()};
  llvm::raw_ostream &os{isInterface ? decls_ : contains_};
  if (isInterface) {
    os << "interface\n";
  }
  PutAttrs(os, prefixAttrs, std::nullopt, ""s, " "s);
  os << (details.isFunction() ? "function " : "subroutine ");
  os << symbol.name() << '(';
  int n = 0;
  for (const auto &dummy : details.dummyArgs()) {
    if (n++ > 0) {
      os << ',';
    }
    if (dummy) {
      os << dummy->name();
    } else {
      os << "*";
    }
  }
  os << ')';
  PutAttrs(os, bindAttrs, details.bindName(), " "s, ""s);
  if (details.isFunction()) {
    const Symbol &result{details.result()};
    if (result.name() != symbol.name()) {
      os << " result(" << result.name() << ')';
    }
  }
  os << '\n';

  // walk symbols, collect ones needed for interface
  const Scope &scope{
      details.entryScope() ? *details.entryScope() : DEREF(symbol.scope())};
  SubprogramSymbolCollector collector{symbol, scope};
  collector.Collect();
  std::string typeBindingsBuf;
  llvm::raw_string_ostream typeBindings{typeBindingsBuf};
  ModFileWriter writer{context_};
  for (const Symbol &need : collector.symbols()) {
    writer.PutSymbol(typeBindings, need);
  }
  CHECK(typeBindings.str().empty());
  os << writer.uses_.str();
  for (const SourceName &import : collector.imports()) {
    decls_ << "import::" << import << "\n";
  }
  os << writer.decls_.str();
  os << "end\n";
  if (isInterface) {
    os << "end interface\n";
  }
}

static bool IsIntrinsicOp(const Symbol &symbol) {
  if (const auto *details{symbol.GetUltimate().detailsIf<GenericDetails>()}) {
    return details->kind().IsIntrinsicOperator();
  } else {
    return false;
  }
}

static llvm::raw_ostream &PutGenericName(
    llvm::raw_ostream &os, const Symbol &symbol) {
  if (IsGenericDefinedOp(symbol)) {
    return os << "operator(" << symbol.name() << ')';
  } else {
    return os << symbol.name();
  }
}

void ModFileWriter::PutGeneric(const Symbol &symbol) {
  auto &details{symbol.get<GenericDetails>()};
  PutGenericName(decls_ << "interface ", symbol) << '\n';
  for (const Symbol &specific : details.specificProcs()) {
    decls_ << "procedure::" << specific.name() << '\n';
  }
  decls_ << "end interface\n";
  if (symbol.attrs().test(Attr::PRIVATE)) {
    PutGenericName(decls_ << "private::", symbol) << '\n';
  }
}

void ModFileWriter::PutUse(const Symbol &symbol) {
  auto &details{symbol.get<UseDetails>()};
  auto &use{details.symbol()};
  uses_ << "use " << GetUsedModule(details).name();
  PutGenericName(uses_ << ",only:", symbol);
  // Can have intrinsic op with different local-name and use-name
  // (e.g. `operator(<)` and `operator(.lt.)`) but rename is not allowed
  if (!IsIntrinsicOp(symbol) && use.name() != symbol.name()) {
    PutGenericName(uses_ << "=>", use);
  }
  uses_ << '\n';
  PutUseExtraAttr(Attr::VOLATILE, symbol, use);
  PutUseExtraAttr(Attr::ASYNCHRONOUS, symbol, use);
}

// We have "USE local => use" in this module. If attr was added locally
// (i.e. on local but not on use), also write it out in the mod file.
void ModFileWriter::PutUseExtraAttr(
    Attr attr, const Symbol &local, const Symbol &use) {
  if (local.attrs().test(attr) && !use.attrs().test(attr)) {
    PutAttr(useExtraAttrs_, attr) << "::";
    useExtraAttrs_ << local.name() << '\n';
  }
}

// Collect the symbols of this scope sorted by their original order, not name.
// Namelists are an exception: they are sorted after other symbols.
SymbolVector CollectSymbols(const Scope &scope) {
  SymbolVector sorted;
  SymbolVector namelist;
  std::size_t commonSize{scope.commonBlocks().size()};
  auto symbols{scope.GetSymbols()};
  sorted.reserve(symbols.size() + commonSize);
  for (SymbolRef symbol : symbols) {
    if (!symbol->test(Symbol::Flag::ParentComp)) {
      if (symbol->has<NamelistDetails>()) {
        namelist.push_back(symbol);
      } else {
        sorted.push_back(symbol);
      }
    }
  }
  sorted.insert(sorted.end(), namelist.begin(), namelist.end());
  for (const auto &pair : scope.commonBlocks()) {
    sorted.push_back(*pair.second);
  }
  std::sort(sorted.end() - commonSize, sorted.end());
  return sorted;
}

void PutEntity(llvm::raw_ostream &os, const Symbol &symbol) {
  std::visit(
      common::visitors{
          [&](const ObjectEntityDetails &) { PutObjectEntity(os, symbol); },
          [&](const ProcEntityDetails &) { PutProcEntity(os, symbol); },
          [&](const TypeParamDetails &) { PutTypeParam(os, symbol); },
          [&](const auto &) {
            common::die("PutEntity: unexpected details: %s",
                DetailsToString(symbol.details()).c_str());
          },
      },
      symbol.details());
}

void PutShapeSpec(llvm::raw_ostream &os, const ShapeSpec &x) {
  if (x.lbound().isAssumed()) {
    CHECK(x.ubound().isAssumed());
    os << "..";
  } else {
    if (!x.lbound().isDeferred()) {
      PutBound(os, x.lbound());
    }
    os << ':';
    if (!x.ubound().isDeferred()) {
      PutBound(os, x.ubound());
    }
  }
}
void PutShape(
    llvm::raw_ostream &os, const ArraySpec &shape, char open, char close) {
  if (!shape.empty()) {
    os << open;
    bool first{true};
    for (const auto &shapeSpec : shape) {
      if (first) {
        first = false;
      } else {
        os << ',';
      }
      PutShapeSpec(os, shapeSpec);
    }
    os << close;
  }
}

void PutObjectEntity(llvm::raw_ostream &os, const Symbol &symbol) {
  auto &details{symbol.get<ObjectEntityDetails>()};
  PutEntity(
      os, symbol, [&]() { PutType(os, DEREF(symbol.GetType())); },
      symbol.attrs());
  PutShape(os, details.shape(), '(', ')');
  PutShape(os, details.coshape(), '[', ']');
  PutInit(os, symbol, details.init());
  os << '\n';
}

void PutProcEntity(llvm::raw_ostream &os, const Symbol &symbol) {
  if (symbol.attrs().test(Attr::INTRINSIC)) {
    os << "intrinsic::" << symbol.name() << '\n';
    return;
  }
  const auto &details{symbol.get<ProcEntityDetails>()};
  const ProcInterface &interface{details.interface()};
  Attrs attrs{symbol.attrs()};
  if (details.passName()) {
    attrs.reset(Attr::PASS);
  }
  PutEntity(
      os, symbol,
      [&]() {
        os << "procedure(";
        if (interface.symbol()) {
          os << interface.symbol()->name();
        } else if (interface.type()) {
          PutType(os, *interface.type());
        }
        os << ')';
        PutPassName(os, details.passName());
      },
      attrs);
  os << '\n';
}

void PutPassName(
    llvm::raw_ostream &os, const std::optional<SourceName> &passName) {
  if (passName) {
    os << ",pass(" << *passName << ')';
  }
}
void PutTypeParam(llvm::raw_ostream &os, const Symbol &symbol) {
  auto &details{symbol.get<TypeParamDetails>()};
  PutEntity(
      os, symbol,
      [&]() {
        PutType(os, DEREF(symbol.GetType()));
        PutLower(os << ',', common::EnumToString(details.attr()));
      },
      symbol.attrs());
  PutInit(os, details.init());
  os << '\n';
}

void PutInit(
    llvm::raw_ostream &os, const Symbol &symbol, const MaybeExpr &init) {
  if (init) {
    if (symbol.attrs().test(Attr::PARAMETER) ||
        symbol.owner().IsDerivedType()) {
      os << (symbol.attrs().test(Attr::POINTER) ? "=>" : "=");
      init->AsFortran(os);
    }
  }
}

void PutInit(llvm::raw_ostream &os, const MaybeIntExpr &init) {
  if (init) {
    init->AsFortran(os << '=');
  }
}

void PutBound(llvm::raw_ostream &os, const Bound &x) {
  if (x.isAssumed()) {
    os << '*';
  } else if (x.isDeferred()) {
    os << ':';
  } else {
    x.GetExplicit()->AsFortran(os);
  }
}

// Write an entity (object or procedure) declaration.
// writeType is called to write out the type.
void PutEntity(llvm::raw_ostream &os, const Symbol &symbol,
    std::function<void()> writeType, Attrs attrs) {
  writeType();
  MaybeExpr bindName;
  std::visit(common::visitors{
                 [&](const SubprogramDetails &x) { bindName = x.bindName(); },
                 [&](const ObjectEntityDetails &x) { bindName = x.bindName(); },
                 [&](const ProcEntityDetails &x) { bindName = x.bindName(); },
                 [&](const auto &) {},
             },
      symbol.details());
  PutAttrs(os, attrs, bindName);
  os << "::" << symbol.name();
}

// Put out each attribute to os, surrounded by `before` and `after` and
// mapped to lower case.
llvm::raw_ostream &PutAttrs(llvm::raw_ostream &os, Attrs attrs,
    const MaybeExpr &bindName, std::string before, std::string after) {
  attrs.set(Attr::PUBLIC, false); // no need to write PUBLIC
  attrs.set(Attr::EXTERNAL, false); // no need to write EXTERNAL
  if (bindName) {
    bindName->AsFortran(os << before << "bind(c, name=") << ')' << after;
    attrs.set(Attr::BIND_C, false);
  }
  for (std::size_t i{0}; i < Attr_enumSize; ++i) {
    Attr attr{static_cast<Attr>(i)};
    if (attrs.test(attr)) {
      PutAttr(os << before, attr) << after;
    }
  }
  return os;
}

llvm::raw_ostream &PutAttr(llvm::raw_ostream &os, Attr attr) {
  return PutLower(os, AttrToString(attr));
}

llvm::raw_ostream &PutType(llvm::raw_ostream &os, const DeclTypeSpec &type) {
  return PutLower(os, type.AsFortran());
}

llvm::raw_ostream &PutLower(llvm::raw_ostream &os, const std::string &str) {
  for (char c : str) {
    os << parser::ToLowerCaseLetter(c);
  }
  return os;
}

struct Temp {
  Temp(int fd, std::string path) : fd{fd}, path{path} {}
  Temp(Temp &&t) : fd{std::exchange(t.fd, -1)}, path{std::move(t.path)} {}
  ~Temp() {
    if (fd >= 0) {
      llvm::sys::fs::file_t native{llvm::sys::fs::convertFDToNativeFile(fd)};
      llvm::sys::fs::closeFile(native);
      llvm::sys::fs::remove(path.c_str());
    }
  }
  int fd;
  std::string path;
};

// Create a temp file in the same directory and with the same suffix as path.
// Return an open file descriptor and its path.
static llvm::ErrorOr<Temp> MkTemp(const std::string &path) {
  auto length{path.length()};
  auto dot{path.find_last_of("./")};
  std::string suffix{
      dot < length && path[dot] == '.' ? path.substr(dot + 1) : ""};
  CHECK(length > suffix.length() &&
      path.substr(length - suffix.length()) == suffix);
  auto prefix{path.substr(0, length - suffix.length())};
  int fd;
  llvm::SmallString<16> tempPath;
  if (std::error_code err{llvm::sys::fs::createUniqueFile(
          prefix + "%%%%%%" + suffix, fd, tempPath)}) {
    return err;
  }
  return Temp{fd, tempPath.c_str()};
}

// Write the module file at path, prepending header. If an error occurs,
// return errno, otherwise 0.
static std::error_code WriteFile(
    const std::string &path, const std::string &contents, bool debug) {
  auto header{std::string{ModHeader::bom} + ModHeader::magic +
      CheckSum(contents) + ModHeader::terminator};
  if (debug) {
    llvm::dbgs() << "Processing module " << path << ": ";
  }
  if (FileContentsMatch(path, header, contents)) {
    if (debug) {
      llvm::dbgs() << "module unchanged, not writing\n";
    }
    return {};
  }
  llvm::ErrorOr<Temp> temp{MkTemp(path)};
  if (!temp) {
    return temp.getError();
  }
  llvm::raw_fd_ostream writer(temp->fd, /*shouldClose=*/false);
  writer << header;
  writer << contents;
  writer.flush();
  if (writer.has_error()) {
    return writer.error();
  }
  if (debug) {
    llvm::dbgs() << "module written\n";
  }
  return llvm::sys::fs::rename(temp->path, path);
}

// Return true if the stream matches what we would write for the mod file.
static bool FileContentsMatch(const std::string &path,
    const std::string &header, const std::string &contents) {
  std::size_t hsize{header.size()};
  std::size_t csize{contents.size()};
  auto buf_or{llvm::MemoryBuffer::getFile(path)};
  if (!buf_or) {
    return false;
  }
  auto buf = std::move(buf_or.get());
  if (buf->getBufferSize() != hsize + csize) {
    return false;
  }
  if (!std::equal(header.begin(), header.end(), buf->getBufferStart(),
          buf->getBufferStart() + hsize)) {
    return false;
  }

  return std::equal(contents.begin(), contents.end(),
      buf->getBufferStart() + hsize, buf->getBufferEnd());
}

// Compute a simple hash of the contents of a module file and
// return it as a string of hex digits.
// This uses the Fowler-Noll-Vo hash function.
static std::string CheckSum(const std::string_view &contents) {
  std::uint64_t hash{0xcbf29ce484222325ull};
  for (char c : contents) {
    hash ^= c & 0xff;
    hash *= 0x100000001b3;
  }
  static const char *digits = "0123456789abcdef";
  std::string result(ModHeader::sumLen, '0');
  for (size_t i{ModHeader::sumLen}; hash != 0; hash >>= 4) {
    result[--i] = digits[hash & 0xf];
  }
  return result;
}

static bool VerifyHeader(llvm::ArrayRef<char> content) {
  std::string_view sv{content.data(), content.size()};
  if (sv.substr(0, ModHeader::magicLen) != ModHeader::magic) {
    return false;
  }
  std::string_view expectSum{sv.substr(ModHeader::magicLen, ModHeader::sumLen)};
  std::string actualSum{CheckSum(sv.substr(ModHeader::len))};
  return expectSum == actualSum;
}

Scope *ModFileReader::Read(const SourceName &name, Scope *ancestor) {
  std::string ancestorName; // empty for module
  if (ancestor) {
    if (auto *scope{ancestor->FindSubmodule(name)}) {
      return scope;
    }
    ancestorName = ancestor->GetName().value().ToString();
  } else {
    auto it{context_.globalScope().find(name)};
    if (it != context_.globalScope().end()) {
      return it->second->scope();
    }
  }
  parser::Parsing parsing{context_.allCookedSources()};
  parser::Options options;
  options.isModuleFile = true;
  options.features.Enable(common::LanguageFeature::BackslashEscapes);
  options.searchDirectories = context_.searchDirectories();
  auto path{ModFileName(name, ancestorName, context_.moduleFileSuffix())};
  const auto *sourceFile{parsing.Prescan(path, options)};
  if (parsing.messages().AnyFatalError()) {
    for (auto &msg : parsing.messages().messages()) {
      std::string str{msg.ToString()};
      Say(name, ancestorName, parser::MessageFixedText{str.c_str(), str.size()},
          path);
    }
    return nullptr;
  }
  CHECK(sourceFile);
  if (!VerifyHeader(sourceFile->content())) {
    Say(name, ancestorName, "File has invalid checksum: %s"_en_US,
        sourceFile->path());
    return nullptr;
  }
  llvm::raw_null_ostream NullStream;
  parsing.Parse(NullStream);
  auto &parseTree{parsing.parseTree()};
  if (!parsing.messages().empty() || !parsing.consumedWholeFile() ||
      !parseTree) {
    Say(name, ancestorName, "Module file is corrupt: %s"_err_en_US,
        sourceFile->path());
    return nullptr;
  }
  Scope *parentScope; // the scope this module/submodule goes into
  if (!ancestor) {
    parentScope = &context_.globalScope();
  } else if (std::optional<SourceName> parent{GetSubmoduleParent(*parseTree)}) {
    parentScope = Read(*parent, ancestor);
  } else {
    parentScope = ancestor;
  }
  ResolveNames(context_, *parseTree);
  const auto &it{parentScope->find(name)};
  if (it == parentScope->end()) {
    return nullptr;
  }
  auto &modSymbol{*it->second};
  modSymbol.set(Symbol::Flag::ModFile);
  return modSymbol.scope();
}

parser::Message &ModFileReader::Say(const SourceName &name,
    const std::string &ancestor, parser::MessageFixedText &&msg,
    const std::string &arg) {
  return context_
      .Say(name,
          ancestor.empty()
              ? "Error reading module file for module '%s'"_err_en_US
              : "Error reading module file for submodule '%s' of module '%s'"_err_en_US,
          name, ancestor)
      .Attach(name, std::move(msg), arg);
}

// program was read from a .mod file for a submodule; return the name of the
// submodule's parent submodule, nullptr if none.
static std::optional<SourceName> GetSubmoduleParent(
    const parser::Program &program) {
  CHECK(program.v.size() == 1);
  auto &unit{program.v.front()};
  auto &submod{std::get<common::Indirection<parser::Submodule>>(unit.u)};
  auto &stmt{
      std::get<parser::Statement<parser::SubmoduleStmt>>(submod.value().t)};
  auto &parentId{std::get<parser::ParentIdentifier>(stmt.statement.t)};
  if (auto &parent{std::get<std::optional<parser::Name>>(parentId.t)}) {
    return parent->source;
  } else {
    return std::nullopt;
  }
}

void SubprogramSymbolCollector::Collect() {
  const auto &details{symbol_.get<SubprogramDetails>()};
  isInterface_ = details.isInterface();
  for (const Symbol *dummyArg : details.dummyArgs()) {
    if (dummyArg) {
      DoSymbol(*dummyArg);
    }
  }
  if (details.isFunction()) {
    DoSymbol(details.result());
  }
  for (const auto &pair : scope_) {
    const Symbol &symbol{*pair.second};
    if (const auto *useDetails{symbol.detailsIf<UseDetails>()}) {
      if (useSet_.count(useDetails->symbol().GetUltimate()) > 0) {
        need_.push_back(symbol);
      }
    }
  }
}

void SubprogramSymbolCollector::DoSymbol(const Symbol &symbol) {
  DoSymbol(symbol.name(), symbol);
}

// Do symbols this one depends on; then add to need_
void SubprogramSymbolCollector::DoSymbol(
    const SourceName &name, const Symbol &symbol) {
  const auto &scope{symbol.owner()};
  if (scope != scope_ && !scope.IsDerivedType()) {
    if (scope != scope_.parent()) {
      useSet_.insert(symbol);
    }
    if (NeedImport(name, symbol)) {
      imports_.insert(name);
    }
    return;
  }
  if (!needSet_.insert(symbol).second) {
    return; // already done
  }
  std::visit(common::visitors{
                 [this](const ObjectEntityDetails &details) {
                   for (const ShapeSpec &spec : details.shape()) {
                     DoBound(spec.lbound());
                     DoBound(spec.ubound());
                   }
                   for (const ShapeSpec &spec : details.coshape()) {
                     DoBound(spec.lbound());
                     DoBound(spec.ubound());
                   }
                   if (const Symbol * commonBlock{details.commonBlock()}) {
                     DoSymbol(*commonBlock);
                   }
                 },
                 [this](const CommonBlockDetails &details) {
                   for (const auto &object : details.objects()) {
                     DoSymbol(*object);
                   }
                 },
                 [](const auto &) {},
             },
      symbol.details());
  if (!symbol.has<UseDetails>()) {
    DoType(symbol.GetType());
  }
  if (!scope.IsDerivedType()) {
    need_.push_back(symbol);
  }
}

void SubprogramSymbolCollector::DoType(const DeclTypeSpec *type) {
  if (!type) {
    return;
  }
  switch (type->category()) {
  case DeclTypeSpec::Numeric:
  case DeclTypeSpec::Logical:
    break; // nothing to do
  case DeclTypeSpec::Character:
    DoParamValue(type->characterTypeSpec().length());
    break;
  default:
    if (const DerivedTypeSpec * derived{type->AsDerived()}) {
      const auto &typeSymbol{derived->typeSymbol()};
      if (const DerivedTypeSpec * extends{typeSymbol.GetParentTypeSpec()}) {
        DoSymbol(extends->name(), extends->typeSymbol());
      }
      for (const auto &pair : derived->parameters()) {
        DoParamValue(pair.second);
      }
      for (const auto &pair : *typeSymbol.scope()) {
        const Symbol &comp{*pair.second};
        DoSymbol(comp);
      }
      DoSymbol(derived->name(), derived->typeSymbol());
    }
  }
}

void SubprogramSymbolCollector::DoBound(const Bound &bound) {
  if (const MaybeSubscriptIntExpr & expr{bound.GetExplicit()}) {
    DoExpr(*expr);
  }
}
void SubprogramSymbolCollector::DoParamValue(const ParamValue &paramValue) {
  if (const auto &expr{paramValue.GetExplicit()}) {
    DoExpr(*expr);
  }
}

// Do we need a IMPORT of this symbol into an interface block?
bool SubprogramSymbolCollector::NeedImport(
    const SourceName &name, const Symbol &symbol) {
  if (!isInterface_) {
    return false;
  } else if (symbol.owner() != scope_.parent()) {
    // detect import from parent of use-associated symbol
    // can be null in the case of a use-associated derived type's parent type
    const auto *found{scope_.FindSymbol(name)};
    CHECK(found || symbol.has<DerivedTypeDetails>());
    return found && found->has<UseDetails>() && found->owner() != scope_;
  } else {
    return true;
  }
}

} // namespace Fortran::semantics
