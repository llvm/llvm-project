//===- ClangAttrEmitter.cpp - Generate Clang attribute handling =-*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These tablegen backends emit Clang attribute processing code
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/StringMatcher.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

namespace {

class FlattenedSpelling {
  std::string V, N, NS;
  bool K;

public:
  FlattenedSpelling(const std::string &Variety, const std::string &Name,
                    const std::string &Namespace, bool KnownToGCC) :
    V(Variety), N(Name), NS(Namespace), K(KnownToGCC) {}
  explicit FlattenedSpelling(const Record &Spelling) :
    V(Spelling.getValueAsString("Variety")),
    N(Spelling.getValueAsString("Name")) {

    assert(V != "GCC" && "Given a GCC spelling, which means this hasn't been"
           "flattened!");
    if (V == "CXX11" || V == "Pragma")
      NS = Spelling.getValueAsString("Namespace");
    bool Unset;
    K = Spelling.getValueAsBitOrUnset("KnownToGCC", Unset);
  }

  const std::string &variety() const { return V; }
  const std::string &name() const { return N; }
  const std::string &nameSpace() const { return NS; }
  bool knownToGCC() const { return K; }
};

} // end anonymous namespace

static std::vector<FlattenedSpelling>
GetFlattenedSpellings(const Record &Attr) {
  std::vector<Record *> Spellings = Attr.getValueAsListOfDefs("Spellings");
  std::vector<FlattenedSpelling> Ret;

  for (const auto &Spelling : Spellings) {
    if (Spelling->getValueAsString("Variety") == "GCC") {
      // Gin up two new spelling objects to add into the list.
      Ret.emplace_back("GNU", Spelling->getValueAsString("Name"), "", true);
      Ret.emplace_back("CXX11", Spelling->getValueAsString("Name"), "gnu",
                       true);
    } else
      Ret.push_back(FlattenedSpelling(*Spelling));
  }

  return Ret;
}

static std::string ReadPCHRecord(StringRef type) {
  return StringSwitch<std::string>(type)
    .EndsWith("Decl *", "GetLocalDeclAs<" 
              + std::string(type, 0, type.size()-1) + ">(F, Record[Idx++])")
    .Case("TypeSourceInfo *", "GetTypeSourceInfo(F, Record, Idx)")
    .Case("Expr *", "ReadExpr(F)")
    .Case("IdentifierInfo *", "GetIdentifierInfo(F, Record, Idx)")
    .Case("StringRef", "ReadString(Record, Idx)")
    .Default("Record[Idx++]");
}

// Get a type that is suitable for storing an object of the specified type.
static StringRef getStorageType(StringRef type) {
  return StringSwitch<StringRef>(type)
    .Case("StringRef", "std::string")
    .Default(type);
}

// Assumes that the way to get the value is SA->getname()
static std::string WritePCHRecord(StringRef type, StringRef name) {
  return "Record." + StringSwitch<std::string>(type)
    .EndsWith("Decl *", "AddDeclRef(" + std::string(name) + ");\n")
    .Case("TypeSourceInfo *", "AddTypeSourceInfo(" + std::string(name) + ");\n")
    .Case("Expr *", "AddStmt(" + std::string(name) + ");\n")
    .Case("IdentifierInfo *", "AddIdentifierRef(" + std::string(name) + ");\n")
    .Case("StringRef", "AddString(" + std::string(name) + ");\n")
    .Default("push_back(" + std::string(name) + ");\n");
}

// Normalize attribute name by removing leading and trailing
// underscores. For example, __foo, foo__, __foo__ would
// become foo.
static StringRef NormalizeAttrName(StringRef AttrName) {
  if (AttrName.startswith("__"))
    AttrName = AttrName.substr(2, AttrName.size());

  if (AttrName.endswith("__"))
    AttrName = AttrName.substr(0, AttrName.size() - 2);

  return AttrName;
}

// Normalize the name by removing any and all leading and trailing underscores.
// This is different from NormalizeAttrName in that it also handles names like
// _pascal and __pascal.
static StringRef NormalizeNameForSpellingComparison(StringRef Name) {
  return Name.trim("_");
}

// Normalize attribute spelling only if the spelling has both leading
// and trailing underscores. For example, __ms_struct__ will be 
// normalized to "ms_struct"; __cdecl will remain intact.
static StringRef NormalizeAttrSpelling(StringRef AttrSpelling) {
  if (AttrSpelling.startswith("__") && AttrSpelling.endswith("__")) {
    AttrSpelling = AttrSpelling.substr(2, AttrSpelling.size() - 4);
  }

  return AttrSpelling;
}

typedef std::vector<std::pair<std::string, const Record *>> ParsedAttrMap;

static ParsedAttrMap getParsedAttrList(const RecordKeeper &Records,
                                       ParsedAttrMap *Dupes = nullptr) {
  std::vector<Record *> Attrs = Records.getAllDerivedDefinitions("Attr");
  std::set<std::string> Seen;
  ParsedAttrMap R;
  for (const auto *Attr : Attrs) {
    if (Attr->getValueAsBit("SemaHandler")) {
      std::string AN;
      if (Attr->isSubClassOf("TargetSpecificAttr") &&
          !Attr->isValueUnset("ParseKind")) {
        AN = Attr->getValueAsString("ParseKind");

        // If this attribute has already been handled, it does not need to be
        // handled again.
        if (Seen.find(AN) != Seen.end()) {
          if (Dupes)
            Dupes->push_back(std::make_pair(AN, Attr));
          continue;
        }
        Seen.insert(AN);
      } else
        AN = NormalizeAttrName(Attr->getName()).str();

      R.push_back(std::make_pair(AN, Attr));
    }
  }
  return R;
}

namespace {

  class Argument {
    std::string lowerName, upperName;
    StringRef attrName;
    bool isOpt;
    bool Fake;

  public:
    Argument(const Record &Arg, StringRef Attr)
      : lowerName(Arg.getValueAsString("Name")), upperName(lowerName),
        attrName(Attr), isOpt(false), Fake(false) {
      if (!lowerName.empty()) {
        lowerName[0] = std::tolower(lowerName[0]);
        upperName[0] = std::toupper(upperName[0]);
      }
      // Work around MinGW's macro definition of 'interface' to 'struct'. We
      // have an attribute argument called 'Interface', so only the lower case
      // name conflicts with the macro definition.
      if (lowerName == "interface")
        lowerName = "interface_";
    }
    virtual ~Argument() = default;

    StringRef getLowerName() const { return lowerName; }
    StringRef getUpperName() const { return upperName; }
    StringRef getAttrName() const { return attrName; }

    bool isOptional() const { return isOpt; }
    void setOptional(bool set) { isOpt = set; }

    bool isFake() const { return Fake; }
    void setFake(bool fake) { Fake = fake; }

    // These functions print the argument contents formatted in different ways.
    virtual void writeAccessors(raw_ostream &OS) const = 0;
    virtual void writeAccessorDefinitions(raw_ostream &OS) const {}
    virtual void writeASTVisitorTraversal(raw_ostream &OS) const {}
    virtual void writeCloneArgs(raw_ostream &OS) const = 0;
    virtual void writeTemplateInstantiationArgs(raw_ostream &OS) const = 0;
    virtual void writeTemplateInstantiation(raw_ostream &OS) const {}
    virtual void writeCtorBody(raw_ostream &OS) const {}
    virtual void writeCtorInitializers(raw_ostream &OS) const = 0;
    virtual void writeCtorDefaultInitializers(raw_ostream &OS) const = 0;
    virtual void writeCtorParameters(raw_ostream &OS) const = 0;
    virtual void writeDeclarations(raw_ostream &OS) const = 0;
    virtual void writePCHReadArgs(raw_ostream &OS) const = 0;
    virtual void writePCHReadDecls(raw_ostream &OS) const = 0;
    virtual void writePCHWrite(raw_ostream &OS) const = 0;
    virtual void writeValue(raw_ostream &OS) const = 0;
    virtual void writeDump(raw_ostream &OS) const = 0;
    virtual void writeDumpChildren(raw_ostream &OS) const {}
    virtual void writeHasChildren(raw_ostream &OS) const { OS << "false"; }

    virtual bool isEnumArg() const { return false; }
    virtual bool isVariadicEnumArg() const { return false; }
    virtual bool isVariadic() const { return false; }

    virtual void writeImplicitCtorArgs(raw_ostream &OS) const {
      OS << getUpperName();
    }
  };

  class SimpleArgument : public Argument {
    std::string type;

  public:
    SimpleArgument(const Record &Arg, StringRef Attr, std::string T)
        : Argument(Arg, Attr), type(std::move(T)) {}

    std::string getType() const { return type; }

    void writeAccessors(raw_ostream &OS) const override {
      OS << "  " << type << " get" << getUpperName() << "() const {\n";
      OS << "    return " << getLowerName() << ";\n";
      OS << "  }";
    }

    void writeCloneArgs(raw_ostream &OS) const override {
      OS << getLowerName();
    }

    void writeTemplateInstantiationArgs(raw_ostream &OS) const override {
      OS << "A->get" << getUpperName() << "()";
    }

    void writeCtorInitializers(raw_ostream &OS) const override {
      OS << getLowerName() << "(" << getUpperName() << ")";
    }

    void writeCtorDefaultInitializers(raw_ostream &OS) const override {
      OS << getLowerName() << "()";
    }

    void writeCtorParameters(raw_ostream &OS) const override {
      OS << type << " " << getUpperName();
    }

    void writeDeclarations(raw_ostream &OS) const override {
      OS << type << " " << getLowerName() << ";";
    }

    void writePCHReadDecls(raw_ostream &OS) const override {
      std::string read = ReadPCHRecord(type);
      OS << "    " << type << " " << getLowerName() << " = " << read << ";\n";
    }

    void writePCHReadArgs(raw_ostream &OS) const override {
      OS << getLowerName();
    }

    void writePCHWrite(raw_ostream &OS) const override {
      OS << "    " << WritePCHRecord(type, "SA->get" +
                                           std::string(getUpperName()) + "()");
    }

    void writeValue(raw_ostream &OS) const override {
      if (type == "FunctionDecl *") {
        OS << "\" << get" << getUpperName()
           << "()->getNameInfo().getAsString() << \"";
      } else if (type == "IdentifierInfo *") {
        OS << "\";\n";
        if (isOptional())
          OS << "    if (get" << getUpperName() << "()) ";
        else
          OS << "    ";
        OS << "OS << get" << getUpperName() << "()->getName();\n";
        OS << "    OS << \"";
      } else if (type == "TypeSourceInfo *") {
        OS << "\" << get" << getUpperName() << "().getAsString() << \"";
      } else {
        OS << "\" << get" << getUpperName() << "() << \"";
      }
    }

    void writeDump(raw_ostream &OS) const override {
      if (type == "FunctionDecl *") {
        OS << "    OS << \" \";\n";
        OS << "    dumpBareDeclRef(SA->get" << getUpperName() << "());\n"; 
      } else if (type == "IdentifierInfo *") {
        if (isOptional())
          OS << "    if (SA->get" << getUpperName() << "())\n  ";
        OS << "    OS << \" \" << SA->get" << getUpperName()
           << "()->getName();\n";
      } else if (type == "TypeSourceInfo *") {
        OS << "    OS << \" \" << SA->get" << getUpperName()
           << "().getAsString();\n";
      } else if (type == "bool") {
        OS << "    if (SA->get" << getUpperName() << "()) OS << \" "
           << getUpperName() << "\";\n";
      } else if (type == "int" || type == "unsigned") {
        OS << "    OS << \" \" << SA->get" << getUpperName() << "();\n";
      } else {
        llvm_unreachable("Unknown SimpleArgument type!");
      }
    }
  };

  class DefaultSimpleArgument : public SimpleArgument {
    int64_t Default;

  public:
    DefaultSimpleArgument(const Record &Arg, StringRef Attr,
                          std::string T, int64_t Default)
      : SimpleArgument(Arg, Attr, T), Default(Default) {}

    void writeAccessors(raw_ostream &OS) const override {
      SimpleArgument::writeAccessors(OS);

      OS << "\n\n  static const " << getType() << " Default" << getUpperName()
         << " = ";
      if (getType() == "bool")
        OS << (Default != 0 ? "true" : "false");
      else
        OS << Default;
      OS << ";";
    }
  };

  class StringArgument : public Argument {
  public:
    StringArgument(const Record &Arg, StringRef Attr)
      : Argument(Arg, Attr)
    {}

    void writeAccessors(raw_ostream &OS) const override {
      OS << "  llvm::StringRef get" << getUpperName() << "() const {\n";
      OS << "    return llvm::StringRef(" << getLowerName() << ", "
         << getLowerName() << "Length);\n";
      OS << "  }\n";
      OS << "  unsigned get" << getUpperName() << "Length() const {\n";
      OS << "    return " << getLowerName() << "Length;\n";
      OS << "  }\n";
      OS << "  void set" << getUpperName()
         << "(ASTContext &C, llvm::StringRef S) {\n";
      OS << "    " << getLowerName() << "Length = S.size();\n";
      OS << "    this->" << getLowerName() << " = new (C, 1) char ["
         << getLowerName() << "Length];\n";
      OS << "    if (!S.empty())\n";
      OS << "      std::memcpy(this->" << getLowerName() << ", S.data(), "
         << getLowerName() << "Length);\n";
      OS << "  }";
    }

    void writeCloneArgs(raw_ostream &OS) const override {
      OS << "get" << getUpperName() << "()";
    }

    void writeTemplateInstantiationArgs(raw_ostream &OS) const override {
      OS << "A->get" << getUpperName() << "()";
    }

    void writeCtorBody(raw_ostream &OS) const override {
      OS << "      if (!" << getUpperName() << ".empty())\n";
      OS << "        std::memcpy(" << getLowerName() << ", " << getUpperName()
         << ".data(), " << getLowerName() << "Length);\n";
    }

    void writeCtorInitializers(raw_ostream &OS) const override {
      OS << getLowerName() << "Length(" << getUpperName() << ".size()),"
         << getLowerName() << "(new (Ctx, 1) char[" << getLowerName()
         << "Length])";
    }

    void writeCtorDefaultInitializers(raw_ostream &OS) const override {
      OS << getLowerName() << "Length(0)," << getLowerName() << "(nullptr)";
    }

    void writeCtorParameters(raw_ostream &OS) const override {
      OS << "llvm::StringRef " << getUpperName();
    }

    void writeDeclarations(raw_ostream &OS) const override {
      OS << "unsigned " << getLowerName() << "Length;\n";
      OS << "char *" << getLowerName() << ";";
    }

    void writePCHReadDecls(raw_ostream &OS) const override {
      OS << "    std::string " << getLowerName()
         << "= ReadString(Record, Idx);\n";
    }

    void writePCHReadArgs(raw_ostream &OS) const override {
      OS << getLowerName();
    }

    void writePCHWrite(raw_ostream &OS) const override {
      OS << "    Record.AddString(SA->get" << getUpperName() << "());\n";
    }

    void writeValue(raw_ostream &OS) const override {
      OS << "\\\"\" << get" << getUpperName() << "() << \"\\\"";
    }

    void writeDump(raw_ostream &OS) const override {
      OS << "    OS << \" \\\"\" << SA->get" << getUpperName()
         << "() << \"\\\"\";\n";
    }
  };

  class AlignedArgument : public Argument {
  public:
    AlignedArgument(const Record &Arg, StringRef Attr)
      : Argument(Arg, Attr)
    {}

    void writeAccessors(raw_ostream &OS) const override {
      OS << "  bool is" << getUpperName() << "Dependent() const;\n";

      OS << "  unsigned get" << getUpperName() << "(ASTContext &Ctx) const;\n";

      OS << "  bool is" << getUpperName() << "Expr() const {\n";
      OS << "    return is" << getLowerName() << "Expr;\n";
      OS << "  }\n";

      OS << "  Expr *get" << getUpperName() << "Expr() const {\n";
      OS << "    assert(is" << getLowerName() << "Expr);\n";
      OS << "    return " << getLowerName() << "Expr;\n";
      OS << "  }\n";

      OS << "  TypeSourceInfo *get" << getUpperName() << "Type() const {\n";
      OS << "    assert(!is" << getLowerName() << "Expr);\n";
      OS << "    return " << getLowerName() << "Type;\n";
      OS << "  }";
    }

    void writeAccessorDefinitions(raw_ostream &OS) const override {
      OS << "bool " << getAttrName() << "Attr::is" << getUpperName()
         << "Dependent() const {\n";
      OS << "  if (is" << getLowerName() << "Expr)\n";
      OS << "    return " << getLowerName() << "Expr && (" << getLowerName()
         << "Expr->isValueDependent() || " << getLowerName()
         << "Expr->isTypeDependent());\n"; 
      OS << "  else\n";
      OS << "    return " << getLowerName()
         << "Type->getType()->isDependentType();\n";
      OS << "}\n";

      // FIXME: Do not do the calculation here
      // FIXME: Handle types correctly
      // A null pointer means maximum alignment
      OS << "unsigned " << getAttrName() << "Attr::get" << getUpperName()
         << "(ASTContext &Ctx) const {\n";
      OS << "  assert(!is" << getUpperName() << "Dependent());\n";
      OS << "  if (is" << getLowerName() << "Expr)\n";
      OS << "    return " << getLowerName() << "Expr ? " << getLowerName()
         << "Expr->EvaluateKnownConstInt(Ctx).getZExtValue()"
         << " * Ctx.getCharWidth() : "
         << "Ctx.getTargetDefaultAlignForAttributeAligned();\n";
      OS << "  else\n";
      OS << "    return 0; // FIXME\n";
      OS << "}\n";
    }

    void writeCloneArgs(raw_ostream &OS) const override {
      OS << "is" << getLowerName() << "Expr, is" << getLowerName()
         << "Expr ? static_cast<void*>(" << getLowerName()
         << "Expr) : " << getLowerName()
         << "Type";
    }

    void writeTemplateInstantiationArgs(raw_ostream &OS) const override {
      // FIXME: move the definition in Sema::InstantiateAttrs to here.
      // In the meantime, aligned attributes are cloned.
    }

    void writeCtorBody(raw_ostream &OS) const override {
      OS << "    if (is" << getLowerName() << "Expr)\n";
      OS << "       " << getLowerName() << "Expr = reinterpret_cast<Expr *>("
         << getUpperName() << ");\n";
      OS << "    else\n";
      OS << "       " << getLowerName()
         << "Type = reinterpret_cast<TypeSourceInfo *>(" << getUpperName()
         << ");\n";
    }

    void writeCtorInitializers(raw_ostream &OS) const override {
      OS << "is" << getLowerName() << "Expr(Is" << getUpperName() << "Expr)";
    }

    void writeCtorDefaultInitializers(raw_ostream &OS) const override {
      OS << "is" << getLowerName() << "Expr(false)";
    }

    void writeCtorParameters(raw_ostream &OS) const override {
      OS << "bool Is" << getUpperName() << "Expr, void *" << getUpperName();
    }

    void writeImplicitCtorArgs(raw_ostream &OS) const override {
      OS << "Is" << getUpperName() << "Expr, " << getUpperName();
    }

    void writeDeclarations(raw_ostream &OS) const override {
      OS << "bool is" << getLowerName() << "Expr;\n";
      OS << "union {\n";
      OS << "Expr *" << getLowerName() << "Expr;\n";
      OS << "TypeSourceInfo *" << getLowerName() << "Type;\n";
      OS << "};";
    }

    void writePCHReadArgs(raw_ostream &OS) const override {
      OS << "is" << getLowerName() << "Expr, " << getLowerName() << "Ptr";
    }

    void writePCHReadDecls(raw_ostream &OS) const override {
      OS << "    bool is" << getLowerName() << "Expr = Record[Idx++];\n";
      OS << "    void *" << getLowerName() << "Ptr;\n";
      OS << "    if (is" << getLowerName() << "Expr)\n";
      OS << "      " << getLowerName() << "Ptr = ReadExpr(F);\n";
      OS << "    else\n";
      OS << "      " << getLowerName()
         << "Ptr = GetTypeSourceInfo(F, Record, Idx);\n";
    }

    void writePCHWrite(raw_ostream &OS) const override {
      OS << "    Record.push_back(SA->is" << getUpperName() << "Expr());\n";
      OS << "    if (SA->is" << getUpperName() << "Expr())\n";
      OS << "      Record.AddStmt(SA->get" << getUpperName() << "Expr());\n";
      OS << "    else\n";
      OS << "      Record.AddTypeSourceInfo(SA->get" << getUpperName()
         << "Type());\n";
    }

    void writeValue(raw_ostream &OS) const override {
      OS << "\";\n";
      // The aligned attribute argument expression is optional.
      OS << "    if (is" << getLowerName() << "Expr && "
         << getLowerName() << "Expr)\n";
      OS << "      " << getLowerName() << "Expr->printPretty(OS, nullptr, Policy);\n";
      OS << "    OS << \"";
    }

    void writeDump(raw_ostream &OS) const override {}

    void writeDumpChildren(raw_ostream &OS) const override {
      OS << "    if (SA->is" << getUpperName() << "Expr())\n";
      OS << "      dumpStmt(SA->get" << getUpperName() << "Expr());\n";
      OS << "    else\n";
      OS << "      dumpType(SA->get" << getUpperName()
         << "Type()->getType());\n";
    }

    void writeHasChildren(raw_ostream &OS) const override {
      OS << "SA->is" << getUpperName() << "Expr()";
    }
  };

  class VariadicArgument : public Argument {
    std::string Type, ArgName, ArgSizeName, RangeName;

  protected:
    // Assumed to receive a parameter: raw_ostream OS.
    virtual void writeValueImpl(raw_ostream &OS) const {
      OS << "    OS << Val;\n";
    }

  public:
    VariadicArgument(const Record &Arg, StringRef Attr, std::string T)
        : Argument(Arg, Attr), Type(std::move(T)),
          ArgName(getLowerName().str() + "_"), ArgSizeName(ArgName + "Size"),
          RangeName(getLowerName()) {}

    const std::string &getType() const { return Type; }
    const std::string &getArgName() const { return ArgName; }
    const std::string &getArgSizeName() const { return ArgSizeName; }
    bool isVariadic() const override { return true; }

    void writeAccessors(raw_ostream &OS) const override {
      std::string IteratorType = getLowerName().str() + "_iterator";
      std::string BeginFn = getLowerName().str() + "_begin()";
      std::string EndFn = getLowerName().str() + "_end()";
      
      OS << "  typedef " << Type << "* " << IteratorType << ";\n";
      OS << "  " << IteratorType << " " << BeginFn << " const {"
         << " return " << ArgName << "; }\n";
      OS << "  " << IteratorType << " " << EndFn << " const {"
         << " return " << ArgName << " + " << ArgSizeName << "; }\n";
      OS << "  unsigned " << getLowerName() << "_size() const {"
         << " return " << ArgSizeName << "; }\n";
      OS << "  llvm::iterator_range<" << IteratorType << "> " << RangeName
         << "() const { return llvm::make_range(" << BeginFn << ", " << EndFn
         << "); }\n";
    }

    void writeCloneArgs(raw_ostream &OS) const override {
      OS << ArgName << ", " << ArgSizeName;
    }

    void writeTemplateInstantiationArgs(raw_ostream &OS) const override {
      // This isn't elegant, but we have to go through public methods...
      OS << "A->" << getLowerName() << "_begin(), "
         << "A->" << getLowerName() << "_size()";
    }

    void writeCtorBody(raw_ostream &OS) const override {
      OS << "    std::copy(" << getUpperName() << ", " << getUpperName()
         << " + " << ArgSizeName << ", " << ArgName << ");\n";
    }

    void writeCtorInitializers(raw_ostream &OS) const override {
      OS << ArgSizeName << "(" << getUpperName() << "Size), "
         << ArgName << "(new (Ctx, 16) " << getType() << "["
         << ArgSizeName << "])";
    }

    void writeCtorDefaultInitializers(raw_ostream &OS) const override {
      OS << ArgSizeName << "(0), " << ArgName << "(nullptr)";
    }

    void writeCtorParameters(raw_ostream &OS) const override {
      OS << getType() << " *" << getUpperName() << ", unsigned "
         << getUpperName() << "Size";
    }

    void writeImplicitCtorArgs(raw_ostream &OS) const override {
      OS << getUpperName() << ", " << getUpperName() << "Size";
    }

    void writeDeclarations(raw_ostream &OS) const override {
      OS << "  unsigned " << ArgSizeName << ";\n";
      OS << "  " << getType() << " *" << ArgName << ";";
    }

    void writePCHReadDecls(raw_ostream &OS) const override {
      OS << "    unsigned " << getLowerName() << "Size = Record[Idx++];\n";
      OS << "    SmallVector<" << getType() << ", 4> "
         << getLowerName() << ";\n";
      OS << "    " << getLowerName() << ".reserve(" << getLowerName()
         << "Size);\n";

      // If we can't store the values in the current type (if it's something
      // like StringRef), store them in a different type and convert the
      // container afterwards.
      std::string StorageType = getStorageType(getType());
      std::string StorageName = getLowerName();
      if (StorageType != getType()) {
        StorageName += "Storage";
        OS << "    SmallVector<" << StorageType << ", 4> "
           << StorageName << ";\n";
        OS << "    " << StorageName << ".reserve(" << getLowerName()
           << "Size);\n";
      }

      OS << "    for (unsigned i = 0; i != " << getLowerName() << "Size; ++i)\n";
      std::string read = ReadPCHRecord(Type);
      OS << "      " << StorageName << ".push_back(" << read << ");\n";

      if (StorageType != getType()) {
        OS << "    for (unsigned i = 0; i != " << getLowerName() << "Size; ++i)\n";
        OS << "      " << getLowerName() << ".push_back("
           << StorageName << "[i]);\n";
      }
    }

    void writePCHReadArgs(raw_ostream &OS) const override {
      OS << getLowerName() << ".data(), " << getLowerName() << "Size";
    }

    void writePCHWrite(raw_ostream &OS) const override {
      OS << "    Record.push_back(SA->" << getLowerName() << "_size());\n";
      OS << "    for (auto &Val : SA->" << RangeName << "())\n";
      OS << "      " << WritePCHRecord(Type, "Val");
    }

    void writeValue(raw_ostream &OS) const override {
      OS << "\";\n";
      OS << "  bool isFirst = true;\n"
         << "  for (const auto &Val : " << RangeName << "()) {\n"
         << "    if (isFirst) isFirst = false;\n"
         << "    else OS << \", \";\n";
      writeValueImpl(OS);
      OS << "  }\n";
      OS << "  OS << \"";
    }

    void writeDump(raw_ostream &OS) const override {
      OS << "    for (const auto &Val : SA->" << RangeName << "())\n";
      OS << "      OS << \" \" << Val;\n";
    }
  };

  // Unique the enums, but maintain the original declaration ordering.
  std::vector<std::string>
  uniqueEnumsInOrder(const std::vector<std::string> &enums) {
    std::vector<std::string> uniques;
    std::set<std::string> unique_set(enums.begin(), enums.end());
    for (const auto &i : enums) {
      auto set_i = unique_set.find(i);
      if (set_i != unique_set.end()) {
        uniques.push_back(i);
        unique_set.erase(set_i);
      }
    }
    return uniques;
  }

  class EnumArgument : public Argument {
    std::string type;
    std::vector<std::string> values, enums, uniques;
  public:
    EnumArgument(const Record &Arg, StringRef Attr)
      : Argument(Arg, Attr), type(Arg.getValueAsString("Type")),
        values(Arg.getValueAsListOfStrings("Values")),
        enums(Arg.getValueAsListOfStrings("Enums")),
        uniques(uniqueEnumsInOrder(enums))
    {
      // FIXME: Emit a proper error
      assert(!uniques.empty());
    }

    bool isEnumArg() const override { return true; }

    void writeAccessors(raw_ostream &OS) const override {
      OS << "  " << type << " get" << getUpperName() << "() const {\n";
      OS << "    return " << getLowerName() << ";\n";
      OS << "  }";
    }

    void writeCloneArgs(raw_ostream &OS) const override {
      OS << getLowerName();
    }

    void writeTemplateInstantiationArgs(raw_ostream &OS) const override {
      OS << "A->get" << getUpperName() << "()";
    }
    void writeCtorInitializers(raw_ostream &OS) const override {
      OS << getLowerName() << "(" << getUpperName() << ")";
    }
    void writeCtorDefaultInitializers(raw_ostream &OS) const override {
      OS << getLowerName() << "(" << type << "(0))";
    }
    void writeCtorParameters(raw_ostream &OS) const override {
      OS << type << " " << getUpperName();
    }
    void writeDeclarations(raw_ostream &OS) const override {
      auto i = uniques.cbegin(), e = uniques.cend();
      // The last one needs to not have a comma.
      --e;

      OS << "public:\n";
      OS << "  enum " << type << " {\n";
      for (; i != e; ++i)
        OS << "    " << *i << ",\n";
      OS << "    " << *e << "\n";
      OS << "  };\n";
      OS << "private:\n";
      OS << "  " << type << " " << getLowerName() << ";";
    }

    void writePCHReadDecls(raw_ostream &OS) const override {
      OS << "    " << getAttrName() << "Attr::" << type << " " << getLowerName()
         << "(static_cast<" << getAttrName() << "Attr::" << type
         << ">(Record[Idx++]));\n";
    }

    void writePCHReadArgs(raw_ostream &OS) const override {
      OS << getLowerName();
    }

    void writePCHWrite(raw_ostream &OS) const override {
      OS << "Record.push_back(SA->get" << getUpperName() << "());\n";
    }

    void writeValue(raw_ostream &OS) const override {
      // FIXME: this isn't 100% correct -- some enum arguments require printing
      // as a string literal, while others require printing as an identifier.
      // Tablegen currently does not distinguish between the two forms.
      OS << "\\\"\" << " << getAttrName() << "Attr::Convert" << type << "ToStr(get"
         << getUpperName() << "()) << \"\\\"";
    }

    void writeDump(raw_ostream &OS) const override {
      OS << "    switch(SA->get" << getUpperName() << "()) {\n";
      for (const auto &I : uniques) {
        OS << "    case " << getAttrName() << "Attr::" << I << ":\n";
        OS << "      OS << \" " << I << "\";\n";
        OS << "      break;\n";
      }
      OS << "    }\n";
    }

    void writeConversion(raw_ostream &OS) const {
      OS << "  static bool ConvertStrTo" << type << "(StringRef Val, ";
      OS << type << " &Out) {\n";
      OS << "    Optional<" << type << "> R = llvm::StringSwitch<Optional<";
      OS << type << ">>(Val)\n";
      for (size_t I = 0; I < enums.size(); ++I) {
        OS << "      .Case(\"" << values[I] << "\", ";
        OS << getAttrName() << "Attr::" << enums[I] << ")\n";
      }
      OS << "      .Default(Optional<" << type << ">());\n";
      OS << "    if (R) {\n";
      OS << "      Out = *R;\n      return true;\n    }\n";
      OS << "    return false;\n";
      OS << "  }\n\n";

      // Mapping from enumeration values back to enumeration strings isn't
      // trivial because some enumeration values have multiple named
      // enumerators, such as type_visibility(internal) and
      // type_visibility(hidden) both mapping to TypeVisibilityAttr::Hidden.
      OS << "  static const char *Convert" << type << "ToStr("
         << type << " Val) {\n"
         << "    switch(Val) {\n";
      std::set<std::string> Uniques;
      for (size_t I = 0; I < enums.size(); ++I) {
        if (Uniques.insert(enums[I]).second)
          OS << "    case " << getAttrName() << "Attr::" << enums[I]
             << ": return \"" << values[I] << "\";\n";       
      }
      OS << "    }\n"
         << "    llvm_unreachable(\"No enumerator with that value\");\n"
         << "  }\n";
    }
  };
  
  class VariadicEnumArgument: public VariadicArgument {
    std::string type, QualifiedTypeName;
    std::vector<std::string> values, enums, uniques;

  protected:
    void writeValueImpl(raw_ostream &OS) const override {
      // FIXME: this isn't 100% correct -- some enum arguments require printing
      // as a string literal, while others require printing as an identifier.
      // Tablegen currently does not distinguish between the two forms.
      OS << "    OS << \"\\\"\" << " << getAttrName() << "Attr::Convert" << type
         << "ToStr(Val)" << "<< \"\\\"\";\n";
    }

  public:
    VariadicEnumArgument(const Record &Arg, StringRef Attr)
      : VariadicArgument(Arg, Attr, Arg.getValueAsString("Type")),
        type(Arg.getValueAsString("Type")),
        values(Arg.getValueAsListOfStrings("Values")),
        enums(Arg.getValueAsListOfStrings("Enums")),
        uniques(uniqueEnumsInOrder(enums))
    {
      QualifiedTypeName = getAttrName().str() + "Attr::" + type;
      
      // FIXME: Emit a proper error
      assert(!uniques.empty());
    }

    bool isVariadicEnumArg() const override { return true; }
    
    void writeDeclarations(raw_ostream &OS) const override {
      auto i = uniques.cbegin(), e = uniques.cend();
      // The last one needs to not have a comma.
      --e;

      OS << "public:\n";
      OS << "  enum " << type << " {\n";
      for (; i != e; ++i)
        OS << "    " << *i << ",\n";
      OS << "    " << *e << "\n";
      OS << "  };\n";
      OS << "private:\n";
      
      VariadicArgument::writeDeclarations(OS);
    }

    void writeDump(raw_ostream &OS) const override {
      OS << "    for (" << getAttrName() << "Attr::" << getLowerName()
         << "_iterator I = SA->" << getLowerName() << "_begin(), E = SA->"
         << getLowerName() << "_end(); I != E; ++I) {\n";
      OS << "      switch(*I) {\n";
      for (const auto &UI : uniques) {
        OS << "    case " << getAttrName() << "Attr::" << UI << ":\n";
        OS << "      OS << \" " << UI << "\";\n";
        OS << "      break;\n";
      }
      OS << "      }\n";
      OS << "    }\n";
    }

    void writePCHReadDecls(raw_ostream &OS) const override {
      OS << "    unsigned " << getLowerName() << "Size = Record[Idx++];\n";
      OS << "    SmallVector<" << QualifiedTypeName << ", 4> " << getLowerName()
         << ";\n";
      OS << "    " << getLowerName() << ".reserve(" << getLowerName()
         << "Size);\n";
      OS << "    for (unsigned i = " << getLowerName() << "Size; i; --i)\n";
      OS << "      " << getLowerName() << ".push_back(" << "static_cast<"
         << QualifiedTypeName << ">(Record[Idx++]));\n";
    }

    void writePCHWrite(raw_ostream &OS) const override {
      OS << "    Record.push_back(SA->" << getLowerName() << "_size());\n";
      OS << "    for (" << getAttrName() << "Attr::" << getLowerName()
         << "_iterator i = SA->" << getLowerName() << "_begin(), e = SA->"
         << getLowerName() << "_end(); i != e; ++i)\n";
      OS << "      " << WritePCHRecord(QualifiedTypeName, "(*i)");
    }

    void writeConversion(raw_ostream &OS) const {
      OS << "  static bool ConvertStrTo" << type << "(StringRef Val, ";
      OS << type << " &Out) {\n";
      OS << "    Optional<" << type << "> R = llvm::StringSwitch<Optional<";
      OS << type << ">>(Val)\n";
      for (size_t I = 0; I < enums.size(); ++I) {
        OS << "      .Case(\"" << values[I] << "\", ";
        OS << getAttrName() << "Attr::" << enums[I] << ")\n";
      }
      OS << "      .Default(Optional<" << type << ">());\n";
      OS << "    if (R) {\n";
      OS << "      Out = *R;\n      return true;\n    }\n";
      OS << "    return false;\n";
      OS << "  }\n\n";

      OS << "  static const char *Convert" << type << "ToStr("
        << type << " Val) {\n"
        << "    switch(Val) {\n";
      std::set<std::string> Uniques;
      for (size_t I = 0; I < enums.size(); ++I) {
        if (Uniques.insert(enums[I]).second)
          OS << "    case " << getAttrName() << "Attr::" << enums[I]
          << ": return \"" << values[I] << "\";\n";
      }
      OS << "    }\n"
        << "    llvm_unreachable(\"No enumerator with that value\");\n"
        << "  }\n";
    }
  };

  class VersionArgument : public Argument {
  public:
    VersionArgument(const Record &Arg, StringRef Attr)
      : Argument(Arg, Attr)
    {}

    void writeAccessors(raw_ostream &OS) const override {
      OS << "  VersionTuple get" << getUpperName() << "() const {\n";
      OS << "    return " << getLowerName() << ";\n";
      OS << "  }\n";
      OS << "  void set" << getUpperName() 
         << "(ASTContext &C, VersionTuple V) {\n";
      OS << "    " << getLowerName() << " = V;\n";
      OS << "  }";
    }

    void writeCloneArgs(raw_ostream &OS) const override {
      OS << "get" << getUpperName() << "()";
    }

    void writeTemplateInstantiationArgs(raw_ostream &OS) const override {
      OS << "A->get" << getUpperName() << "()";
    }

    void writeCtorInitializers(raw_ostream &OS) const override {
      OS << getLowerName() << "(" << getUpperName() << ")";
    }

    void writeCtorDefaultInitializers(raw_ostream &OS) const override {
      OS << getLowerName() << "()";
    }

    void writeCtorParameters(raw_ostream &OS) const override {
      OS << "VersionTuple " << getUpperName();
    }

    void writeDeclarations(raw_ostream &OS) const override {
      OS << "VersionTuple " << getLowerName() << ";\n";
    }

    void writePCHReadDecls(raw_ostream &OS) const override {
      OS << "    VersionTuple " << getLowerName()
         << "= ReadVersionTuple(Record, Idx);\n";
    }

    void writePCHReadArgs(raw_ostream &OS) const override {
      OS << getLowerName();
    }

    void writePCHWrite(raw_ostream &OS) const override {
      OS << "    Record.AddVersionTuple(SA->get" << getUpperName() << "());\n";
    }

    void writeValue(raw_ostream &OS) const override {
      OS << getLowerName() << "=\" << get" << getUpperName() << "() << \"";
    }

    void writeDump(raw_ostream &OS) const override {
      OS << "    OS << \" \" << SA->get" << getUpperName() << "();\n";
    }
  };

  class ExprArgument : public SimpleArgument {
  public:
    ExprArgument(const Record &Arg, StringRef Attr)
      : SimpleArgument(Arg, Attr, "Expr *")
    {}

    void writeASTVisitorTraversal(raw_ostream &OS) const override {
      OS << "  if (!"
         << "getDerived().TraverseStmt(A->get" << getUpperName() << "()))\n";
      OS << "    return false;\n";
    }

    void writeTemplateInstantiationArgs(raw_ostream &OS) const override {
      OS << "tempInst" << getUpperName();
    }

    void writeTemplateInstantiation(raw_ostream &OS) const override {
      OS << "      " << getType() << " tempInst" << getUpperName() << ";\n";
      OS << "      {\n";
      OS << "        EnterExpressionEvaluationContext "
         << "Unevaluated(S, Sema::Unevaluated);\n";
      OS << "        ExprResult " << "Result = S.SubstExpr("
         << "A->get" << getUpperName() << "(), TemplateArgs);\n";
      OS << "        tempInst" << getUpperName() << " = "
         << "Result.getAs<Expr>();\n";
      OS << "      }\n";
    }

    void writeDump(raw_ostream &OS) const override {}

    void writeDumpChildren(raw_ostream &OS) const override {
      OS << "    dumpStmt(SA->get" << getUpperName() << "());\n";
    }

    void writeHasChildren(raw_ostream &OS) const override { OS << "true"; }
  };

  class VariadicExprArgument : public VariadicArgument {
  public:
    VariadicExprArgument(const Record &Arg, StringRef Attr)
      : VariadicArgument(Arg, Attr, "Expr *")
    {}

    void writeASTVisitorTraversal(raw_ostream &OS) const override {
      OS << "  {\n";
      OS << "    " << getType() << " *I = A->" << getLowerName()
         << "_begin();\n";
      OS << "    " << getType() << " *E = A->" << getLowerName()
         << "_end();\n";
      OS << "    for (; I != E; ++I) {\n";
      OS << "      if (!getDerived().TraverseStmt(*I))\n";
      OS << "        return false;\n";
      OS << "    }\n";
      OS << "  }\n";
    }

    void writeTemplateInstantiationArgs(raw_ostream &OS) const override {
      OS << "tempInst" << getUpperName() << ", "
         << "A->" << getLowerName() << "_size()";
    }

    void writeTemplateInstantiation(raw_ostream &OS) const override {
      OS << "      auto *tempInst" << getUpperName()
         << " = new (C, 16) " << getType()
         << "[A->" << getLowerName() << "_size()];\n";
      OS << "      {\n";
      OS << "        EnterExpressionEvaluationContext "
         << "Unevaluated(S, Sema::Unevaluated);\n";
      OS << "        " << getType() << " *TI = tempInst" << getUpperName()
         << ";\n";
      OS << "        " << getType() << " *I = A->" << getLowerName()
         << "_begin();\n";
      OS << "        " << getType() << " *E = A->" << getLowerName()
         << "_end();\n";
      OS << "        for (; I != E; ++I, ++TI) {\n";
      OS << "          ExprResult Result = S.SubstExpr(*I, TemplateArgs);\n";
      OS << "          *TI = Result.getAs<Expr>();\n";
      OS << "        }\n";
      OS << "      }\n";
    }

    void writeDump(raw_ostream &OS) const override {}

    void writeDumpChildren(raw_ostream &OS) const override {
      OS << "    for (" << getAttrName() << "Attr::" << getLowerName()
         << "_iterator I = SA->" << getLowerName() << "_begin(), E = SA->"
         << getLowerName() << "_end(); I != E; ++I)\n";
      OS << "      dumpStmt(*I);\n";
    }

    void writeHasChildren(raw_ostream &OS) const override {
      OS << "SA->" << getLowerName() << "_begin() != "
         << "SA->" << getLowerName() << "_end()";
    }
  };

  class VariadicStringArgument : public VariadicArgument {
  public:
    VariadicStringArgument(const Record &Arg, StringRef Attr)
      : VariadicArgument(Arg, Attr, "StringRef")
    {}

    void writeCtorBody(raw_ostream &OS) const override {
      OS << "    for (size_t I = 0, E = " << getArgSizeName() << "; I != E;\n"
            "         ++I) {\n"
            "      StringRef Ref = " << getUpperName() << "[I];\n"
            "      if (!Ref.empty()) {\n"
            "        char *Mem = new (Ctx, 1) char[Ref.size()];\n"
            "        std::memcpy(Mem, Ref.data(), Ref.size());\n"
            "        " << getArgName() << "[I] = StringRef(Mem, Ref.size());\n"
            "      }\n"
            "    }\n";
    }

    void writeValueImpl(raw_ostream &OS) const override {
      OS << "    OS << \"\\\"\" << Val << \"\\\"\";\n";
    }
  };

  class TypeArgument : public SimpleArgument {
  public:
    TypeArgument(const Record &Arg, StringRef Attr)
      : SimpleArgument(Arg, Attr, "TypeSourceInfo *")
    {}

    void writeAccessors(raw_ostream &OS) const override {
      OS << "  QualType get" << getUpperName() << "() const {\n";
      OS << "    return " << getLowerName() << "->getType();\n";
      OS << "  }";
      OS << "  " << getType() << " get" << getUpperName() << "Loc() const {\n";
      OS << "    return " << getLowerName() << ";\n";
      OS << "  }";
    }

    void writeTemplateInstantiationArgs(raw_ostream &OS) const override {
      OS << "A->get" << getUpperName() << "Loc()";
    }

    void writePCHWrite(raw_ostream &OS) const override {
      OS << "    " << WritePCHRecord(
          getType(), "SA->get" + std::string(getUpperName()) + "Loc()");
    }
  };

} // end anonymous namespace

static std::unique_ptr<Argument>
createArgument(const Record &Arg, StringRef Attr,
               const Record *Search = nullptr) {
  if (!Search)
    Search = &Arg;

  std::unique_ptr<Argument> Ptr;
  llvm::StringRef ArgName = Search->getName();

  if (ArgName == "AlignedArgument")
    Ptr = llvm::make_unique<AlignedArgument>(Arg, Attr);
  else if (ArgName == "EnumArgument")
    Ptr = llvm::make_unique<EnumArgument>(Arg, Attr);
  else if (ArgName == "ExprArgument")
    Ptr = llvm::make_unique<ExprArgument>(Arg, Attr);
  else if (ArgName == "FunctionArgument")
    Ptr = llvm::make_unique<SimpleArgument>(Arg, Attr, "FunctionDecl *");
  else if (ArgName == "IdentifierArgument")
    Ptr = llvm::make_unique<SimpleArgument>(Arg, Attr, "IdentifierInfo *");
  else if (ArgName == "DefaultBoolArgument")
    Ptr = llvm::make_unique<DefaultSimpleArgument>(
        Arg, Attr, "bool", Arg.getValueAsBit("Default"));
  else if (ArgName == "BoolArgument")
    Ptr = llvm::make_unique<SimpleArgument>(Arg, Attr, "bool");
  else if (ArgName == "DefaultIntArgument")
    Ptr = llvm::make_unique<DefaultSimpleArgument>(
        Arg, Attr, "int", Arg.getValueAsInt("Default"));
  else if (ArgName == "IntArgument")
    Ptr = llvm::make_unique<SimpleArgument>(Arg, Attr, "int");
  else if (ArgName == "StringArgument")
    Ptr = llvm::make_unique<StringArgument>(Arg, Attr);
  else if (ArgName == "TypeArgument")
    Ptr = llvm::make_unique<TypeArgument>(Arg, Attr);
  else if (ArgName == "UnsignedArgument")
    Ptr = llvm::make_unique<SimpleArgument>(Arg, Attr, "unsigned");
  else if (ArgName == "VariadicUnsignedArgument")
    Ptr = llvm::make_unique<VariadicArgument>(Arg, Attr, "unsigned");
  else if (ArgName == "VariadicStringArgument")
    Ptr = llvm::make_unique<VariadicStringArgument>(Arg, Attr);
  else if (ArgName == "VariadicEnumArgument")
    Ptr = llvm::make_unique<VariadicEnumArgument>(Arg, Attr);
  else if (ArgName == "VariadicExprArgument")
    Ptr = llvm::make_unique<VariadicExprArgument>(Arg, Attr);
  else if (ArgName == "VersionArgument")
    Ptr = llvm::make_unique<VersionArgument>(Arg, Attr);

  if (!Ptr) {
    // Search in reverse order so that the most-derived type is handled first.
    ArrayRef<std::pair<Record*, SMRange>> Bases = Search->getSuperClasses();
    for (const auto &Base : llvm::reverse(Bases)) {
      if ((Ptr = createArgument(Arg, Attr, Base.first)))
        break;
    }
  }

  if (Ptr && Arg.getValueAsBit("Optional"))
    Ptr->setOptional(true);

  if (Ptr && Arg.getValueAsBit("Fake"))
    Ptr->setFake(true);

  return Ptr;
}

static void writeAvailabilityValue(raw_ostream &OS) {
  OS << "\" << getPlatform()->getName();\n"
     << "  if (getStrict()) OS << \", strict\";\n"
     << "  if (!getIntroduced().empty()) OS << \", introduced=\" << getIntroduced();\n"
     << "  if (!getDeprecated().empty()) OS << \", deprecated=\" << getDeprecated();\n"
     << "  if (!getObsoleted().empty()) OS << \", obsoleted=\" << getObsoleted();\n"
     << "  if (getUnavailable()) OS << \", unavailable\";\n"
     << "  OS << \"";
}

static void writeDeprecatedAttrValue(raw_ostream &OS, std::string &Variety) {
  OS << "\\\"\" << getMessage() << \"\\\"\";\n";
  // Only GNU deprecated has an optional fixit argument at the second position.
  if (Variety == "GNU")
     OS << "    if (!getReplacement().empty()) OS << \", \\\"\""
           " << getReplacement() << \"\\\"\";\n";
  OS << "    OS << \"";
}

static void writeGetSpellingFunction(Record &R, raw_ostream &OS) {
  std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(R);

  OS << "const char *" << R.getName() << "Attr::getSpelling() const {\n";
  if (Spellings.empty()) {
    OS << "  return \"(No spelling)\";\n}\n\n";
    return;
  }

  OS << "  switch (SpellingListIndex) {\n"
        "  default:\n"
        "    llvm_unreachable(\"Unknown attribute spelling!\");\n"
        "    return \"(No spelling)\";\n";

  for (unsigned I = 0; I < Spellings.size(); ++I)
    OS << "  case " << I << ":\n"
          "    return \"" << Spellings[I].name() << "\";\n";
  // End of the switch statement.
  OS << "  }\n";
  // End of the getSpelling function.
  OS << "}\n\n";
}

static void
writePrettyPrintFunction(Record &R,
                         const std::vector<std::unique_ptr<Argument>> &Args,
                         raw_ostream &OS) {
  std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(R);

  OS << "void " << R.getName() << "Attr::printPretty("
    << "raw_ostream &OS, const PrintingPolicy &Policy) const {\n";

  if (Spellings.empty()) {
    OS << "}\n\n";
    return;
  }

  OS <<
    "  switch (SpellingListIndex) {\n"
    "  default:\n"
    "    llvm_unreachable(\"Unknown attribute spelling!\");\n"
    "    break;\n";

  for (unsigned I = 0; I < Spellings.size(); ++ I) {
    llvm::SmallString<16> Prefix;
    llvm::SmallString<8> Suffix;
    // The actual spelling of the name and namespace (if applicable)
    // of an attribute without considering prefix and suffix.
    llvm::SmallString<64> Spelling;
    std::string Name = Spellings[I].name();
    std::string Variety = Spellings[I].variety();

    if (Variety == "GNU") {
      Prefix = " __attribute__((";
      Suffix = "))";
    } else if (Variety == "CXX11") {
      Prefix = " [[";
      Suffix = "]]";
      std::string Namespace = Spellings[I].nameSpace();
      if (!Namespace.empty()) {
        Spelling += Namespace;
        Spelling += "::";
      }
    } else if (Variety == "Declspec") {
      Prefix = " __declspec(";
      Suffix = ")";
    } else if (Variety == "Microsoft") {
      Prefix = "[";
      Suffix = "]";
    } else if (Variety == "Keyword") {
      Prefix = " ";
      Suffix = "";
    } else if (Variety == "Pragma") {
      Prefix = "#pragma ";
      Suffix = "\n";
      std::string Namespace = Spellings[I].nameSpace();
      if (!Namespace.empty()) {
        Spelling += Namespace;
        Spelling += " ";
      }
    } else {
      llvm_unreachable("Unknown attribute syntax variety!");
    }

    Spelling += Name;

    OS <<
      "  case " << I << " : {\n"
      "    OS << \"" << Prefix << Spelling;

    if (Variety == "Pragma") {
      OS << " \";\n";
      OS << "    printPrettyPragma(OS, Policy);\n";
      OS << "    OS << \"\\n\";";
      OS << "    break;\n";
      OS << "  }\n";
      continue;
    }

    // Fake arguments aren't part of the parsed form and should not be
    // pretty-printed.
    bool hasNonFakeArgs = false;
    for (const auto &arg : Args) {
      if (arg->isFake()) continue;
      hasNonFakeArgs = true;
    }

    // FIXME: always printing the parenthesis isn't the correct behavior for
    // attributes which have optional arguments that were not provided. For
    // instance: __attribute__((aligned)) will be pretty printed as
    // __attribute__((aligned())). The logic should check whether there is only
    // a single argument, and if it is optional, whether it has been provided.
    if (hasNonFakeArgs)
      OS << "(";
    if (Spelling == "availability") {
      writeAvailabilityValue(OS);
    } else if (Spelling == "deprecated" || Spelling == "gnu::deprecated") {
        writeDeprecatedAttrValue(OS, Variety);
    } else {
      unsigned index = 0;
      for (const auto &arg : Args) {
        if (arg->isFake()) continue;
        if (index++) OS << ", ";
        arg->writeValue(OS);
      }
    }

    if (hasNonFakeArgs)
      OS << ")";
    OS << Suffix + "\";\n";

    OS <<
      "    break;\n"
      "  }\n";
  }

  // End of the switch statement.
  OS << "}\n";
  // End of the print function.
  OS << "}\n\n";
}

/// \brief Return the index of a spelling in a spelling list.
static unsigned
getSpellingListIndex(const std::vector<FlattenedSpelling> &SpellingList,
                     const FlattenedSpelling &Spelling) {
  assert(!SpellingList.empty() && "Spelling list is empty!");

  for (unsigned Index = 0; Index < SpellingList.size(); ++Index) {
    const FlattenedSpelling &S = SpellingList[Index];
    if (S.variety() != Spelling.variety())
      continue;
    if (S.nameSpace() != Spelling.nameSpace())
      continue;
    if (S.name() != Spelling.name())
      continue;

    return Index;
  }

  llvm_unreachable("Unknown spelling!");
}

static void writeAttrAccessorDefinition(const Record &R, raw_ostream &OS) {
  std::vector<Record*> Accessors = R.getValueAsListOfDefs("Accessors");
  for (const auto *Accessor : Accessors) {
    std::string Name = Accessor->getValueAsString("Name");
    std::vector<FlattenedSpelling> Spellings = 
      GetFlattenedSpellings(*Accessor);
    std::vector<FlattenedSpelling> SpellingList = GetFlattenedSpellings(R);
    assert(!SpellingList.empty() &&
           "Attribute with empty spelling list can't have accessors!");

    OS << "  bool " << Name << "() const { return SpellingListIndex == ";
    for (unsigned Index = 0; Index < Spellings.size(); ++Index) {
      OS << getSpellingListIndex(SpellingList, Spellings[Index]);
      if (Index != Spellings.size() -1)
        OS << " ||\n    SpellingListIndex == ";
      else
        OS << "; }\n";
    }
  }
}

static bool
SpellingNamesAreCommon(const std::vector<FlattenedSpelling>& Spellings) {
  assert(!Spellings.empty() && "An empty list of spellings was provided");
  std::string FirstName = NormalizeNameForSpellingComparison(
    Spellings.front().name());
  for (const auto &Spelling :
       llvm::make_range(std::next(Spellings.begin()), Spellings.end())) {
    std::string Name = NormalizeNameForSpellingComparison(Spelling.name());
    if (Name != FirstName)
      return false;
  }
  return true;
}

typedef std::map<unsigned, std::string> SemanticSpellingMap;
static std::string
CreateSemanticSpellings(const std::vector<FlattenedSpelling> &Spellings,
                        SemanticSpellingMap &Map) {
  // The enumerants are automatically generated based on the variety,
  // namespace (if present) and name for each attribute spelling. However,
  // care is taken to avoid trampling on the reserved namespace due to
  // underscores.
  std::string Ret("  enum Spelling {\n");
  std::set<std::string> Uniques;
  unsigned Idx = 0;
  for (auto I = Spellings.begin(), E = Spellings.end(); I != E; ++I, ++Idx) {
    const FlattenedSpelling &S = *I;
    const std::string &Variety = S.variety();
    const std::string &Spelling = S.name();
    const std::string &Namespace = S.nameSpace();
    std::string EnumName;

    EnumName += (Variety + "_");
    if (!Namespace.empty())
      EnumName += (NormalizeNameForSpellingComparison(Namespace).str() +
      "_");
    EnumName += NormalizeNameForSpellingComparison(Spelling);

    // Even if the name is not unique, this spelling index corresponds to a
    // particular enumerant name that we've calculated.
    Map[Idx] = EnumName;

    // Since we have been stripping underscores to avoid trampling on the
    // reserved namespace, we may have inadvertently created duplicate
    // enumerant names. These duplicates are not considered part of the
    // semantic spelling, and can be elided.
    if (Uniques.find(EnumName) != Uniques.end())
      continue;

    Uniques.insert(EnumName);
    if (I != Spellings.begin())
      Ret += ",\n";
    // Duplicate spellings are not considered part of the semantic spelling
    // enumeration, but the spelling index and semantic spelling values are
    // meant to be equivalent, so we must specify a concrete value for each
    // enumerator.
    Ret += "    " + EnumName + " = " + llvm::utostr(Idx);
  }
  Ret += "\n  };\n\n";
  return Ret;
}

void WriteSemanticSpellingSwitch(const std::string &VarName,
                                 const SemanticSpellingMap &Map,
                                 raw_ostream &OS) {
  OS << "  switch (" << VarName << ") {\n    default: "
    << "llvm_unreachable(\"Unknown spelling list index\");\n";
  for (const auto &I : Map)
    OS << "    case " << I.first << ": return " << I.second << ";\n";
  OS << "  }\n";
}

// Emits the LateParsed property for attributes.
static void emitClangAttrLateParsedList(RecordKeeper &Records, raw_ostream &OS) {
  OS << "#if defined(CLANG_ATTR_LATE_PARSED_LIST)\n";
  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr");

  for (const auto *Attr : Attrs) {
    bool LateParsed = Attr->getValueAsBit("LateParsed");

    if (LateParsed) {
      std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(*Attr);

      // FIXME: Handle non-GNU attributes
      for (const auto &I : Spellings) {
        if (I.variety() != "GNU")
          continue;
        OS << ".Case(\"" << I.name() << "\", " << LateParsed << ")\n";
      }
    }
  }
  OS << "#endif // CLANG_ATTR_LATE_PARSED_LIST\n\n";
}

/// \brief Emits the first-argument-is-type property for attributes.
static void emitClangAttrTypeArgList(RecordKeeper &Records, raw_ostream &OS) {
  OS << "#if defined(CLANG_ATTR_TYPE_ARG_LIST)\n";
  std::vector<Record *> Attrs = Records.getAllDerivedDefinitions("Attr");

  for (const auto *Attr : Attrs) {
    // Determine whether the first argument is a type.
    std::vector<Record *> Args = Attr->getValueAsListOfDefs("Args");
    if (Args.empty())
      continue;

    if (Args[0]->getSuperClasses().back().first->getName() != "TypeArgument")
      continue;

    // All these spellings take a single type argument.
    std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(*Attr);
    std::set<std::string> Emitted;
    for (const auto &S : Spellings) {
      if (Emitted.insert(S.name()).second)
        OS << ".Case(\"" << S.name() << "\", " << "true" << ")\n";
    }
  }
  OS << "#endif // CLANG_ATTR_TYPE_ARG_LIST\n\n";
}

/// \brief Emits the parse-arguments-in-unevaluated-context property for
/// attributes.
static void emitClangAttrArgContextList(RecordKeeper &Records, raw_ostream &OS) {
  OS << "#if defined(CLANG_ATTR_ARG_CONTEXT_LIST)\n";
  ParsedAttrMap Attrs = getParsedAttrList(Records);
  for (const auto &I : Attrs) {
    const Record &Attr = *I.second;

    if (!Attr.getValueAsBit("ParseArgumentsAsUnevaluated"))
      continue;

    // All these spellings take are parsed unevaluated.
    std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(Attr);
    std::set<std::string> Emitted;
    for (const auto &S : Spellings) {
      if (Emitted.insert(S.name()).second)
        OS << ".Case(\"" << S.name() << "\", " << "true" << ")\n";
    }
  }
  OS << "#endif // CLANG_ATTR_ARG_CONTEXT_LIST\n\n";
}

static bool isIdentifierArgument(Record *Arg) {
  return !Arg->getSuperClasses().empty() &&
    llvm::StringSwitch<bool>(Arg->getSuperClasses().back().first->getName())
    .Case("IdentifierArgument", true)
    .Case("EnumArgument", true)
    .Case("VariadicEnumArgument", true)
    .Default(false);
}

// Emits the first-argument-is-identifier property for attributes.
static void emitClangAttrIdentifierArgList(RecordKeeper &Records, raw_ostream &OS) {
  OS << "#if defined(CLANG_ATTR_IDENTIFIER_ARG_LIST)\n";
  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr");

  for (const auto *Attr : Attrs) {
    // Determine whether the first argument is an identifier.
    std::vector<Record *> Args = Attr->getValueAsListOfDefs("Args");
    if (Args.empty() || !isIdentifierArgument(Args[0]))
      continue;

    // All these spellings take an identifier argument.
    std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(*Attr);
    std::set<std::string> Emitted;
    for (const auto &S : Spellings) {
      if (Emitted.insert(S.name()).second)
        OS << ".Case(\"" << S.name() << "\", " << "true" << ")\n";
    }
  }
  OS << "#endif // CLANG_ATTR_IDENTIFIER_ARG_LIST\n\n";
}

namespace clang {

// Emits the class definitions for attributes.
void EmitClangAttrClass(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Attribute classes' definitions", OS);

  OS << "#ifndef LLVM_CLANG_ATTR_CLASSES_INC\n";
  OS << "#define LLVM_CLANG_ATTR_CLASSES_INC\n\n";

  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr");

  for (const auto *Attr : Attrs) {
    const Record &R = *Attr;

    // FIXME: Currently, documentation is generated as-needed due to the fact
    // that there is no way to allow a generated project "reach into" the docs
    // directory (for instance, it may be an out-of-tree build). However, we want
    // to ensure that every attribute has a Documentation field, and produce an
    // error if it has been neglected. Otherwise, the on-demand generation which
    // happens server-side will fail. This code is ensuring that functionality,
    // even though this Emitter doesn't technically need the documentation.
    // When attribute documentation can be generated as part of the build
    // itself, this code can be removed.
    (void)R.getValueAsListOfDefs("Documentation");
    
    if (!R.getValueAsBit("ASTNode"))
      continue;
    
    ArrayRef<std::pair<Record *, SMRange>> Supers = R.getSuperClasses();
    assert(!Supers.empty() && "Forgot to specify a superclass for the attr");
    std::string SuperName;
    for (const auto &Super : llvm::reverse(Supers)) {
      const Record *R = Super.first;
      if (R->getName() != "TargetSpecificAttr" && SuperName.empty())
        SuperName = R->getName();
    }

    OS << "class " << R.getName() << "Attr : public " << SuperName << " {\n";

    std::vector<Record*> ArgRecords = R.getValueAsListOfDefs("Args");
    std::vector<std::unique_ptr<Argument>> Args;
    Args.reserve(ArgRecords.size());

    bool HasOptArg = false;
    bool HasFakeArg = false;
    for (const auto *ArgRecord : ArgRecords) {
      Args.emplace_back(createArgument(*ArgRecord, R.getName()));
      Args.back()->writeDeclarations(OS);
      OS << "\n\n";

      // For these purposes, fake takes priority over optional.
      if (Args.back()->isFake()) {
        HasFakeArg = true;
      } else if (Args.back()->isOptional()) {
        HasOptArg = true;
      }
    }

    OS << "public:\n";

    std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(R);

    // If there are zero or one spellings, all spelling-related functionality
    // can be elided. If all of the spellings share the same name, the spelling
    // functionality can also be elided.
    bool ElideSpelling = (Spellings.size() <= 1) ||
                         SpellingNamesAreCommon(Spellings);

    // This maps spelling index values to semantic Spelling enumerants.
    SemanticSpellingMap SemanticToSyntacticMap;

    if (!ElideSpelling)
      OS << CreateSemanticSpellings(Spellings, SemanticToSyntacticMap);

    // Emit CreateImplicit factory methods.
    auto emitCreateImplicit = [&](bool emitFake) {
      OS << "  static " << R.getName() << "Attr *CreateImplicit(";
      OS << "ASTContext &Ctx";
      if (!ElideSpelling)
        OS << ", Spelling S";
      for (auto const &ai : Args) {
        if (ai->isFake() && !emitFake) continue;
        OS << ", ";
        ai->writeCtorParameters(OS);
      }
      OS << ", SourceRange Loc = SourceRange()";
      OS << ") {\n";
      OS << "    auto *A = new (Ctx) " << R.getName();
      OS << "Attr(Loc, Ctx, ";
      for (auto const &ai : Args) {
        if (ai->isFake() && !emitFake) continue;
        ai->writeImplicitCtorArgs(OS);
        OS << ", ";
      }
      OS << (ElideSpelling ? "0" : "S") << ");\n";
      OS << "    A->setImplicit(true);\n";
      OS << "    return A;\n  }\n\n";
    };

    // Emit a CreateImplicit that takes all the arguments.
    emitCreateImplicit(true);

    // Emit a CreateImplicit that takes all the non-fake arguments.
    if (HasFakeArg) {
      emitCreateImplicit(false);
    }

    // Emit constructors.
    auto emitCtor = [&](bool emitOpt, bool emitFake) {
      auto shouldEmitArg = [=](const std::unique_ptr<Argument> &arg) {
        if (arg->isFake()) return emitFake;
        if (arg->isOptional()) return emitOpt;
        return true;
      };

      OS << "  " << R.getName() << "Attr(SourceRange R, ASTContext &Ctx\n";
      for (auto const &ai : Args) {
        if (!shouldEmitArg(ai)) continue;
        OS << "              , ";
        ai->writeCtorParameters(OS);
        OS << "\n";
      }

      OS << "              , ";
      OS << "unsigned SI\n";

      OS << "             )\n";
      OS << "    : " << SuperName << "(attr::" << R.getName() << ", R, SI, "
         << ( R.getValueAsBit("LateParsed") ? "true" : "false" ) << ", "
         << ( R.getValueAsBit("DuplicatesAllowedWhileMerging") ? "true" : "false" ) << ")\n";

      for (auto const &ai : Args) {
        OS << "              , ";
        if (!shouldEmitArg(ai)) {
          ai->writeCtorDefaultInitializers(OS);
        } else {
          ai->writeCtorInitializers(OS);
        }
        OS << "\n";
      }

      OS << "  {\n";
  
      for (auto const &ai : Args) {
        if (!shouldEmitArg(ai)) continue;
        ai->writeCtorBody(OS);
      }
      OS << "  }\n\n";
    };

    // Emit a constructor that includes all the arguments.
    // This is necessary for cloning.
    emitCtor(true, true);

    // Emit a constructor that takes all the non-fake arguments.
    if (HasFakeArg) {
      emitCtor(true, false);
    }
 
    // Emit a constructor that takes all the non-fake, non-optional arguments.
    if (HasOptArg) {
      emitCtor(false, false);
    }

    OS << "  " << R.getName() << "Attr *clone(ASTContext &C) const;\n";
    OS << "  void printPretty(raw_ostream &OS,\n"
       << "                   const PrintingPolicy &Policy) const;\n";
    OS << "  const char *getSpelling() const;\n";
    
    if (!ElideSpelling) {
      assert(!SemanticToSyntacticMap.empty() && "Empty semantic mapping list");
      OS << "  Spelling getSemanticSpelling() const {\n";
      WriteSemanticSpellingSwitch("SpellingListIndex", SemanticToSyntacticMap,
                                  OS);
      OS << "  }\n";
    }

    writeAttrAccessorDefinition(R, OS);

    for (auto const &ai : Args) {
      ai->writeAccessors(OS);
      OS << "\n\n";

      // Don't write conversion routines for fake arguments.
      if (ai->isFake()) continue;

      if (ai->isEnumArg())
        static_cast<const EnumArgument *>(ai.get())->writeConversion(OS);
      else if (ai->isVariadicEnumArg())
        static_cast<const VariadicEnumArgument *>(ai.get())
            ->writeConversion(OS);
    }

    OS << R.getValueAsString("AdditionalMembers");
    OS << "\n\n";

    OS << "  static bool classof(const Attr *A) { return A->getKind() == "
       << "attr::" << R.getName() << "; }\n";

    OS << "};\n\n";
  }

  OS << "#endif // LLVM_CLANG_ATTR_CLASSES_INC\n";
}

// Emits the class method definitions for attributes.
void EmitClangAttrImpl(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Attribute classes' member function definitions", OS);

  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr");

  for (auto *Attr : Attrs) {
    Record &R = *Attr;
    
    if (!R.getValueAsBit("ASTNode"))
      continue;

    std::vector<Record*> ArgRecords = R.getValueAsListOfDefs("Args");
    std::vector<std::unique_ptr<Argument>> Args;
    for (const auto *Arg : ArgRecords)
      Args.emplace_back(createArgument(*Arg, R.getName()));

    for (auto const &ai : Args)
      ai->writeAccessorDefinitions(OS);

    OS << R.getName() << "Attr *" << R.getName()
       << "Attr::clone(ASTContext &C) const {\n";
    OS << "  auto *A = new (C) " << R.getName() << "Attr(getLocation(), C";
    for (auto const &ai : Args) {
      OS << ", ";
      ai->writeCloneArgs(OS);
    }
    OS << ", getSpellingListIndex());\n";
    OS << "  A->Inherited = Inherited;\n";
    OS << "  A->IsPackExpansion = IsPackExpansion;\n";
    OS << "  A->Implicit = Implicit;\n";
    OS << "  return A;\n}\n\n";

    writePrettyPrintFunction(R, Args, OS);
    writeGetSpellingFunction(R, OS);
  }

  // Instead of relying on virtual dispatch we just create a huge dispatch
  // switch. This is both smaller and faster than virtual functions.
  auto EmitFunc = [&](const char *Method) {
    OS << "  switch (getKind()) {\n";
    for (const auto *Attr : Attrs) {
      const Record &R = *Attr;
      if (!R.getValueAsBit("ASTNode"))
        continue;

      OS << "  case attr::" << R.getName() << ":\n";
      OS << "    return cast<" << R.getName() << "Attr>(this)->" << Method
         << ";\n";
    }
    OS << "  }\n";
    OS << "  llvm_unreachable(\"Unexpected attribute kind!\");\n";
    OS << "}\n\n";
  };

  OS << "const char *Attr::getSpelling() const {\n";
  EmitFunc("getSpelling()");

  OS << "Attr *Attr::clone(ASTContext &C) const {\n";
  EmitFunc("clone(C)");

  OS << "void Attr::printPretty(raw_ostream &OS, "
        "const PrintingPolicy &Policy) const {\n";
  EmitFunc("printPretty(OS, Policy)");
}

} // end namespace clang

static void emitAttrList(raw_ostream &OS, StringRef Class,
                         const std::vector<Record*> &AttrList) {
  for (auto Cur : AttrList) {
    OS << Class << "(" << Cur->getName() << ")\n";
  }
}

// Determines if an attribute has a Pragma spelling.
static bool AttrHasPragmaSpelling(const Record *R) {
  std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(*R);
  return std::find_if(Spellings.begin(), Spellings.end(),
                      [](const FlattenedSpelling &S) {
           return S.variety() == "Pragma";
         }) != Spellings.end();
}

namespace {

  struct AttrClassDescriptor {
    const char * const MacroName;
    const char * const TableGenName;
  };

} // end anonymous namespace

static const AttrClassDescriptor AttrClassDescriptors[] = {
  { "ATTR", "Attr" },
  { "STMT_ATTR", "StmtAttr" },
  { "INHERITABLE_ATTR", "InheritableAttr" },
  { "INHERITABLE_PARAM_ATTR", "InheritableParamAttr" },
  { "PARAMETER_ABI_ATTR", "ParameterABIAttr" }
};

static void emitDefaultDefine(raw_ostream &OS, StringRef name,
                              const char *superName) {
  OS << "#ifndef " << name << "\n";
  OS << "#define " << name << "(NAME) ";
  if (superName) OS << superName << "(NAME)";
  OS << "\n#endif\n\n";
}

namespace {

  /// A class of attributes.
  struct AttrClass {
    const AttrClassDescriptor &Descriptor;
    Record *TheRecord;
    AttrClass *SuperClass = nullptr;
    std::vector<AttrClass*> SubClasses;
    std::vector<Record*> Attrs;

    AttrClass(const AttrClassDescriptor &Descriptor, Record *R)
      : Descriptor(Descriptor), TheRecord(R) {}

    void emitDefaultDefines(raw_ostream &OS) const {
      // Default the macro unless this is a root class (i.e. Attr).
      if (SuperClass) {
        emitDefaultDefine(OS, Descriptor.MacroName,
                          SuperClass->Descriptor.MacroName);
      }
    }

    void emitUndefs(raw_ostream &OS) const {
      OS << "#undef " << Descriptor.MacroName << "\n";
    }

    void emitAttrList(raw_ostream &OS) const {
      for (auto SubClass : SubClasses) {
        SubClass->emitAttrList(OS);
      }

      ::emitAttrList(OS, Descriptor.MacroName, Attrs);
    }

    void classifyAttrOnRoot(Record *Attr) {
      bool result = classifyAttr(Attr);
      assert(result && "failed to classify on root"); (void) result;
    }

    void emitAttrRange(raw_ostream &OS) const {
      OS << "ATTR_RANGE(" << Descriptor.TableGenName
         << ", " << getFirstAttr()->getName()
         << ", " << getLastAttr()->getName() << ")\n";
    }

  private:
    bool classifyAttr(Record *Attr) {
      // Check all the subclasses.
      for (auto SubClass : SubClasses) {
        if (SubClass->classifyAttr(Attr))
          return true;
      }

      // It's not more specific than this class, but it might still belong here.
      if (Attr->isSubClassOf(TheRecord)) {
        Attrs.push_back(Attr);
        return true;
      }

      return false;
    }

    Record *getFirstAttr() const {
      if (!SubClasses.empty())
        return SubClasses.front()->getFirstAttr();
      return Attrs.front();
    }

    Record *getLastAttr() const {
      if (!Attrs.empty())
        return Attrs.back();
      return SubClasses.back()->getLastAttr();
    }
  };

  /// The entire hierarchy of attribute classes.
  class AttrClassHierarchy {
    std::vector<std::unique_ptr<AttrClass>> Classes;

  public:
    AttrClassHierarchy(RecordKeeper &Records) {
      // Find records for all the classes.
      for (auto &Descriptor : AttrClassDescriptors) {
        Record *ClassRecord = Records.getClass(Descriptor.TableGenName);
        AttrClass *Class = new AttrClass(Descriptor, ClassRecord);
        Classes.emplace_back(Class);
      }

      // Link up the hierarchy.
      for (auto &Class : Classes) {
        if (AttrClass *SuperClass = findSuperClass(Class->TheRecord)) {
          Class->SuperClass = SuperClass;
          SuperClass->SubClasses.push_back(Class.get());
        }
      }

#ifndef NDEBUG
      for (auto i = Classes.begin(), e = Classes.end(); i != e; ++i) {
        assert((i == Classes.begin()) == ((*i)->SuperClass == nullptr) &&
               "only the first class should be a root class!");
      }
#endif
    }

    void emitDefaultDefines(raw_ostream &OS) const {
      for (auto &Class : Classes) {
        Class->emitDefaultDefines(OS);
      }
    }

    void emitUndefs(raw_ostream &OS) const {
      for (auto &Class : Classes) {
        Class->emitUndefs(OS);
      }
    }

    void emitAttrLists(raw_ostream &OS) const {
      // Just start from the root class.
      Classes[0]->emitAttrList(OS);
    }

    void emitAttrRanges(raw_ostream &OS) const {
      for (auto &Class : Classes)
        Class->emitAttrRange(OS);
    }

    void classifyAttr(Record *Attr) {
      // Add the attribute to the root class.
      Classes[0]->classifyAttrOnRoot(Attr);
    }

  private:
    AttrClass *findClassByRecord(Record *R) const {
      for (auto &Class : Classes) {
        if (Class->TheRecord == R)
          return Class.get();
      }
      return nullptr;
    }

    AttrClass *findSuperClass(Record *R) const {
      // TableGen flattens the superclass list, so we just need to walk it
      // in reverse.
      auto SuperClasses = R->getSuperClasses();
      for (signed i = 0, e = SuperClasses.size(); i != e; ++i) {
        auto SuperClass = findClassByRecord(SuperClasses[e - i - 1].first);
        if (SuperClass) return SuperClass;
      }
      return nullptr;
    }
  };

} // end anonymous namespace

namespace clang {

// Emits the enumeration list for attributes.
void EmitClangAttrList(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("List of all attributes that Clang recognizes", OS);

  AttrClassHierarchy Hierarchy(Records);

  // Add defaulting macro definitions.
  Hierarchy.emitDefaultDefines(OS);
  emitDefaultDefine(OS, "PRAGMA_SPELLING_ATTR", nullptr);

  std::vector<Record *> Attrs = Records.getAllDerivedDefinitions("Attr");
  std::vector<Record *> PragmaAttrs;
  for (auto *Attr : Attrs) {
    if (!Attr->getValueAsBit("ASTNode"))
      continue;

    // Add the attribute to the ad-hoc groups.
    if (AttrHasPragmaSpelling(Attr))
      PragmaAttrs.push_back(Attr);

    // Place it in the hierarchy.
    Hierarchy.classifyAttr(Attr);
  }

  // Emit the main attribute list.
  Hierarchy.emitAttrLists(OS);

  // Emit the ad hoc groups.
  emitAttrList(OS, "PRAGMA_SPELLING_ATTR", PragmaAttrs);

  // Emit the attribute ranges.
  OS << "#ifdef ATTR_RANGE\n";
  Hierarchy.emitAttrRanges(OS);
  OS << "#undef ATTR_RANGE\n";
  OS << "#endif\n";

  Hierarchy.emitUndefs(OS);
  OS << "#undef PRAGMA_SPELLING_ATTR\n";
}

// Emits the code to read an attribute from a precompiled header.
void EmitClangAttrPCHRead(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Attribute deserialization code", OS);

  Record *InhClass = Records.getClass("InheritableAttr");
  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr"),
                       ArgRecords;
  std::vector<std::unique_ptr<Argument>> Args;

  OS << "  switch (Kind) {\n";
  for (const auto *Attr : Attrs) {
    const Record &R = *Attr;
    if (!R.getValueAsBit("ASTNode"))
      continue;
    
    OS << "  case attr::" << R.getName() << ": {\n";
    if (R.isSubClassOf(InhClass))
      OS << "    bool isInherited = Record[Idx++];\n";
    OS << "    bool isImplicit = Record[Idx++];\n";
    OS << "    unsigned Spelling = Record[Idx++];\n";
    ArgRecords = R.getValueAsListOfDefs("Args");
    Args.clear();
    for (const auto *Arg : ArgRecords) {
      Args.emplace_back(createArgument(*Arg, R.getName()));
      Args.back()->writePCHReadDecls(OS);
    }
    OS << "    New = new (Context) " << R.getName() << "Attr(Range, Context";
    for (auto const &ri : Args) {
      OS << ", ";
      ri->writePCHReadArgs(OS);
    }
    OS << ", Spelling);\n";
    if (R.isSubClassOf(InhClass))
      OS << "    cast<InheritableAttr>(New)->setInherited(isInherited);\n";
    OS << "    New->setImplicit(isImplicit);\n";
    OS << "    break;\n";
    OS << "  }\n";
  }
  OS << "  }\n";
}

// Emits the code to write an attribute to a precompiled header.
void EmitClangAttrPCHWrite(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Attribute serialization code", OS);

  Record *InhClass = Records.getClass("InheritableAttr");
  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr"), Args;

  OS << "  switch (A->getKind()) {\n";
  for (const auto *Attr : Attrs) {
    const Record &R = *Attr;
    if (!R.getValueAsBit("ASTNode"))
      continue;
    OS << "  case attr::" << R.getName() << ": {\n";
    Args = R.getValueAsListOfDefs("Args");
    if (R.isSubClassOf(InhClass) || !Args.empty())
      OS << "    const auto *SA = cast<" << R.getName()
         << "Attr>(A);\n";
    if (R.isSubClassOf(InhClass))
      OS << "    Record.push_back(SA->isInherited());\n";
    OS << "    Record.push_back(A->isImplicit());\n";
    OS << "    Record.push_back(A->getSpellingListIndex());\n";

    for (const auto *Arg : Args)
      createArgument(*Arg, R.getName())->writePCHWrite(OS);
    OS << "    break;\n";
    OS << "  }\n";
  }
  OS << "  }\n";
}

// Generate a conditional expression to check if the current target satisfies
// the conditions for a TargetSpecificAttr record, and append the code for
// those checks to the Test string. If the FnName string pointer is non-null,
// append a unique suffix to distinguish this set of target checks from other
// TargetSpecificAttr records.
static void GenerateTargetSpecificAttrChecks(const Record *R,
                                             std::vector<std::string> &Arches,
                                             std::string &Test,
                                             std::string *FnName) {
  // It is assumed that there will be an llvm::Triple object
  // named "T" and a TargetInfo object named "Target" within
  // scope that can be used to determine whether the attribute exists in
  // a given target.
  Test += "(";

  for (auto I = Arches.begin(), E = Arches.end(); I != E; ++I) {
    std::string Part = *I;
    Test += "T.getArch() == llvm::Triple::" + Part;
    if (I + 1 != E)
      Test += " || ";
    if (FnName)
      *FnName += Part;
  }
  Test += ")";

  // If the attribute is specific to particular OSes, check those.
  if (!R->isValueUnset("OSes")) {
    // We know that there was at least one arch test, so we need to and in the
    // OS tests.
    Test += " && (";
    std::vector<std::string> OSes = R->getValueAsListOfStrings("OSes");
    for (auto I = OSes.begin(), E = OSes.end(); I != E; ++I) {
      std::string Part = *I;

      Test += "T.getOS() == llvm::Triple::" + Part;
      if (I + 1 != E)
        Test += " || ";
      if (FnName)
        *FnName += Part;
    }
    Test += ")";
  }

  // If one or more CXX ABIs are specified, check those as well.
  if (!R->isValueUnset("CXXABIs")) {
    Test += " && (";
    std::vector<std::string> CXXABIs = R->getValueAsListOfStrings("CXXABIs");
    for (auto I = CXXABIs.begin(), E = CXXABIs.end(); I != E; ++I) {
      std::string Part = *I;
      Test += "Target.getCXXABI().getKind() == TargetCXXABI::" + Part;
      if (I + 1 != E)
        Test += " || ";
      if (FnName)
        *FnName += Part;
    }
    Test += ")";
  }
}

static void GenerateHasAttrSpellingStringSwitch(
    const std::vector<Record *> &Attrs, raw_ostream &OS,
    const std::string &Variety = "", const std::string &Scope = "") {
  for (const auto *Attr : Attrs) {
    // C++11-style attributes have specific version information associated with
    // them. If the attribute has no scope, the version information must not
    // have the default value (1), as that's incorrect. Instead, the unscoped
    // attribute version information should be taken from the SD-6 standing
    // document, which can be found at: 
    // https://isocpp.org/std/standing-documents/sd-6-sg10-feature-test-recommendations
    int Version = 1;

    if (Variety == "CXX11") {
        std::vector<Record *> Spellings = Attr->getValueAsListOfDefs("Spellings");
        for (const auto &Spelling : Spellings) {
          if (Spelling->getValueAsString("Variety") == "CXX11") {
            Version = static_cast<int>(Spelling->getValueAsInt("Version"));
            if (Scope.empty() && Version == 1)
              PrintError(Spelling->getLoc(), "C++ standard attributes must "
              "have valid version information.");
            break;
          }
      }
    }

    std::string Test;
    if (Attr->isSubClassOf("TargetSpecificAttr")) {
      const Record *R = Attr->getValueAsDef("Target");
      std::vector<std::string> Arches = R->getValueAsListOfStrings("Arches");
      GenerateTargetSpecificAttrChecks(R, Arches, Test, nullptr);

      // If this is the C++11 variety, also add in the LangOpts test.
      if (Variety == "CXX11")
        Test += " && LangOpts.CPlusPlus11";
    } else if (Variety == "CXX11")
      // C++11 mode should be checked against LangOpts, which is presumed to be
      // present in the caller.
      Test = "LangOpts.CPlusPlus11";

    std::string TestStr =
        !Test.empty() ? Test + " ? " + llvm::itostr(Version) + " : 0" : "1";
    std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(*Attr);
    for (const auto &S : Spellings)
      if (Variety.empty() || (Variety == S.variety() &&
                              (Scope.empty() || Scope == S.nameSpace())))
        OS << "    .Case(\"" << S.name() << "\", " << TestStr << ")\n";
  }
  OS << "    .Default(0);\n";
}

// Emits the list of spellings for attributes.
void EmitClangAttrHasAttrImpl(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Code to implement the __has_attribute logic", OS);

  // Separate all of the attributes out into four group: generic, C++11, GNU,
  // and declspecs. Then generate a big switch statement for each of them.
  std::vector<Record *> Attrs = Records.getAllDerivedDefinitions("Attr");
  std::vector<Record *> Declspec, Microsoft, GNU, Pragma;
  std::map<std::string, std::vector<Record *>> CXX;

  // Walk over the list of all attributes, and split them out based on the
  // spelling variety.
  for (auto *R : Attrs) {
    std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(*R);
    for (const auto &SI : Spellings) {
      const std::string &Variety = SI.variety();
      if (Variety == "GNU")
        GNU.push_back(R);
      else if (Variety == "Declspec")
        Declspec.push_back(R);
      else if (Variety == "Microsoft")
        Microsoft.push_back(R);
      else if (Variety == "CXX11")
        CXX[SI.nameSpace()].push_back(R);
      else if (Variety == "Pragma")
        Pragma.push_back(R);
    }
  }

  OS << "const llvm::Triple &T = Target.getTriple();\n";
  OS << "switch (Syntax) {\n";
  OS << "case AttrSyntax::GNU:\n";
  OS << "  return llvm::StringSwitch<int>(Name)\n";
  GenerateHasAttrSpellingStringSwitch(GNU, OS, "GNU");
  OS << "case AttrSyntax::Declspec:\n";
  OS << "  return llvm::StringSwitch<int>(Name)\n";
  GenerateHasAttrSpellingStringSwitch(Declspec, OS, "Declspec");
  OS << "case AttrSyntax::Microsoft:\n";
  OS << "  return llvm::StringSwitch<int>(Name)\n";
  GenerateHasAttrSpellingStringSwitch(Microsoft, OS, "Microsoft");
  OS << "case AttrSyntax::Pragma:\n";
  OS << "  return llvm::StringSwitch<int>(Name)\n";
  GenerateHasAttrSpellingStringSwitch(Pragma, OS, "Pragma");
  OS << "case AttrSyntax::CXX: {\n";
  // C++11-style attributes are further split out based on the Scope.
  for (auto I = CXX.cbegin(), E = CXX.cend(); I != E; ++I) {
    if (I != CXX.begin())
      OS << " else ";
    if (I->first.empty())
      OS << "if (!Scope || Scope->getName() == \"\") {\n";
    else
      OS << "if (Scope->getName() == \"" << I->first << "\") {\n";
    OS << "  return llvm::StringSwitch<int>(Name)\n";
    GenerateHasAttrSpellingStringSwitch(I->second, OS, "CXX11", I->first);
    OS << "}";
  }
  OS << "\n}\n";
  OS << "}\n";
}

void EmitClangAttrSpellingListIndex(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Code to translate different attribute spellings "
                       "into internal identifiers", OS);

  OS << "  switch (AttrKind) {\n";

  ParsedAttrMap Attrs = getParsedAttrList(Records);
  for (const auto &I : Attrs) {
    const Record &R = *I.second;
    std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(R);
    OS << "  case AT_" << I.first << ": {\n";
    for (unsigned I = 0; I < Spellings.size(); ++ I) {
      OS << "    if (Name == \"" << Spellings[I].name() << "\" && "
         << "SyntaxUsed == "
         << StringSwitch<unsigned>(Spellings[I].variety())
                .Case("GNU", 0)
                .Case("CXX11", 1)
                .Case("Declspec", 2)
                .Case("Microsoft", 3)
                .Case("Keyword", 4)
                .Case("Pragma", 5)
                .Default(0)
         << " && Scope == \"" << Spellings[I].nameSpace() << "\")\n"
         << "        return " << I << ";\n";
    }

    OS << "    break;\n";
    OS << "  }\n";
  }

  OS << "  }\n";
  OS << "  return 0;\n";
}

// Emits code used by RecursiveASTVisitor to visit attributes
void EmitClangAttrASTVisitor(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Used by RecursiveASTVisitor to visit attributes.", OS);

  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr");

  // Write method declarations for Traverse* methods.
  // We emit this here because we only generate methods for attributes that
  // are declared as ASTNodes.
  OS << "#ifdef ATTR_VISITOR_DECLS_ONLY\n\n";
  for (const auto *Attr : Attrs) {
    const Record &R = *Attr;
    if (!R.getValueAsBit("ASTNode"))
      continue;
    OS << "  bool Traverse"
       << R.getName() << "Attr(" << R.getName() << "Attr *A);\n";
    OS << "  bool Visit"
       << R.getName() << "Attr(" << R.getName() << "Attr *A) {\n"
       << "    return true; \n"
       << "  }\n";
  }
  OS << "\n#else // ATTR_VISITOR_DECLS_ONLY\n\n";

  // Write individual Traverse* methods for each attribute class.
  for (const auto *Attr : Attrs) {
    const Record &R = *Attr;
    if (!R.getValueAsBit("ASTNode"))
      continue;

    OS << "template <typename Derived>\n"
       << "bool VISITORCLASS<Derived>::Traverse"
       << R.getName() << "Attr(" << R.getName() << "Attr *A) {\n"
       << "  if (!getDerived().VisitAttr(A))\n"
       << "    return false;\n"
       << "  if (!getDerived().Visit" << R.getName() << "Attr(A))\n"
       << "    return false;\n";

    std::vector<Record*> ArgRecords = R.getValueAsListOfDefs("Args");
    for (const auto *Arg : ArgRecords)
      createArgument(*Arg, R.getName())->writeASTVisitorTraversal(OS);

    OS << "  return true;\n";
    OS << "}\n\n";
  }

  // Write generic Traverse routine
  OS << "template <typename Derived>\n"
     << "bool VISITORCLASS<Derived>::TraverseAttr(Attr *A) {\n"
     << "  if (!A)\n"
     << "    return true;\n"
     << "\n"
     << "  switch (A->getKind()) {\n";

  for (const auto *Attr : Attrs) {
    const Record &R = *Attr;
    if (!R.getValueAsBit("ASTNode"))
      continue;

    OS << "    case attr::" << R.getName() << ":\n"
       << "      return getDerived().Traverse" << R.getName() << "Attr("
       << "cast<" << R.getName() << "Attr>(A));\n";
  }
  OS << "  }\n";  // end switch
  OS << "  llvm_unreachable(\"bad attribute kind\");\n";
  OS << "}\n";  // end function
  OS << "#endif  // ATTR_VISITOR_DECLS_ONLY\n";
}

// Emits code to instantiate dependent attributes on templates.
void EmitClangAttrTemplateInstantiate(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Template instantiation code for attributes", OS);

  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr");

  OS << "namespace clang {\n"
     << "namespace sema {\n\n"
     << "Attr *instantiateTemplateAttribute(const Attr *At, ASTContext &C, "
     << "Sema &S,\n"
     << "        const MultiLevelTemplateArgumentList &TemplateArgs) {\n"
     << "  switch (At->getKind()) {\n";

  for (const auto *Attr : Attrs) {
    const Record &R = *Attr;
    if (!R.getValueAsBit("ASTNode"))
      continue;

    OS << "    case attr::" << R.getName() << ": {\n";
    bool ShouldClone = R.getValueAsBit("Clone");

    if (!ShouldClone) {
      OS << "      return nullptr;\n";
      OS << "    }\n";
      continue;
    }

    OS << "      const auto *A = cast<"
       << R.getName() << "Attr>(At);\n";
    bool TDependent = R.getValueAsBit("TemplateDependent");

    if (!TDependent) {
      OS << "      return A->clone(C);\n";
      OS << "    }\n";
      continue;
    }

    std::vector<Record*> ArgRecords = R.getValueAsListOfDefs("Args");
    std::vector<std::unique_ptr<Argument>> Args;
    Args.reserve(ArgRecords.size());

    for (const auto *ArgRecord : ArgRecords)
      Args.emplace_back(createArgument(*ArgRecord, R.getName()));

    for (auto const &ai : Args)
      ai->writeTemplateInstantiation(OS);

    OS << "      return new (C) " << R.getName() << "Attr(A->getLocation(), C";
    for (auto const &ai : Args) {
      OS << ", ";
      ai->writeTemplateInstantiationArgs(OS);
    }
    OS << ", A->getSpellingListIndex());\n    }\n";
  }
  OS << "  } // end switch\n"
     << "  llvm_unreachable(\"Unknown attribute!\");\n"
     << "  return nullptr;\n"
     << "}\n\n"
     << "} // end namespace sema\n"
     << "} // end namespace clang\n";
}

// Emits the list of parsed attributes.
void EmitClangAttrParsedAttrList(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("List of all attributes that Clang recognizes", OS);

  OS << "#ifndef PARSED_ATTR\n";
  OS << "#define PARSED_ATTR(NAME) NAME\n";
  OS << "#endif\n\n";
  
  ParsedAttrMap Names = getParsedAttrList(Records);
  for (const auto &I : Names) {
    OS << "PARSED_ATTR(" << I.first << ")\n";
  }
}

static bool isArgVariadic(const Record &R, StringRef AttrName) {
  return createArgument(R, AttrName)->isVariadic();
}

static void emitArgInfo(const Record &R, std::stringstream &OS) {
  // This function will count the number of arguments specified for the
  // attribute and emit the number of required arguments followed by the
  // number of optional arguments.
  std::vector<Record *> Args = R.getValueAsListOfDefs("Args");
  unsigned ArgCount = 0, OptCount = 0;
  bool HasVariadic = false;
  for (const auto *Arg : Args) {
    Arg->getValueAsBit("Optional") ? ++OptCount : ++ArgCount;
    if (!HasVariadic && isArgVariadic(*Arg, R.getName()))
      HasVariadic = true;
  }

  // If there is a variadic argument, we will set the optional argument count
  // to its largest value. Since it's currently a 4-bit number, we set it to 15.
  OS << ArgCount << ", " << (HasVariadic ? 15 : OptCount);
}

static void GenerateDefaultAppertainsTo(raw_ostream &OS) {
  OS << "static bool defaultAppertainsTo(Sema &, const AttributeList &,";
  OS << "const Decl *) {\n";
  OS << "  return true;\n";
  OS << "}\n\n";
}

static std::string CalculateDiagnostic(const Record &S) {
  // If the SubjectList object has a custom diagnostic associated with it,
  // return that directly.
  std::string CustomDiag = S.getValueAsString("CustomDiag");
  if (!CustomDiag.empty())
    return CustomDiag;

  // Given the list of subjects, determine what diagnostic best fits.
  enum {
    Func = 1U << 0,
    Var = 1U << 1,
    ObjCMethod = 1U << 2,
    Param = 1U << 3,
    Class = 1U << 4,
    GenericRecord = 1U << 5,
    Type = 1U << 6,
    ObjCIVar = 1U << 7,
    ObjCProp = 1U << 8,
    ObjCInterface = 1U << 9,
    Block = 1U << 10,
    Namespace = 1U << 11,
    Field = 1U << 12,
    CXXMethod = 1U << 13,
    ObjCProtocol = 1U << 14,
    Enum = 1U << 15
  };
  uint32_t SubMask = 0;

  std::vector<Record *> Subjects = S.getValueAsListOfDefs("Subjects");
  for (const auto *Subject : Subjects) {
    const Record &R = *Subject;
    std::string Name;

    if (R.isSubClassOf("SubsetSubject")) {
      PrintError(R.getLoc(), "SubsetSubjects should use a custom diagnostic");
      // As a fallback, look through the SubsetSubject to see what its base
      // type is, and use that. This needs to be updated if SubsetSubjects
      // are allowed within other SubsetSubjects.
      Name = R.getValueAsDef("Base")->getName();
    } else
      Name = R.getName();

    uint32_t V = StringSwitch<uint32_t>(Name)
                   .Case("Function", Func)
                   .Case("Var", Var)
                   .Case("ObjCMethod", ObjCMethod)
                   .Case("ParmVar", Param)
                   .Case("TypedefName", Type)
                   .Case("ObjCIvar", ObjCIVar)
                   .Case("ObjCProperty", ObjCProp)
                   .Case("Record", GenericRecord)
                   .Case("ObjCInterface", ObjCInterface)
                   .Case("ObjCProtocol", ObjCProtocol)
                   .Case("Block", Block)
                   .Case("CXXRecord", Class)
                   .Case("Namespace", Namespace)
                   .Case("Field", Field)
                   .Case("CXXMethod", CXXMethod)
                   .Case("Enum", Enum)
                   .Default(0);
    if (!V) {
      // Something wasn't in our mapping, so be helpful and let the developer
      // know about it.
      PrintFatalError(R.getLoc(), "Unknown subject type: " + R.getName());
      return "";
    }

    SubMask |= V;
  }

  switch (SubMask) {
    // For the simple cases where there's only a single entry in the mask, we
    // don't have to resort to bit fiddling.
    case Func:  return "ExpectedFunction";
    case Var:   return "ExpectedVariable";
    case Param: return "ExpectedParameter";
    case Class: return "ExpectedClass";
    case Enum:  return "ExpectedEnum";
    case CXXMethod:
      // FIXME: Currently, this maps to ExpectedMethod based on existing code,
      // but should map to something a bit more accurate at some point.
    case ObjCMethod:  return "ExpectedMethod";
    case Type:  return "ExpectedType";
    case ObjCInterface: return "ExpectedObjectiveCInterface";
    case ObjCProtocol: return "ExpectedObjectiveCProtocol";
    
    // "GenericRecord" means struct, union or class; check the language options
    // and if not compiling for C++, strip off the class part. Note that this
    // relies on the fact that the context for this declares "Sema &S".
    case GenericRecord:
      return "(S.getLangOpts().CPlusPlus ? ExpectedStructOrUnionOrClass : "
                                           "ExpectedStructOrUnion)";
    case Func | ObjCMethod | Block: return "ExpectedFunctionMethodOrBlock";
    case Func | ObjCMethod | Class: return "ExpectedFunctionMethodOrClass";
    case Func | Param:
    case Func | ObjCMethod | Param: return "ExpectedFunctionMethodOrParameter";
    case Func | ObjCMethod: return "ExpectedFunctionOrMethod";
    case Func | Var: return "ExpectedVariableOrFunction";

    // If not compiling for C++, the class portion does not apply.
    case Func | Var | Class:
      return "(S.getLangOpts().CPlusPlus ? ExpectedFunctionVariableOrClass : "
                                           "ExpectedVariableOrFunction)";

    case Func | Var | Class | ObjCInterface:
      return "(S.getLangOpts().CPlusPlus"
             "     ? ((S.getLangOpts().ObjC1 || S.getLangOpts().ObjC2)"
             "            ? ExpectedFunctionVariableClassOrObjCInterface"
             "            : ExpectedFunctionVariableOrClass)"
             "     : ((S.getLangOpts().ObjC1 || S.getLangOpts().ObjC2)"
             "            ? ExpectedFunctionVariableOrObjCInterface"
             "            : ExpectedVariableOrFunction))";

    case ObjCMethod | ObjCProp: return "ExpectedMethodOrProperty";
    case ObjCProtocol | ObjCInterface:
      return "ExpectedObjectiveCInterfaceOrProtocol";
    case Field | Var: return "ExpectedFieldOrGlobalVar";
  }

  PrintFatalError(S.getLoc(),
                  "Could not deduce diagnostic argument for Attr subjects");

  return "";
}

static std::string GetSubjectWithSuffix(const Record *R) {
  std::string B = R->getName();
  if (B == "DeclBase")
    return "Decl";
  return B + "Decl";
}

static std::string GenerateCustomAppertainsTo(const Record &Subject,
                                              raw_ostream &OS) {
  std::string FnName = "is" + Subject.getName();

  // If this code has already been generated, simply return the previous
  // instance of it.
  static std::set<std::string> CustomSubjectSet;
  auto I = CustomSubjectSet.find(FnName);
  if (I != CustomSubjectSet.end())
    return *I;

  Record *Base = Subject.getValueAsDef("Base");

  // Not currently support custom subjects within custom subjects.
  if (Base->isSubClassOf("SubsetSubject")) {
    PrintFatalError(Subject.getLoc(),
                    "SubsetSubjects within SubsetSubjects is not supported");
    return "";
  }

  OS << "static bool " << FnName << "(const Decl *D) {\n";
  OS << "  if (const auto *S = dyn_cast<";
  OS << GetSubjectWithSuffix(Base);
  OS << ">(D))\n";
  OS << "    return " << Subject.getValueAsString("CheckCode") << ";\n";
  OS << "  return false;\n";
  OS << "}\n\n";

  CustomSubjectSet.insert(FnName);
  return FnName;
}

static std::string GenerateAppertainsTo(const Record &Attr, raw_ostream &OS) {
  // If the attribute does not contain a Subjects definition, then use the
  // default appertainsTo logic.
  if (Attr.isValueUnset("Subjects"))
    return "defaultAppertainsTo";

  const Record *SubjectObj = Attr.getValueAsDef("Subjects");
  std::vector<Record*> Subjects = SubjectObj->getValueAsListOfDefs("Subjects");

  // If the list of subjects is empty, it is assumed that the attribute
  // appertains to everything.
  if (Subjects.empty())
    return "defaultAppertainsTo";

  bool Warn = SubjectObj->getValueAsDef("Diag")->getValueAsBit("Warn");

  // Otherwise, generate an appertainsTo check specific to this attribute which
  // checks all of the given subjects against the Decl passed in. Return the
  // name of that check to the caller.
  std::string FnName = "check" + Attr.getName() + "AppertainsTo";
  std::stringstream SS;
  SS << "static bool " << FnName << "(Sema &S, const AttributeList &Attr, ";
  SS << "const Decl *D) {\n";
  SS << "  if (";
  for (auto I = Subjects.begin(), E = Subjects.end(); I != E; ++I) {
    // If the subject has custom code associated with it, generate a function
    // for it. The function cannot be inlined into this check (yet) because it
    // requires the subject to be of a specific type, and were that information
    // inlined here, it would not support an attribute with multiple custom
    // subjects.
    if ((*I)->isSubClassOf("SubsetSubject")) {
      SS << "!" << GenerateCustomAppertainsTo(**I, OS) << "(D)";
    } else {
      SS << "!isa<" << GetSubjectWithSuffix(*I) << ">(D)";
    }

    if (I + 1 != E)
      SS << " && ";
  }
  SS << ") {\n";
  SS << "    S.Diag(Attr.getLoc(), diag::";
  SS << (Warn ? "warn_attribute_wrong_decl_type" :
               "err_attribute_wrong_decl_type");
  SS << ")\n";
  SS << "      << Attr.getName() << ";
  SS << CalculateDiagnostic(*SubjectObj) << ";\n";
  SS << "    return false;\n";
  SS << "  }\n";
  SS << "  return true;\n";
  SS << "}\n\n";

  OS << SS.str();
  return FnName;
}

static void GenerateDefaultLangOptRequirements(raw_ostream &OS) {
  OS << "static bool defaultDiagnoseLangOpts(Sema &, ";
  OS << "const AttributeList &) {\n";
  OS << "  return true;\n";
  OS << "}\n\n";
}

static std::string GenerateLangOptRequirements(const Record &R,
                                               raw_ostream &OS) {
  // If the attribute has an empty or unset list of language requirements,
  // return the default handler.
  std::vector<Record *> LangOpts = R.getValueAsListOfDefs("LangOpts");
  if (LangOpts.empty())
    return "defaultDiagnoseLangOpts";

  // Generate the test condition, as well as a unique function name for the
  // diagnostic test. The list of options should usually be short (one or two
  // options), and the uniqueness isn't strictly necessary (it is just for
  // codegen efficiency).
  std::string FnName = "check", Test;
  for (auto I = LangOpts.begin(), E = LangOpts.end(); I != E; ++I) {
    std::string Part = (*I)->getValueAsString("Name");
    if ((*I)->getValueAsBit("Negated")) {
      FnName += "Not";
      Test += "!";
    }
    Test += "S.LangOpts." + Part;
    if (I + 1 != E)
      Test += " || ";
    FnName += Part;
  }
  FnName += "LangOpts";

  // If this code has already been generated, simply return the previous
  // instance of it.
  static std::set<std::string> CustomLangOptsSet;
  auto I = CustomLangOptsSet.find(FnName);
  if (I != CustomLangOptsSet.end())
    return *I;

  OS << "static bool " << FnName << "(Sema &S, const AttributeList &Attr) {\n";
  OS << "  if (" << Test << ")\n";
  OS << "    return true;\n\n";
  OS << "  S.Diag(Attr.getLoc(), diag::warn_attribute_ignored) ";
  OS << "<< Attr.getName();\n";
  OS << "  return false;\n";
  OS << "}\n\n";

  CustomLangOptsSet.insert(FnName);
  return FnName;
}

static void GenerateDefaultTargetRequirements(raw_ostream &OS) {
  OS << "static bool defaultTargetRequirements(const TargetInfo &) {\n";
  OS << "  return true;\n";
  OS << "}\n\n";
}

static std::string GenerateTargetRequirements(const Record &Attr,
                                              const ParsedAttrMap &Dupes,
                                              raw_ostream &OS) {
  // If the attribute is not a target specific attribute, return the default
  // target handler.
  if (!Attr.isSubClassOf("TargetSpecificAttr"))
    return "defaultTargetRequirements";

  // Get the list of architectures to be tested for.
  const Record *R = Attr.getValueAsDef("Target");
  std::vector<std::string> Arches = R->getValueAsListOfStrings("Arches");
  if (Arches.empty()) {
    PrintError(Attr.getLoc(), "Empty list of target architectures for a "
                              "target-specific attr");
    return "defaultTargetRequirements";
  }

  // If there are other attributes which share the same parsed attribute kind,
  // such as target-specific attributes with a shared spelling, collapse the
  // duplicate architectures. This is required because a shared target-specific
  // attribute has only one AttributeList::Kind enumeration value, but it
  // applies to multiple target architectures. In order for the attribute to be
  // considered valid, all of its architectures need to be included.
  if (!Attr.isValueUnset("ParseKind")) {
    std::string APK = Attr.getValueAsString("ParseKind");
    for (const auto &I : Dupes) {
      if (I.first == APK) {
        std::vector<std::string> DA = I.second->getValueAsDef("Target")
                                          ->getValueAsListOfStrings("Arches");
        std::copy(DA.begin(), DA.end(), std::back_inserter(Arches));
      }
    }
  }

  std::string FnName = "isTarget";
  std::string Test;
  GenerateTargetSpecificAttrChecks(R, Arches, Test, &FnName);

  // If this code has already been generated, simply return the previous
  // instance of it.
  static std::set<std::string> CustomTargetSet;
  auto I = CustomTargetSet.find(FnName);
  if (I != CustomTargetSet.end())
    return *I;

  OS << "static bool " << FnName << "(const TargetInfo &Target) {\n";
  OS << "  const llvm::Triple &T = Target.getTriple();\n";
  OS << "  return " << Test << ";\n";
  OS << "}\n\n";

  CustomTargetSet.insert(FnName);
  return FnName;
}

static void GenerateDefaultSpellingIndexToSemanticSpelling(raw_ostream &OS) {
  OS << "static unsigned defaultSpellingIndexToSemanticSpelling("
     << "const AttributeList &Attr) {\n";
  OS << "  return UINT_MAX;\n";
  OS << "}\n\n";
}

static std::string GenerateSpellingIndexToSemanticSpelling(const Record &Attr,
                                                           raw_ostream &OS) {
  // If the attribute does not have a semantic form, we can bail out early.
  if (!Attr.getValueAsBit("ASTNode"))
    return "defaultSpellingIndexToSemanticSpelling";

  std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(Attr);

  // If there are zero or one spellings, or all of the spellings share the same
  // name, we can also bail out early.
  if (Spellings.size() <= 1 || SpellingNamesAreCommon(Spellings))
    return "defaultSpellingIndexToSemanticSpelling";

  // Generate the enumeration we will use for the mapping.
  SemanticSpellingMap SemanticToSyntacticMap;
  std::string Enum = CreateSemanticSpellings(Spellings, SemanticToSyntacticMap);
  std::string Name = Attr.getName() + "AttrSpellingMap";

  OS << "static unsigned " << Name << "(const AttributeList &Attr) {\n";
  OS << Enum;
  OS << "  unsigned Idx = Attr.getAttributeSpellingListIndex();\n";
  WriteSemanticSpellingSwitch("Idx", SemanticToSyntacticMap, OS);
  OS << "}\n\n";

  return Name;
}

static bool IsKnownToGCC(const Record &Attr) {
  // Look at the spellings for this subject; if there are any spellings which
  // claim to be known to GCC, the attribute is known to GCC.
  std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(Attr);
  for (const auto &I : Spellings) {
    if (I.knownToGCC())
      return true;
  }
  return false;
}

/// Emits the parsed attribute helpers
void EmitClangAttrParsedAttrImpl(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Parsed attribute helpers", OS);

  // Get the list of parsed attributes, and accept the optional list of
  // duplicates due to the ParseKind.
  ParsedAttrMap Dupes;
  ParsedAttrMap Attrs = getParsedAttrList(Records, &Dupes);

  // Generate the default appertainsTo, target and language option diagnostic,
  // and spelling list index mapping methods.
  GenerateDefaultAppertainsTo(OS);
  GenerateDefaultLangOptRequirements(OS);
  GenerateDefaultTargetRequirements(OS);
  GenerateDefaultSpellingIndexToSemanticSpelling(OS);

  // Generate the appertainsTo diagnostic methods and write their names into
  // another mapping. At the same time, generate the AttrInfoMap object
  // contents. Due to the reliance on generated code, use separate streams so
  // that code will not be interleaved.
  std::stringstream SS;
  for (auto I = Attrs.begin(), E = Attrs.end(); I != E; ++I) {
    // TODO: If the attribute's kind appears in the list of duplicates, that is
    // because it is a target-specific attribute that appears multiple times.
    // It would be beneficial to test whether the duplicates are "similar
    // enough" to each other to not cause problems. For instance, check that
    // the spellings are identical, and custom parsing rules match, etc.

    // We need to generate struct instances based off ParsedAttrInfo from
    // AttributeList.cpp.
    SS << "  { ";
    emitArgInfo(*I->second, SS);
    SS << ", " << I->second->getValueAsBit("HasCustomParsing");
    SS << ", " << I->second->isSubClassOf("TargetSpecificAttr");
    SS << ", " << I->second->isSubClassOf("TypeAttr");
    SS << ", " << I->second->isSubClassOf("StmtAttr");
    SS << ", " << IsKnownToGCC(*I->second);
    SS << ", " << GenerateAppertainsTo(*I->second, OS);
    SS << ", " << GenerateLangOptRequirements(*I->second, OS);
    SS << ", " << GenerateTargetRequirements(*I->second, Dupes, OS);
    SS << ", " << GenerateSpellingIndexToSemanticSpelling(*I->second, OS);
    SS << " }";

    if (I + 1 != E)
      SS << ",";

    SS << "  // AT_" << I->first << "\n";
  }

  OS << "static const ParsedAttrInfo AttrInfoMap[AttributeList::UnknownAttribute + 1] = {\n";
  OS << SS.str();
  OS << "};\n\n";
}

// Emits the kind list of parsed attributes
void EmitClangAttrParsedAttrKinds(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Attribute name matcher", OS);

  std::vector<Record *> Attrs = Records.getAllDerivedDefinitions("Attr");
  std::vector<StringMatcher::StringPair> GNU, Declspec, Microsoft, CXX11,
      Keywords, Pragma;
  std::set<std::string> Seen;
  for (const auto *A : Attrs) {
    const Record &Attr = *A;

    bool SemaHandler = Attr.getValueAsBit("SemaHandler");
    bool Ignored = Attr.getValueAsBit("Ignored");
    if (SemaHandler || Ignored) {
      // Attribute spellings can be shared between target-specific attributes,
      // and can be shared between syntaxes for the same attribute. For
      // instance, an attribute can be spelled GNU<"interrupt"> for an ARM-
      // specific attribute, or MSP430-specific attribute. Additionally, an
      // attribute can be spelled GNU<"dllexport"> and Declspec<"dllexport">
      // for the same semantic attribute. Ultimately, we need to map each of
      // these to a single AttributeList::Kind value, but the StringMatcher
      // class cannot handle duplicate match strings. So we generate a list of
      // string to match based on the syntax, and emit multiple string matchers
      // depending on the syntax used.
      std::string AttrName;
      if (Attr.isSubClassOf("TargetSpecificAttr") &&
          !Attr.isValueUnset("ParseKind")) {
        AttrName = Attr.getValueAsString("ParseKind");
        if (Seen.find(AttrName) != Seen.end())
          continue;
        Seen.insert(AttrName);
      } else
        AttrName = NormalizeAttrName(StringRef(Attr.getName())).str();

      std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(Attr);
      for (const auto &S : Spellings) {
        const std::string &RawSpelling = S.name();
        std::vector<StringMatcher::StringPair> *Matches = nullptr;
        std::string Spelling;
        const std::string &Variety = S.variety();
        if (Variety == "CXX11") {
          Matches = &CXX11;
          Spelling += S.nameSpace();
          Spelling += "::";
        } else if (Variety == "GNU")
          Matches = &GNU;
        else if (Variety == "Declspec")
          Matches = &Declspec;
        else if (Variety == "Microsoft")
          Matches = &Microsoft;
        else if (Variety == "Keyword")
          Matches = &Keywords;
        else if (Variety == "Pragma")
          Matches = &Pragma;

        assert(Matches && "Unsupported spelling variety found");

        Spelling += NormalizeAttrSpelling(RawSpelling);
        if (SemaHandler)
          Matches->push_back(StringMatcher::StringPair(Spelling,
                              "return AttributeList::AT_" + AttrName + ";"));
        else
          Matches->push_back(StringMatcher::StringPair(Spelling,
                              "return AttributeList::IgnoredAttribute;"));
      }
    }
  }
  
  OS << "static AttributeList::Kind getAttrKind(StringRef Name, ";
  OS << "AttributeList::Syntax Syntax) {\n";
  OS << "  if (AttributeList::AS_GNU == Syntax) {\n";
  StringMatcher("Name", GNU, OS).Emit();
  OS << "  } else if (AttributeList::AS_Declspec == Syntax) {\n";
  StringMatcher("Name", Declspec, OS).Emit();
  OS << "  } else if (AttributeList::AS_Microsoft == Syntax) {\n";
  StringMatcher("Name", Microsoft, OS).Emit();
  OS << "  } else if (AttributeList::AS_CXX11 == Syntax) {\n";
  StringMatcher("Name", CXX11, OS).Emit();
  OS << "  } else if (AttributeList::AS_Keyword == Syntax || ";
  OS << "AttributeList::AS_ContextSensitiveKeyword == Syntax) {\n";
  StringMatcher("Name", Keywords, OS).Emit();
  OS << "  } else if (AttributeList::AS_Pragma == Syntax) {\n";
  StringMatcher("Name", Pragma, OS).Emit();
  OS << "  }\n";
  OS << "  return AttributeList::UnknownAttribute;\n"
     << "}\n";
}

// Emits the code to dump an attribute.
void EmitClangAttrDump(RecordKeeper &Records, raw_ostream &OS) {
  emitSourceFileHeader("Attribute dumper", OS);

  OS << "  switch (A->getKind()) {\n";
  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr"), Args;
  for (const auto *Attr : Attrs) {
    const Record &R = *Attr;
    if (!R.getValueAsBit("ASTNode"))
      continue;
    OS << "  case attr::" << R.getName() << ": {\n";

    // If the attribute has a semantically-meaningful name (which is determined
    // by whether there is a Spelling enumeration for it), then write out the
    // spelling used for the attribute.
    std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(R);
    if (Spellings.size() > 1 && !SpellingNamesAreCommon(Spellings))
      OS << "    OS << \" \" << A->getSpelling();\n";

    Args = R.getValueAsListOfDefs("Args");
    if (!Args.empty()) {
      OS << "    const auto *SA = cast<" << R.getName()
         << "Attr>(A);\n";
      for (const auto *Arg : Args)
        createArgument(*Arg, R.getName())->writeDump(OS);

      for (const auto *AI : Args)
        createArgument(*AI, R.getName())->writeDumpChildren(OS);
    }
    OS <<
      "    break;\n"
      "  }\n";
  }
  OS << "  }\n";
}

void EmitClangAttrParserStringSwitches(RecordKeeper &Records,
                                       raw_ostream &OS) {
  emitSourceFileHeader("Parser-related llvm::StringSwitch cases", OS);
  emitClangAttrArgContextList(Records, OS);
  emitClangAttrIdentifierArgList(Records, OS);
  emitClangAttrTypeArgList(Records, OS);
  emitClangAttrLateParsedList(Records, OS);
}

class DocumentationData {
public:
  const Record *Documentation;
  const Record *Attribute;

  DocumentationData(const Record &Documentation, const Record &Attribute)
      : Documentation(&Documentation), Attribute(&Attribute) {}
};

static void WriteCategoryHeader(const Record *DocCategory,
                                raw_ostream &OS) {
  const std::string &Name = DocCategory->getValueAsString("Name");
  OS << Name << "\n" << std::string(Name.length(), '=') << "\n";

  // If there is content, print that as well.
  std::string ContentStr = DocCategory->getValueAsString("Content");
  // Trim leading and trailing newlines and spaces.
  OS << StringRef(ContentStr).trim();

  OS << "\n\n";
}

enum SpellingKind {
  GNU = 1 << 0,
  CXX11 = 1 << 1,
  Declspec = 1 << 2,
  Microsoft = 1 << 3,
  Keyword = 1 << 4,
  Pragma = 1 << 5
};

static void WriteDocumentation(const DocumentationData &Doc,
                               raw_ostream &OS) {
  // FIXME: there is no way to have a per-spelling category for the attribute
  // documentation. This may not be a limiting factor since the spellings
  // should generally be consistently applied across the category.

  std::vector<FlattenedSpelling> Spellings = GetFlattenedSpellings(*Doc.Attribute);

  // Determine the heading to be used for this attribute.
  std::string Heading = Doc.Documentation->getValueAsString("Heading");
  bool CustomHeading = !Heading.empty();
  if (Heading.empty()) {
    // If there's only one spelling, we can simply use that.
    if (Spellings.size() == 1)
      Heading = Spellings.begin()->name();
    else {
      std::set<std::string> Uniques;
      for (auto I = Spellings.begin(), E = Spellings.end();
           I != E && Uniques.size() <= 1; ++I) {
        std::string Spelling = NormalizeNameForSpellingComparison(I->name());
        Uniques.insert(Spelling);
      }
      // If the semantic map has only one spelling, that is sufficient for our
      // needs.
      if (Uniques.size() == 1)
        Heading = *Uniques.begin();
    }
  }

  // If the heading is still empty, it is an error.
  if (Heading.empty())
    PrintFatalError(Doc.Attribute->getLoc(),
                    "This attribute requires a heading to be specified");

  // Gather a list of unique spellings; this is not the same as the semantic
  // spelling for the attribute. Variations in underscores and other non-
  // semantic characters are still acceptable.
  std::vector<std::string> Names;

  unsigned SupportedSpellings = 0;
  for (const auto &I : Spellings) {
    SpellingKind Kind = StringSwitch<SpellingKind>(I.variety())
                            .Case("GNU", GNU)
                            .Case("CXX11", CXX11)
                            .Case("Declspec", Declspec)
                            .Case("Microsoft", Microsoft)
                            .Case("Keyword", Keyword)
                            .Case("Pragma", Pragma);

    // Mask in the supported spelling.
    SupportedSpellings |= Kind;

    std::string Name;
    if (Kind == CXX11 && !I.nameSpace().empty())
      Name = I.nameSpace() + "::";
    Name += I.name();

    // If this name is the same as the heading, do not add it.
    if (Name != Heading)
      Names.push_back(Name);
  }

  // Print out the heading for the attribute. If there are alternate spellings,
  // then display those after the heading.
  if (!CustomHeading && !Names.empty()) {
    Heading += " (";
    for (auto I = Names.begin(), E = Names.end(); I != E; ++I) {
      if (I != Names.begin())
        Heading += ", ";
      Heading += *I;
    }
    Heading += ")";
  }
  OS << Heading << "\n" << std::string(Heading.length(), '-') << "\n";

  if (!SupportedSpellings)
    PrintFatalError(Doc.Attribute->getLoc(),
                    "Attribute has no supported spellings; cannot be "
                    "documented");

  // List what spelling syntaxes the attribute supports.
  OS << ".. csv-table:: Supported Syntaxes\n";
  OS << "   :header: \"GNU\", \"C++11\", \"__declspec\", \"Keyword\",";
  OS << " \"Pragma\"\n\n";
  OS << "   \"";
  if (SupportedSpellings & GNU) OS << "X";
  OS << "\",\"";
  if (SupportedSpellings & CXX11) OS << "X";
  OS << "\",\"";
  if (SupportedSpellings & Declspec) OS << "X";
  OS << "\",\"";
  if (SupportedSpellings & Keyword) OS << "X";
  OS << "\", \"";
  if (SupportedSpellings & Pragma) OS << "X";
  OS << "\"\n\n";

  // If the attribute is deprecated, print a message about it, and possibly
  // provide a replacement attribute.
  if (!Doc.Documentation->isValueUnset("Deprecated")) {
    OS << "This attribute has been deprecated, and may be removed in a future "
       << "version of Clang.";
    const Record &Deprecated = *Doc.Documentation->getValueAsDef("Deprecated");
    std::string Replacement = Deprecated.getValueAsString("Replacement");
    if (!Replacement.empty())
      OS << "  This attribute has been superseded by ``"
         << Replacement << "``.";
    OS << "\n\n";
  }

  std::string ContentStr = Doc.Documentation->getValueAsString("Content");
  // Trim leading and trailing newlines and spaces.
  OS << StringRef(ContentStr).trim();

  OS << "\n\n\n";
}

void EmitClangAttrDocs(RecordKeeper &Records, raw_ostream &OS) {
  // Get the documentation introduction paragraph.
  const Record *Documentation = Records.getDef("GlobalDocumentation");
  if (!Documentation) {
    PrintFatalError("The Documentation top-level definition is missing, "
                    "no documentation will be generated.");
    return;
  }

  OS << Documentation->getValueAsString("Intro") << "\n";

  // Gather the Documentation lists from each of the attributes, based on the
  // category provided.
  std::vector<Record *> Attrs = Records.getAllDerivedDefinitions("Attr");
  std::map<const Record *, std::vector<DocumentationData>> SplitDocs;
  for (const auto *A : Attrs) {
    const Record &Attr = *A;
    std::vector<Record *> Docs = Attr.getValueAsListOfDefs("Documentation");
    for (const auto *D : Docs) {
      const Record &Doc = *D;
      const Record *Category = Doc.getValueAsDef("Category");
      // If the category is "undocumented", then there cannot be any other
      // documentation categories (otherwise, the attribute would become
      // documented).
      std::string Cat = Category->getValueAsString("Name");
      bool Undocumented = Cat == "Undocumented";
      if (Undocumented && Docs.size() > 1)
        PrintFatalError(Doc.getLoc(),
                        "Attribute is \"Undocumented\", but has multiple "
                        "documentation categories");      

      if (!Undocumented)
        SplitDocs[Category].push_back(DocumentationData(Doc, Attr));
    }
  }

  // Having split the attributes out based on what documentation goes where,
  // we can begin to generate sections of documentation.
  for (const auto &I : SplitDocs) {
    WriteCategoryHeader(I.first, OS);

    // Walk over each of the attributes in the category and write out their
    // documentation.
    for (const auto &Doc : I.second)
      WriteDocumentation(Doc, OS);
  }
}

} // end namespace clang
