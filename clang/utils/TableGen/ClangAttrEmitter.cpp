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

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/StringMatcher.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <algorithm>
#include <cctype>

using namespace llvm;

static const std::vector<StringRef>
getValueAsListOfStrings(Record &R, StringRef FieldName) {
  ListInit *List = R.getValueAsListInit(FieldName);
  assert (List && "Got a null ListInit");

  std::vector<StringRef> Strings;
  Strings.reserve(List->getSize());

  for (ListInit::const_iterator i = List->begin(), e = List->end();
       i != e;
       ++i) {
    assert(*i && "Got a null element in a ListInit");
    if (StringInit *S = dyn_cast<StringInit>(*i))
      Strings.push_back(S->getValue());
    else
      assert(false && "Got a non-string, non-code element in a ListInit");
  }

  return Strings;
}

static std::string ReadPCHRecord(StringRef type) {
  return StringSwitch<std::string>(type)
    .EndsWith("Decl *", "GetLocalDeclAs<" 
              + std::string(type, 0, type.size()-1) + ">(F, Record[Idx++])")
    .Case("QualType", "getLocalType(F, Record[Idx++])")
    .Case("Expr *", "ReadSubExpr()")
    .Case("IdentifierInfo *", "GetIdentifierInfo(F, Record, Idx)")
    .Case("SourceLocation", "ReadSourceLocation(F, Record, Idx)")
    .Default("Record[Idx++]");
}

// Assumes that the way to get the value is SA->getname()
static std::string WritePCHRecord(StringRef type, StringRef name) {
  return StringSwitch<std::string>(type)
    .EndsWith("Decl *", "AddDeclRef(" + std::string(name) +
                        ", Record);\n")
    .Case("QualType", "AddTypeRef(" + std::string(name) + ", Record);\n")
    .Case("Expr *", "AddStmt(" + std::string(name) + ");\n")
    .Case("IdentifierInfo *", 
          "AddIdentifierRef(" + std::string(name) + ", Record);\n")
    .Case("SourceLocation", 
          "AddSourceLocation(" + std::string(name) + ", Record);\n")
    .Default("Record.push_back(" + std::string(name) + ");\n");
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

// Normalize attribute spelling only if the spelling has both leading
// and trailing underscores. For example, __ms_struct__ will be 
// normalized to "ms_struct"; __cdecl will remain intact.
static StringRef NormalizeAttrSpelling(StringRef AttrSpelling) {
  if (AttrSpelling.startswith("__") && AttrSpelling.endswith("__")) {
    AttrSpelling = AttrSpelling.substr(2, AttrSpelling.size() - 4);
  }

  return AttrSpelling;
}

namespace {
  class Argument {
    std::string lowerName, upperName;
    StringRef attrName;

  public:
    Argument(Record &Arg, StringRef Attr)
      : lowerName(Arg.getValueAsString("Name")), upperName(lowerName),
        attrName(Attr) {
      if (!lowerName.empty()) {
        lowerName[0] = std::tolower(lowerName[0]);
        upperName[0] = std::toupper(upperName[0]);
      }
    }
    virtual ~Argument() {}

    StringRef getLowerName() const { return lowerName; }
    StringRef getUpperName() const { return upperName; }
    StringRef getAttrName() const { return attrName; }

    // These functions print the argument contents formatted in different ways.
    virtual void writeAccessors(raw_ostream &OS) const = 0;
    virtual void writeAccessorDefinitions(raw_ostream &OS) const {}
    virtual void writeCloneArgs(raw_ostream &OS) const = 0;
    virtual void writeTemplateInstantiationArgs(raw_ostream &OS) const = 0;
    virtual void writeTemplateInstantiation(raw_ostream &OS) const {}
    virtual void writeCtorBody(raw_ostream &OS) const {}
    virtual void writeCtorInitializers(raw_ostream &OS) const = 0;
    virtual void writeCtorParameters(raw_ostream &OS) const = 0;
    virtual void writeDeclarations(raw_ostream &OS) const = 0;
    virtual void writePCHReadArgs(raw_ostream &OS) const = 0;
    virtual void writePCHReadDecls(raw_ostream &OS) const = 0;
    virtual void writePCHWrite(raw_ostream &OS) const = 0;
    virtual void writeValue(raw_ostream &OS) const = 0;
  };

  class SimpleArgument : public Argument {
    std::string type;

  public:
    SimpleArgument(Record &Arg, StringRef Attr, std::string T)
      : Argument(Arg, Attr), type(T)
    {}

    std::string getType() const { return type; }

    void writeAccessors(raw_ostream &OS) const {
      OS << "  " << type << " get" << getUpperName() << "() const {\n";
      OS << "    return " << getLowerName() << ";\n";
      OS << "  }";
    }
    void writeCloneArgs(raw_ostream &OS) const {
      OS << getLowerName();
    }
    void writeTemplateInstantiationArgs(raw_ostream &OS) const {
      OS << "A->get" << getUpperName() << "()";
    }
    void writeCtorInitializers(raw_ostream &OS) const {
      OS << getLowerName() << "(" << getUpperName() << ")";
    }
    void writeCtorParameters(raw_ostream &OS) const {
      OS << type << " " << getUpperName();
    }
    void writeDeclarations(raw_ostream &OS) const {
      OS << type << " " << getLowerName() << ";";
    }
    void writePCHReadDecls(raw_ostream &OS) const {
      std::string read = ReadPCHRecord(type);
      OS << "    " << type << " " << getLowerName() << " = " << read << ";\n";
    }
    void writePCHReadArgs(raw_ostream &OS) const {
      OS << getLowerName();
    }
    void writePCHWrite(raw_ostream &OS) const {
      OS << "    " << WritePCHRecord(type, "SA->get" +
                                           std::string(getUpperName()) + "()");
    }
    void writeValue(raw_ostream &OS) const {
      if (type == "FunctionDecl *") {
        OS << "\" << get" << getUpperName() << "()->getNameInfo().getAsString() << \"";
      } else if (type == "IdentifierInfo *") {
        OS << "\" << get" << getUpperName() << "()->getName() << \"";
      } else if (type == "QualType") {
        OS << "\" << get" << getUpperName() << "().getAsString() << \"";
      } else if (type == "SourceLocation") {
        OS << "\" << get" << getUpperName() << "().getRawEncoding() << \"";
      } else {
        OS << "\" << get" << getUpperName() << "() << \"";
      }
    }
  };

  class StringArgument : public Argument {
  public:
    StringArgument(Record &Arg, StringRef Attr)
      : Argument(Arg, Attr)
    {}

    void writeAccessors(raw_ostream &OS) const {
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
      OS << "    std::memcpy(this->" << getLowerName() << ", S.data(), "
         << getLowerName() << "Length);\n";
      OS << "  }";
    }
    void writeCloneArgs(raw_ostream &OS) const {
      OS << "get" << getUpperName() << "()";
    }
    void writeTemplateInstantiationArgs(raw_ostream &OS) const {
      OS << "A->get" << getUpperName() << "()";
    }
    void writeCtorBody(raw_ostream &OS) const {
      OS << "      std::memcpy(" << getLowerName() << ", " << getUpperName()
         << ".data(), " << getLowerName() << "Length);";
    }
    void writeCtorInitializers(raw_ostream &OS) const {
      OS << getLowerName() << "Length(" << getUpperName() << ".size()),"
         << getLowerName() << "(new (Ctx, 1) char[" << getLowerName()
         << "Length])";
    }
    void writeCtorParameters(raw_ostream &OS) const {
      OS << "llvm::StringRef " << getUpperName();
    }
    void writeDeclarations(raw_ostream &OS) const {
      OS << "unsigned " << getLowerName() << "Length;\n";
      OS << "char *" << getLowerName() << ";";
    }
    void writePCHReadDecls(raw_ostream &OS) const {
      OS << "    std::string " << getLowerName()
         << "= ReadString(Record, Idx);\n";
    }
    void writePCHReadArgs(raw_ostream &OS) const {
      OS << getLowerName();
    }
    void writePCHWrite(raw_ostream &OS) const {
      OS << "    AddString(SA->get" << getUpperName() << "(), Record);\n";
    }
    void writeValue(raw_ostream &OS) const {
      OS << "\\\"\" << get" << getUpperName() << "() << \"\\\"";
    }
  };

  class AlignedArgument : public Argument {
  public:
    AlignedArgument(Record &Arg, StringRef Attr)
      : Argument(Arg, Attr)
    {}

    void writeAccessors(raw_ostream &OS) const {
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
    void writeAccessorDefinitions(raw_ostream &OS) const {
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
      // FIXME: Load the platform-specific maximum alignment, rather than
      //        16, the x86 max.
      OS << "unsigned " << getAttrName() << "Attr::get" << getUpperName()
         << "(ASTContext &Ctx) const {\n";
      OS << "  assert(!is" << getUpperName() << "Dependent());\n";
      OS << "  if (is" << getLowerName() << "Expr)\n";
      OS << "    return (" << getLowerName() << "Expr ? " << getLowerName()
         << "Expr->EvaluateKnownConstInt(Ctx).getZExtValue() : 16)"
         << "* Ctx.getCharWidth();\n";
      OS << "  else\n";
      OS << "    return 0; // FIXME\n";
      OS << "}\n";
    }
    void writeCloneArgs(raw_ostream &OS) const {
      OS << "is" << getLowerName() << "Expr, is" << getLowerName()
         << "Expr ? static_cast<void*>(" << getLowerName()
         << "Expr) : " << getLowerName()
         << "Type";
    }
    void writeTemplateInstantiationArgs(raw_ostream &OS) const {
      // FIXME: move the definition in Sema::InstantiateAttrs to here.
      // In the meantime, aligned attributes are cloned.
    }
    void writeCtorBody(raw_ostream &OS) const {
      OS << "    if (is" << getLowerName() << "Expr)\n";
      OS << "       " << getLowerName() << "Expr = reinterpret_cast<Expr *>("
         << getUpperName() << ");\n";
      OS << "    else\n";
      OS << "       " << getLowerName()
         << "Type = reinterpret_cast<TypeSourceInfo *>(" << getUpperName()
         << ");";
    }
    void writeCtorInitializers(raw_ostream &OS) const {
      OS << "is" << getLowerName() << "Expr(Is" << getUpperName() << "Expr)";
    }
    void writeCtorParameters(raw_ostream &OS) const {
      OS << "bool Is" << getUpperName() << "Expr, void *" << getUpperName();
    }
    void writeDeclarations(raw_ostream &OS) const {
      OS << "bool is" << getLowerName() << "Expr;\n";
      OS << "union {\n";
      OS << "Expr *" << getLowerName() << "Expr;\n";
      OS << "TypeSourceInfo *" << getLowerName() << "Type;\n";
      OS << "};";
    }
    void writePCHReadArgs(raw_ostream &OS) const {
      OS << "is" << getLowerName() << "Expr, " << getLowerName() << "Ptr";
    }
    void writePCHReadDecls(raw_ostream &OS) const {
      OS << "    bool is" << getLowerName() << "Expr = Record[Idx++];\n";
      OS << "    void *" << getLowerName() << "Ptr;\n";
      OS << "    if (is" << getLowerName() << "Expr)\n";
      OS << "      " << getLowerName() << "Ptr = ReadExpr(F);\n";
      OS << "    else\n";
      OS << "      " << getLowerName()
         << "Ptr = GetTypeSourceInfo(F, Record, Idx);\n";
    }
    void writePCHWrite(raw_ostream &OS) const {
      OS << "    Record.push_back(SA->is" << getUpperName() << "Expr());\n";
      OS << "    if (SA->is" << getUpperName() << "Expr())\n";
      OS << "      AddStmt(SA->get" << getUpperName() << "Expr());\n";
      OS << "    else\n";
      OS << "      AddTypeSourceInfo(SA->get" << getUpperName()
         << "Type(), Record);\n";
    }
    void writeValue(raw_ostream &OS) const {
      OS << "\";\n"
         << "  " << getLowerName() << "Expr->printPretty(OS, 0, Policy);\n"
         << "  OS << \"";
    }
  };

  class VariadicArgument : public Argument {
    std::string type;

  public:
    VariadicArgument(Record &Arg, StringRef Attr, std::string T)
      : Argument(Arg, Attr), type(T)
    {}

    std::string getType() const { return type; }

    void writeAccessors(raw_ostream &OS) const {
      OS << "  typedef " << type << "* " << getLowerName() << "_iterator;\n";
      OS << "  " << getLowerName() << "_iterator " << getLowerName()
         << "_begin() const {\n";
      OS << "    return " << getLowerName() << ";\n";
      OS << "  }\n";
      OS << "  " << getLowerName() << "_iterator " << getLowerName()
         << "_end() const {\n";
      OS << "    return " << getLowerName() << " + " << getLowerName()
         << "Size;\n";
      OS << "  }\n";
      OS << "  unsigned " << getLowerName() << "_size() const {\n"
         << "    return " << getLowerName() << "Size;\n";
      OS << "  }";
    }
    void writeCloneArgs(raw_ostream &OS) const {
      OS << getLowerName() << ", " << getLowerName() << "Size";
    }
    void writeTemplateInstantiationArgs(raw_ostream &OS) const {
      // This isn't elegant, but we have to go through public methods...
      OS << "A->" << getLowerName() << "_begin(), "
         << "A->" << getLowerName() << "_size()";
    }
    void writeCtorBody(raw_ostream &OS) const {
      // FIXME: memcpy is not safe on non-trivial types.
      OS << "    std::memcpy(" << getLowerName() << ", " << getUpperName()
         << ", " << getLowerName() << "Size * sizeof(" << getType() << "));\n";
    }
    void writeCtorInitializers(raw_ostream &OS) const {
      OS << getLowerName() << "Size(" << getUpperName() << "Size), "
         << getLowerName() << "(new (Ctx, 16) " << getType() << "["
         << getLowerName() << "Size])";
    }
    void writeCtorParameters(raw_ostream &OS) const {
      OS << getType() << " *" << getUpperName() << ", unsigned "
         << getUpperName() << "Size";
    }
    void writeDeclarations(raw_ostream &OS) const {
      OS << "  unsigned " << getLowerName() << "Size;\n";
      OS << "  " << getType() << " *" << getLowerName() << ";";
    }
    void writePCHReadDecls(raw_ostream &OS) const {
      OS << "  unsigned " << getLowerName() << "Size = Record[Idx++];\n";
      OS << "  llvm::SmallVector<" << type << ", 4> " << getLowerName()
         << ";\n";
      OS << "  " << getLowerName() << ".reserve(" << getLowerName()
         << "Size);\n";
      OS << "  for (unsigned i = " << getLowerName() << "Size; i; --i)\n";
      
      std::string read = ReadPCHRecord(type);
      OS << "    " << getLowerName() << ".push_back(" << read << ");\n";
    }
    void writePCHReadArgs(raw_ostream &OS) const {
      OS << getLowerName() << ".data(), " << getLowerName() << "Size";
    }
    void writePCHWrite(raw_ostream &OS) const{
      OS << "    Record.push_back(SA->" << getLowerName() << "_size());\n";
      OS << "    for (" << getAttrName() << "Attr::" << getLowerName()
         << "_iterator i = SA->" << getLowerName() << "_begin(), e = SA->"
         << getLowerName() << "_end(); i != e; ++i)\n";
      OS << "      " << WritePCHRecord(type, "(*i)");
    }
    void writeValue(raw_ostream &OS) const {
      OS << "\";\n";
      OS << "  bool isFirst = true;\n"
         << "  for (" << getAttrName() << "Attr::" << getLowerName()
         << "_iterator i = " << getLowerName() << "_begin(), e = "
         << getLowerName() << "_end(); i != e; ++i) {\n"
         << "    if (isFirst) isFirst = false;\n"
         << "    else OS << \", \";\n"
         << "    OS << *i;\n"
         << "  }\n";
      OS << "  OS << \"";
    }
  };

  class EnumArgument : public Argument {
    std::string type;
    std::vector<StringRef> values, enums;
  public:
    EnumArgument(Record &Arg, StringRef Attr)
      : Argument(Arg, Attr), type(Arg.getValueAsString("Type")),
        values(getValueAsListOfStrings(Arg, "Values")),
        enums(getValueAsListOfStrings(Arg, "Enums"))
    {}

    void writeAccessors(raw_ostream &OS) const {
      OS << "  " << type << " get" << getUpperName() << "() const {\n";
      OS << "    return " << getLowerName() << ";\n";
      OS << "  }";
    }
    void writeCloneArgs(raw_ostream &OS) const {
      OS << getLowerName();
    }
    void writeTemplateInstantiationArgs(raw_ostream &OS) const {
      OS << "A->get" << getUpperName() << "()";
    }
    void writeCtorInitializers(raw_ostream &OS) const {
      OS << getLowerName() << "(" << getUpperName() << ")";
    }
    void writeCtorParameters(raw_ostream &OS) const {
      OS << type << " " << getUpperName();
    }
    void writeDeclarations(raw_ostream &OS) const {
      // Calculate the various enum values
      std::vector<StringRef> uniques(enums);
      std::sort(uniques.begin(), uniques.end());
      uniques.erase(std::unique(uniques.begin(), uniques.end()),
                    uniques.end());
      // FIXME: Emit a proper error
      assert(!uniques.empty());

      std::vector<StringRef>::iterator i = uniques.begin(),
                                       e = uniques.end();
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
    void writePCHReadDecls(raw_ostream &OS) const {
      OS << "    " << getAttrName() << "Attr::" << type << " " << getLowerName()
         << "(static_cast<" << getAttrName() << "Attr::" << type
         << ">(Record[Idx++]));\n";
    }
    void writePCHReadArgs(raw_ostream &OS) const {
      OS << getLowerName();
    }
    void writePCHWrite(raw_ostream &OS) const {
      OS << "Record.push_back(SA->get" << getUpperName() << "());\n";
    }
    void writeValue(raw_ostream &OS) const {
      OS << "\" << get" << getUpperName() << "() << \"";
    }
  };

  class VersionArgument : public Argument {
  public:
    VersionArgument(Record &Arg, StringRef Attr)
      : Argument(Arg, Attr)
    {}

    void writeAccessors(raw_ostream &OS) const {
      OS << "  VersionTuple get" << getUpperName() << "() const {\n";
      OS << "    return " << getLowerName() << ";\n";
      OS << "  }\n";
      OS << "  void set" << getUpperName() 
         << "(ASTContext &C, VersionTuple V) {\n";
      OS << "    " << getLowerName() << " = V;\n";
      OS << "  }";
    }
    void writeCloneArgs(raw_ostream &OS) const {
      OS << "get" << getUpperName() << "()";
    }
    void writeTemplateInstantiationArgs(raw_ostream &OS) const {
      OS << "A->get" << getUpperName() << "()";
    }
    void writeCtorBody(raw_ostream &OS) const {
    }
    void writeCtorInitializers(raw_ostream &OS) const {
      OS << getLowerName() << "(" << getUpperName() << ")";
    }
    void writeCtorParameters(raw_ostream &OS) const {
      OS << "VersionTuple " << getUpperName();
    }
    void writeDeclarations(raw_ostream &OS) const {
      OS << "VersionTuple " << getLowerName() << ";\n";
    }
    void writePCHReadDecls(raw_ostream &OS) const {
      OS << "    VersionTuple " << getLowerName()
         << "= ReadVersionTuple(Record, Idx);\n";
    }
    void writePCHReadArgs(raw_ostream &OS) const {
      OS << getLowerName();
    }
    void writePCHWrite(raw_ostream &OS) const {
      OS << "    AddVersionTuple(SA->get" << getUpperName() << "(), Record);\n";
    }
    void writeValue(raw_ostream &OS) const {
      OS << getLowerName() << "=\" << get" << getUpperName() << "() << \"";
    }
  };

  class ExprArgument : public SimpleArgument {
  public:
    ExprArgument(Record &Arg, StringRef Attr)
      : SimpleArgument(Arg, Attr, "Expr *")
    {}

    void writeTemplateInstantiationArgs(raw_ostream &OS) const {
      OS << "tempInst" << getUpperName();
    }

    void writeTemplateInstantiation(raw_ostream &OS) const {
      OS << "      " << getType() << " tempInst" << getUpperName() << ";\n";
      OS << "      {\n";
      OS << "        EnterExpressionEvaluationContext "
         << "Unevaluated(S, Sema::Unevaluated);\n";
      OS << "        ExprResult " << "Result = S.SubstExpr("
         << "A->get" << getUpperName() << "(), TemplateArgs);\n";
      OS << "        tempInst" << getUpperName() << " = "
         << "Result.takeAs<Expr>();\n";
      OS << "      }\n";
    }
  };

  class VariadicExprArgument : public VariadicArgument {
  public:
    VariadicExprArgument(Record &Arg, StringRef Attr)
      : VariadicArgument(Arg, Attr, "Expr *")
    {}

    void writeTemplateInstantiationArgs(raw_ostream &OS) const {
      OS << "tempInst" << getUpperName() << ", "
         << "A->" << getLowerName() << "_size()";
    }

    void writeTemplateInstantiation(raw_ostream &OS) const {
      OS << "      " << getType() << " *tempInst" << getUpperName()
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
      OS << "          *TI = Result.takeAs<Expr>();\n";
      OS << "        }\n";
      OS << "      }\n";
    }
  };
}

static Argument *createArgument(Record &Arg, StringRef Attr,
                                Record *Search = 0) {
  if (!Search)
    Search = &Arg;

  Argument *Ptr = 0;
  llvm::StringRef ArgName = Search->getName();

  if (ArgName == "AlignedArgument") Ptr = new AlignedArgument(Arg, Attr);
  else if (ArgName == "EnumArgument") Ptr = new EnumArgument(Arg, Attr);
  else if (ArgName == "ExprArgument") Ptr = new ExprArgument(Arg, Attr);
  else if (ArgName == "FunctionArgument")
    Ptr = new SimpleArgument(Arg, Attr, "FunctionDecl *");
  else if (ArgName == "IdentifierArgument")
    Ptr = new SimpleArgument(Arg, Attr, "IdentifierInfo *");
  else if (ArgName == "BoolArgument") Ptr = new SimpleArgument(Arg, Attr, 
                                                               "bool");
  else if (ArgName == "IntArgument") Ptr = new SimpleArgument(Arg, Attr, "int");
  else if (ArgName == "StringArgument") Ptr = new StringArgument(Arg, Attr);
  else if (ArgName == "TypeArgument")
    Ptr = new SimpleArgument(Arg, Attr, "QualType");
  else if (ArgName == "UnsignedArgument")
    Ptr = new SimpleArgument(Arg, Attr, "unsigned");
  else if (ArgName == "SourceLocArgument")
    Ptr = new SimpleArgument(Arg, Attr, "SourceLocation");
  else if (ArgName == "VariadicUnsignedArgument")
    Ptr = new VariadicArgument(Arg, Attr, "unsigned");
  else if (ArgName == "VariadicExprArgument")
    Ptr = new VariadicExprArgument(Arg, Attr);
  else if (ArgName == "VersionArgument")
    Ptr = new VersionArgument(Arg, Attr);

  if (!Ptr) {
    std::vector<Record*> Bases = Search->getSuperClasses();
    for (std::vector<Record*>::iterator i = Bases.begin(), e = Bases.end();
         i != e; ++i) {
      Ptr = createArgument(Arg, Attr, *i);
      if (Ptr)
        break;
    }
  }
  return Ptr;
}

static void writeAvailabilityValue(raw_ostream &OS) {
  OS << "\" << getPlatform()->getName();\n"
     << "  if (!getIntroduced().empty()) OS << \", introduced=\" << getIntroduced();\n"
     << "  if (!getDeprecated().empty()) OS << \", deprecated=\" << getDeprecated();\n"
     << "  if (!getObsoleted().empty()) OS << \", obsoleted=\" << getObsoleted();\n"
     << "  if (getUnavailable()) OS << \", unavailable\";\n"
     << "  OS << \"";
}

namespace clang {

// Emits the class definitions for attributes.
void EmitClangAttrClass(RecordKeeper &Records, raw_ostream &OS) {
  OS << "// This file is generated by TableGen. Do not edit.\n\n";
  OS << "#ifndef LLVM_CLANG_ATTR_CLASSES_INC\n";
  OS << "#define LLVM_CLANG_ATTR_CLASSES_INC\n\n";

  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr");

  for (std::vector<Record*>::iterator i = Attrs.begin(), e = Attrs.end();
       i != e; ++i) {
    Record &R = **i;
    
    if (!R.getValueAsBit("ASTNode"))
      continue;
    
    const std::string &SuperName = R.getSuperClasses().back()->getName();

    OS << "class " << R.getName() << "Attr : public " << SuperName << " {\n";

    std::vector<Record*> ArgRecords = R.getValueAsListOfDefs("Args");
    std::vector<Argument*> Args;
    std::vector<Argument*>::iterator ai, ae;
    Args.reserve(ArgRecords.size());

    for (std::vector<Record*>::iterator ri = ArgRecords.begin(),
                                        re = ArgRecords.end();
         ri != re; ++ri) {
      Record &ArgRecord = **ri;
      Argument *Arg = createArgument(ArgRecord, R.getName());
      assert(Arg);
      Args.push_back(Arg);

      Arg->writeDeclarations(OS);
      OS << "\n\n";
    }

    ae = Args.end();

    OS << "\n public:\n";
    OS << "  " << R.getName() << "Attr(SourceRange R, ASTContext &Ctx\n";
    
    for (ai = Args.begin(); ai != ae; ++ai) {
      OS << "              , ";
      (*ai)->writeCtorParameters(OS);
      OS << "\n";
    }
    
    OS << "             )\n";
    OS << "    : " << SuperName << "(attr::" << R.getName() << ", R)\n";

    for (ai = Args.begin(); ai != ae; ++ai) {
      OS << "              , ";
      (*ai)->writeCtorInitializers(OS);
      OS << "\n";
    }

    OS << "  {\n";
  
    for (ai = Args.begin(); ai != ae; ++ai) {
      (*ai)->writeCtorBody(OS);
      OS << "\n";
    }
    OS << "  }\n\n";

    OS << "  virtual " << R.getName() << "Attr *clone (ASTContext &C) const;\n";
    OS << "  virtual void printPretty(llvm::raw_ostream &OS,"
       << "                           const PrintingPolicy &Policy) const;\n";

    for (ai = Args.begin(); ai != ae; ++ai) {
      (*ai)->writeAccessors(OS);
      OS << "\n\n";
    }

    OS << R.getValueAsString("AdditionalMembers");
    OS << "\n\n";

    OS << "  static bool classof(const Attr *A) { return A->getKind() == "
       << "attr::" << R.getName() << "; }\n";

    bool LateParsed = R.getValueAsBit("LateParsed");
    OS << "  virtual bool isLateParsed() const { return "
       << LateParsed << "; }\n";

    OS << "};\n\n";
  }

  OS << "#endif\n";
}

// Emits the class method definitions for attributes.
void EmitClangAttrImpl(RecordKeeper &Records, raw_ostream &OS) {
  OS << "// This file is generated by TableGen. Do not edit.\n\n";

  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr");
  std::vector<Record*>::iterator i = Attrs.begin(), e = Attrs.end(), ri, re;
  std::vector<Argument*>::iterator ai, ae;

  for (; i != e; ++i) {
    Record &R = **i;
    
    if (!R.getValueAsBit("ASTNode"))
      continue;
    
    std::vector<Record*> ArgRecords = R.getValueAsListOfDefs("Args");
    std::vector<Record*> Spellings = R.getValueAsListOfDefs("Spellings");
    std::vector<Argument*> Args;
    for (ri = ArgRecords.begin(), re = ArgRecords.end(); ri != re; ++ri)
      Args.push_back(createArgument(**ri, R.getName()));

    for (ai = Args.begin(), ae = Args.end(); ai != ae; ++ai)
      (*ai)->writeAccessorDefinitions(OS);

    OS << R.getName() << "Attr *" << R.getName()
       << "Attr::clone(ASTContext &C) const {\n";
    OS << "  return new (C) " << R.getName() << "Attr(getLocation(), C";
    for (ai = Args.begin(); ai != ae; ++ai) {
      OS << ", ";
      (*ai)->writeCloneArgs(OS);
    }
    OS << ");\n}\n\n";

    OS << "void " << R.getName() << "Attr::printPretty("
       << "llvm::raw_ostream &OS, const PrintingPolicy &Policy) const {\n";
    if (Spellings.begin() != Spellings.end()) {
      std::string Spelling = (*Spellings.begin())->getValueAsString("Name");
      OS << "  OS << \" __attribute__((" << Spelling;
      if (Args.size()) OS << "(";
      if (Spelling == "availability") {
        writeAvailabilityValue(OS);
      } else {
        for (ai = Args.begin(); ai != ae; ++ai) {
          if (ai!=Args.begin()) OS <<", ";
          (*ai)->writeValue(OS);
        }
      }
      if (Args.size()) OS << ")";
      OS << "))\";\n";
    }
    OS << "}\n\n";
  }
}

} // end namespace clang

static void EmitAttrList(raw_ostream &OS, StringRef Class,
                         const std::vector<Record*> &AttrList) {
  std::vector<Record*>::const_iterator i = AttrList.begin(), e = AttrList.end();

  if (i != e) {
    // Move the end iterator back to emit the last attribute.
    for(--e; i != e; ++i) {
      if (!(*i)->getValueAsBit("ASTNode"))
        continue;
      
      OS << Class << "(" << (*i)->getName() << ")\n";
    }
    
    OS << "LAST_" << Class << "(" << (*i)->getName() << ")\n\n";
  }
}

namespace clang {

// Emits the enumeration list for attributes.
void EmitClangAttrList(RecordKeeper &Records, raw_ostream &OS) {
  OS << "// This file is generated by TableGen. Do not edit.\n\n";

  OS << "#ifndef LAST_ATTR\n";
  OS << "#define LAST_ATTR(NAME) ATTR(NAME)\n";
  OS << "#endif\n\n";

  OS << "#ifndef INHERITABLE_ATTR\n";
  OS << "#define INHERITABLE_ATTR(NAME) ATTR(NAME)\n";
  OS << "#endif\n\n";

  OS << "#ifndef LAST_INHERITABLE_ATTR\n";
  OS << "#define LAST_INHERITABLE_ATTR(NAME) INHERITABLE_ATTR(NAME)\n";
  OS << "#endif\n\n";

  OS << "#ifndef INHERITABLE_PARAM_ATTR\n";
  OS << "#define INHERITABLE_PARAM_ATTR(NAME) ATTR(NAME)\n";
  OS << "#endif\n\n";

  OS << "#ifndef LAST_INHERITABLE_PARAM_ATTR\n";
  OS << "#define LAST_INHERITABLE_PARAM_ATTR(NAME)"
        " INHERITABLE_PARAM_ATTR(NAME)\n";
  OS << "#endif\n\n";

  Record *InhClass = Records.getClass("InheritableAttr");
  Record *InhParamClass = Records.getClass("InheritableParamAttr");
  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr"),
                       NonInhAttrs, InhAttrs, InhParamAttrs;
  for (std::vector<Record*>::iterator i = Attrs.begin(), e = Attrs.end();
       i != e; ++i) {
    if (!(*i)->getValueAsBit("ASTNode"))
      continue;
    
    if ((*i)->isSubClassOf(InhParamClass))
      InhParamAttrs.push_back(*i);
    else if ((*i)->isSubClassOf(InhClass))
      InhAttrs.push_back(*i);
    else
      NonInhAttrs.push_back(*i);
  }

  EmitAttrList(OS, "INHERITABLE_PARAM_ATTR", InhParamAttrs);
  EmitAttrList(OS, "INHERITABLE_ATTR", InhAttrs);
  EmitAttrList(OS, "ATTR", NonInhAttrs);

  OS << "#undef LAST_ATTR\n";
  OS << "#undef INHERITABLE_ATTR\n";
  OS << "#undef LAST_INHERITABLE_ATTR\n";
  OS << "#undef LAST_INHERITABLE_PARAM_ATTR\n";
  OS << "#undef ATTR\n";
}

// Emits the code to read an attribute from a precompiled header.
void EmitClangAttrPCHRead(RecordKeeper &Records, raw_ostream &OS) {
  OS << "// This file is generated by TableGen. Do not edit.\n\n";

  Record *InhClass = Records.getClass("InheritableAttr");
  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr"),
                       ArgRecords;
  std::vector<Record*>::iterator i = Attrs.begin(), e = Attrs.end(), ai, ae;
  std::vector<Argument*> Args;
  std::vector<Argument*>::iterator ri, re;

  OS << "  switch (Kind) {\n";
  OS << "  default:\n";
  OS << "    assert(0 && \"Unknown attribute!\");\n";
  OS << "    break;\n";
  for (; i != e; ++i) {
    Record &R = **i;
    if (!R.getValueAsBit("ASTNode"))
      continue;
    
    OS << "  case attr::" << R.getName() << ": {\n";
    if (R.isSubClassOf(InhClass))
      OS << "    bool isInherited = Record[Idx++];\n";
    ArgRecords = R.getValueAsListOfDefs("Args");
    Args.clear();
    for (ai = ArgRecords.begin(), ae = ArgRecords.end(); ai != ae; ++ai) {
      Argument *A = createArgument(**ai, R.getName());
      Args.push_back(A);
      A->writePCHReadDecls(OS);
    }
    OS << "    New = new (Context) " << R.getName() << "Attr(Range, Context";
    for (ri = Args.begin(), re = Args.end(); ri != re; ++ri) {
      OS << ", ";
      (*ri)->writePCHReadArgs(OS);
    }
    OS << ");\n";
    if (R.isSubClassOf(InhClass))
      OS << "    cast<InheritableAttr>(New)->setInherited(isInherited);\n";
    OS << "    break;\n";
    OS << "  }\n";
  }
  OS << "  }\n";
}

// Emits the code to write an attribute to a precompiled header.
void EmitClangAttrPCHWrite(RecordKeeper &Records, raw_ostream &OS) {
  Record *InhClass = Records.getClass("InheritableAttr");
  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr"), Args;
  std::vector<Record*>::iterator i = Attrs.begin(), e = Attrs.end(), ai, ae;

  OS << "  switch (A->getKind()) {\n";
  OS << "  default:\n";
  OS << "    llvm_unreachable(\"Unknown attribute kind!\");\n";
  OS << "    break;\n";
  for (; i != e; ++i) {
    Record &R = **i;
    if (!R.getValueAsBit("ASTNode"))
      continue;
    OS << "  case attr::" << R.getName() << ": {\n";
    Args = R.getValueAsListOfDefs("Args");
    if (R.isSubClassOf(InhClass) || !Args.empty())
      OS << "    const " << R.getName() << "Attr *SA = cast<" << R.getName()
         << "Attr>(A);\n";
    if (R.isSubClassOf(InhClass))
      OS << "    Record.push_back(SA->isInherited());\n";
    for (ai = Args.begin(), ae = Args.end(); ai != ae; ++ai)
      createArgument(**ai, R.getName())->writePCHWrite(OS);
    OS << "    break;\n";
    OS << "  }\n";
  }
  OS << "  }\n";
}

// Emits the list of spellings for attributes.
void EmitClangAttrSpellingList(RecordKeeper &Records, raw_ostream &OS) {
  OS << "// This file is generated by TableGen. Do not edit.\n\n";

  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr");
  
  for (std::vector<Record*>::iterator I = Attrs.begin(), E = Attrs.end(); I != E; ++I) {
    Record &Attr = **I;

    std::vector<Record*> Spellings = Attr.getValueAsListOfDefs("Spellings");

    for (std::vector<Record*>::const_iterator I = Spellings.begin(), E = Spellings.end(); I != E; ++I) {
      OS << ".Case(\"" << (*I)->getValueAsString("Name") << "\", true)\n";
    }
  }

}

// Emits the LateParsed property for attributes.
void EmitClangAttrLateParsedList(RecordKeeper &Records, raw_ostream &OS) {
  OS << "// This file is generated by TableGen. Do not edit.\n\n";

  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr");

  for (std::vector<Record*>::iterator I = Attrs.begin(), E = Attrs.end();
       I != E; ++I) {
    Record &Attr = **I;

    bool LateParsed = Attr.getValueAsBit("LateParsed");

    if (LateParsed) {
      std::vector<Record*> Spellings =
        Attr.getValueAsListOfDefs("Spellings");

      // FIXME: Handle non-GNU attributes
      for (std::vector<Record*>::const_iterator I = Spellings.begin(),
           E = Spellings.end(); I != E; ++I) {
        if ((*I)->getValueAsString("Variety") != "GNU")
          continue;
        OS << ".Case(\"" << (*I)->getValueAsString("Name") << "\", "
           << LateParsed << ")\n";
      }
    }
  }
}

// Emits code to instantiate dependent attributes on templates.
void EmitClangAttrTemplateInstantiate(RecordKeeper &Records, raw_ostream &OS) {
  OS << "// This file is generated by TableGen. Do not edit.\n\n";

  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr");

  OS << "namespace clang {\n"
     << "namespace sema {\n\n"
     << "Attr *instantiateTemplateAttribute(const Attr *At, ASTContext &C, "
     << "Sema &S,\n"
     << "        const MultiLevelTemplateArgumentList &TemplateArgs) {\n"
     << "  switch (At->getKind()) {\n"
     << "    default:\n"
     << "      break;\n";

  for (std::vector<Record*>::iterator I = Attrs.begin(), E = Attrs.end();
       I != E; ++I) {
    Record &R = **I;
    if (!R.getValueAsBit("ASTNode"))
      continue;

    OS << "    case attr::" << R.getName() << ": {\n";
    bool ShouldClone = R.getValueAsBit("Clone");

    if (!ShouldClone) {
      OS << "      return NULL;\n";
      OS << "    }\n";
      continue;
    }

    OS << "      const " << R.getName() << "Attr *A = cast<"
       << R.getName() << "Attr>(At);\n";
    bool TDependent = R.getValueAsBit("TemplateDependent");

    if (!TDependent) {
      OS << "      return A->clone(C);\n";
      OS << "    }\n";
      continue;
    }

    std::vector<Record*> ArgRecords = R.getValueAsListOfDefs("Args");
    std::vector<Argument*> Args;
    std::vector<Argument*>::iterator ai, ae;
    Args.reserve(ArgRecords.size());

    for (std::vector<Record*>::iterator ri = ArgRecords.begin(),
                                        re = ArgRecords.end();
         ri != re; ++ri) {
      Record &ArgRecord = **ri;
      Argument *Arg = createArgument(ArgRecord, R.getName());
      assert(Arg);
      Args.push_back(Arg);
    }
    ae = Args.end();

    for (ai = Args.begin(); ai != ae; ++ai) {
      (*ai)->writeTemplateInstantiation(OS);
    }
    OS << "      return new (C) " << R.getName() << "Attr(A->getLocation(), C";
    for (ai = Args.begin(); ai != ae; ++ai) {
      OS << ", ";
      (*ai)->writeTemplateInstantiationArgs(OS);
    }
    OS << ");\n    }\n";
  }
  OS << "  } // end switch\n"
     << "  llvm_unreachable(\"Unknown attribute!\");\n"
     << "  return 0;\n"
     << "}\n\n"
     << "} // end namespace sema\n"
     << "} // end namespace clang\n";
}

// Emits the list of parsed attributes.
void EmitClangAttrParsedAttrList(RecordKeeper &Records, raw_ostream &OS) {
  OS << "// This file is generated by TableGen. Do not edit.\n\n";
  
  OS << "#ifndef PARSED_ATTR\n";
  OS << "#define PARSED_ATTR(NAME) NAME\n";
  OS << "#endif\n\n";
  
  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr");

  for (std::vector<Record*>::iterator I = Attrs.begin(), E = Attrs.end();
       I != E; ++I) {
    Record &Attr = **I;
    
    bool SemaHandler = Attr.getValueAsBit("SemaHandler");
    bool DistinctSpellings = Attr.getValueAsBit("DistinctSpellings");

    if (SemaHandler) {
      if (DistinctSpellings) {
        std::vector<Record*> Spellings = Attr.getValueAsListOfDefs("Spellings");
        
        for (std::vector<Record*>::const_iterator I = Spellings.begin(),
             E = Spellings.end(); I != E; ++I) {
          std::string AttrName = (*I)->getValueAsString("Name");

          StringRef Spelling = NormalizeAttrName(AttrName);

          OS << "PARSED_ATTR(" << Spelling << ")\n";
        }
      } else {
        StringRef AttrName = Attr.getName();
        AttrName = NormalizeAttrName(AttrName);
        OS << "PARSED_ATTR(" << AttrName << ")\n";
      }
    }
  }
}

// Emits the kind list of parsed attributes
void EmitClangAttrParsedAttrKinds(RecordKeeper &Records, raw_ostream &OS) {
  OS << "// This file is generated by TableGen. Do not edit.\n\n";
  OS << "\n";
  
  std::vector<Record*> Attrs = Records.getAllDerivedDefinitions("Attr");

  std::vector<StringMatcher::StringPair> Matches;
  for (std::vector<Record*>::iterator I = Attrs.begin(), E = Attrs.end();
       I != E; ++I) {
    Record &Attr = **I;
    
    bool SemaHandler = Attr.getValueAsBit("SemaHandler");
    bool Ignored = Attr.getValueAsBit("Ignored");
    bool DistinctSpellings = Attr.getValueAsBit("DistinctSpellings");
    if (SemaHandler || Ignored) {
      std::vector<Record*> Spellings = Attr.getValueAsListOfDefs("Spellings");

      for (std::vector<Record*>::const_iterator I = Spellings.begin(),
           E = Spellings.end(); I != E; ++I) {
        std::string RawSpelling = (*I)->getValueAsString("Name");
        StringRef AttrName = NormalizeAttrName(DistinctSpellings
                                                 ? StringRef(RawSpelling)
                                                 : StringRef(Attr.getName()));

        SmallString<64> Spelling;
        if ((*I)->getValueAsString("Variety") == "CXX11") {
          Spelling += (*I)->getValueAsString("Namespace");
          Spelling += "::";
        }
        Spelling += NormalizeAttrSpelling(RawSpelling);

        if (SemaHandler)
          Matches.push_back(
            StringMatcher::StringPair(
              StringRef(Spelling),
              "return AttributeList::AT_" + AttrName.str() + ";"));
        else
          Matches.push_back(
            StringMatcher::StringPair(
              StringRef(Spelling),
              "return AttributeList::IgnoredAttribute;"));
      }
    }
  }
  
  OS << "static AttributeList::Kind getAttrKind(StringRef Name) {\n";
  StringMatcher("Name", Matches, OS).Emit();
  OS << "return AttributeList::UnknownAttribute;\n"
     << "}\n";
}

} // end namespace clang
