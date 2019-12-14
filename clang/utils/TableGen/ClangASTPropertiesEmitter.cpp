//=== ClangASTPropsEmitter.cpp - Generate Clang AST properties --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits code for working with Clang AST properties.
//
//===----------------------------------------------------------------------===//

#include "ASTTableGen.h"
#include "TableGenBackends.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <cctype>
#include <map>
#include <set>
#include <string>
using namespace llvm;
using namespace clang;
using namespace clang::tblgen;

static StringRef getReaderResultType(TypeNode _) { return "QualType"; }

namespace {

struct ReaderWriterInfo {
  bool IsReader;

  /// The name of the node hierarchy.  Not actually sensitive to IsReader,
  /// but useful to cache here anyway.
  StringRef HierarchyName;

  /// The suffix on classes: Reader/Writer
  StringRef ClassSuffix;

  /// The base name of methods: read/write
  StringRef MethodPrefix;

  /// The name of the property helper member: R/W
  StringRef HelperVariable;

  /// The result type of methods on the class.
  StringRef ResultType;

  template <class NodeClass>
  static ReaderWriterInfo forReader() {
    return ReaderWriterInfo{
      true,
      NodeClass::getASTHierarchyName(),
      "Reader",
      "read",
      "R",
      getReaderResultType(NodeClass())
    };
  }

  template <class NodeClass>
  static ReaderWriterInfo forWriter() {
    return ReaderWriterInfo{
      false,
      NodeClass::getASTHierarchyName(),
      "Writer",
      "write",
      "W",
      "void"
    };
  }
};

struct NodeInfo {
  std::vector<Property> Properties;
  CreationRule Creator = nullptr;
  OverrideRule Override = nullptr;
};

class ASTPropsEmitter {
	raw_ostream &Out;
	RecordKeeper &Records;
	std::map<ASTNode, NodeInfo> NodeInfos;

public:
	ASTPropsEmitter(RecordKeeper &records, raw_ostream &out)
		: Out(out), Records(records) {

		// Find all the properties.
		for (Property property :
           records.getAllDerivedDefinitions(PropertyClassName)) {
			ASTNode node = property.getClass();
			NodeInfos[node].Properties.push_back(property);
		}

    // Find all the creation rules.
    for (CreationRule creationRule :
           records.getAllDerivedDefinitions(CreationRuleClassName)) {
      ASTNode node = creationRule.getClass();

      auto &info = NodeInfos[node];
      if (info.Creator) {
        PrintFatalError(creationRule.getLoc(),
                        "multiple creator rules for \"" + node.getName()
                          + "\"");
      }
      info.Creator = creationRule;
    }

    // Find all the override rules.
    for (OverrideRule overrideRule :
           records.getAllDerivedDefinitions(OverrideRuleClassName)) {
      ASTNode node = overrideRule.getClass();

      auto &info = NodeInfos[node];
      if (info.Override) {
        PrintFatalError(overrideRule.getLoc(),
                        "multiple override rules for \"" + node.getName()
                          + "\"");
      }
      info.Override = overrideRule;
    }

    Validator(*this).validate();
	}

  void visitAllProperties(ASTNode derived, const NodeInfo &derivedInfo,
                          function_ref<void (Property)> visit) {
    std::set<StringRef> ignoredProperties;

    auto overrideRule = derivedInfo.Override;
    if (overrideRule) {
      auto list = overrideRule.getIgnoredProperties();
      ignoredProperties.insert(list.begin(), list.end());
    }

    for (ASTNode node = derived; node; node = node.getBase()) {
      auto it = NodeInfos.find(node);

      // Ignore intermediate nodes that don't add interesting properties.
      if (it == NodeInfos.end()) continue;
      auto &info = it->second;

      for (Property prop : info.Properties) {
        if (ignoredProperties.count(prop.getName()))
          continue;

        visit(prop);
      }
    }
  }

  template <class NodeClass>
  void emitReaderClass() {
    auto info = ReaderWriterInfo::forReader<NodeClass>();
    emitReaderWriterClass<NodeClass>(info);
  }

  template <class NodeClass>
  void emitWriterClass() {
    auto info = ReaderWriterInfo::forWriter<NodeClass>();
    emitReaderWriterClass<NodeClass>(info);
  }

  template <class NodeClass>
  void emitReaderWriterClass(const ReaderWriterInfo &info);

  template <class NodeClass>
  void emitNodeReaderWriterMethod(NodeClass node,
                                  const ReaderWriterInfo &info);

  void emitReadOfProperty(Property property);
  void emitWriteOfProperty(Property property);

private:
  class Validator {
    const ASTPropsEmitter &Emitter;
    std::set<ASTNode> ValidatedNodes;

  public:
    Validator(const ASTPropsEmitter &emitter) : Emitter(emitter) {}
    void validate();

  private:
    void validateNode(ASTNode node, const NodeInfo &nodeInfo);
    void validateType(PropertyType type, WrappedRecord context);
  };
};

} // end anonymous namespace

void ASTPropsEmitter::Validator::validate() {
  for (auto &entry : Emitter.NodeInfos) {
    validateNode(entry.first, entry.second);
  }

  if (ErrorsPrinted > 0) {
    PrintFatalError("property validation failed");
  }
}

void ASTPropsEmitter::Validator::validateNode(ASTNode node,
                                              const NodeInfo &nodeInfo) {
  if (!ValidatedNodes.insert(node).second) return;

  // A map from property name to property.
  std::map<StringRef, Property> allProperties;

  // Walk the hierarchy, ignoring nodes that don't declare anything
  // interesting.
  for (auto base = node; base; base = base.getBase()) {
    auto it = Emitter.NodeInfos.find(base);
    if (it == Emitter.NodeInfos.end()) continue;

    auto &baseInfo = it->second;
    for (Property property : baseInfo.Properties) {
      validateType(property.getType(), property);

      auto result = allProperties.insert(
                      std::make_pair(property.getName(), property));

      // Diagnose non-unique properties.
      if (!result.second) {
        // The existing property is more likely to be associated with a
        // derived node, so use it as the error.
        Property existingProperty = result.first->second;
        PrintError(existingProperty.getLoc(),
                   "multiple properties named \"" + property.getName()
                      + "\" in hierarchy of " + node.getName());
        PrintNote(property.getLoc(), "existing property");
      }
    }
  }
}

void ASTPropsEmitter::Validator::validateType(PropertyType type,
                                              WrappedRecord context) {
  if (!type.isGenericSpecialization()) {
    if (type.getCXXTypeName() == "") {
      PrintError(type.getLoc(),
                 "type is not generic but has no C++ type name");
      if (context) PrintNote(context.getLoc(), "type used here");
    }
  } else if (auto eltType = type.getArrayElementType()) {
    validateType(eltType, context);
  } else if (auto valueType = type.getOptionalElementType()) {
    validateType(valueType, context);

    if (valueType.getPackOptionalCode().empty()) {
      PrintError(valueType.getLoc(),
                 "type doesn't provide optional-packing code");
      if (context) PrintNote(context.getLoc(), "type used here");
    } else if (valueType.getUnpackOptionalCode().empty()) {
      PrintError(valueType.getLoc(),
                 "type doesn't provide optional-unpacking code");
      if (context) PrintNote(context.getLoc(), "type used here");
    }
  } else {
    PrintError(type.getLoc(), "unknown generic property type");
    if (context) PrintNote(context.getLoc(), "type used here");
  }
}

/****************************************************************************/
/**************************** AST READER/WRITERS ****************************/
/****************************************************************************/

template <class NodeClass>
void ASTPropsEmitter::emitReaderWriterClass(const ReaderWriterInfo &info) {
  StringRef suffix = info.ClassSuffix;
  StringRef var = info.HelperVariable;

  // Enter the class declaration.
  Out << "template <class Property" << suffix << ">\n"
         "class Abstract" << info.HierarchyName << suffix << " {\n"
         "public:\n"
         "  Property" << suffix << " &" << var << ";\n\n";

  // Emit the constructor.
  Out << "  Abstract" << info.HierarchyName << suffix
                      << "(Property" << suffix << " &" << var << ") : "
                      << var << "(" << var << ") {}\n\n";

  // Emit a method that dispatches on a kind to the appropriate node-specific
  // method.
  Out << "  " << info.ResultType << " " << info.MethodPrefix << "(";
  if (info.IsReader)
    Out       << NodeClass::getASTIdTypeName() << " kind";
  else
    Out       << "const " << info.HierarchyName << " *node";
  Out         << ") {\n"
         "    switch (";
  if (info.IsReader)
    Out         << "kind";
  else
    Out         << "node->" << NodeClass::getASTIdAccessorName() << "()";
  Out           << ") {\n";
  visitASTNodeHierarchy<NodeClass>(Records, [&](NodeClass node, NodeClass _) {
    if (node.isAbstract()) return;
    Out << "    case " << info.HierarchyName << "::" << node.getId() << ":\n"
           "      return " << info.MethodPrefix << node.getClassName() << "(";
    if (!info.IsReader)
      Out                  << "static_cast<const " << node.getClassName()
                           << " *>(node)";
    Out                    << ");\n";
  });
  Out << "    }\n"
         "    llvm_unreachable(\"bad kind\");\n"
         "  }\n\n";

  // Emit node-specific methods for all the concrete nodes.
  visitASTNodeHierarchy<NodeClass>(Records,
                                   [&](NodeClass node, NodeClass base) {
    if (node.isAbstract()) return;
    emitNodeReaderWriterMethod(node, info);
  });

  // Finish the class.
  Out << "};\n\n";
}

/// Emit a reader method for the given concrete AST node class.
template <class NodeClass>
void ASTPropsEmitter::emitNodeReaderWriterMethod(NodeClass node,
                                           const ReaderWriterInfo &info) {
  // Declare and start the method.
  Out << "  " << info.ResultType << " "
              << info.MethodPrefix << node.getClassName() << "(";
  if (!info.IsReader)
    Out <<       "const " << node.getClassName() << " *node";
  Out <<         ") {\n";
  if (info.IsReader)
    Out << "    auto &ctx = " << info.HelperVariable << ".getASTContext();\n";

  // Find the information for this node.
  auto it = NodeInfos.find(node);
  if (it == NodeInfos.end())
    PrintFatalError(node.getLoc(),
                    "no information about how to deserialize \""
                      + node.getName() + "\"");
  auto &nodeInfo = it->second;

  StringRef creationCode;
  if (info.IsReader) {
    // We should have a creation rule.
    if (!nodeInfo.Creator)
      PrintFatalError(node.getLoc(),
                      "no " CreationRuleClassName " for \""
                        + node.getName() + "\"");

    creationCode = nodeInfo.Creator.getCreationCode();
  }

  // Emit code to read all the properties.
  visitAllProperties(node, nodeInfo, [&](Property prop) {
    // Verify that the creation code refers to this property.
    if (info.IsReader && creationCode.find(prop.getName()) == StringRef::npos)
      PrintFatalError(nodeInfo.Creator.getLoc(),
                      "creation code for " + node.getName()
                        + " doesn't refer to property \""
                        + prop.getName() + "\"");

    // Emit code to read or write this property.
    if (info.IsReader)
      emitReadOfProperty(prop);
    else
      emitWriteOfProperty(prop);
  });

  // Emit the final creation code.
  if (info.IsReader)
    Out << "    " << creationCode << "\n";

  // Finish the method declaration.
  Out << "  }\n\n";
}

static void emitBasicReaderWriterMethodSuffix(raw_ostream &out,
                                              PropertyType type,
                                              bool isForRead) {
  if (!type.isGenericSpecialization()) {
    out << type.getAbstractTypeName();
  } else if (auto eltType = type.getArrayElementType()) {
    out << "Array";
    // We only include an explicit template argument for reads so that
    // we don't cause spurious const mismatches.
    if (isForRead) {
      out << "<";
      eltType.emitCXXValueTypeName(isForRead, out);
      out << ">";
    }
  } else if (auto valueType = type.getOptionalElementType()) {
    out << "Optional";
    // We only include an explicit template argument for reads so that
    // we don't cause spurious const mismatches.
    if (isForRead) {
      out << "<";
      valueType.emitCXXValueTypeName(isForRead, out);
      out << ">";
    }
  } else {
    PrintFatalError(type.getLoc(), "unexpected generic property type");
  }
}

/// Emit code to read the given property in a node-reader method.
void ASTPropsEmitter::emitReadOfProperty(Property property) {
  PropertyType type = property.getType();
  auto name = property.getName();

  // Declare all the necessary buffers.
  auto bufferTypes = type.getBufferElementTypes();
  for (size_t i = 0, e = bufferTypes.size(); i != e; ++i) {
    Out << "    llvm::SmallVector<";
    PropertyType(bufferTypes[i]).emitCXXValueTypeName(/*for read*/ true, Out);
    Out << ", 8> " << name << "_buffer_" << i << ";\n";
  }

  //   T prop = R.find("prop").read##ValueType(buffers...);
  // We intentionally ignore shouldPassByReference here: we're going to
  // get a pr-value back from read(), and we should be able to forward
  // that in the creation rule.
  Out << "    ";
  type.emitCXXValueTypeName(true, Out);
  Out << " " << name << " = R.find(\"" << name << "\")."
      << (type.isGenericSpecialization() ? "template " : "") << "read";
  emitBasicReaderWriterMethodSuffix(Out, type, /*for read*/ true);
  Out << "(";
  for (size_t i = 0, e = bufferTypes.size(); i != e; ++i) {
    Out << (i > 0 ? ", " : "") << name << "_buffer_" << i;
  }
  Out << ");\n";
}

/// Emit code to write the given property in a node-writer method.
void ASTPropsEmitter::emitWriteOfProperty(Property property) {
  // Focus down to the property:
  //   W.find("prop").write##ValueType(value);
  Out << "    W.find(\"" << property.getName() << "\").write";
  emitBasicReaderWriterMethodSuffix(Out, property.getType(),
                                    /*for read*/ false);
  Out << "(" << property.getReadCode() << ");\n";
}

/// Emit an .inc file that defines the AbstractFooReader class
/// for the given AST class hierarchy.
template <class NodeClass>
static void emitASTReader(RecordKeeper &records, raw_ostream &out,
                          StringRef description) {
  emitSourceFileHeader(description, out);

  ASTPropsEmitter(records, out).emitReaderClass<NodeClass>();
}

void clang::EmitClangTypeReader(RecordKeeper &records, raw_ostream &out) {
  emitASTReader<TypeNode>(records, out, "A CRTP reader for Clang Type nodes");
}

/// Emit an .inc file that defines the AbstractFooWriter class
/// for the given AST class hierarchy.
template <class NodeClass>
static void emitASTWriter(RecordKeeper &records, raw_ostream &out,
                          StringRef description) {
  emitSourceFileHeader(description, out);

  ASTPropsEmitter(records, out).emitWriterClass<NodeClass>();
}

void clang::EmitClangTypeWriter(RecordKeeper &records, raw_ostream &out) {
  emitASTWriter<TypeNode>(records, out, "A CRTP writer for Clang Type nodes");
}

/****************************************************************************/
/*************************** BASIC READER/WRITERS ***************************/
/****************************************************************************/

static void emitDispatcherTemplate(ArrayRef<Record*> types, raw_ostream &out,
                                   const ReaderWriterInfo &info) {
  // Declare the {Read,Write}Dispatcher template.
  StringRef dispatcherPrefix = (info.IsReader ? "Read" : "Write");
  out << "template <class ValueType>\n"
         "struct " << dispatcherPrefix << "Dispatcher;\n";

  // Declare a specific specialization of the dispatcher template.
  auto declareSpecialization =
    [&](StringRef specializationParameters,
        const Twine &cxxTypeName,
        StringRef methodSuffix) {
    StringRef var = info.HelperVariable;
    out << "template " << specializationParameters << "\n"
           "struct " << dispatcherPrefix << "Dispatcher<"
                     << cxxTypeName << "> {\n";
    out << "  template <class Basic" << info.ClassSuffix << ", class... Args>\n"
           "  static " << (info.IsReader ? cxxTypeName : "void") << " "
                       << info.MethodPrefix
                       << "(Basic" << info.ClassSuffix << " &" << var
                       << ", Args &&... args) {\n"
           "    return " << var << "."
                         << info.MethodPrefix << methodSuffix
                         << "(std::forward<Args>(args)...);\n"
           "  }\n"
           "};\n";
  };

  // Declare explicit specializations for each of the concrete types.
  for (PropertyType type : types) {
    declareSpecialization("<>",
                          type.getCXXTypeName(),
                          type.getAbstractTypeName());
    // Also declare a specialization for the const type when appropriate.
    if (!info.IsReader && type.isConstWhenWriting()) {
      declareSpecialization("<>",
                            "const " + type.getCXXTypeName(),
                            type.getAbstractTypeName());
    }
  }
  // Declare partial specializations for ArrayRef and Optional.
  declareSpecialization("<class T>",
                        "llvm::ArrayRef<T>",
                        "Array");
  declareSpecialization("<class T>",
                        "llvm::Optional<T>",
                        "Optional");
  out << "\n";
}

static void emitPackUnpackOptionalTemplate(ArrayRef<Record*> types,
                                           raw_ostream &out,
                                           const ReaderWriterInfo &info) {
  StringRef classPrefix = (info.IsReader ? "Unpack" : "Pack");
  StringRef methodName = (info.IsReader ? "unpack" : "pack");

  // Declare the {Pack,Unpack}OptionalValue template.
  out << "template <class ValueType>\n"
         "struct " << classPrefix << "OptionalValue;\n";

  auto declareSpecialization = [&](const Twine &typeName,
                                   StringRef code) {
    out << "template <>\n"
           "struct " << classPrefix << "OptionalValue<" << typeName << "> {\n"
           "  static " << (info.IsReader ? "Optional<" : "") << typeName
                       << (info.IsReader ? "> " : " ") << methodName << "("
                       << (info.IsReader ? "" : "Optional<") << typeName
                       << (info.IsReader ? "" : ">") << " value) {\n"
           "    return " << code << ";\n"
           "  }\n"
           "};\n";
  };

  for (PropertyType type : types) {
    StringRef code = (info.IsReader ? type.getUnpackOptionalCode()
                                    : type.getPackOptionalCode());
    if (code.empty()) continue;

    StringRef typeName = type.getCXXTypeName();
    declareSpecialization(typeName, code);
    if (type.isConstWhenWriting() && !info.IsReader)
      declareSpecialization("const " + typeName, code);
  }
  out << "\n";
}

static void emitBasicReaderWriterTemplate(ArrayRef<Record*> types,
                                          raw_ostream &out,
                                          const ReaderWriterInfo &info) {
  // Emit the Basic{Reader,Writer}Base template.
  out << "template <class Impl>\n"
         "class Basic" << info.ClassSuffix << "Base {\n";
  if (info.IsReader)
    out << "  ASTContext &C;\n";
  out << "protected:\n"
         "  Basic" << info.ClassSuffix << "Base"
                   << (info.IsReader ? "(ASTContext &ctx) : C(ctx)" : "()")
                   << " {}\n"
         "public:\n";
  if (info.IsReader)
    out << "  ASTContext &getASTContext() { return C; }\n";
  out << "  Impl &asImpl() { return static_cast<Impl&>(*this); }\n";

  auto enterReaderWriterMethod = [&](StringRef cxxTypeName,
                                     StringRef abstractTypeName,
                                     bool shouldPassByReference,
                                     bool constWhenWriting) {
    out << "  " << (info.IsReader ? cxxTypeName : "void")
                << " " << info.MethodPrefix << abstractTypeName << "(";
    if (!info.IsReader)
      out       << (shouldPassByReference || constWhenWriting ? "const " : "")
                << cxxTypeName
                << (shouldPassByReference ? " &" : "") << " value";
    out         << ") {\n";
  };

  // Emit {read,write}ValueType methods for all the enum and subclass types
  // that default to using the integer/base-class implementations.
  for (PropertyType type : types) {
    if (type.isEnum()) {
      enterReaderWriterMethod(type.getCXXTypeName(),
                              type.getAbstractTypeName(),
                              /*pass by reference*/ false,
                              /*const when writing*/ false);
      if (info.IsReader)
        out << "    return " << type.getCXXTypeName()
                             << "(asImpl().readUInt32());\n";
      else
        out << "    asImpl().writeUInt32(uint32_t(value));\n";
      out << "  }\n";
    } else if (PropertyType superclass = type.getSuperclassType()) {
      enterReaderWriterMethod(type.getCXXTypeName(),
                              type.getAbstractTypeName(),
                              /*pass by reference*/ false,
                              /*const when writing*/ type.isConstWhenWriting());
      if (info.IsReader)
        out << "    return cast_or_null<" << type.getSubclassClassName()
                                          << ">(asImpl().read"
                                          << superclass.getAbstractTypeName()
                                          << "());\n";
      else
        out << "    asImpl().write" << superclass.getAbstractTypeName()
                                    << "(value);\n";
      out << "  }\n";
    } else {
      // The other types can't be handled as trivially.
    }
  }
  out << "};\n\n";
}

static void emitBasicReaderWriterFile(RecordKeeper &records, raw_ostream &out,
                                      const ReaderWriterInfo &info) {
  auto types = records.getAllDerivedDefinitions(PropertyTypeClassName);

  emitDispatcherTemplate(types, out, info);
  emitPackUnpackOptionalTemplate(types, out, info);
  emitBasicReaderWriterTemplate(types, out, info);
}

/// Emit an .inc file that defines some helper classes for reading
/// basic values.
void clang::EmitClangBasicReader(RecordKeeper &records, raw_ostream &out) {
  emitSourceFileHeader("Helper classes for BasicReaders", out);

  // Use any property, we won't be using those properties.
  auto info = ReaderWriterInfo::forReader<TypeNode>();
  emitBasicReaderWriterFile(records, out, info);
}

/// Emit an .inc file that defines some helper classes for writing
/// basic values.
void clang::EmitClangBasicWriter(RecordKeeper &records, raw_ostream &out) {
  emitSourceFileHeader("Helper classes for BasicWriters", out);

  // Use any property, we won't be using those properties.
  auto info = ReaderWriterInfo::forWriter<TypeNode>();
  emitBasicReaderWriterFile(records, out, info);
}
