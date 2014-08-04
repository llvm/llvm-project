//===--- ASTDiagnostic.cpp - Diagnostic Printing Hooks for AST Nodes ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a diagnostic formatting hook for AST elements.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

// Returns a desugared version of the QualType, and marks ShouldAKA as true
// whenever we remove significant sugar from the type.
static QualType Desugar(ASTContext &Context, QualType QT, bool &ShouldAKA) {
  QualifierCollector QC;

  while (true) {
    const Type *Ty = QC.strip(QT);

    // Don't aka just because we saw an elaborated type...
    if (const ElaboratedType *ET = dyn_cast<ElaboratedType>(Ty)) {
      QT = ET->desugar();
      continue;
    }
    // ... or a paren type ...
    if (const ParenType *PT = dyn_cast<ParenType>(Ty)) {
      QT = PT->desugar();
      continue;
    }
    // ...or a substituted template type parameter ...
    if (const SubstTemplateTypeParmType *ST =
          dyn_cast<SubstTemplateTypeParmType>(Ty)) {
      QT = ST->desugar();
      continue;
    }
    // ...or an attributed type...
    if (const AttributedType *AT = dyn_cast<AttributedType>(Ty)) {
      QT = AT->desugar();
      continue;
    }
    // ...or an adjusted type...
    if (const AdjustedType *AT = dyn_cast<AdjustedType>(Ty)) {
      QT = AT->desugar();
      continue;
    }
    // ... or an auto type.
    if (const AutoType *AT = dyn_cast<AutoType>(Ty)) {
      if (!AT->isSugared())
        break;
      QT = AT->desugar();
      continue;
    }

    // Don't desugar template specializations, unless it's an alias template.
    if (const TemplateSpecializationType *TST
          = dyn_cast<TemplateSpecializationType>(Ty))
      if (!TST->isTypeAlias())
        break;

    // Don't desugar magic Objective-C types.
    if (QualType(Ty,0) == Context.getObjCIdType() ||
        QualType(Ty,0) == Context.getObjCClassType() ||
        QualType(Ty,0) == Context.getObjCSelType() ||
        QualType(Ty,0) == Context.getObjCProtoType())
      break;

    // Don't desugar va_list.
    if (QualType(Ty,0) == Context.getBuiltinVaListType())
      break;

    // Otherwise, do a single-step desugar.
    QualType Underlying;
    bool IsSugar = false;
    switch (Ty->getTypeClass()) {
#define ABSTRACT_TYPE(Class, Base)
#define TYPE(Class, Base) \
case Type::Class: { \
const Class##Type *CTy = cast<Class##Type>(Ty); \
if (CTy->isSugared()) { \
IsSugar = true; \
Underlying = CTy->desugar(); \
} \
break; \
}
#include "clang/AST/TypeNodes.def"
    }

    // If it wasn't sugared, we're done.
    if (!IsSugar)
      break;

    // If the desugared type is a vector type, we don't want to expand
    // it, it will turn into an attribute mess. People want their "vec4".
    if (isa<VectorType>(Underlying))
      break;

    // Don't desugar through the primary typedef of an anonymous type.
    if (const TagType *UTT = Underlying->getAs<TagType>())
      if (const TypedefType *QTT = dyn_cast<TypedefType>(QT))
        if (UTT->getDecl()->getTypedefNameForAnonDecl() == QTT->getDecl())
          break;

    // Record that we actually looked through an opaque type here.
    ShouldAKA = true;
    QT = Underlying;
  }

  // If we have a pointer-like type, desugar the pointee as well.
  // FIXME: Handle other pointer-like types.
  if (const PointerType *Ty = QT->getAs<PointerType>()) {
    QT = Context.getPointerType(Desugar(Context, Ty->getPointeeType(),
                                        ShouldAKA));
  } else if (const LValueReferenceType *Ty = QT->getAs<LValueReferenceType>()) {
    QT = Context.getLValueReferenceType(Desugar(Context, Ty->getPointeeType(),
                                                ShouldAKA));
  } else if (const RValueReferenceType *Ty = QT->getAs<RValueReferenceType>()) {
    QT = Context.getRValueReferenceType(Desugar(Context, Ty->getPointeeType(),
                                                ShouldAKA));
  }

  return QC.apply(Context, QT);
}

/// \brief Convert the given type to a string suitable for printing as part of 
/// a diagnostic.
///
/// There are four main criteria when determining whether we should have an
/// a.k.a. clause when pretty-printing a type:
///
/// 1) Some types provide very minimal sugar that doesn't impede the
///    user's understanding --- for example, elaborated type
///    specifiers.  If this is all the sugar we see, we don't want an
///    a.k.a. clause.
/// 2) Some types are technically sugared but are much more familiar
///    when seen in their sugared form --- for example, va_list,
///    vector types, and the magic Objective C types.  We don't
///    want to desugar these, even if we do produce an a.k.a. clause.
/// 3) Some types may have already been desugared previously in this diagnostic.
///    if this is the case, doing another "aka" would just be clutter.
/// 4) Two different types within the same diagnostic have the same output
///    string.  In this case, force an a.k.a with the desugared type when
///    doing so will provide additional information.
///
/// \param Context the context in which the type was allocated
/// \param Ty the type to print
/// \param QualTypeVals pointer values to QualTypes which are used in the
/// diagnostic message
static std::string
ConvertTypeToDiagnosticString(ASTContext &Context, QualType Ty,
                            ArrayRef<DiagnosticsEngine::ArgumentValue> PrevArgs,
                            ArrayRef<intptr_t> QualTypeVals) {
  // FIXME: Playing with std::string is really slow.
  bool ForceAKA = false;
  QualType CanTy = Ty.getCanonicalType();
  std::string S = Ty.getAsString(Context.getPrintingPolicy());
  std::string CanS = CanTy.getAsString(Context.getPrintingPolicy());

  for (unsigned I = 0, E = QualTypeVals.size(); I != E; ++I) {
    QualType CompareTy =
        QualType::getFromOpaquePtr(reinterpret_cast<void*>(QualTypeVals[I]));
    if (CompareTy.isNull())
      continue;
    if (CompareTy == Ty)
      continue;  // Same types
    QualType CompareCanTy = CompareTy.getCanonicalType();
    if (CompareCanTy == CanTy)
      continue;  // Same canonical types
    std::string CompareS = CompareTy.getAsString(Context.getPrintingPolicy());
    bool aka;
    QualType CompareDesugar = Desugar(Context, CompareTy, aka);
    std::string CompareDesugarStr =
        CompareDesugar.getAsString(Context.getPrintingPolicy());
    if (CompareS != S && CompareDesugarStr != S)
      continue;  // The type string is different than the comparison string
                 // and the desugared comparison string.
    std::string CompareCanS =
        CompareCanTy.getAsString(Context.getPrintingPolicy());
    
    if (CompareCanS == CanS)
      continue;  // No new info from canonical type

    ForceAKA = true;
    break;
  }

  // Check to see if we already desugared this type in this
  // diagnostic.  If so, don't do it again.
  bool Repeated = false;
  for (unsigned i = 0, e = PrevArgs.size(); i != e; ++i) {
    // TODO: Handle ak_declcontext case.
    if (PrevArgs[i].first == DiagnosticsEngine::ak_qualtype) {
      void *Ptr = (void*)PrevArgs[i].second;
      QualType PrevTy(QualType::getFromOpaquePtr(Ptr));
      if (PrevTy == Ty) {
        Repeated = true;
        break;
      }
    }
  }

  // Consider producing an a.k.a. clause if removing all the direct
  // sugar gives us something "significantly different".
  if (!Repeated) {
    bool ShouldAKA = false;
    QualType DesugaredTy = Desugar(Context, Ty, ShouldAKA);
    if (ShouldAKA || ForceAKA) {
      if (DesugaredTy == Ty) {
        DesugaredTy = Ty.getCanonicalType();
      }
      std::string akaStr = DesugaredTy.getAsString(Context.getPrintingPolicy());
      if (akaStr != S) {
        S = "'" + S + "' (aka '" + akaStr + "')";
        return S;
      }
    }

    // Give some additional info on vector types. These are either not desugared
    // or displaying complex __attribute__ expressions so add details of the
    // type and element count.
    if (Ty->isVectorType()) {
      const VectorType *VTy = Ty->getAs<VectorType>();
      std::string DecoratedString;
      llvm::raw_string_ostream OS(DecoratedString);
      const char *Values = VTy->getNumElements() > 1 ? "values" : "value";
      OS << "'" << S << "' (vector of " << VTy->getNumElements() << " '"
         << VTy->getElementType().getAsString(Context.getPrintingPolicy())
         << "' " << Values << ")";
      return OS.str();
    }
  }

  S = "'" + S + "'";
  return S;
}

static bool FormatTemplateTypeDiff(ASTContext &Context, QualType FromType,
                                   QualType ToType, bool PrintTree,
                                   bool PrintFromType, bool ElideType,
                                   bool ShowColors, raw_ostream &OS);

void clang::FormatASTNodeDiagnosticArgument(
    DiagnosticsEngine::ArgumentKind Kind,
    intptr_t Val,
    StringRef Modifier,
    StringRef Argument,
    ArrayRef<DiagnosticsEngine::ArgumentValue> PrevArgs,
    SmallVectorImpl<char> &Output,
    void *Cookie,
    ArrayRef<intptr_t> QualTypeVals) {
  ASTContext &Context = *static_cast<ASTContext*>(Cookie);
  
  size_t OldEnd = Output.size();
  llvm::raw_svector_ostream OS(Output);
  bool NeedQuotes = true;
  
  switch (Kind) {
    default: llvm_unreachable("unknown ArgumentKind");
    case DiagnosticsEngine::ak_qualtype_pair: {
      TemplateDiffTypes &TDT = *reinterpret_cast<TemplateDiffTypes*>(Val);
      QualType FromType =
          QualType::getFromOpaquePtr(reinterpret_cast<void*>(TDT.FromType));
      QualType ToType =
          QualType::getFromOpaquePtr(reinterpret_cast<void*>(TDT.ToType));

      if (FormatTemplateTypeDiff(Context, FromType, ToType, TDT.PrintTree,
                                 TDT.PrintFromType, TDT.ElideType,
                                 TDT.ShowColors, OS)) {
        NeedQuotes = !TDT.PrintTree;
        TDT.TemplateDiffUsed = true;
        break;
      }

      // Don't fall-back during tree printing.  The caller will handle
      // this case.
      if (TDT.PrintTree)
        return;

      // Attempting to do a template diff on non-templates.  Set the variables
      // and continue with regular type printing of the appropriate type.
      Val = TDT.PrintFromType ? TDT.FromType : TDT.ToType;
      Modifier = StringRef();
      Argument = StringRef();
      // Fall through
    }
    case DiagnosticsEngine::ak_qualtype: {
      assert(Modifier.empty() && Argument.empty() &&
             "Invalid modifier for QualType argument");
      
      QualType Ty(QualType::getFromOpaquePtr(reinterpret_cast<void*>(Val)));
      OS << ConvertTypeToDiagnosticString(Context, Ty, PrevArgs, QualTypeVals);
      NeedQuotes = false;
      break;
    }
    case DiagnosticsEngine::ak_declarationname: {
      if (Modifier == "objcclass" && Argument.empty())
        OS << '+';
      else if (Modifier == "objcinstance" && Argument.empty())
        OS << '-';
      else
        assert(Modifier.empty() && Argument.empty() &&
               "Invalid modifier for DeclarationName argument");

      OS << DeclarationName::getFromOpaqueInteger(Val);
      break;
    }
    case DiagnosticsEngine::ak_nameddecl: {
      bool Qualified;
      if (Modifier == "q" && Argument.empty())
        Qualified = true;
      else {
        assert(Modifier.empty() && Argument.empty() &&
               "Invalid modifier for NamedDecl* argument");
        Qualified = false;
      }
      const NamedDecl *ND = reinterpret_cast<const NamedDecl*>(Val);
      ND->getNameForDiagnostic(OS, Context.getPrintingPolicy(), Qualified);
      break;
    }
    case DiagnosticsEngine::ak_nestednamespec: {
      NestedNameSpecifier *NNS = reinterpret_cast<NestedNameSpecifier*>(Val);
      NNS->print(OS, Context.getPrintingPolicy());
      NeedQuotes = false;
      break;
    }
    case DiagnosticsEngine::ak_declcontext: {
      DeclContext *DC = reinterpret_cast<DeclContext *> (Val);
      assert(DC && "Should never have a null declaration context");
      NeedQuotes = false;

      if (DC->isTranslationUnit()) {
        // FIXME: Get these strings from some localized place
        if (Context.getLangOpts().CPlusPlus)
          OS << "the global namespace";
        else
          OS << "the global scope";
      } else if (TypeDecl *Type = dyn_cast<TypeDecl>(DC)) {
        OS << ConvertTypeToDiagnosticString(Context,
                                            Context.getTypeDeclType(Type),
                                            PrevArgs, QualTypeVals);
      } else {
        // FIXME: Get these strings from some localized place
        if (isa<BlockDecl>(DC)) {
          OS << "block literal";
          break;
        }
        if (isLambdaCallOperator(DC)) {
          OS << "lambda expression";
          break;
        }
        NamedDecl *ND = cast<NamedDecl>(DC);
        if (isa<NamespaceDecl>(ND))
          OS << "namespace ";
        else if (isa<ObjCMethodDecl>(ND))
          OS << "method ";
        else if (isa<FunctionDecl>(ND))
          OS << "function ";

        OS << '\'';
        ND->getNameForDiagnostic(OS, Context.getPrintingPolicy(), true);
        OS << '\'';
      }
      break;
    }
    case DiagnosticsEngine::ak_attr: {
      const Attr *At = reinterpret_cast<Attr *>(Val);
      assert(At && "Received null Attr object!");
      OS << '\'' << At->getSpelling() << '\'';
      NeedQuotes = false;
      break;
    }

  }

  OS.flush();

  if (NeedQuotes) {
    Output.insert(Output.begin()+OldEnd, '\'');
    Output.push_back('\'');
  }
}

/// TemplateDiff - A class that constructs a pretty string for a pair of
/// QualTypes.  For the pair of types, a diff tree will be created containing
/// all the information about the templates and template arguments.  Afterwards,
/// the tree is transformed to a string according to the options passed in.
namespace {
class TemplateDiff {
  /// Context - The ASTContext which is used for comparing template arguments.
  ASTContext &Context;

  /// Policy - Used during expression printing.
  PrintingPolicy Policy;

  /// ElideType - Option to elide identical types.
  bool ElideType;

  /// PrintTree - Format output string as a tree.
  bool PrintTree;

  /// ShowColor - Diagnostics support color, so bolding will be used.
  bool ShowColor;

  /// FromType - When single type printing is selected, this is the type to be
  /// be printed.  When tree printing is selected, this type will show up first
  /// in the tree.
  QualType FromType;

  /// ToType - The type that FromType is compared to.  Only in tree printing
  /// will this type be outputed.
  QualType ToType;

  /// OS - The stream used to construct the output strings.
  raw_ostream &OS;

  /// IsBold - Keeps track of the bold formatting for the output string.
  bool IsBold;

  /// DiffTree - A tree representation the differences between two types.
  class DiffTree {
  public:
    /// DiffKind - The difference in a DiffNode and which fields are used.
    enum DiffKind {
      /// Incomplete or invalid node.
      Invalid,
      /// Another level of templates, uses TemplateDecl and Qualifiers
      Template,
      /// Type difference, uses QualType
      Type,
      /// Expression difference, uses Expr
      Expression,
      /// Template argument difference, uses TemplateDecl
      TemplateTemplate,
      /// Integer difference, uses APSInt and Expr
      Integer,
      /// Declaration difference, uses ValueDecl
      Declaration
    };
  private:
    /// DiffNode - The root node stores the original type.  Each child node
    /// stores template arguments of their parents.  For templated types, the
    /// template decl is also stored.
    struct DiffNode {
      DiffKind Kind;

      /// NextNode - The index of the next sibling node or 0.
      unsigned NextNode;

      /// ChildNode - The index of the first child node or 0.
      unsigned ChildNode;

      /// ParentNode - The index of the parent node.
      unsigned ParentNode;

      /// FromType, ToType - The type arguments.
      QualType FromType, ToType;

      /// FromExpr, ToExpr - The expression arguments.
      Expr *FromExpr, *ToExpr;

      /// FromTD, ToTD - The template decl for template template
      /// arguments or the type arguments that are templates.
      TemplateDecl *FromTD, *ToTD;

      /// FromQual, ToQual - Qualifiers for template types.
      Qualifiers FromQual, ToQual;

      /// FromInt, ToInt - APSInt's for integral arguments.
      llvm::APSInt FromInt, ToInt;

      /// IsValidFromInt, IsValidToInt - Whether the APSInt's are valid.
      bool IsValidFromInt, IsValidToInt;

      /// FromValueDecl, ToValueDecl - Whether the argument is a decl.
      ValueDecl *FromValueDecl, *ToValueDecl;

      /// FromAddressOf, ToAddressOf - Whether the ValueDecl needs an address of
      /// operator before it.
      bool FromAddressOf, ToAddressOf;

      /// FromDefault, ToDefault - Whether the argument is a default argument.
      bool FromDefault, ToDefault;

      /// Same - Whether the two arguments evaluate to the same value.
      bool Same;

      DiffNode(unsigned ParentNode = 0)
        : Kind(Invalid), NextNode(0), ChildNode(0), ParentNode(ParentNode),
          FromType(), ToType(), FromExpr(nullptr), ToExpr(nullptr),
          FromTD(nullptr), ToTD(nullptr), IsValidFromInt(false),
          IsValidToInt(false), FromValueDecl(nullptr), ToValueDecl(nullptr),
          FromAddressOf(false), ToAddressOf(false), FromDefault(false),
          ToDefault(false), Same(false) {}
    };

    /// FlatTree - A flattened tree used to store the DiffNodes.
    SmallVector<DiffNode, 16> FlatTree;

    /// CurrentNode - The index of the current node being used.
    unsigned CurrentNode;

    /// NextFreeNode - The index of the next unused node.  Used when creating
    /// child nodes.
    unsigned NextFreeNode;

    /// ReadNode - The index of the current node being read.
    unsigned ReadNode;
  
  public:
    DiffTree() :
        CurrentNode(0), NextFreeNode(1) {
      FlatTree.push_back(DiffNode());
    }

    // Node writing functions.
    /// SetNode - Sets FromTD and ToTD of the current node.
    void SetNode(TemplateDecl *FromTD, TemplateDecl *ToTD) {
      FlatTree[CurrentNode].FromTD = FromTD;
      FlatTree[CurrentNode].ToTD = ToTD;
    }

    /// SetNode - Sets FromType and ToType of the current node.
    void SetNode(QualType FromType, QualType ToType) {
      FlatTree[CurrentNode].FromType = FromType;
      FlatTree[CurrentNode].ToType = ToType;
    }

    /// SetNode - Set FromExpr and ToExpr of the current node.
    void SetNode(Expr *FromExpr, Expr *ToExpr) {
      FlatTree[CurrentNode].FromExpr = FromExpr;
      FlatTree[CurrentNode].ToExpr = ToExpr;
    }

    /// SetNode - Set FromInt and ToInt of the current node.
    void SetNode(llvm::APSInt FromInt, llvm::APSInt ToInt,
                 bool IsValidFromInt, bool IsValidToInt) {
      FlatTree[CurrentNode].FromInt = FromInt;
      FlatTree[CurrentNode].ToInt = ToInt;
      FlatTree[CurrentNode].IsValidFromInt = IsValidFromInt;
      FlatTree[CurrentNode].IsValidToInt = IsValidToInt;
    }

    /// SetNode - Set FromQual and ToQual of the current node.
    void SetNode(Qualifiers FromQual, Qualifiers ToQual) {
      FlatTree[CurrentNode].FromQual = FromQual;
      FlatTree[CurrentNode].ToQual = ToQual;
    }

    /// SetNode - Set FromValueDecl and ToValueDecl of the current node.
    void SetNode(ValueDecl *FromValueDecl, ValueDecl *ToValueDecl,
                 bool FromAddressOf, bool ToAddressOf) {
      FlatTree[CurrentNode].FromValueDecl = FromValueDecl;
      FlatTree[CurrentNode].ToValueDecl = ToValueDecl;
      FlatTree[CurrentNode].FromAddressOf = FromAddressOf;
      FlatTree[CurrentNode].ToAddressOf = ToAddressOf;
    }

    /// SetSame - Sets the same flag of the current node.
    void SetSame(bool Same) {
      FlatTree[CurrentNode].Same = Same;
    }

    /// SetDefault - Sets FromDefault and ToDefault flags of the current node.
    void SetDefault(bool FromDefault, bool ToDefault) {
      FlatTree[CurrentNode].FromDefault = FromDefault;
      FlatTree[CurrentNode].ToDefault = ToDefault;
    }

    /// SetKind - Sets the current node's type.
    void SetKind(DiffKind Kind) {
      FlatTree[CurrentNode].Kind = Kind;
    }

    /// Up - Changes the node to the parent of the current node.
    void Up() {
      CurrentNode = FlatTree[CurrentNode].ParentNode;
    }

    /// AddNode - Adds a child node to the current node, then sets that node
    /// node as the current node.
    void AddNode() {
      FlatTree.push_back(DiffNode(CurrentNode));
      DiffNode &Node = FlatTree[CurrentNode];
      if (Node.ChildNode == 0) {
        // If a child node doesn't exist, add one.
        Node.ChildNode = NextFreeNode;
      } else {
        // If a child node exists, find the last child node and add a
        // next node to it.
        unsigned i;
        for (i = Node.ChildNode; FlatTree[i].NextNode != 0;
             i = FlatTree[i].NextNode) {
        }
        FlatTree[i].NextNode = NextFreeNode;
      }
      CurrentNode = NextFreeNode;
      ++NextFreeNode;
    }

    // Node reading functions.
    /// StartTraverse - Prepares the tree for recursive traversal.
    void StartTraverse() {
      ReadNode = 0;
      CurrentNode = NextFreeNode;
      NextFreeNode = 0;
    }

    /// Parent - Move the current read node to its parent.
    void Parent() {
      ReadNode = FlatTree[ReadNode].ParentNode;
    }

    /// GetNode - Gets the FromType and ToType.
    void GetNode(QualType &FromType, QualType &ToType) {
      FromType = FlatTree[ReadNode].FromType;
      ToType = FlatTree[ReadNode].ToType;
    }

    /// GetNode - Gets the FromExpr and ToExpr.
    void GetNode(Expr *&FromExpr, Expr *&ToExpr) {
      FromExpr = FlatTree[ReadNode].FromExpr;
      ToExpr = FlatTree[ReadNode].ToExpr;
    }

    /// GetNode - Gets the FromTD and ToTD.
    void GetNode(TemplateDecl *&FromTD, TemplateDecl *&ToTD) {
      FromTD = FlatTree[ReadNode].FromTD;
      ToTD = FlatTree[ReadNode].ToTD;
    }

    /// GetNode - Gets the FromInt and ToInt.
    void GetNode(llvm::APSInt &FromInt, llvm::APSInt &ToInt,
                 bool &IsValidFromInt, bool &IsValidToInt) {
      FromInt = FlatTree[ReadNode].FromInt;
      ToInt = FlatTree[ReadNode].ToInt;
      IsValidFromInt = FlatTree[ReadNode].IsValidFromInt;
      IsValidToInt = FlatTree[ReadNode].IsValidToInt;
    }

    /// GetNode - Gets the FromQual and ToQual.
    void GetNode(Qualifiers &FromQual, Qualifiers &ToQual) {
      FromQual = FlatTree[ReadNode].FromQual;
      ToQual = FlatTree[ReadNode].ToQual;
    }

    /// GetNode - Gets the FromValueDecl and ToValueDecl.
    void GetNode(ValueDecl *&FromValueDecl, ValueDecl *&ToValueDecl,
                 bool &FromAddressOf, bool &ToAddressOf) {
      FromValueDecl = FlatTree[ReadNode].FromValueDecl;
      ToValueDecl = FlatTree[ReadNode].ToValueDecl;
      FromAddressOf = FlatTree[ReadNode].FromAddressOf;
      ToAddressOf = FlatTree[ReadNode].ToAddressOf;
    }

    /// NodeIsSame - Returns true the arguments are the same.
    bool NodeIsSame() {
      return FlatTree[ReadNode].Same;
    }

    /// HasChildrend - Returns true if the node has children.
    bool HasChildren() {
      return FlatTree[ReadNode].ChildNode != 0;
    }

    /// MoveToChild - Moves from the current node to its child.
    void MoveToChild() {
      ReadNode = FlatTree[ReadNode].ChildNode;
    }

    /// AdvanceSibling - If there is a next sibling, advance to it and return
    /// true.  Otherwise, return false.
    bool AdvanceSibling() {
      if (FlatTree[ReadNode].NextNode == 0)
        return false;

      ReadNode = FlatTree[ReadNode].NextNode;
      return true;
    }

    /// HasNextSibling - Return true if the node has a next sibling.
    bool HasNextSibling() {
      return FlatTree[ReadNode].NextNode != 0;
    }

    /// FromDefault - Return true if the from argument is the default.
    bool FromDefault() {
      return FlatTree[ReadNode].FromDefault;
    }

    /// ToDefault - Return true if the to argument is the default.
    bool ToDefault() {
      return FlatTree[ReadNode].ToDefault;
    }

    /// Empty - Returns true if the tree has no information.
    bool Empty() {
      return GetKind() == Invalid;
    }

    /// GetKind - Returns the current node's type.
    DiffKind GetKind() {
      return FlatTree[ReadNode].Kind;
    }
  };

  DiffTree Tree;

  /// TSTiterator - an iterator that is used to enter a
  /// TemplateSpecializationType and read TemplateArguments inside template
  /// parameter packs in order with the rest of the TemplateArguments.
  struct TSTiterator {
    typedef const TemplateArgument& reference;
    typedef const TemplateArgument* pointer;

    /// TST - the template specialization whose arguments this iterator
    /// traverse over.
    const TemplateSpecializationType *TST;

    /// DesugarTST - desugared template specialization used to extract
    /// default argument information
    const TemplateSpecializationType *DesugarTST;

    /// Index - the index of the template argument in TST.
    unsigned Index;

    /// CurrentTA - if CurrentTA is not the same as EndTA, then CurrentTA
    /// points to a TemplateArgument within a parameter pack.
    TemplateArgument::pack_iterator CurrentTA;

    /// EndTA - the end iterator of a parameter pack
    TemplateArgument::pack_iterator EndTA;

    /// TSTiterator - Constructs an iterator and sets it to the first template
    /// argument.
    TSTiterator(ASTContext &Context, const TemplateSpecializationType *TST)
        : TST(TST),
          DesugarTST(GetTemplateSpecializationType(Context, TST->desugar())),
          Index(0), CurrentTA(nullptr), EndTA(nullptr) {
      if (isEnd()) return;

      // Set to first template argument.  If not a parameter pack, done.
      TemplateArgument TA = TST->getArg(0);
      if (TA.getKind() != TemplateArgument::Pack) return;

      // Start looking into the parameter pack.
      CurrentTA = TA.pack_begin();
      EndTA = TA.pack_end();

      // Found a valid template argument.
      if (CurrentTA != EndTA) return;

      // Parameter pack is empty, use the increment to get to a valid
      // template argument.
      ++(*this);
    }

    /// isEnd - Returns true if the iterator is one past the end.
    bool isEnd() const {
      return Index >= TST->getNumArgs();
    }

    /// &operator++ - Increment the iterator to the next template argument.
    TSTiterator &operator++() {
      // After the end, Index should be the default argument position in
      // DesugarTST, if it exists.
      if (isEnd()) {
        ++Index;
        return *this;
      }

      // If in a parameter pack, advance in the parameter pack.
      if (CurrentTA != EndTA) {
        ++CurrentTA;
        if (CurrentTA != EndTA)
          return *this;
      }

      // Loop until a template argument is found, or the end is reached.
      while (true) {
        // Advance to the next template argument.  Break if reached the end.
        if (++Index == TST->getNumArgs()) break;

        // If the TemplateArgument is not a parameter pack, done.
        TemplateArgument TA = TST->getArg(Index);
        if (TA.getKind() != TemplateArgument::Pack) break;

        // Handle parameter packs.
        CurrentTA = TA.pack_begin();
        EndTA = TA.pack_end();

        // If the parameter pack is empty, try to advance again.
        if (CurrentTA != EndTA) break;
      }
      return *this;
    }

    /// operator* - Returns the appropriate TemplateArgument.
    reference operator*() const {
      assert(!isEnd() && "Index exceeds number of arguments.");
      if (CurrentTA == EndTA)
        return TST->getArg(Index);
      else
        return *CurrentTA;
    }

    /// operator-> - Allow access to the underlying TemplateArgument.
    pointer operator->() const {
      return &operator*();
    }

    /// getDesugar - Returns the deduced template argument from DesguarTST
    reference getDesugar() const {
      return DesugarTST->getArg(Index);
    }
  };

  // These functions build up the template diff tree, including functions to
  // retrieve and compare template arguments. 

  static const TemplateSpecializationType * GetTemplateSpecializationType(
      ASTContext &Context, QualType Ty) {
    if (const TemplateSpecializationType *TST =
            Ty->getAs<TemplateSpecializationType>())
      return TST;

    const RecordType *RT = Ty->getAs<RecordType>();

    if (!RT)
      return nullptr;

    const ClassTemplateSpecializationDecl *CTSD =
        dyn_cast<ClassTemplateSpecializationDecl>(RT->getDecl());

    if (!CTSD)
      return nullptr;

    Ty = Context.getTemplateSpecializationType(
             TemplateName(CTSD->getSpecializedTemplate()),
             CTSD->getTemplateArgs().data(),
             CTSD->getTemplateArgs().size(),
             Ty.getLocalUnqualifiedType().getCanonicalType());

    return Ty->getAs<TemplateSpecializationType>();
  }

  /// DiffTemplate - recursively visits template arguments and stores the
  /// argument info into a tree.
  void DiffTemplate(const TemplateSpecializationType *FromTST,
                    const TemplateSpecializationType *ToTST) {
    // Begin descent into diffing template tree.
    TemplateParameterList *ParamsFrom =
        FromTST->getTemplateName().getAsTemplateDecl()->getTemplateParameters();
    TemplateParameterList *ParamsTo =
        ToTST->getTemplateName().getAsTemplateDecl()->getTemplateParameters();
    unsigned TotalArgs = 0;
    for (TSTiterator FromIter(Context, FromTST), ToIter(Context, ToTST);
         !FromIter.isEnd() || !ToIter.isEnd(); ++TotalArgs) {
      Tree.AddNode();

      // Get the parameter at index TotalArgs.  If index is larger
      // than the total number of parameters, then there is an
      // argument pack, so re-use the last parameter.
      unsigned ParamIndex = std::min(TotalArgs, ParamsFrom->size() - 1);
      NamedDecl *ParamND = ParamsFrom->getParam(ParamIndex);

      // Handle Types
      if (TemplateTypeParmDecl *DefaultTTPD =
              dyn_cast<TemplateTypeParmDecl>(ParamND)) {
        QualType FromType, ToType;
        FromType = GetType(FromIter, DefaultTTPD);
        // A forward declaration can have no default arg but the actual class
        // can, don't mix up iterators and get the original parameter.
        ToType = GetType(
            ToIter, cast<TemplateTypeParmDecl>(ParamsTo->getParam(ParamIndex)));
        Tree.SetNode(FromType, ToType);
        Tree.SetDefault(FromIter.isEnd() && !FromType.isNull(),
                        ToIter.isEnd() && !ToType.isNull());
        Tree.SetKind(DiffTree::Type);
        if (!FromType.isNull() && !ToType.isNull()) {
          if (Context.hasSameType(FromType, ToType)) {
            Tree.SetSame(true);
          } else {
            Qualifiers FromQual = FromType.getQualifiers(),
                       ToQual = ToType.getQualifiers();
            const TemplateSpecializationType *FromArgTST =
                GetTemplateSpecializationType(Context, FromType);
            const TemplateSpecializationType *ToArgTST =
                GetTemplateSpecializationType(Context, ToType);

            if (FromArgTST && ToArgTST &&
                hasSameTemplate(FromArgTST, ToArgTST)) {
              FromQual -= QualType(FromArgTST, 0).getQualifiers();
              ToQual -= QualType(ToArgTST, 0).getQualifiers();
              Tree.SetNode(FromArgTST->getTemplateName().getAsTemplateDecl(),
                           ToArgTST->getTemplateName().getAsTemplateDecl());
              Tree.SetNode(FromQual, ToQual);
              Tree.SetKind(DiffTree::Template);
              DiffTemplate(FromArgTST, ToArgTST);
            }
          }
        }
      }

      // Handle Expressions
      if (NonTypeTemplateParmDecl *DefaultNTTPD =
              dyn_cast<NonTypeTemplateParmDecl>(ParamND)) {
        Expr *FromExpr = nullptr, *ToExpr = nullptr;
        llvm::APSInt FromInt, ToInt;
        ValueDecl *FromValueDecl = nullptr, *ToValueDecl = nullptr;
        unsigned ParamWidth = 128; // Safe default
        if (DefaultNTTPD->getType()->isIntegralOrEnumerationType())
          ParamWidth = Context.getIntWidth(DefaultNTTPD->getType());
        bool HasFromInt = !FromIter.isEnd() &&
                          FromIter->getKind() == TemplateArgument::Integral;
        bool HasToInt = !ToIter.isEnd() &&
                        ToIter->getKind() == TemplateArgument::Integral;
        bool HasFromValueDecl =
            !FromIter.isEnd() &&
            FromIter->getKind() == TemplateArgument::Declaration;
        bool HasToValueDecl =
            !ToIter.isEnd() &&
            ToIter->getKind() == TemplateArgument::Declaration;

        assert(((!HasFromInt && !HasToInt) ||
                (!HasFromValueDecl && !HasToValueDecl)) &&
               "Template argument cannot be both integer and declaration");

        if (HasFromInt)
          FromInt = FromIter->getAsIntegral();
        else if (HasFromValueDecl)
          FromValueDecl = FromIter->getAsDecl();
        else
          FromExpr = GetExpr(FromIter, DefaultNTTPD);

        if (HasToInt)
          ToInt = ToIter->getAsIntegral();
        else if (HasToValueDecl)
          ToValueDecl = ToIter->getAsDecl();
        else
          ToExpr = GetExpr(ToIter, DefaultNTTPD);

        if (!HasFromInt && !HasToInt && !HasFromValueDecl && !HasToValueDecl) {
          Tree.SetNode(FromExpr, ToExpr);
          Tree.SetDefault(FromIter.isEnd() && FromExpr,
                          ToIter.isEnd() && ToExpr);
          if (DefaultNTTPD->getType()->isIntegralOrEnumerationType()) {
            if (FromExpr)
              HasFromInt = GetInt(FromIter, FromExpr, FromInt);
            if (ToExpr)
              HasToInt = GetInt(ToIter, ToExpr, ToInt);
          }
          if (HasFromInt && HasToInt) {
            Tree.SetNode(FromInt, ToInt, HasFromInt, HasToInt);
            Tree.SetSame(IsSameConvertedInt(ParamWidth, FromInt, ToInt));
            Tree.SetKind(DiffTree::Integer);
          } else if (HasFromInt || HasToInt) {
            Tree.SetNode(FromInt, ToInt, HasFromInt, HasToInt);
            Tree.SetSame(false);
            Tree.SetKind(DiffTree::Integer);
          } else {
            Tree.SetSame(IsEqualExpr(Context, ParamWidth, FromExpr, ToExpr));
            Tree.SetKind(DiffTree::Expression);
          }
        } else if (HasFromInt || HasToInt) {
          if (!HasFromInt && FromExpr)
            HasFromInt = GetInt(FromIter, FromExpr, FromInt);
          if (!HasToInt && ToExpr)
            HasToInt = GetInt(ToIter, ToExpr, ToInt);
          Tree.SetNode(FromInt, ToInt, HasFromInt, HasToInt);
          Tree.SetSame(IsSameConvertedInt(ParamWidth, FromInt, ToInt));
          Tree.SetDefault(FromIter.isEnd() && HasFromInt,
                          ToIter.isEnd() && HasToInt);
          Tree.SetKind(DiffTree::Integer);
        } else {
          if (!HasFromValueDecl && FromExpr)
            FromValueDecl = GetValueDecl(FromIter, FromExpr);
          if (!HasToValueDecl && ToExpr)
            ToValueDecl = GetValueDecl(ToIter, ToExpr);
          QualType ArgumentType = DefaultNTTPD->getType();
          bool FromAddressOf = false;
          if (FromValueDecl) {
            if (FromExpr) {
              if (UnaryOperator *UO = dyn_cast<UnaryOperator>(FromExpr)) {
                if (UO->getOpcode() == UO_AddrOf)
                  FromAddressOf = true;
              }
            } else {
              if (!ArgumentType->isReferenceType()) {
                FromAddressOf = true;
              }
            }
          }
          bool ToAddressOf = false;
          if (ToValueDecl) {
            if (ToExpr) {
              if (UnaryOperator *UO = dyn_cast<UnaryOperator>(ToExpr)) {
                if (UO->getOpcode() == UO_AddrOf) {
                  ToAddressOf = true;
                }
              }
            } else {
              if (!ArgumentType->isReferenceType())
                ToAddressOf = true;
            }
          }
          Tree.SetNode(FromValueDecl, ToValueDecl, FromAddressOf, ToAddressOf);
          Tree.SetSame(FromValueDecl && ToValueDecl &&
                       FromValueDecl->getCanonicalDecl() ==
                       ToValueDecl->getCanonicalDecl());
          Tree.SetDefault(FromIter.isEnd() && FromValueDecl,
                          ToIter.isEnd() && ToValueDecl);
          Tree.SetKind(DiffTree::Declaration);
        }
      }

      // Handle Templates
      if (TemplateTemplateParmDecl *DefaultTTPD =
              dyn_cast<TemplateTemplateParmDecl>(ParamND)) {
        TemplateDecl *FromDecl, *ToDecl;
        FromDecl = GetTemplateDecl(FromIter, DefaultTTPD);
        ToDecl = GetTemplateDecl(ToIter, DefaultTTPD);
        Tree.SetNode(FromDecl, ToDecl);
        Tree.SetSame(
            FromDecl && ToDecl &&
            FromDecl->getCanonicalDecl() == ToDecl->getCanonicalDecl());
        Tree.SetKind(DiffTree::TemplateTemplate);
      }

      ++FromIter;
      ++ToIter;
      Tree.Up();
    }
  }

  /// makeTemplateList - Dump every template alias into the vector.
  static void makeTemplateList(
      SmallVectorImpl<const TemplateSpecializationType *> &TemplateList,
      const TemplateSpecializationType *TST) {
    while (TST) {
      TemplateList.push_back(TST);
      if (!TST->isTypeAlias())
        return;
      TST = TST->getAliasedType()->getAs<TemplateSpecializationType>();
    }
  }

  /// hasSameBaseTemplate - Returns true when the base templates are the same,
  /// even if the template arguments are not.
  static bool hasSameBaseTemplate(const TemplateSpecializationType *FromTST,
                                  const TemplateSpecializationType *ToTST) {
    return FromTST->getTemplateName().getAsTemplateDecl()->getCanonicalDecl() ==
           ToTST->getTemplateName().getAsTemplateDecl()->getCanonicalDecl();
  }

  /// hasSameTemplate - Returns true if both types are specialized from the
  /// same template declaration.  If they come from different template aliases,
  /// do a parallel ascension search to determine the highest template alias in
  /// common and set the arguments to them.
  static bool hasSameTemplate(const TemplateSpecializationType *&FromTST,
                              const TemplateSpecializationType *&ToTST) {
    // Check the top templates if they are the same.
    if (hasSameBaseTemplate(FromTST, ToTST))
      return true;

    // Create vectors of template aliases.
    SmallVector<const TemplateSpecializationType*, 1> FromTemplateList,
                                                      ToTemplateList;

    makeTemplateList(FromTemplateList, FromTST);
    makeTemplateList(ToTemplateList, ToTST);

    SmallVectorImpl<const TemplateSpecializationType *>::reverse_iterator
        FromIter = FromTemplateList.rbegin(), FromEnd = FromTemplateList.rend(),
        ToIter = ToTemplateList.rbegin(), ToEnd = ToTemplateList.rend();

    // Check if the lowest template types are the same.  If not, return.
    if (!hasSameBaseTemplate(*FromIter, *ToIter))
      return false;

    // Begin searching up the template aliases.  The bottom most template
    // matches so move up until one pair does not match.  Use the template
    // right before that one.
    for (; FromIter != FromEnd && ToIter != ToEnd; ++FromIter, ++ToIter) {
      if (!hasSameBaseTemplate(*FromIter, *ToIter))
        break;
    }

    FromTST = FromIter[-1];
    ToTST = ToIter[-1];

    return true;
  }

  /// GetType - Retrieves the template type arguments, including default
  /// arguments.
  QualType GetType(const TSTiterator &Iter, TemplateTypeParmDecl *DefaultTTPD) {
    bool isVariadic = DefaultTTPD->isParameterPack();

    if (!Iter.isEnd())
      return Iter->getAsType();
    if (isVariadic)
      return QualType();

    QualType ArgType = DefaultTTPD->getDefaultArgument();
    if (ArgType->isDependentType())
      return Iter.getDesugar().getAsType();

    return ArgType;
  }

  /// GetExpr - Retrieves the template expression argument, including default
  /// arguments.
  Expr *GetExpr(const TSTiterator &Iter, NonTypeTemplateParmDecl *DefaultNTTPD) {
    Expr *ArgExpr = nullptr;
    bool isVariadic = DefaultNTTPD->isParameterPack();

    if (!Iter.isEnd())
      ArgExpr = Iter->getAsExpr();
    else if (!isVariadic)
      ArgExpr = DefaultNTTPD->getDefaultArgument();

    if (ArgExpr)
      while (SubstNonTypeTemplateParmExpr *SNTTPE =
                 dyn_cast<SubstNonTypeTemplateParmExpr>(ArgExpr))
        ArgExpr = SNTTPE->getReplacement();

    return ArgExpr;
  }

  /// GetInt - Retrieves the template integer argument, including evaluating
  /// default arguments.
  bool GetInt(const TSTiterator &Iter, Expr *ArgExpr, llvm::APInt &Int) {
    // Default, value-depenedent expressions require fetching
    // from the desugared TemplateArgument, otherwise expression needs to
    // be evaluatable.
    if (Iter.isEnd() && ArgExpr->isValueDependent()) {
      switch (Iter.getDesugar().getKind()) {
        case TemplateArgument::Integral:
          Int = Iter.getDesugar().getAsIntegral();
          return true;
        case TemplateArgument::Expression:
          ArgExpr = Iter.getDesugar().getAsExpr();
          Int = ArgExpr->EvaluateKnownConstInt(Context);
          return true;
        default:
          llvm_unreachable("Unexpected template argument kind");
      }
    } else if (ArgExpr->isEvaluatable(Context)) {
      Int = ArgExpr->EvaluateKnownConstInt(Context);
      return true;
    }

    return false;
  }

  /// GetValueDecl - Retrieves the template Decl argument, including
  /// default expression argument.
  ValueDecl *GetValueDecl(const TSTiterator &Iter, Expr *ArgExpr) {
    // Default, value-depenedent expressions require fetching
    // from the desugared TemplateArgument
    if (Iter.isEnd() && ArgExpr->isValueDependent())
      switch (Iter.getDesugar().getKind()) {
        case TemplateArgument::Declaration:
          return Iter.getDesugar().getAsDecl();
        case TemplateArgument::Expression:
          ArgExpr = Iter.getDesugar().getAsExpr();
          return cast<DeclRefExpr>(ArgExpr)->getDecl();
        default:
          llvm_unreachable("Unexpected template argument kind");
      }
    DeclRefExpr *DRE = dyn_cast<DeclRefExpr>(ArgExpr);
    if (!DRE) {
      DRE = cast<DeclRefExpr>(cast<UnaryOperator>(ArgExpr)->getSubExpr());
    }

    return DRE->getDecl();
  }

  /// GetTemplateDecl - Retrieves the template template arguments, including
  /// default arguments.
  TemplateDecl *GetTemplateDecl(const TSTiterator &Iter,
                                TemplateTemplateParmDecl *DefaultTTPD) {
    bool isVariadic = DefaultTTPD->isParameterPack();

    TemplateArgument TA = DefaultTTPD->getDefaultArgument().getArgument();
    TemplateDecl *DefaultTD = nullptr;
    if (TA.getKind() != TemplateArgument::Null)
      DefaultTD = TA.getAsTemplate().getAsTemplateDecl();

    if (!Iter.isEnd())
      return Iter->getAsTemplate().getAsTemplateDecl();
    if (!isVariadic)
      return DefaultTD;

    return nullptr;
  }

  /// IsSameConvertedInt - Returns true if both integers are equal when
  /// converted to an integer type with the given width.
  static bool IsSameConvertedInt(unsigned Width, const llvm::APSInt &X,
                                 const llvm::APSInt &Y) {
    llvm::APInt ConvertedX = X.extOrTrunc(Width);
    llvm::APInt ConvertedY = Y.extOrTrunc(Width);
    return ConvertedX == ConvertedY;
  }

  /// IsEqualExpr - Returns true if the expressions evaluate to the same value.
  static bool IsEqualExpr(ASTContext &Context, unsigned ParamWidth,
                          Expr *FromExpr, Expr *ToExpr) {
    if (FromExpr == ToExpr)
      return true;

    if (!FromExpr || !ToExpr)
      return false;

    FromExpr = FromExpr->IgnoreParens();
    ToExpr = ToExpr->IgnoreParens();

    DeclRefExpr *FromDRE = dyn_cast<DeclRefExpr>(FromExpr),
                *ToDRE = dyn_cast<DeclRefExpr>(ToExpr);

    if (FromDRE || ToDRE) {
      if (!FromDRE || !ToDRE)
        return false;
      return FromDRE->getDecl() == ToDRE->getDecl();
    }

    Expr::EvalResult FromResult, ToResult;
    if (!FromExpr->EvaluateAsRValue(FromResult, Context) ||
        !ToExpr->EvaluateAsRValue(ToResult, Context))
      return false;

    APValue &FromVal = FromResult.Val;
    APValue &ToVal = ToResult.Val;

    if (FromVal.getKind() != ToVal.getKind()) return false;

    switch (FromVal.getKind()) {
      case APValue::Int:
        return IsSameConvertedInt(ParamWidth, FromVal.getInt(), ToVal.getInt());
      case APValue::LValue: {
        APValue::LValueBase FromBase = FromVal.getLValueBase();
        APValue::LValueBase ToBase = ToVal.getLValueBase();
        if (FromBase.isNull() && ToBase.isNull())
          return true;
        if (FromBase.isNull() || ToBase.isNull())
          return false;
        return FromBase.get<const ValueDecl*>() ==
               ToBase.get<const ValueDecl*>();
      }
      case APValue::MemberPointer:
        return FromVal.getMemberPointerDecl() == ToVal.getMemberPointerDecl();
      default:
        llvm_unreachable("Unknown template argument expression.");
    }
  }

  // These functions converts the tree representation of the template
  // differences into the internal character vector.

  /// TreeToString - Converts the Tree object into a character stream which
  /// will later be turned into the output string.
  void TreeToString(int Indent = 1) {
    if (PrintTree) {
      OS << '\n';
      OS.indent(2 * Indent);
      ++Indent;
    }

    // Handle cases where the difference is not templates with different
    // arguments.
    switch (Tree.GetKind()) {
      case DiffTree::Invalid:
        llvm_unreachable("Template diffing failed with bad DiffNode");
      case DiffTree::Type: {
        QualType FromType, ToType;
        Tree.GetNode(FromType, ToType);
        PrintTypeNames(FromType, ToType, Tree.FromDefault(), Tree.ToDefault(),
                       Tree.NodeIsSame());
        return;
      }
      case DiffTree::Expression: {
        Expr *FromExpr, *ToExpr;
        Tree.GetNode(FromExpr, ToExpr);
        PrintExpr(FromExpr, ToExpr, Tree.FromDefault(), Tree.ToDefault(),
                  Tree.NodeIsSame());
        return;
      }
      case DiffTree::TemplateTemplate: {
        TemplateDecl *FromTD, *ToTD;
        Tree.GetNode(FromTD, ToTD);
        PrintTemplateTemplate(FromTD, ToTD, Tree.FromDefault(),
                              Tree.ToDefault(), Tree.NodeIsSame());
        return;
      }
      case DiffTree::Integer: {
        llvm::APSInt FromInt, ToInt;
        Expr *FromExpr, *ToExpr;
        bool IsValidFromInt, IsValidToInt;
        Tree.GetNode(FromExpr, ToExpr);
        Tree.GetNode(FromInt, ToInt, IsValidFromInt, IsValidToInt);
        PrintAPSInt(FromInt, ToInt, IsValidFromInt, IsValidToInt,
                    FromExpr, ToExpr, Tree.FromDefault(), Tree.ToDefault(),
                    Tree.NodeIsSame());
        return;
      }
      case DiffTree::Declaration: {
        ValueDecl *FromValueDecl, *ToValueDecl;
        bool FromAddressOf, ToAddressOf;
        Tree.GetNode(FromValueDecl, ToValueDecl, FromAddressOf, ToAddressOf);
        PrintValueDecl(FromValueDecl, ToValueDecl, FromAddressOf, ToAddressOf,
                       Tree.FromDefault(), Tree.ToDefault(), Tree.NodeIsSame());
        return;
      }
      case DiffTree::Template: {
        // Node is root of template.  Recurse on children.
        TemplateDecl *FromTD, *ToTD;
        Tree.GetNode(FromTD, ToTD);

        if (!Tree.HasChildren()) {
          // If we're dealing with a template specialization with zero
          // arguments, there are no children; special-case this.
          OS << FromTD->getNameAsString() << "<>";
          return;
        }

        Qualifiers FromQual, ToQual;
        Tree.GetNode(FromQual, ToQual);
        PrintQualifiers(FromQual, ToQual);

        OS << FromTD->getNameAsString() << '<'; 
        Tree.MoveToChild();
        unsigned NumElideArgs = 0;
        do {
          if (ElideType) {
            if (Tree.NodeIsSame()) {
              ++NumElideArgs;
              continue;
            }
            if (NumElideArgs > 0) {
              PrintElideArgs(NumElideArgs, Indent);
              NumElideArgs = 0;
              OS << ", ";
            }
          }
          TreeToString(Indent);
          if (Tree.HasNextSibling())
            OS << ", ";
        } while (Tree.AdvanceSibling());
        if (NumElideArgs > 0)
          PrintElideArgs(NumElideArgs, Indent);

        Tree.Parent();
        OS << ">";
        return;
      }
    }
  }

  // To signal to the text printer that a certain text needs to be bolded,
  // a special character is injected into the character stream which the
  // text printer will later strip out.

  /// Bold - Start bolding text.
  void Bold() {
    assert(!IsBold && "Attempting to bold text that is already bold.");
    IsBold = true;
    if (ShowColor)
      OS << ToggleHighlight;
  }

  /// Unbold - Stop bolding text.
  void Unbold() {
    assert(IsBold && "Attempting to remove bold from unbold text.");
    IsBold = false;
    if (ShowColor)
      OS << ToggleHighlight;
  }

  // Functions to print out the arguments and highlighting the difference.

  /// PrintTypeNames - prints the typenames, bolding differences.  Will detect
  /// typenames that are the same and attempt to disambiguate them by using
  /// canonical typenames.
  void PrintTypeNames(QualType FromType, QualType ToType,
                      bool FromDefault, bool ToDefault, bool Same) {
    assert((!FromType.isNull() || !ToType.isNull()) &&
           "Only one template argument may be missing.");

    if (Same) {
      OS << FromType.getAsString();
      return;
    }

    if (!FromType.isNull() && !ToType.isNull() &&
        FromType.getLocalUnqualifiedType() ==
        ToType.getLocalUnqualifiedType()) {
      Qualifiers FromQual = FromType.getLocalQualifiers(),
                 ToQual = ToType.getLocalQualifiers();
      PrintQualifiers(FromQual, ToQual);
      FromType.getLocalUnqualifiedType().print(OS, Policy);
      return;
    }

    std::string FromTypeStr = FromType.isNull() ? "(no argument)"
                                                : FromType.getAsString();
    std::string ToTypeStr = ToType.isNull() ? "(no argument)"
                                            : ToType.getAsString();
    // Switch to canonical typename if it is better.
    // TODO: merge this with other aka printing above.
    if (FromTypeStr == ToTypeStr) {
      std::string FromCanTypeStr = FromType.getCanonicalType().getAsString();
      std::string ToCanTypeStr = ToType.getCanonicalType().getAsString();
      if (FromCanTypeStr != ToCanTypeStr) {
        FromTypeStr = FromCanTypeStr;
        ToTypeStr = ToCanTypeStr;
      }
    }

    if (PrintTree) OS << '[';
    OS << (FromDefault ? "(default) " : "");
    Bold();
    OS << FromTypeStr;
    Unbold();
    if (PrintTree) {
      OS << " != " << (ToDefault ? "(default) " : "");
      Bold();
      OS << ToTypeStr;
      Unbold();
      OS << "]";
    }
    return;
  }

  /// PrintExpr - Prints out the expr template arguments, highlighting argument
  /// differences.
  void PrintExpr(const Expr *FromExpr, const Expr *ToExpr,
                 bool FromDefault, bool ToDefault, bool Same) {
    assert((FromExpr || ToExpr) &&
            "Only one template argument may be missing.");
    if (Same) {
      PrintExpr(FromExpr);
    } else if (!PrintTree) {
      OS << (FromDefault ? "(default) " : "");
      Bold();
      PrintExpr(FromExpr);
      Unbold();
    } else {
      OS << (FromDefault ? "[(default) " : "[");
      Bold();
      PrintExpr(FromExpr);
      Unbold();
      OS << " != " << (ToDefault ? "(default) " : "");
      Bold();
      PrintExpr(ToExpr);
      Unbold();
      OS << ']';
    }
  }

  /// PrintExpr - Actual formatting and printing of expressions.
  void PrintExpr(const Expr *E) {
    if (!E)
      OS << "(no argument)";
    else
      E->printPretty(OS, nullptr, Policy);
  }

  /// PrintTemplateTemplate - Handles printing of template template arguments,
  /// highlighting argument differences.
  void PrintTemplateTemplate(TemplateDecl *FromTD, TemplateDecl *ToTD,
                             bool FromDefault, bool ToDefault, bool Same) {
    assert((FromTD || ToTD) && "Only one template argument may be missing.");

    std::string FromName = FromTD ? FromTD->getName() : "(no argument)";
    std::string ToName = ToTD ? ToTD->getName() : "(no argument)";
    if (FromTD && ToTD && FromName == ToName) {
      FromName = FromTD->getQualifiedNameAsString();
      ToName = ToTD->getQualifiedNameAsString();
    }

    if (Same) {
      OS << "template " << FromTD->getNameAsString();
    } else if (!PrintTree) {
      OS << (FromDefault ? "(default) template " : "template ");
      Bold();
      OS << FromName;
      Unbold();
    } else {
      OS << (FromDefault ? "[(default) template " : "[template ");
      Bold();
      OS << FromName;
      Unbold();
      OS << " != " << (ToDefault ? "(default) template " : "template ");
      Bold();
      OS << ToName;
      Unbold();
      OS << ']';
    }
  }

  /// PrintAPSInt - Handles printing of integral arguments, highlighting
  /// argument differences.
  void PrintAPSInt(llvm::APSInt FromInt, llvm::APSInt ToInt,
                   bool IsValidFromInt, bool IsValidToInt, Expr *FromExpr,
                   Expr *ToExpr, bool FromDefault, bool ToDefault, bool Same) {
    assert((IsValidFromInt || IsValidToInt) &&
           "Only one integral argument may be missing.");

    if (Same) {
      OS << FromInt.toString(10);
    } else if (!PrintTree) {
      OS << (FromDefault ? "(default) " : "");
      PrintAPSInt(FromInt, FromExpr, IsValidFromInt);
    } else {
      OS << (FromDefault ? "[(default) " : "[");
      PrintAPSInt(FromInt, FromExpr, IsValidFromInt);
      OS << " != " << (ToDefault ? "(default) " : "");
      PrintAPSInt(ToInt, ToExpr, IsValidToInt);
      OS << ']';
    }
  }

  /// PrintAPSInt - If valid, print the APSInt.  If the expression is
  /// gives more information, print it too.
  void PrintAPSInt(llvm::APSInt Val, Expr *E, bool Valid) {
    Bold();
    if (Valid) {
      if (HasExtraInfo(E)) {
        PrintExpr(E);
        Unbold();
        OS << " aka ";
        Bold();
      }
      OS << Val.toString(10);
    } else if (E) {
      PrintExpr(E);
    } else {
      OS << "(no argument)";
    }
    Unbold();
  }
  
  /// HasExtraInfo - Returns true if E is not an integer literal or the
  /// negation of an integer literal
  bool HasExtraInfo(Expr *E) {
    if (!E) return false;
    if (isa<IntegerLiteral>(E)) return false;

    if (UnaryOperator *UO = dyn_cast<UnaryOperator>(E))
      if (UO->getOpcode() == UO_Minus)
        if (isa<IntegerLiteral>(UO->getSubExpr()))
          return false;

    return true;
  }

  /// PrintDecl - Handles printing of Decl arguments, highlighting
  /// argument differences.
  void PrintValueDecl(ValueDecl *FromValueDecl, ValueDecl *ToValueDecl,
                      bool FromAddressOf, bool ToAddressOf, bool FromDefault,
                      bool ToDefault, bool Same) {
    assert((FromValueDecl || ToValueDecl) &&
           "Only one Decl argument may be NULL");

    if (Same) {
      OS << FromValueDecl->getName();
    } else if (!PrintTree) {
      OS << (FromDefault ? "(default) " : "");
      Bold();
      if (FromAddressOf)
        OS << "&";
      OS << (FromValueDecl ? FromValueDecl->getName() : "(no argument)");
      Unbold();
    } else {
      OS << (FromDefault ? "[(default) " : "[");
      Bold();
      if (FromAddressOf)
        OS << "&";
      OS << (FromValueDecl ? FromValueDecl->getName() : "(no argument)");
      Unbold();
      OS << " != " << (ToDefault ? "(default) " : "");
      Bold();
      if (ToAddressOf)
        OS << "&";
      OS << (ToValueDecl ? ToValueDecl->getName() : "(no argument)");
      Unbold();
      OS << ']';
    }

  }

  // Prints the appropriate placeholder for elided template arguments.
  void PrintElideArgs(unsigned NumElideArgs, unsigned Indent) {
    if (PrintTree) {
      OS << '\n';
      for (unsigned i = 0; i < Indent; ++i)
        OS << "  ";
    }
    if (NumElideArgs == 0) return;
    if (NumElideArgs == 1)
      OS << "[...]";
    else
      OS << "[" << NumElideArgs << " * ...]";
  }

  // Prints and highlights differences in Qualifiers.
  void PrintQualifiers(Qualifiers FromQual, Qualifiers ToQual) {
    // Both types have no qualifiers
    if (FromQual.empty() && ToQual.empty())
      return;

    // Both types have same qualifiers
    if (FromQual == ToQual) {
      PrintQualifier(FromQual, /*ApplyBold*/false);
      return;
    }

    // Find common qualifiers and strip them from FromQual and ToQual.
    Qualifiers CommonQual = Qualifiers::removeCommonQualifiers(FromQual,
                                                               ToQual);

    // The qualifiers are printed before the template name.
    // Inline printing:
    // The common qualifiers are printed.  Then, qualifiers only in this type
    // are printed and highlighted.  Finally, qualifiers only in the other
    // type are printed and highlighted inside parentheses after "missing".
    // Tree printing:
    // Qualifiers are printed next to each other, inside brackets, and
    // separated by "!=".  The printing order is:
    // common qualifiers, highlighted from qualifiers, "!=",
    // common qualifiers, highlighted to qualifiers
    if (PrintTree) {
      OS << "[";
      if (CommonQual.empty() && FromQual.empty()) {
        Bold();
        OS << "(no qualifiers) ";
        Unbold();
      } else {
        PrintQualifier(CommonQual, /*ApplyBold*/false);
        PrintQualifier(FromQual, /*ApplyBold*/true);
      }
      OS << "!= ";
      if (CommonQual.empty() && ToQual.empty()) {
        Bold();
        OS << "(no qualifiers)";
        Unbold();
      } else {
        PrintQualifier(CommonQual, /*ApplyBold*/false,
                       /*appendSpaceIfNonEmpty*/!ToQual.empty());
        PrintQualifier(ToQual, /*ApplyBold*/true,
                       /*appendSpaceIfNonEmpty*/false);
      }
      OS << "] ";
    } else {
      PrintQualifier(CommonQual, /*ApplyBold*/false);
      PrintQualifier(FromQual, /*ApplyBold*/true);
    }
  }

  void PrintQualifier(Qualifiers Q, bool ApplyBold,
                      bool AppendSpaceIfNonEmpty = true) {
    if (Q.empty()) return;
    if (ApplyBold) Bold();
    Q.print(OS, Policy, AppendSpaceIfNonEmpty);
    if (ApplyBold) Unbold();
  }

public:

  TemplateDiff(raw_ostream &OS, ASTContext &Context, QualType FromType,
               QualType ToType, bool PrintTree, bool PrintFromType,
               bool ElideType, bool ShowColor)
    : Context(Context),
      Policy(Context.getLangOpts()),
      ElideType(ElideType),
      PrintTree(PrintTree),
      ShowColor(ShowColor),
      // When printing a single type, the FromType is the one printed.
      FromType(PrintFromType ? FromType : ToType),
      ToType(PrintFromType ? ToType : FromType),
      OS(OS),
      IsBold(false) {
  }

  /// DiffTemplate - Start the template type diffing.
  void DiffTemplate() {
    Qualifiers FromQual = FromType.getQualifiers(),
               ToQual = ToType.getQualifiers();

    const TemplateSpecializationType *FromOrigTST =
        GetTemplateSpecializationType(Context, FromType);
    const TemplateSpecializationType *ToOrigTST =
        GetTemplateSpecializationType(Context, ToType);

    // Only checking templates.
    if (!FromOrigTST || !ToOrigTST)
      return;

    // Different base templates.
    if (!hasSameTemplate(FromOrigTST, ToOrigTST)) {
      return;
    }

    FromQual -= QualType(FromOrigTST, 0).getQualifiers();
    ToQual -= QualType(ToOrigTST, 0).getQualifiers();
    Tree.SetNode(FromType, ToType);
    Tree.SetNode(FromQual, ToQual);
    Tree.SetKind(DiffTree::Template);

    // Same base template, but different arguments.
    Tree.SetNode(FromOrigTST->getTemplateName().getAsTemplateDecl(),
                 ToOrigTST->getTemplateName().getAsTemplateDecl());

    DiffTemplate(FromOrigTST, ToOrigTST);
  }

  /// Emit - When the two types given are templated types with the same
  /// base template, a string representation of the type difference will be
  /// emitted to the stream and return true.  Otherwise, return false.
  bool Emit() {
    Tree.StartTraverse();
    if (Tree.Empty())
      return false;

    TreeToString();
    assert(!IsBold && "Bold is applied to end of string.");
    return true;
  }
}; // end class TemplateDiff
}  // end namespace

/// FormatTemplateTypeDiff - A helper static function to start the template
/// diff and return the properly formatted string.  Returns true if the diff
/// is successful.
static bool FormatTemplateTypeDiff(ASTContext &Context, QualType FromType,
                                   QualType ToType, bool PrintTree,
                                   bool PrintFromType, bool ElideType, 
                                   bool ShowColors, raw_ostream &OS) {
  if (PrintTree)
    PrintFromType = true;
  TemplateDiff TD(OS, Context, FromType, ToType, PrintTree, PrintFromType,
                  ElideType, ShowColors);
  TD.DiffTemplate();
  return TD.Emit();
}
