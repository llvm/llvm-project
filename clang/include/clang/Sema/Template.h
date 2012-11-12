//===------- SemaTemplate.h - C++ Templates ---------------------*- C++ -*-===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===/
//
//  This file provides types used in the semantic analysis of C++ templates.
//
//===----------------------------------------------------------------------===/
#ifndef LLVM_CLANG_SEMA_TEMPLATE_H
#define LLVM_CLANG_SEMA_TEMPLATE_H

#include "clang/AST/DeclTemplate.h"
#include "clang/AST/DeclVisitor.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <utility>

namespace clang {
  /// \brief Data structure that captures multiple levels of template argument
  /// lists for use in template instantiation.
  ///
  /// Multiple levels of template arguments occur when instantiating the 
  /// definitions of member templates. For example:
  ///
  /// \code
  /// template<typename T>
  /// struct X {
  ///   template<T Value>
  ///   struct Y {
  ///     void f();
  ///   };
  /// };
  /// \endcode
  ///
  /// When instantiating X<int>::Y<17>::f, the multi-level template argument
  /// list will contain a template argument list (int) at depth 0 and a
  /// template argument list (17) at depth 1.
  class MultiLevelTemplateArgumentList {
  public:
    typedef std::pair<const TemplateArgument *, unsigned> ArgList;
    
  private:
    /// \brief The template argument lists, stored from the innermost template
    /// argument list (first) to the outermost template argument list (last).
    SmallVector<ArgList, 4> TemplateArgumentLists;
    
  public:
    /// \brief Construct an empty set of template argument lists.
    MultiLevelTemplateArgumentList() { }
    
    /// \brief Construct a single-level template argument list.
    explicit 
    MultiLevelTemplateArgumentList(const TemplateArgumentList &TemplateArgs) {
      addOuterTemplateArguments(&TemplateArgs);
    }
    
    /// \brief Determine the number of levels in this template argument
    /// list.
    unsigned getNumLevels() const { return TemplateArgumentLists.size(); }
    
    /// \brief Retrieve the template argument at a given depth and index.
    const TemplateArgument &operator()(unsigned Depth, unsigned Index) const {
      assert(Depth < TemplateArgumentLists.size());
      assert(Index < TemplateArgumentLists[getNumLevels() - Depth - 1].second);
      return TemplateArgumentLists[getNumLevels() - Depth - 1].first[Index];
    }
    
    /// \brief Determine whether there is a non-NULL template argument at the
    /// given depth and index.
    ///
    /// There must exist a template argument list at the given depth.
    bool hasTemplateArgument(unsigned Depth, unsigned Index) const {
      assert(Depth < TemplateArgumentLists.size());
      
      if (Index >= TemplateArgumentLists[getNumLevels() - Depth - 1].second)
        return false;
      
      return !(*this)(Depth, Index).isNull();
    }
    
    /// \brief Clear out a specific template argument.
    void setArgument(unsigned Depth, unsigned Index,
                     TemplateArgument Arg) {
      assert(Depth < TemplateArgumentLists.size());
      assert(Index < TemplateArgumentLists[getNumLevels() - Depth - 1].second);
      const_cast<TemplateArgument&>(
                TemplateArgumentLists[getNumLevels() - Depth - 1].first[Index])
        = Arg;
    }
    
    /// \brief Add a new outermost level to the multi-level template argument 
    /// list.
    void addOuterTemplateArguments(const TemplateArgumentList *TemplateArgs) {
      TemplateArgumentLists.push_back(ArgList(TemplateArgs->data(),
                                              TemplateArgs->size()));
    }
    
    /// \brief Add a new outmost level to the multi-level template argument
    /// list.
    void addOuterTemplateArguments(const TemplateArgument *Args, 
                                   unsigned NumArgs) {
      TemplateArgumentLists.push_back(ArgList(Args, NumArgs));
    }
    
    /// \brief Retrieve the innermost template argument list.
    const ArgList &getInnermost() const { 
      return TemplateArgumentLists.front(); 
    }
  };
  
  /// \brief The context in which partial ordering of function templates occurs.
  enum TPOC {
    /// \brief Partial ordering of function templates for a function call.
    TPOC_Call,
    /// \brief Partial ordering of function templates for a call to a 
    /// conversion function.
    TPOC_Conversion,
    /// \brief Partial ordering of function templates in other contexts, e.g.,
    /// taking the address of a function template or matching a function 
    /// template specialization to a function template.
    TPOC_Other
  };

  // This is lame but unavoidable in a world without forward
  // declarations of enums.  The alternatives are to either pollute
  // Sema.h (by including this file) or sacrifice type safety (by
  // making Sema.h declare things as enums).
  class TemplatePartialOrderingContext {
    TPOC Value;
  public:
    TemplatePartialOrderingContext(TPOC Value) : Value(Value) {}
    operator TPOC() const { return Value; }
  };

  /// \brief Captures a template argument whose value has been deduced
  /// via c++ template argument deduction.
  class DeducedTemplateArgument : public TemplateArgument {
    /// \brief For a non-type template argument, whether the value was
    /// deduced from an array bound.
    bool DeducedFromArrayBound;

  public:
    DeducedTemplateArgument()
      : TemplateArgument(), DeducedFromArrayBound(false) { }

    DeducedTemplateArgument(const TemplateArgument &Arg,
                            bool DeducedFromArrayBound = false)
      : TemplateArgument(Arg), DeducedFromArrayBound(DeducedFromArrayBound) { }

    /// \brief Construct an integral non-type template argument that
    /// has been deduced, possibly from an array bound.
    DeducedTemplateArgument(ASTContext &Ctx,
                            const llvm::APSInt &Value,
                            QualType ValueType,
                            bool DeducedFromArrayBound)
      : TemplateArgument(Ctx, Value, ValueType),
        DeducedFromArrayBound(DeducedFromArrayBound) { }

    /// \brief For a non-type template argument, determine whether the
    /// template argument was deduced from an array bound.
    bool wasDeducedFromArrayBound() const { return DeducedFromArrayBound; }

    /// \brief Specify whether the given non-type template argument
    /// was deduced from an array bound.
    void setDeducedFromArrayBound(bool Deduced) {
      DeducedFromArrayBound = Deduced;
    }
  };

  /// \brief A stack-allocated class that identifies which local
  /// variable declaration instantiations are present in this scope.
  ///
  /// A new instance of this class type will be created whenever we
  /// instantiate a new function declaration, which will have its own
  /// set of parameter declarations.
  class LocalInstantiationScope {
  public:
    /// \brief A set of declarations.
    typedef SmallVector<Decl *, 4> DeclArgumentPack;
    
  private:
    /// \brief Reference to the semantic analysis that is performing
    /// this template instantiation.
    Sema &SemaRef;

    typedef llvm::DenseMap<const Decl *, 
                           llvm::PointerUnion<Decl *, DeclArgumentPack *> >
      LocalDeclsMap;
    
    /// \brief A mapping from local declarations that occur
    /// within a template to their instantiations.
    ///
    /// This mapping is used during instantiation to keep track of,
    /// e.g., function parameter and variable declarations. For example,
    /// given:
    ///
    /// \code
    ///   template<typename T> T add(T x, T y) { return x + y; }
    /// \endcode
    ///
    /// when we instantiate add<int>, we will introduce a mapping from
    /// the ParmVarDecl for 'x' that occurs in the template to the
    /// instantiated ParmVarDecl for 'x'.
    ///
    /// For a parameter pack, the local instantiation scope may contain a
    /// set of instantiated parameters. This is stored as a DeclArgumentPack
    /// pointer.
    LocalDeclsMap LocalDecls;

    /// \brief The set of argument packs we've allocated.
    SmallVector<DeclArgumentPack *, 1> ArgumentPacks;
    
    /// \brief The outer scope, which contains local variable
    /// definitions from some other instantiation (that may not be
    /// relevant to this particular scope).
    LocalInstantiationScope *Outer;

    /// \brief Whether we have already exited this scope.
    bool Exited;

    /// \brief Whether to combine this scope with the outer scope, such that
    /// lookup will search our outer scope.
    bool CombineWithOuterScope;
    
    /// \brief If non-NULL, the template parameter pack that has been
    /// partially substituted per C++0x [temp.arg.explicit]p9.
    NamedDecl *PartiallySubstitutedPack;
    
    /// \brief If \c PartiallySubstitutedPack is non-null, the set of
    /// explicitly-specified template arguments in that pack.
    const TemplateArgument *ArgsInPartiallySubstitutedPack;    
    
    /// \brief If \c PartiallySubstitutedPack, the number of 
    /// explicitly-specified template arguments in 
    /// ArgsInPartiallySubstitutedPack.
    unsigned NumArgsInPartiallySubstitutedPack;

    // This class is non-copyable
    LocalInstantiationScope(
      const LocalInstantiationScope &) LLVM_DELETED_FUNCTION;
    void operator=(const LocalInstantiationScope &) LLVM_DELETED_FUNCTION;

  public:
    LocalInstantiationScope(Sema &SemaRef, bool CombineWithOuterScope = false)
      : SemaRef(SemaRef), Outer(SemaRef.CurrentInstantiationScope),
        Exited(false), CombineWithOuterScope(CombineWithOuterScope),
        PartiallySubstitutedPack(0)
    {
      SemaRef.CurrentInstantiationScope = this;
    }

    ~LocalInstantiationScope() {
      Exit();
    }
    
    const Sema &getSema() const { return SemaRef; }

    /// \brief Exit this local instantiation scope early.
    void Exit() {
      if (Exited)
        return;
      
      for (unsigned I = 0, N = ArgumentPacks.size(); I != N; ++I)
        delete ArgumentPacks[I];
        
      SemaRef.CurrentInstantiationScope = Outer;
      Exited = true;
    }

    /// \brief Clone this scope, and all outer scopes, down to the given
    /// outermost scope.
    LocalInstantiationScope *cloneScopes(LocalInstantiationScope *Outermost) {
      if (this == Outermost) return this;
      LocalInstantiationScope *newScope =
        new LocalInstantiationScope(SemaRef, CombineWithOuterScope);

      newScope->Outer = 0;
      if (Outer)
        newScope->Outer = Outer->cloneScopes(Outermost);

      newScope->PartiallySubstitutedPack = PartiallySubstitutedPack;
      newScope->ArgsInPartiallySubstitutedPack = ArgsInPartiallySubstitutedPack;
      newScope->NumArgsInPartiallySubstitutedPack =
        NumArgsInPartiallySubstitutedPack;

      for (LocalDeclsMap::iterator I = LocalDecls.begin(), E = LocalDecls.end();
           I != E; ++I) {
        const Decl *D = I->first;
        llvm::PointerUnion<Decl *, DeclArgumentPack *> &Stored =
          newScope->LocalDecls[D];
        if (I->second.is<Decl *>()) {
          Stored = I->second.get<Decl *>();
        } else {
          DeclArgumentPack *OldPack = I->second.get<DeclArgumentPack *>();
          DeclArgumentPack *NewPack = new DeclArgumentPack(*OldPack);
          Stored = NewPack;
          newScope->ArgumentPacks.push_back(NewPack);
        }
      }
      return newScope;
    }

    /// \brief deletes the given scope, and all otuer scopes, down to the
    /// given outermost scope.
    static void deleteScopes(LocalInstantiationScope *Scope,
                             LocalInstantiationScope *Outermost) {
      while (Scope && Scope != Outermost) {
        LocalInstantiationScope *Out = Scope->Outer;
        delete Scope;
        Scope = Out;
      }
    }

    /// \brief Find the instantiation of the declaration D within the current
    /// instantiation scope.
    ///
    /// \param D The declaration whose instantiation we are searching for.
    ///
    /// \returns A pointer to the declaration or argument pack of declarations
    /// to which the declaration \c D is instantiataed, if found. Otherwise,
    /// returns NULL.
    llvm::PointerUnion<Decl *, DeclArgumentPack *> *
    findInstantiationOf(const Decl *D);

    void InstantiatedLocal(const Decl *D, Decl *Inst);
    void InstantiatedLocalPackArg(const Decl *D, Decl *Inst);
    void MakeInstantiatedLocalArgPack(const Decl *D);
    
    /// \brief Note that the given parameter pack has been partially substituted
    /// via explicit specification of template arguments 
    /// (C++0x [temp.arg.explicit]p9).
    ///
    /// \param Pack The parameter pack, which will always be a template
    /// parameter pack.
    ///
    /// \param ExplicitArgs The explicitly-specified template arguments provided
    /// for this parameter pack.
    ///
    /// \param NumExplicitArgs The number of explicitly-specified template
    /// arguments provided for this parameter pack.
    void SetPartiallySubstitutedPack(NamedDecl *Pack, 
                                     const TemplateArgument *ExplicitArgs,
                                     unsigned NumExplicitArgs);
    
    /// \brief Retrieve the partially-substitued template parameter pack.
    ///
    /// If there is no partially-substituted parameter pack, returns NULL.
    NamedDecl *getPartiallySubstitutedPack(
                                      const TemplateArgument **ExplicitArgs = 0,
                                           unsigned *NumExplicitArgs = 0) const;
  };

  class TemplateDeclInstantiator
    : public DeclVisitor<TemplateDeclInstantiator, Decl *> 
  {
    Sema &SemaRef;
    Sema::ArgumentPackSubstitutionIndexRAII SubstIndex;
    DeclContext *Owner;
    const MultiLevelTemplateArgumentList &TemplateArgs;
    Sema::LateInstantiatedAttrVec* LateAttrs;
    LocalInstantiationScope *StartingScope;

    /// \brief A list of out-of-line class template partial
    /// specializations that will need to be instantiated after the
    /// enclosing class's instantiation is complete.
    SmallVector<std::pair<ClassTemplateDecl *,
                                ClassTemplatePartialSpecializationDecl *>, 4>
      OutOfLinePartialSpecs;

  public:
    TemplateDeclInstantiator(Sema &SemaRef, DeclContext *Owner,
                             const MultiLevelTemplateArgumentList &TemplateArgs)
      : SemaRef(SemaRef),
        SubstIndex(SemaRef, SemaRef.ArgumentPackSubstitutionIndex),
        Owner(Owner), TemplateArgs(TemplateArgs), LateAttrs(0), StartingScope(0)
    { }

    // FIXME: Once we get closer to completion, replace these manually-written
    // declarations with automatically-generated ones from
    // clang/AST/DeclNodes.inc.
    Decl *VisitTranslationUnitDecl(TranslationUnitDecl *D);
    Decl *VisitLabelDecl(LabelDecl *D);
    Decl *VisitNamespaceDecl(NamespaceDecl *D);
    Decl *VisitNamespaceAliasDecl(NamespaceAliasDecl *D);
    Decl *VisitTypedefDecl(TypedefDecl *D);
    Decl *VisitTypeAliasDecl(TypeAliasDecl *D);
    Decl *VisitTypeAliasTemplateDecl(TypeAliasTemplateDecl *D);
    Decl *VisitVarDecl(VarDecl *D);
    Decl *VisitAccessSpecDecl(AccessSpecDecl *D);
    Decl *VisitFieldDecl(FieldDecl *D);
    Decl *VisitIndirectFieldDecl(IndirectFieldDecl *D);
    Decl *VisitStaticAssertDecl(StaticAssertDecl *D);
    Decl *VisitEnumDecl(EnumDecl *D);
    Decl *VisitEnumConstantDecl(EnumConstantDecl *D);
    Decl *VisitFriendDecl(FriendDecl *D);
    Decl *VisitFunctionDecl(FunctionDecl *D,
                            TemplateParameterList *TemplateParams = 0);
    Decl *VisitCXXRecordDecl(CXXRecordDecl *D);
    Decl *VisitCXXMethodDecl(CXXMethodDecl *D,
                             TemplateParameterList *TemplateParams = 0,
                             bool IsClassScopeSpecialization = false);
    Decl *VisitCXXConstructorDecl(CXXConstructorDecl *D);
    Decl *VisitCXXDestructorDecl(CXXDestructorDecl *D);
    Decl *VisitCXXConversionDecl(CXXConversionDecl *D);
    ParmVarDecl *VisitParmVarDecl(ParmVarDecl *D);
    Decl *VisitClassTemplateDecl(ClassTemplateDecl *D);
    Decl *VisitClassTemplatePartialSpecializationDecl(
                                    ClassTemplatePartialSpecializationDecl *D);
    Decl *VisitFunctionTemplateDecl(FunctionTemplateDecl *D);
    Decl *VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D);
    Decl *VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D);
    Decl *VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D);
    Decl *VisitUsingDirectiveDecl(UsingDirectiveDecl *D);
    Decl *VisitUsingDecl(UsingDecl *D);
    Decl *VisitUsingShadowDecl(UsingShadowDecl *D);
    Decl *VisitUnresolvedUsingValueDecl(UnresolvedUsingValueDecl *D);
    Decl *VisitUnresolvedUsingTypenameDecl(UnresolvedUsingTypenameDecl *D);
    Decl *VisitClassScopeFunctionSpecializationDecl(
                                      ClassScopeFunctionSpecializationDecl *D);

    // Base case. FIXME: Remove once we can instantiate everything.
    Decl *VisitDecl(Decl *D) {
      unsigned DiagID = SemaRef.getDiagnostics().getCustomDiagID(
                                                   DiagnosticsEngine::Error,
                                                   "cannot instantiate %0 yet");
      SemaRef.Diag(D->getLocation(), DiagID)
        << D->getDeclKindName();
      
      return 0;
    }
    
    // Enable late instantiation of attributes.  Late instantiated attributes
    // will be stored in LA.
    void enableLateAttributeInstantiation(Sema::LateInstantiatedAttrVec *LA) {
      LateAttrs = LA;
      StartingScope = SemaRef.CurrentInstantiationScope;
    }

    // Disable late instantiation of attributes.
    void disableLateAttributeInstantiation() {
      LateAttrs = 0;
      StartingScope = 0;
    }

    LocalInstantiationScope *getStartingScope() const { return StartingScope; }

    typedef 
      SmallVectorImpl<std::pair<ClassTemplateDecl *,
                                     ClassTemplatePartialSpecializationDecl *> >
        ::iterator
      delayed_partial_spec_iterator;

    /// \brief Return an iterator to the beginning of the set of
    /// "delayed" partial specializations, which must be passed to
    /// InstantiateClassTemplatePartialSpecialization once the class
    /// definition has been completed.
    delayed_partial_spec_iterator delayed_partial_spec_begin() {
      return OutOfLinePartialSpecs.begin();
    }

    /// \brief Return an iterator to the end of the set of
    /// "delayed" partial specializations, which must be passed to
    /// InstantiateClassTemplatePartialSpecialization once the class
    /// definition has been completed.
    delayed_partial_spec_iterator delayed_partial_spec_end() {
      return OutOfLinePartialSpecs.end();
    }

    // Helper functions for instantiating methods.
    TypeSourceInfo *SubstFunctionType(FunctionDecl *D,
                             SmallVectorImpl<ParmVarDecl *> &Params);
    bool InitFunctionInstantiation(FunctionDecl *New, FunctionDecl *Tmpl);
    bool InitMethodInstantiation(CXXMethodDecl *New, CXXMethodDecl *Tmpl);

    TemplateParameterList *
      SubstTemplateParams(TemplateParameterList *List);

    bool SubstQualifier(const DeclaratorDecl *OldDecl,
                        DeclaratorDecl *NewDecl);
    bool SubstQualifier(const TagDecl *OldDecl,
                        TagDecl *NewDecl);
      
    Decl *InstantiateTypedefNameDecl(TypedefNameDecl *D, bool IsTypeAlias);
    ClassTemplatePartialSpecializationDecl *
    InstantiateClassTemplatePartialSpecialization(
                                              ClassTemplateDecl *ClassTemplate,
                           ClassTemplatePartialSpecializationDecl *PartialSpec);
    void InstantiateEnumDefinition(EnumDecl *Enum, EnumDecl *Pattern);
  };  
}

#endif // LLVM_CLANG_SEMA_TEMPLATE_H
