//===--- DeclObjC.h - Classes for representing declarations -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the DeclObjC interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECLOBJC_H
#define LLVM_CLANG_AST_DECLOBJC_H

#include "clang/AST/Decl.h"
#include "clang/AST/SelectorLocationsKind.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"

namespace clang {
class Expr;
class Stmt;
class FunctionDecl;
class RecordDecl;
class ObjCIvarDecl;
class ObjCMethodDecl;
class ObjCProtocolDecl;
class ObjCCategoryDecl;
class ObjCPropertyDecl;
class ObjCPropertyImplDecl;
class CXXCtorInitializer;

class ObjCListBase {
  ObjCListBase(const ObjCListBase &) LLVM_DELETED_FUNCTION;
  void operator=(const ObjCListBase &) LLVM_DELETED_FUNCTION;
protected:
  /// List is an array of pointers to objects that are not owned by this object.
  void **List;
  unsigned NumElts;

public:
  ObjCListBase() : List(0), NumElts(0) {}
  unsigned size() const { return NumElts; }
  bool empty() const { return NumElts == 0; }

protected:
  void set(void *const* InList, unsigned Elts, ASTContext &Ctx);
};


/// ObjCList - This is a simple template class used to hold various lists of
/// decls etc, which is heavily used by the ObjC front-end.  This only use case
/// this supports is setting the list all at once and then reading elements out
/// of it.
template <typename T>
class ObjCList : public ObjCListBase {
public:
  void set(T* const* InList, unsigned Elts, ASTContext &Ctx) {
    ObjCListBase::set(reinterpret_cast<void*const*>(InList), Elts, Ctx);
  }

  typedef T* const * iterator;
  iterator begin() const { return (iterator)List; }
  iterator end() const { return (iterator)List+NumElts; }

  T* operator[](unsigned Idx) const {
    assert(Idx < NumElts && "Invalid access");
    return (T*)List[Idx];
  }
};

/// \brief A list of Objective-C protocols, along with the source
/// locations at which they were referenced.
class ObjCProtocolList : public ObjCList<ObjCProtocolDecl> {
  SourceLocation *Locations;

  using ObjCList<ObjCProtocolDecl>::set;

public:
  ObjCProtocolList() : ObjCList<ObjCProtocolDecl>(), Locations(0) { }

  typedef const SourceLocation *loc_iterator;
  loc_iterator loc_begin() const { return Locations; }
  loc_iterator loc_end() const { return Locations + size(); }

  void set(ObjCProtocolDecl* const* InList, unsigned Elts,
           const SourceLocation *Locs, ASTContext &Ctx);
};


/// ObjCMethodDecl - Represents an instance or class method declaration.
/// ObjC methods can be declared within 4 contexts: class interfaces,
/// categories, protocols, and class implementations. While C++ member
/// functions leverage C syntax, Objective-C method syntax is modeled after
/// Smalltalk (using colons to specify argument types/expressions).
/// Here are some brief examples:
///
/// Setter/getter instance methods:
/// - (void)setMenu:(NSMenu *)menu;
/// - (NSMenu *)menu;
///
/// Instance method that takes 2 NSView arguments:
/// - (void)replaceSubview:(NSView *)oldView with:(NSView *)newView;
///
/// Getter class method:
/// + (NSMenu *)defaultMenu;
///
/// A selector represents a unique name for a method. The selector names for
/// the above methods are setMenu:, menu, replaceSubview:with:, and defaultMenu.
///
class ObjCMethodDecl : public NamedDecl, public DeclContext {
public:
  enum ImplementationControl { None, Required, Optional };
private:
  // The conventional meaning of this method; an ObjCMethodFamily.
  // This is not serialized; instead, it is computed on demand and
  // cached.
  mutable unsigned Family : ObjCMethodFamilyBitWidth;

  /// instance (true) or class (false) method.
  unsigned IsInstance : 1;
  unsigned IsVariadic : 1;

  /// True if this method is the getter or setter for an explicit property.
  unsigned IsPropertyAccessor : 1;

  // Method has a definition.
  unsigned IsDefined : 1;

  /// \brief Method redeclaration in the same interface.
  unsigned IsRedeclaration : 1;

  /// \brief Is redeclared in the same interface.
  mutable unsigned HasRedeclaration : 1;

  // NOTE: VC++ treats enums as signed, avoid using ImplementationControl enum
  /// \@required/\@optional
  unsigned DeclImplementation : 2;

  // NOTE: VC++ treats enums as signed, avoid using the ObjCDeclQualifier enum
  /// in, inout, etc.
  unsigned objcDeclQualifier : 6;

  /// \brief Indicates whether this method has a related result type.
  unsigned RelatedResultType : 1;

  /// \brief Whether the locations of the selector identifiers are in a
  /// "standard" position, a enum SelectorLocationsKind.
  unsigned SelLocsKind : 2;

  /// \brief Whether this method overrides any other in the class hierarchy.
  ///
  /// A method is said to override any method in the class's
  /// base classes, its protocols, or its categories' protocols, that has
  /// the same selector and is of the same kind (class or instance).
  /// A method in an implementation is not considered as overriding the same
  /// method in the interface or its categories.
  unsigned IsOverriding : 1;

  // Result type of this method.
  QualType MethodDeclType;

  // Type source information for the result type.
  TypeSourceInfo *ResultTInfo;

  /// \brief Array of ParmVarDecls for the formal parameters of this method
  /// and optionally followed by selector locations.
  void *ParamsAndSelLocs;
  unsigned NumParams;

  /// List of attributes for this method declaration.
  SourceLocation DeclEndLoc; // the location of the ';' or '{'.

  // The following are only used for method definitions, null otherwise.
  LazyDeclStmtPtr Body;

  /// SelfDecl - Decl for the implicit self parameter. This is lazily
  /// constructed by createImplicitParams.
  ImplicitParamDecl *SelfDecl;
  /// CmdDecl - Decl for the implicit _cmd parameter. This is lazily
  /// constructed by createImplicitParams.
  ImplicitParamDecl *CmdDecl;

  SelectorLocationsKind getSelLocsKind() const {
    return (SelectorLocationsKind)SelLocsKind;
  }
  bool hasStandardSelLocs() const {
    return getSelLocsKind() != SelLoc_NonStandard;
  }

  /// \brief Get a pointer to the stored selector identifiers locations array.
  /// No locations will be stored if HasStandardSelLocs is true.
  SourceLocation *getStoredSelLocs() {
    return reinterpret_cast<SourceLocation*>(getParams() + NumParams);
  }
  const SourceLocation *getStoredSelLocs() const {
    return reinterpret_cast<const SourceLocation*>(getParams() + NumParams);
  }

  /// \brief Get a pointer to the stored selector identifiers locations array.
  /// No locations will be stored if HasStandardSelLocs is true.
  ParmVarDecl **getParams() {
    return reinterpret_cast<ParmVarDecl **>(ParamsAndSelLocs);
  }
  const ParmVarDecl *const *getParams() const {
    return reinterpret_cast<const ParmVarDecl *const *>(ParamsAndSelLocs);
  }

  /// \brief Get the number of stored selector identifiers locations.
  /// No locations will be stored if HasStandardSelLocs is true.
  unsigned getNumStoredSelLocs() const {
    if (hasStandardSelLocs())
      return 0;
    return getNumSelectorLocs();
  }

  void setParamsAndSelLocs(ASTContext &C,
                           ArrayRef<ParmVarDecl*> Params,
                           ArrayRef<SourceLocation> SelLocs);

  ObjCMethodDecl(SourceLocation beginLoc, SourceLocation endLoc,
                 Selector SelInfo, QualType T,
                 TypeSourceInfo *ResultTInfo,
                 DeclContext *contextDecl,
                 bool isInstance = true,
                 bool isVariadic = false,
                 bool isPropertyAccessor = false,
                 bool isImplicitlyDeclared = false,
                 bool isDefined = false,
                 ImplementationControl impControl = None,
                 bool HasRelatedResultType = false)
  : NamedDecl(ObjCMethod, contextDecl, beginLoc, SelInfo),
    DeclContext(ObjCMethod), Family(InvalidObjCMethodFamily),
    IsInstance(isInstance), IsVariadic(isVariadic),
    IsPropertyAccessor(isPropertyAccessor),
    IsDefined(isDefined), IsRedeclaration(0), HasRedeclaration(0),
    DeclImplementation(impControl), objcDeclQualifier(OBJC_TQ_None),
    RelatedResultType(HasRelatedResultType),
    SelLocsKind(SelLoc_StandardNoSpace), IsOverriding(0),
    MethodDeclType(T), ResultTInfo(ResultTInfo),
    ParamsAndSelLocs(0), NumParams(0),
    DeclEndLoc(endLoc), Body(), SelfDecl(0), CmdDecl(0) {
    setImplicit(isImplicitlyDeclared);
  }

  /// \brief A definition will return its interface declaration.
  /// An interface declaration will return its definition.
  /// Otherwise it will return itself.
  virtual ObjCMethodDecl *getNextRedeclaration();

public:
  static ObjCMethodDecl *Create(ASTContext &C,
                                SourceLocation beginLoc,
                                SourceLocation endLoc,
                                Selector SelInfo,
                                QualType T,
                                TypeSourceInfo *ResultTInfo,
                                DeclContext *contextDecl,
                                bool isInstance = true,
                                bool isVariadic = false,
                                bool isPropertyAccessor = false,
                                bool isImplicitlyDeclared = false,
                                bool isDefined = false,
                                ImplementationControl impControl = None,
                                bool HasRelatedResultType = false);

  static ObjCMethodDecl *CreateDeserialized(ASTContext &C, unsigned ID);
  
  virtual ObjCMethodDecl *getCanonicalDecl();
  const ObjCMethodDecl *getCanonicalDecl() const {
    return const_cast<ObjCMethodDecl*>(this)->getCanonicalDecl();
  }

  ObjCDeclQualifier getObjCDeclQualifier() const {
    return ObjCDeclQualifier(objcDeclQualifier);
  }
  void setObjCDeclQualifier(ObjCDeclQualifier QV) { objcDeclQualifier = QV; }

  /// \brief Determine whether this method has a result type that is related
  /// to the message receiver's type.
  bool hasRelatedResultType() const { return RelatedResultType; }

  /// \brief Note whether this method has a related result type.
  void SetRelatedResultType(bool RRT = true) { RelatedResultType = RRT; }

  /// \brief True if this is a method redeclaration in the same interface.
  bool isRedeclaration() const { return IsRedeclaration; }
  void setAsRedeclaration(const ObjCMethodDecl *PrevMethod);

  /// \brief Returns the location where the declarator ends. It will be
  /// the location of ';' for a method declaration and the location of '{'
  /// for a method definition.
  SourceLocation getDeclaratorEndLoc() const { return DeclEndLoc; }

  // Location information, modeled after the Stmt API.
  SourceLocation getLocStart() const LLVM_READONLY { return getLocation(); }
  SourceLocation getLocEnd() const LLVM_READONLY;
  virtual SourceRange getSourceRange() const LLVM_READONLY {
    return SourceRange(getLocation(), getLocEnd());
  }

  SourceLocation getSelectorStartLoc() const {
    if (isImplicit())
      return getLocStart();
    return getSelectorLoc(0);
  }
  SourceLocation getSelectorLoc(unsigned Index) const {
    assert(Index < getNumSelectorLocs() && "Index out of range!");
    if (hasStandardSelLocs())
      return getStandardSelectorLoc(Index, getSelector(),
                                   getSelLocsKind() == SelLoc_StandardWithSpace,
                      llvm::makeArrayRef(const_cast<ParmVarDecl**>(getParams()),
                                         NumParams),
                                   DeclEndLoc);
    return getStoredSelLocs()[Index];
  }

  void getSelectorLocs(SmallVectorImpl<SourceLocation> &SelLocs) const;

  unsigned getNumSelectorLocs() const {
    if (isImplicit())
      return 0;
    Selector Sel = getSelector();
    if (Sel.isUnarySelector())
      return 1;
    return Sel.getNumArgs();
  }

  ObjCInterfaceDecl *getClassInterface();
  const ObjCInterfaceDecl *getClassInterface() const {
    return const_cast<ObjCMethodDecl*>(this)->getClassInterface();
  }

  Selector getSelector() const { return getDeclName().getObjCSelector(); }

  QualType getResultType() const { return MethodDeclType; }
  void setResultType(QualType T) { MethodDeclType = T; }

  /// \brief Determine the type of an expression that sends a message to this
  /// function.
  QualType getSendResultType() const {
    return getResultType().getNonLValueExprType(getASTContext());
  }

  TypeSourceInfo *getResultTypeSourceInfo() const { return ResultTInfo; }
  void setResultTypeSourceInfo(TypeSourceInfo *TInfo) { ResultTInfo = TInfo; }

  // Iterator access to formal parameters.
  unsigned param_size() const { return NumParams; }
  typedef const ParmVarDecl *const *param_const_iterator;
  typedef ParmVarDecl *const *param_iterator;
  param_const_iterator param_begin() const { return getParams(); }
  param_const_iterator param_end() const { return getParams() + NumParams; }
  param_iterator param_begin() { return getParams(); }
  param_iterator param_end() { return getParams() + NumParams; }
  // This method returns and of the parameters which are part of the selector
  // name mangling requirements.
  param_const_iterator sel_param_end() const {
    return param_begin() + getSelector().getNumArgs();
  }

  /// \brief Sets the method's parameters and selector source locations.
  /// If the method is implicit (not coming from source) \p SelLocs is
  /// ignored.
  void setMethodParams(ASTContext &C,
                       ArrayRef<ParmVarDecl*> Params,
                       ArrayRef<SourceLocation> SelLocs =
                           ArrayRef<SourceLocation>());

  // Iterator access to parameter types.
  typedef std::const_mem_fun_t<QualType, ParmVarDecl> deref_fun;
  typedef llvm::mapped_iterator<param_const_iterator, deref_fun>
      arg_type_iterator;

  arg_type_iterator arg_type_begin() const {
    return llvm::map_iterator(param_begin(), deref_fun(&ParmVarDecl::getType));
  }
  arg_type_iterator arg_type_end() const {
    return llvm::map_iterator(param_end(), deref_fun(&ParmVarDecl::getType));
  }

  /// createImplicitParams - Used to lazily create the self and cmd
  /// implict parameters. This must be called prior to using getSelfDecl()
  /// or getCmdDecl(). The call is ignored if the implicit paramters
  /// have already been created.
  void createImplicitParams(ASTContext &Context, const ObjCInterfaceDecl *ID);

  ImplicitParamDecl * getSelfDecl() const { return SelfDecl; }
  void setSelfDecl(ImplicitParamDecl *SD) { SelfDecl = SD; }
  ImplicitParamDecl * getCmdDecl() const { return CmdDecl; }
  void setCmdDecl(ImplicitParamDecl *CD) { CmdDecl = CD; }

  /// Determines the family of this method.
  ObjCMethodFamily getMethodFamily() const;

  bool isInstanceMethod() const { return IsInstance; }
  void setInstanceMethod(bool isInst) { IsInstance = isInst; }
  bool isVariadic() const { return IsVariadic; }
  void setVariadic(bool isVar) { IsVariadic = isVar; }

  bool isClassMethod() const { return !IsInstance; }

  bool isPropertyAccessor() const { return IsPropertyAccessor; }
  void setPropertyAccessor(bool isAccessor) { IsPropertyAccessor = isAccessor; }

  bool isDefined() const { return IsDefined; }
  void setDefined(bool isDefined) { IsDefined = isDefined; }

  /// \brief Whether this method overrides any other in the class hierarchy.
  ///
  /// A method is said to override any method in the class's
  /// base classes, its protocols, or its categories' protocols, that has
  /// the same selector and is of the same kind (class or instance).
  /// A method in an implementation is not considered as overriding the same
  /// method in the interface or its categories.
  bool isOverriding() const { return IsOverriding; }
  void setOverriding(bool isOverriding) { IsOverriding = isOverriding; }

  /// \brief Return overridden methods for the given \p Method.
  ///
  /// An ObjC method is considered to override any method in the class's
  /// base classes (and base's categories), its protocols, or its categories'
  /// protocols, that has
  /// the same selector and is of the same kind (class or instance).
  /// A method in an implementation is not considered as overriding the same
  /// method in the interface or its categories.
  void getOverriddenMethods(
                     SmallVectorImpl<const ObjCMethodDecl *> &Overridden) const;

  /// \brief Returns the property associated with this method's selector.
  ///
  /// Note that even if this particular method is not marked as a property
  /// accessor, it is still possible for it to match a property declared in a
  /// superclass. Pass \c false if you only want to check the current class.
  const ObjCPropertyDecl *findPropertyDecl(bool CheckOverrides = true) const;

  // Related to protocols declared in  \@protocol
  void setDeclImplementation(ImplementationControl ic) {
    DeclImplementation = ic;
  }
  ImplementationControl getImplementationControl() const {
    return ImplementationControl(DeclImplementation);
  }

  /// \brief Determine whether this method has a body.
  virtual bool hasBody() const { return Body; }

  /// \brief Retrieve the body of this method, if it has one.
  virtual Stmt *getBody() const;

  void setLazyBody(uint64_t Offset) { Body = Offset; }

  CompoundStmt *getCompoundBody() { return (CompoundStmt*)getBody(); }
  void setBody(Stmt *B) { Body = B; }

  /// \brief Returns whether this specific method is a definition.
  bool isThisDeclarationADefinition() const { return Body; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == ObjCMethod; }
  static DeclContext *castToDeclContext(const ObjCMethodDecl *D) {
    return static_cast<DeclContext *>(const_cast<ObjCMethodDecl*>(D));
  }
  static ObjCMethodDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<ObjCMethodDecl *>(const_cast<DeclContext*>(DC));
  }

  friend class ASTDeclReader;
  friend class ASTDeclWriter;
};

/// ObjCContainerDecl - Represents a container for method declarations.
/// Current sub-classes are ObjCInterfaceDecl, ObjCCategoryDecl,
/// ObjCProtocolDecl, and ObjCImplDecl.
///
class ObjCContainerDecl : public NamedDecl, public DeclContext {
  virtual void anchor();

  SourceLocation AtStart;

  // These two locations in the range mark the end of the method container.
  // The first points to the '@' token, and the second to the 'end' token.
  SourceRange AtEnd;
public:

  ObjCContainerDecl(Kind DK, DeclContext *DC,
                    IdentifierInfo *Id, SourceLocation nameLoc,
                    SourceLocation atStartLoc)
    : NamedDecl(DK, DC, nameLoc, Id), DeclContext(DK), AtStart(atStartLoc) {}

  // Iterator access to properties.
  typedef specific_decl_iterator<ObjCPropertyDecl> prop_iterator;
  prop_iterator prop_begin() const {
    return prop_iterator(decls_begin());
  }
  prop_iterator prop_end() const {
    return prop_iterator(decls_end());
  }

  // Iterator access to instance/class methods.
  typedef specific_decl_iterator<ObjCMethodDecl> method_iterator;
  method_iterator meth_begin() const {
    return method_iterator(decls_begin());
  }
  method_iterator meth_end() const {
    return method_iterator(decls_end());
  }

  typedef filtered_decl_iterator<ObjCMethodDecl,
                                 &ObjCMethodDecl::isInstanceMethod>
    instmeth_iterator;
  instmeth_iterator instmeth_begin() const {
    return instmeth_iterator(decls_begin());
  }
  instmeth_iterator instmeth_end() const {
    return instmeth_iterator(decls_end());
  }

  typedef filtered_decl_iterator<ObjCMethodDecl,
                                 &ObjCMethodDecl::isClassMethod>
    classmeth_iterator;
  classmeth_iterator classmeth_begin() const {
    return classmeth_iterator(decls_begin());
  }
  classmeth_iterator classmeth_end() const {
    return classmeth_iterator(decls_end());
  }

  // Get the local instance/class method declared in this interface.
  ObjCMethodDecl *getMethod(Selector Sel, bool isInstance) const;
  ObjCMethodDecl *getInstanceMethod(Selector Sel) const {
    return getMethod(Sel, true/*isInstance*/);
  }
  ObjCMethodDecl *getClassMethod(Selector Sel) const {
    return getMethod(Sel, false/*isInstance*/);
  }
  ObjCIvarDecl *getIvarDecl(IdentifierInfo *Id) const;

  ObjCPropertyDecl *FindPropertyDeclaration(IdentifierInfo *PropertyId) const;

  typedef llvm::DenseMap<IdentifierInfo*, ObjCPropertyDecl*> PropertyMap;

  /// This routine collects list of properties to be implemented in the class.
  /// This includes, class's and its conforming protocols' properties.
  /// Note, the superclass's properties are not included in the list.
  virtual void collectPropertiesToImplement(PropertyMap &PM) const {}

  SourceLocation getAtStartLoc() const { return AtStart; }
  void setAtStartLoc(SourceLocation Loc) { AtStart = Loc; }

  // Marks the end of the container.
  SourceRange getAtEndRange() const {
    return AtEnd;
  }
  void setAtEndRange(SourceRange atEnd) {
    AtEnd = atEnd;
  }

  virtual SourceRange getSourceRange() const LLVM_READONLY {
    return SourceRange(AtStart, getAtEndRange().getEnd());
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) {
    return K >= firstObjCContainer &&
           K <= lastObjCContainer;
  }

  static DeclContext *castToDeclContext(const ObjCContainerDecl *D) {
    return static_cast<DeclContext *>(const_cast<ObjCContainerDecl*>(D));
  }
  static ObjCContainerDecl *castFromDeclContext(const DeclContext *DC) {
    return static_cast<ObjCContainerDecl *>(const_cast<DeclContext*>(DC));
  }
};

/// \brief Represents an ObjC class declaration.
///
/// For example:
///
/// \code
///   // MostPrimitive declares no super class (not particularly useful).
///   \@interface MostPrimitive
///     // no instance variables or methods.
///   \@end
///
///   // NSResponder inherits from NSObject & implements NSCoding (a protocol).
///   \@interface NSResponder : NSObject \<NSCoding>
///   { // instance variables are represented by ObjCIvarDecl.
///     id nextResponder; // nextResponder instance variable.
///   }
///   - (NSResponder *)nextResponder; // return a pointer to NSResponder.
///   - (void)mouseMoved:(NSEvent *)theEvent; // return void, takes a pointer
///   \@end                                    // to an NSEvent.
/// \endcode
///
///   Unlike C/C++, forward class declarations are accomplished with \@class.
///   Unlike C/C++, \@class allows for a list of classes to be forward declared.
///   Unlike C++, ObjC is a single-rooted class model. In Cocoa, classes
///   typically inherit from NSObject (an exception is NSProxy).
///
class ObjCInterfaceDecl : public ObjCContainerDecl
                        , public Redeclarable<ObjCInterfaceDecl> {
  virtual void anchor();

  /// TypeForDecl - This indicates the Type object that represents this
  /// TypeDecl.  It is a cache maintained by ASTContext::getObjCInterfaceType
  mutable const Type *TypeForDecl;
  friend class ASTContext;
  
  struct DefinitionData {
    /// \brief The definition of this class, for quick access from any 
    /// declaration.
    ObjCInterfaceDecl *Definition;
    
    /// Class's super class.
    ObjCInterfaceDecl *SuperClass;

    /// Protocols referenced in the \@interface  declaration
    ObjCProtocolList ReferencedProtocols;

    /// Protocols reference in both the \@interface and class extensions.
    ObjCList<ObjCProtocolDecl> AllReferencedProtocols;

    /// \brief List of categories and class extensions defined for this class.
    ///
    /// Categories are stored as a linked list in the AST, since the categories
    /// and class extensions come long after the initial interface declaration,
    /// and we avoid dynamically-resized arrays in the AST wherever possible.
    ObjCCategoryDecl *CategoryList;

    /// IvarList - List of all ivars defined by this class; including class
    /// extensions and implementation. This list is built lazily.
    ObjCIvarDecl *IvarList;

    /// \brief Indicates that the contents of this Objective-C class will be
    /// completed by the external AST source when required.
    mutable bool ExternallyCompleted : 1;

    /// \brief The location of the superclass, if any.
    SourceLocation SuperClassLoc;
    
    /// \brief The location of the last location in this declaration, before
    /// the properties/methods. For example, this will be the '>', '}', or 
    /// identifier, 
    SourceLocation EndLoc; 

    DefinitionData() : Definition(), SuperClass(), CategoryList(), IvarList(), 
                       ExternallyCompleted() { }
  };

  ObjCInterfaceDecl(DeclContext *DC, SourceLocation atLoc, IdentifierInfo *Id,
                    SourceLocation CLoc, ObjCInterfaceDecl *PrevDecl,
                    bool isInternal);

  void LoadExternalDefinition() const;

  /// \brief Contains a pointer to the data associated with this class,
  /// which will be NULL if this class has not yet been defined.
  DefinitionData *Data;

  DefinitionData &data() const {
    assert(Data != 0 && "Declaration has no definition!");
    return *Data;
  }

  /// \brief Allocate the definition data for this class.
  void allocateDefinitionData();
  
  typedef Redeclarable<ObjCInterfaceDecl> redeclarable_base;
  virtual ObjCInterfaceDecl *getNextRedeclaration() { 
    return RedeclLink.getNext(); 
  }
  virtual ObjCInterfaceDecl *getPreviousDeclImpl() {
    return getPreviousDecl();
  }
  virtual ObjCInterfaceDecl *getMostRecentDeclImpl() {
    return getMostRecentDecl();
  }

public:
  static ObjCInterfaceDecl *Create(const ASTContext &C, DeclContext *DC,
                                   SourceLocation atLoc,
                                   IdentifierInfo *Id,
                                   ObjCInterfaceDecl *PrevDecl,
                                   SourceLocation ClassLoc = SourceLocation(),
                                   bool isInternal = false);

  static ObjCInterfaceDecl *CreateDeserialized(ASTContext &C, unsigned ID);

  virtual SourceRange getSourceRange() const LLVM_READONLY {
    if (isThisDeclarationADefinition())
      return ObjCContainerDecl::getSourceRange();
    
    return SourceRange(getAtStartLoc(), getLocation());
  }

  /// \brief Indicate that this Objective-C class is complete, but that
  /// the external AST source will be responsible for filling in its contents
  /// when a complete class is required.
  void setExternallyCompleted();

  const ObjCProtocolList &getReferencedProtocols() const {
    assert(hasDefinition() && "Caller did not check for forward reference!");
    if (data().ExternallyCompleted)
      LoadExternalDefinition();

    return data().ReferencedProtocols;
  }

  ObjCImplementationDecl *getImplementation() const;
  void setImplementation(ObjCImplementationDecl *ImplD);

  ObjCCategoryDecl *FindCategoryDeclaration(IdentifierInfo *CategoryId) const;

  // Get the local instance/class method declared in a category.
  ObjCMethodDecl *getCategoryInstanceMethod(Selector Sel) const;
  ObjCMethodDecl *getCategoryClassMethod(Selector Sel) const;
  ObjCMethodDecl *getCategoryMethod(Selector Sel, bool isInstance) const {
    return isInstance ? getInstanceMethod(Sel)
                      : getClassMethod(Sel);
  }

  typedef ObjCProtocolList::iterator protocol_iterator;

  protocol_iterator protocol_begin() const {
    // FIXME: Should make sure no callers ever do this.
    if (!hasDefinition())
      return protocol_iterator();
    
    if (data().ExternallyCompleted)
      LoadExternalDefinition();

    return data().ReferencedProtocols.begin();
  }
  protocol_iterator protocol_end() const {
    // FIXME: Should make sure no callers ever do this.
    if (!hasDefinition())
      return protocol_iterator();

    if (data().ExternallyCompleted)
      LoadExternalDefinition();

    return data().ReferencedProtocols.end();
  }

  typedef ObjCProtocolList::loc_iterator protocol_loc_iterator;

  protocol_loc_iterator protocol_loc_begin() const {
    // FIXME: Should make sure no callers ever do this.
    if (!hasDefinition())
      return protocol_loc_iterator();

    if (data().ExternallyCompleted)
      LoadExternalDefinition();

    return data().ReferencedProtocols.loc_begin();
  }

  protocol_loc_iterator protocol_loc_end() const {
    // FIXME: Should make sure no callers ever do this.
    if (!hasDefinition())
      return protocol_loc_iterator();

    if (data().ExternallyCompleted)
      LoadExternalDefinition();

    return data().ReferencedProtocols.loc_end();
  }

  typedef ObjCList<ObjCProtocolDecl>::iterator all_protocol_iterator;

  all_protocol_iterator all_referenced_protocol_begin() const {
    // FIXME: Should make sure no callers ever do this.
    if (!hasDefinition())
      return all_protocol_iterator();

    if (data().ExternallyCompleted)
      LoadExternalDefinition();

    return data().AllReferencedProtocols.empty()  
             ? protocol_begin()
             : data().AllReferencedProtocols.begin();
  }
  all_protocol_iterator all_referenced_protocol_end() const {
    // FIXME: Should make sure no callers ever do this.
    if (!hasDefinition())
      return all_protocol_iterator();
    
    if (data().ExternallyCompleted)
      LoadExternalDefinition();

    return data().AllReferencedProtocols.empty() 
             ? protocol_end()
             : data().AllReferencedProtocols.end();
  }

  typedef specific_decl_iterator<ObjCIvarDecl> ivar_iterator;

  ivar_iterator ivar_begin() const { 
    if (const ObjCInterfaceDecl *Def = getDefinition())
      return ivar_iterator(Def->decls_begin()); 
    
    // FIXME: Should make sure no callers ever do this.
    return ivar_iterator();
  }
  ivar_iterator ivar_end() const { 
    if (const ObjCInterfaceDecl *Def = getDefinition())
      return ivar_iterator(Def->decls_end()); 

    // FIXME: Should make sure no callers ever do this.
    return ivar_iterator();
  }

  unsigned ivar_size() const {
    return std::distance(ivar_begin(), ivar_end());
  }

  bool ivar_empty() const { return ivar_begin() == ivar_end(); }

  ObjCIvarDecl *all_declared_ivar_begin();
  const ObjCIvarDecl *all_declared_ivar_begin() const {
    // Even though this modifies IvarList, it's conceptually const:
    // the ivar chain is essentially a cached property of ObjCInterfaceDecl.
    return const_cast<ObjCInterfaceDecl *>(this)->all_declared_ivar_begin();
  }
  void setIvarList(ObjCIvarDecl *ivar) { data().IvarList = ivar; }

  /// setProtocolList - Set the list of protocols that this interface
  /// implements.
  void setProtocolList(ObjCProtocolDecl *const* List, unsigned Num,
                       const SourceLocation *Locs, ASTContext &C) {
    data().ReferencedProtocols.set(List, Num, Locs, C);
  }

  /// mergeClassExtensionProtocolList - Merge class extension's protocol list
  /// into the protocol list for this class.
  void mergeClassExtensionProtocolList(ObjCProtocolDecl *const* List,
                                       unsigned Num,
                                       ASTContext &C);

  /// \brief Determine whether this particular declaration of this class is
  /// actually also a definition.
  bool isThisDeclarationADefinition() const { 
    return Data && Data->Definition == this;
  }
                          
  /// \brief Determine whether this class has been defined.
  bool hasDefinition() const { return Data; }
                        
  /// \brief Retrieve the definition of this class, or NULL if this class 
  /// has been forward-declared (with \@class) but not yet defined (with 
  /// \@interface).
  ObjCInterfaceDecl *getDefinition() {
    return hasDefinition()? Data->Definition : 0;
  }

  /// \brief Retrieve the definition of this class, or NULL if this class 
  /// has been forward-declared (with \@class) but not yet defined (with 
  /// \@interface).
  const ObjCInterfaceDecl *getDefinition() const {
    return hasDefinition()? Data->Definition : 0;
  }

  /// \brief Starts the definition of this Objective-C class, taking it from
  /// a forward declaration (\@class) to a definition (\@interface).
  void startDefinition();
  
  ObjCInterfaceDecl *getSuperClass() const {
    // FIXME: Should make sure no callers ever do this.
    if (!hasDefinition())
      return 0;
    
    if (data().ExternallyCompleted)
      LoadExternalDefinition();

    return data().SuperClass;
  }

  void setSuperClass(ObjCInterfaceDecl * superCls) { 
    data().SuperClass = 
      (superCls && superCls->hasDefinition()) ? superCls->getDefinition() 
                                              : superCls; 
  }

  ObjCCategoryDecl* getCategoryList() const {
    // FIXME: Should make sure no callers ever do this.
    if (!hasDefinition())
      return 0;
    
    if (data().ExternallyCompleted)
      LoadExternalDefinition();

    return data().CategoryList;
  }

  void setCategoryList(ObjCCategoryDecl *category) {
    data().CategoryList = category;
  }

  ObjCCategoryDecl* getFirstClassExtension() const;

  ObjCPropertyDecl
    *FindPropertyVisibleInPrimaryClass(IdentifierInfo *PropertyId) const;

  virtual void collectPropertiesToImplement(PropertyMap &PM) const;

  /// isSuperClassOf - Return true if this class is the specified class or is a
  /// super class of the specified interface class.
  bool isSuperClassOf(const ObjCInterfaceDecl *I) const {
    // If RHS is derived from LHS it is OK; else it is not OK.
    while (I != NULL) {
      if (declaresSameEntity(this, I))
        return true;
      
      I = I->getSuperClass();
    }
    return false;
  }

  /// isArcWeakrefUnavailable - Checks for a class or one of its super classes
  /// to be incompatible with __weak references. Returns true if it is.
  bool isArcWeakrefUnavailable() const {
    const ObjCInterfaceDecl *Class = this;
    while (Class) {
      if (Class->hasAttr<ArcWeakrefUnavailableAttr>())
        return true;
      Class = Class->getSuperClass();
   }
   return false;
  }

  /// isObjCRequiresPropertyDefs - Checks that a class or one of its super 
  /// classes must not be auto-synthesized. Returns class decl. if it must not
  /// be; 0, otherwise.
  const ObjCInterfaceDecl *isObjCRequiresPropertyDefs() const {
    const ObjCInterfaceDecl *Class = this;
    while (Class) {
      if (Class->hasAttr<ObjCRequiresPropertyDefsAttr>())
        return Class;
      Class = Class->getSuperClass();
   }
   return 0;
  }

  ObjCIvarDecl *lookupInstanceVariable(IdentifierInfo *IVarName,
                                       ObjCInterfaceDecl *&ClassDeclared);
  ObjCIvarDecl *lookupInstanceVariable(IdentifierInfo *IVarName) {
    ObjCInterfaceDecl *ClassDeclared;
    return lookupInstanceVariable(IVarName, ClassDeclared);
  }

  // Lookup a method. First, we search locally. If a method isn't
  // found, we search referenced protocols and class categories.
  ObjCMethodDecl *lookupMethod(Selector Sel, bool isInstance,
                               bool shallowCategoryLookup= false) const;
  ObjCMethodDecl *lookupInstanceMethod(Selector Sel,
                            bool shallowCategoryLookup = false) const {
    return lookupMethod(Sel, true/*isInstance*/, shallowCategoryLookup);
  }
  ObjCMethodDecl *lookupClassMethod(Selector Sel,
                     bool shallowCategoryLookup = false) const {
    return lookupMethod(Sel, false/*isInstance*/, shallowCategoryLookup);
  }
  ObjCInterfaceDecl *lookupInheritedClass(const IdentifierInfo *ICName);

  /// \brief Lookup a method in the classes implementation hierarchy.
  ObjCMethodDecl *lookupPrivateMethod(const Selector &Sel,
                                      bool Instance=true) const;

  ObjCMethodDecl *lookupPrivateClassMethod(const Selector &Sel) {
    return lookupPrivateMethod(Sel, false);
  }

  SourceLocation getEndOfDefinitionLoc() const { 
    if (!hasDefinition())
      return getLocation();
    
    return data().EndLoc; 
  }
                          
  void setEndOfDefinitionLoc(SourceLocation LE) { data().EndLoc = LE; }

  void setSuperClassLoc(SourceLocation Loc) { data().SuperClassLoc = Loc; }
  SourceLocation getSuperClassLoc() const { return data().SuperClassLoc; }

  /// isImplicitInterfaceDecl - check that this is an implicitly declared
  /// ObjCInterfaceDecl node. This is for legacy objective-c \@implementation
  /// declaration without an \@interface declaration.
  bool isImplicitInterfaceDecl() const { 
    return hasDefinition() ? Data->Definition->isImplicit() : isImplicit(); 
  }

  /// ClassImplementsProtocol - Checks that 'lProto' protocol
  /// has been implemented in IDecl class, its super class or categories (if
  /// lookupCategory is true).
  bool ClassImplementsProtocol(ObjCProtocolDecl *lProto,
                               bool lookupCategory,
                               bool RHSIsQualifiedID = false);

  typedef redeclarable_base::redecl_iterator redecl_iterator;
  using redeclarable_base::redecls_begin;
  using redeclarable_base::redecls_end;
  using redeclarable_base::getPreviousDecl;
  using redeclarable_base::getMostRecentDecl;

  /// Retrieves the canonical declaration of this Objective-C class.
  ObjCInterfaceDecl *getCanonicalDecl() {
    return getFirstDeclaration();
  }
  const ObjCInterfaceDecl *getCanonicalDecl() const {
    return getFirstDeclaration();
  }

  // Low-level accessor
  const Type *getTypeForDecl() const { return TypeForDecl; }
  void setTypeForDecl(const Type *TD) const { TypeForDecl = TD; }

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == ObjCInterface; }

  friend class ASTReader;
  friend class ASTDeclReader;
  friend class ASTDeclWriter;
};

/// ObjCIvarDecl - Represents an ObjC instance variable. In general, ObjC
/// instance variables are identical to C. The only exception is Objective-C
/// supports C++ style access control. For example:
///
///   \@interface IvarExample : NSObject
///   {
///     id defaultToProtected;
///   \@public:
///     id canBePublic; // same as C++.
///   \@protected:
///     id canBeProtected; // same as C++.
///   \@package:
///     id canBePackage; // framework visibility (not available in C++).
///   }
///
class ObjCIvarDecl : public FieldDecl {
  virtual void anchor();

public:
  enum AccessControl {
    None, Private, Protected, Public, Package
  };

private:
  ObjCIvarDecl(ObjCContainerDecl *DC, SourceLocation StartLoc,
               SourceLocation IdLoc, IdentifierInfo *Id,
               QualType T, TypeSourceInfo *TInfo, AccessControl ac, Expr *BW,
               bool synthesized)
    : FieldDecl(ObjCIvar, DC, StartLoc, IdLoc, Id, T, TInfo, BW,
                /*Mutable=*/false, /*HasInit=*/ICIS_NoInit),
      NextIvar(0), DeclAccess(ac), Synthesized(synthesized) {}

public:
  static ObjCIvarDecl *Create(ASTContext &C, ObjCContainerDecl *DC,
                              SourceLocation StartLoc, SourceLocation IdLoc,
                              IdentifierInfo *Id, QualType T,
                              TypeSourceInfo *TInfo,
                              AccessControl ac, Expr *BW = NULL,
                              bool synthesized=false);

  static ObjCIvarDecl *CreateDeserialized(ASTContext &C, unsigned ID);
  
  /// \brief Return the class interface that this ivar is logically contained
  /// in; this is either the interface where the ivar was declared, or the
  /// interface the ivar is conceptually a part of in the case of synthesized
  /// ivars.
  const ObjCInterfaceDecl *getContainingInterface() const;

  ObjCIvarDecl *getNextIvar() { return NextIvar; }
  const ObjCIvarDecl *getNextIvar() const { return NextIvar; }
  void setNextIvar(ObjCIvarDecl *ivar) { NextIvar = ivar; }

  void setAccessControl(AccessControl ac) { DeclAccess = ac; }

  AccessControl getAccessControl() const { return AccessControl(DeclAccess); }

  AccessControl getCanonicalAccessControl() const {
    return DeclAccess == None ? Protected : AccessControl(DeclAccess);
  }

  void setSynthesize(bool synth) { Synthesized = synth; }
  bool getSynthesize() const { return Synthesized; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == ObjCIvar; }
private:
  /// NextIvar - Next Ivar in the list of ivars declared in class; class's
  /// extensions and class's implementation
  ObjCIvarDecl *NextIvar;

  // NOTE: VC++ treats enums as signed, avoid using the AccessControl enum
  unsigned DeclAccess : 3;
  unsigned Synthesized : 1;
};


/// \brief Represents a field declaration created by an \@defs(...).
class ObjCAtDefsFieldDecl : public FieldDecl {
  virtual void anchor();
  ObjCAtDefsFieldDecl(DeclContext *DC, SourceLocation StartLoc,
                      SourceLocation IdLoc, IdentifierInfo *Id,
                      QualType T, Expr *BW)
    : FieldDecl(ObjCAtDefsField, DC, StartLoc, IdLoc, Id, T,
                /*TInfo=*/0, // FIXME: Do ObjCAtDefs have declarators ?
                BW, /*Mutable=*/false, /*HasInit=*/ICIS_NoInit) {}

public:
  static ObjCAtDefsFieldDecl *Create(ASTContext &C, DeclContext *DC,
                                     SourceLocation StartLoc,
                                     SourceLocation IdLoc, IdentifierInfo *Id,
                                     QualType T, Expr *BW);

  static ObjCAtDefsFieldDecl *CreateDeserialized(ASTContext &C, unsigned ID);
  
  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == ObjCAtDefsField; }
};

/// \brief Represents an Objective-C protocol declaration.
///
/// Objective-C protocols declare a pure abstract type (i.e., no instance
/// variables are permitted).  Protocols originally drew inspiration from
/// C++ pure virtual functions (a C++ feature with nice semantics and lousy
/// syntax:-). Here is an example:
///
/// \code
/// \@protocol NSDraggingInfo <refproto1, refproto2>
/// - (NSWindow *)draggingDestinationWindow;
/// - (NSImage *)draggedImage;
/// \@end
/// \endcode
///
/// This says that NSDraggingInfo requires two methods and requires everything
/// that the two "referenced protocols" 'refproto1' and 'refproto2' require as
/// well.
///
/// \code
/// \@interface ImplementsNSDraggingInfo : NSObject \<NSDraggingInfo>
/// \@end
/// \endcode
///
/// ObjC protocols inspired Java interfaces. Unlike Java, ObjC classes and
/// protocols are in distinct namespaces. For example, Cocoa defines both
/// an NSObject protocol and class (which isn't allowed in Java). As a result,
/// protocols are referenced using angle brackets as follows:
///
/// id \<NSDraggingInfo> anyObjectThatImplementsNSDraggingInfo;
///
class ObjCProtocolDecl : public ObjCContainerDecl,
                         public Redeclarable<ObjCProtocolDecl> {
  virtual void anchor();

  struct DefinitionData {
    // \brief The declaration that defines this protocol.
    ObjCProtocolDecl *Definition;

    /// \brief Referenced protocols
    ObjCProtocolList ReferencedProtocols;    
  };
  
  DefinitionData *Data;

  DefinitionData &data() const {
    assert(Data && "Objective-C protocol has no definition!");
    return *Data;
  }
  
  ObjCProtocolDecl(DeclContext *DC, IdentifierInfo *Id,
                   SourceLocation nameLoc, SourceLocation atStartLoc,
                   ObjCProtocolDecl *PrevDecl);

  void allocateDefinitionData();

  typedef Redeclarable<ObjCProtocolDecl> redeclarable_base;
  virtual ObjCProtocolDecl *getNextRedeclaration() { 
    return RedeclLink.getNext(); 
  }
  virtual ObjCProtocolDecl *getPreviousDeclImpl() {
    return getPreviousDecl();
  }
  virtual ObjCProtocolDecl *getMostRecentDeclImpl() {
    return getMostRecentDecl();
  }
                           
public:
  static ObjCProtocolDecl *Create(ASTContext &C, DeclContext *DC,
                                  IdentifierInfo *Id,
                                  SourceLocation nameLoc,
                                  SourceLocation atStartLoc,
                                  ObjCProtocolDecl *PrevDecl);

  static ObjCProtocolDecl *CreateDeserialized(ASTContext &C, unsigned ID);
                           
  const ObjCProtocolList &getReferencedProtocols() const {
    assert(hasDefinition() && "No definition available!");
    return data().ReferencedProtocols;
  }
  typedef ObjCProtocolList::iterator protocol_iterator;
  protocol_iterator protocol_begin() const {
    if (!hasDefinition())
      return protocol_iterator();
    
    return data().ReferencedProtocols.begin();
  }
  protocol_iterator protocol_end() const { 
    if (!hasDefinition())
      return protocol_iterator();
    
    return data().ReferencedProtocols.end(); 
  }
  typedef ObjCProtocolList::loc_iterator protocol_loc_iterator;
  protocol_loc_iterator protocol_loc_begin() const {
    if (!hasDefinition())
      return protocol_loc_iterator();
    
    return data().ReferencedProtocols.loc_begin();
  }
  protocol_loc_iterator protocol_loc_end() const {
    if (!hasDefinition())
      return protocol_loc_iterator();
    
    return data().ReferencedProtocols.loc_end();
  }
  unsigned protocol_size() const { 
    if (!hasDefinition())
      return 0;
    
    return data().ReferencedProtocols.size(); 
  }

  /// setProtocolList - Set the list of protocols that this interface
  /// implements.
  void setProtocolList(ObjCProtocolDecl *const*List, unsigned Num,
                       const SourceLocation *Locs, ASTContext &C) {
    assert(Data && "Protocol is not defined");
    data().ReferencedProtocols.set(List, Num, Locs, C);
  }

  ObjCProtocolDecl *lookupProtocolNamed(IdentifierInfo *PName);

  // Lookup a method. First, we search locally. If a method isn't
  // found, we search referenced protocols and class categories.
  ObjCMethodDecl *lookupMethod(Selector Sel, bool isInstance) const;
  ObjCMethodDecl *lookupInstanceMethod(Selector Sel) const {
    return lookupMethod(Sel, true/*isInstance*/);
  }
  ObjCMethodDecl *lookupClassMethod(Selector Sel) const {
    return lookupMethod(Sel, false/*isInstance*/);
  }

  /// \brief Determine whether this protocol has a definition.
  bool hasDefinition() const { return Data != 0; }

  /// \brief Retrieve the definition of this protocol, if any.
  ObjCProtocolDecl *getDefinition() {
    return Data? Data->Definition : 0;
  }

  /// \brief Retrieve the definition of this protocol, if any.
  const ObjCProtocolDecl *getDefinition() const {
    return Data? Data->Definition : 0;
  }

  /// \brief Determine whether this particular declaration is also the 
  /// definition.
  bool isThisDeclarationADefinition() const {
    return getDefinition() == this;
  }
  
  /// \brief Starts the definition of this Objective-C protocol.
  void startDefinition();

  virtual SourceRange getSourceRange() const LLVM_READONLY {
    if (isThisDeclarationADefinition())
      return ObjCContainerDecl::getSourceRange();
   
    return SourceRange(getAtStartLoc(), getLocation());
  }
   
  typedef redeclarable_base::redecl_iterator redecl_iterator;
  using redeclarable_base::redecls_begin;
  using redeclarable_base::redecls_end;
  using redeclarable_base::getPreviousDecl;
  using redeclarable_base::getMostRecentDecl;

  /// Retrieves the canonical declaration of this Objective-C protocol.
  ObjCProtocolDecl *getCanonicalDecl() {
    return getFirstDeclaration();
  }
  const ObjCProtocolDecl *getCanonicalDecl() const {
    return getFirstDeclaration();
  }

  virtual void collectPropertiesToImplement(PropertyMap &PM) const;

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == ObjCProtocol; }

  friend class ASTReader;
  friend class ASTDeclReader;
  friend class ASTDeclWriter;
};

/// ObjCCategoryDecl - Represents a category declaration. A category allows
/// you to add methods to an existing class (without subclassing or modifying
/// the original class interface or implementation:-). Categories don't allow
/// you to add instance data. The following example adds "myMethod" to all
/// NSView's within a process:
///
/// \@interface NSView (MyViewMethods)
/// - myMethod;
/// \@end
///
/// Categories also allow you to split the implementation of a class across
/// several files (a feature more naturally supported in C++).
///
/// Categories were originally inspired by dynamic languages such as Common
/// Lisp and Smalltalk.  More traditional class-based languages (C++, Java)
/// don't support this level of dynamism, which is both powerful and dangerous.
///
class ObjCCategoryDecl : public ObjCContainerDecl {
  virtual void anchor();

  /// Interface belonging to this category
  ObjCInterfaceDecl *ClassInterface;

  /// referenced protocols in this category.
  ObjCProtocolList ReferencedProtocols;

  /// Next category belonging to this class.
  /// FIXME: this should not be a singly-linked list.  Move storage elsewhere.
  ObjCCategoryDecl *NextClassCategory;

  /// \brief The location of the category name in this declaration.
  SourceLocation CategoryNameLoc;

  /// class extension may have private ivars.
  SourceLocation IvarLBraceLoc;
  SourceLocation IvarRBraceLoc;
  
  ObjCCategoryDecl(DeclContext *DC, SourceLocation AtLoc,
                   SourceLocation ClassNameLoc, SourceLocation CategoryNameLoc,
                   IdentifierInfo *Id, ObjCInterfaceDecl *IDecl,
                   SourceLocation IvarLBraceLoc=SourceLocation(),
                   SourceLocation IvarRBraceLoc=SourceLocation())
    : ObjCContainerDecl(ObjCCategory, DC, Id, ClassNameLoc, AtLoc),
      ClassInterface(IDecl), NextClassCategory(0),
      CategoryNameLoc(CategoryNameLoc),
      IvarLBraceLoc(IvarLBraceLoc), IvarRBraceLoc(IvarRBraceLoc) {
  }
public:

  static ObjCCategoryDecl *Create(ASTContext &C, DeclContext *DC,
                                  SourceLocation AtLoc,
                                  SourceLocation ClassNameLoc,
                                  SourceLocation CategoryNameLoc,
                                  IdentifierInfo *Id,
                                  ObjCInterfaceDecl *IDecl,
                                  SourceLocation IvarLBraceLoc=SourceLocation(),
                                  SourceLocation IvarRBraceLoc=SourceLocation());
  static ObjCCategoryDecl *CreateDeserialized(ASTContext &C, unsigned ID);

  ObjCInterfaceDecl *getClassInterface() { return ClassInterface; }
  const ObjCInterfaceDecl *getClassInterface() const { return ClassInterface; }

  ObjCCategoryImplDecl *getImplementation() const;
  void setImplementation(ObjCCategoryImplDecl *ImplD);

  /// setProtocolList - Set the list of protocols that this interface
  /// implements.
  void setProtocolList(ObjCProtocolDecl *const*List, unsigned Num,
                       const SourceLocation *Locs, ASTContext &C) {
    ReferencedProtocols.set(List, Num, Locs, C);
  }

  const ObjCProtocolList &getReferencedProtocols() const {
    return ReferencedProtocols;
  }

  typedef ObjCProtocolList::iterator protocol_iterator;
  protocol_iterator protocol_begin() const {return ReferencedProtocols.begin();}
  protocol_iterator protocol_end() const { return ReferencedProtocols.end(); }
  unsigned protocol_size() const { return ReferencedProtocols.size(); }
  typedef ObjCProtocolList::loc_iterator protocol_loc_iterator;
  protocol_loc_iterator protocol_loc_begin() const {
    return ReferencedProtocols.loc_begin();
  }
  protocol_loc_iterator protocol_loc_end() const {
    return ReferencedProtocols.loc_end();
  }

  ObjCCategoryDecl *getNextClassCategory() const { return NextClassCategory; }

  bool IsClassExtension() const { return getIdentifier() == 0; }
  const ObjCCategoryDecl *getNextClassExtension() const;

  typedef specific_decl_iterator<ObjCIvarDecl> ivar_iterator;
  ivar_iterator ivar_begin() const {
    return ivar_iterator(decls_begin());
  }
  ivar_iterator ivar_end() const {
    return ivar_iterator(decls_end());
  }
  unsigned ivar_size() const {
    return std::distance(ivar_begin(), ivar_end());
  }
  bool ivar_empty() const {
    return ivar_begin() == ivar_end();
  }

  SourceLocation getCategoryNameLoc() const { return CategoryNameLoc; }
  void setCategoryNameLoc(SourceLocation Loc) { CategoryNameLoc = Loc; }
  
  void setIvarLBraceLoc(SourceLocation Loc) { IvarLBraceLoc = Loc; }
  SourceLocation getIvarLBraceLoc() const { return IvarLBraceLoc; }
  void setIvarRBraceLoc(SourceLocation Loc) { IvarRBraceLoc = Loc; }
  SourceLocation getIvarRBraceLoc() const { return IvarRBraceLoc; }

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == ObjCCategory; }

  friend class ASTDeclReader;
  friend class ASTDeclWriter;
};

class ObjCImplDecl : public ObjCContainerDecl {
  virtual void anchor();

  /// Class interface for this class/category implementation
  ObjCInterfaceDecl *ClassInterface;

protected:
  ObjCImplDecl(Kind DK, DeclContext *DC,
               ObjCInterfaceDecl *classInterface,
               SourceLocation nameLoc, SourceLocation atStartLoc)
    : ObjCContainerDecl(DK, DC,
                        classInterface? classInterface->getIdentifier() : 0,
                        nameLoc, atStartLoc),
      ClassInterface(classInterface) {}

public:
  const ObjCInterfaceDecl *getClassInterface() const { return ClassInterface; }
  ObjCInterfaceDecl *getClassInterface() { return ClassInterface; }
  void setClassInterface(ObjCInterfaceDecl *IFace);

  void addInstanceMethod(ObjCMethodDecl *method) {
    // FIXME: Context should be set correctly before we get here.
    method->setLexicalDeclContext(this);
    addDecl(method);
  }
  void addClassMethod(ObjCMethodDecl *method) {
    // FIXME: Context should be set correctly before we get here.
    method->setLexicalDeclContext(this);
    addDecl(method);
  }

  void addPropertyImplementation(ObjCPropertyImplDecl *property);

  ObjCPropertyImplDecl *FindPropertyImplDecl(IdentifierInfo *propertyId) const;
  ObjCPropertyImplDecl *FindPropertyImplIvarDecl(IdentifierInfo *ivarId) const;

  // Iterator access to properties.
  typedef specific_decl_iterator<ObjCPropertyImplDecl> propimpl_iterator;
  propimpl_iterator propimpl_begin() const {
    return propimpl_iterator(decls_begin());
  }
  propimpl_iterator propimpl_end() const {
    return propimpl_iterator(decls_end());
  }

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) {
    return K >= firstObjCImpl && K <= lastObjCImpl;
  }
};

/// ObjCCategoryImplDecl - An object of this class encapsulates a category
/// \@implementation declaration. If a category class has declaration of a
/// property, its implementation must be specified in the category's
/// \@implementation declaration. Example:
/// \@interface I \@end
/// \@interface I(CATEGORY)
///    \@property int p1, d1;
/// \@end
/// \@implementation I(CATEGORY)
///  \@dynamic p1,d1;
/// \@end
///
/// ObjCCategoryImplDecl
class ObjCCategoryImplDecl : public ObjCImplDecl {
  virtual void anchor();

  // Category name
  IdentifierInfo *Id;

  // Category name location
  SourceLocation CategoryNameLoc;

  ObjCCategoryImplDecl(DeclContext *DC, IdentifierInfo *Id,
                       ObjCInterfaceDecl *classInterface,
                       SourceLocation nameLoc, SourceLocation atStartLoc,
                       SourceLocation CategoryNameLoc)
    : ObjCImplDecl(ObjCCategoryImpl, DC, classInterface, nameLoc, atStartLoc),
      Id(Id), CategoryNameLoc(CategoryNameLoc) {}
public:
  static ObjCCategoryImplDecl *Create(ASTContext &C, DeclContext *DC,
                                      IdentifierInfo *Id,
                                      ObjCInterfaceDecl *classInterface,
                                      SourceLocation nameLoc,
                                      SourceLocation atStartLoc,
                                      SourceLocation CategoryNameLoc);
  static ObjCCategoryImplDecl *CreateDeserialized(ASTContext &C, unsigned ID);

  /// getIdentifier - Get the identifier that names the category
  /// interface associated with this implementation.
  /// FIXME: This is a bad API, we are overriding the NamedDecl::getIdentifier()
  /// to mean something different. For example:
  /// ((NamedDecl *)SomeCategoryImplDecl)->getIdentifier()
  /// returns the class interface name, whereas
  /// ((ObjCCategoryImplDecl *)SomeCategoryImplDecl)->getIdentifier()
  /// returns the category name.
  IdentifierInfo *getIdentifier() const {
    return Id;
  }
  void setIdentifier(IdentifierInfo *II) { Id = II; }

  ObjCCategoryDecl *getCategoryDecl() const;

  SourceLocation getCategoryNameLoc() const { return CategoryNameLoc; }

  /// getName - Get the name of identifier for the class interface associated
  /// with this implementation as a StringRef.
  //
  // FIXME: This is a bad API, we are overriding the NamedDecl::getName, to mean
  // something different.
  StringRef getName() const {
    return Id ? Id->getNameStart() : "";
  }

  /// @brief Get the name of the class associated with this interface.
  //
  // FIXME: Deprecated, move clients to getName().
  std::string getNameAsString() const {
    return getName();
  }

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == ObjCCategoryImpl;}

  friend class ASTDeclReader;
  friend class ASTDeclWriter;
};

raw_ostream &operator<<(raw_ostream &OS, const ObjCCategoryImplDecl &CID);

/// ObjCImplementationDecl - Represents a class definition - this is where
/// method definitions are specified. For example:
///
/// @code
/// \@implementation MyClass
/// - (void)myMethod { /* do something */ }
/// \@end
/// @endcode
///
/// Typically, instance variables are specified in the class interface,
/// *not* in the implementation. Nevertheless (for legacy reasons), we
/// allow instance variables to be specified in the implementation.  When
/// specified, they need to be *identical* to the interface.
///
class ObjCImplementationDecl : public ObjCImplDecl {
  virtual void anchor();
  /// Implementation Class's super class.
  ObjCInterfaceDecl *SuperClass;
  /// \@implementation may have private ivars.
  SourceLocation IvarLBraceLoc;
  SourceLocation IvarRBraceLoc;
  
  /// Support for ivar initialization.
  /// IvarInitializers - The arguments used to initialize the ivars
  CXXCtorInitializer **IvarInitializers;
  unsigned NumIvarInitializers;

  /// Do the ivars of this class require initialization other than
  /// zero-initialization?
  bool HasNonZeroConstructors : 1;

  /// Do the ivars of this class require non-trivial destruction?
  bool HasDestructors : 1;

  ObjCImplementationDecl(DeclContext *DC,
                         ObjCInterfaceDecl *classInterface,
                         ObjCInterfaceDecl *superDecl,
                         SourceLocation nameLoc, SourceLocation atStartLoc,
                         SourceLocation IvarLBraceLoc=SourceLocation(), 
                         SourceLocation IvarRBraceLoc=SourceLocation())
    : ObjCImplDecl(ObjCImplementation, DC, classInterface, nameLoc, atStartLoc),
       SuperClass(superDecl), IvarLBraceLoc(IvarLBraceLoc), 
       IvarRBraceLoc(IvarRBraceLoc),
       IvarInitializers(0), NumIvarInitializers(0),
       HasNonZeroConstructors(false), HasDestructors(false) {}
public:
  static ObjCImplementationDecl *Create(ASTContext &C, DeclContext *DC,
                                        ObjCInterfaceDecl *classInterface,
                                        ObjCInterfaceDecl *superDecl,
                                        SourceLocation nameLoc,
                                        SourceLocation atStartLoc,
                                        SourceLocation IvarLBraceLoc=SourceLocation(), 
                                        SourceLocation IvarRBraceLoc=SourceLocation());

  static ObjCImplementationDecl *CreateDeserialized(ASTContext &C, unsigned ID);

  /// init_iterator - Iterates through the ivar initializer list.
  typedef CXXCtorInitializer **init_iterator;

  /// init_const_iterator - Iterates through the ivar initializer list.
  typedef CXXCtorInitializer * const * init_const_iterator;

  /// init_begin() - Retrieve an iterator to the first initializer.
  init_iterator       init_begin()       { return IvarInitializers; }
  /// begin() - Retrieve an iterator to the first initializer.
  init_const_iterator init_begin() const { return IvarInitializers; }

  /// init_end() - Retrieve an iterator past the last initializer.
  init_iterator       init_end()       {
    return IvarInitializers + NumIvarInitializers;
  }
  /// end() - Retrieve an iterator past the last initializer.
  init_const_iterator init_end() const {
    return IvarInitializers + NumIvarInitializers;
  }
  /// getNumArgs - Number of ivars which must be initialized.
  unsigned getNumIvarInitializers() const {
    return NumIvarInitializers;
  }

  void setNumIvarInitializers(unsigned numNumIvarInitializers) {
    NumIvarInitializers = numNumIvarInitializers;
  }

  void setIvarInitializers(ASTContext &C,
                           CXXCtorInitializer ** initializers,
                           unsigned numInitializers);

  /// Do any of the ivars of this class (not counting its base classes)
  /// require construction other than zero-initialization?
  bool hasNonZeroConstructors() const { return HasNonZeroConstructors; }
  void setHasNonZeroConstructors(bool val) { HasNonZeroConstructors = val; }

  /// Do any of the ivars of this class (not counting its base classes)
  /// require non-trivial destruction?
  bool hasDestructors() const { return HasDestructors; }
  void setHasDestructors(bool val) { HasDestructors = val; }

  /// getIdentifier - Get the identifier that names the class
  /// interface associated with this implementation.
  IdentifierInfo *getIdentifier() const {
    return getClassInterface()->getIdentifier();
  }

  /// getName - Get the name of identifier for the class interface associated
  /// with this implementation as a StringRef.
  //
  // FIXME: This is a bad API, we are overriding the NamedDecl::getName, to mean
  // something different.
  StringRef getName() const {
    assert(getIdentifier() && "Name is not a simple identifier");
    return getIdentifier()->getName();
  }

  /// @brief Get the name of the class associated with this interface.
  //
  // FIXME: Move to StringRef API.
  std::string getNameAsString() const {
    return getName();
  }

  const ObjCInterfaceDecl *getSuperClass() const { return SuperClass; }
  ObjCInterfaceDecl *getSuperClass() { return SuperClass; }

  void setSuperClass(ObjCInterfaceDecl * superCls) { SuperClass = superCls; }

  void setIvarLBraceLoc(SourceLocation Loc) { IvarLBraceLoc = Loc; }
  SourceLocation getIvarLBraceLoc() const { return IvarLBraceLoc; }
  void setIvarRBraceLoc(SourceLocation Loc) { IvarRBraceLoc = Loc; }
  SourceLocation getIvarRBraceLoc() const { return IvarRBraceLoc; }
  
  typedef specific_decl_iterator<ObjCIvarDecl> ivar_iterator;
  ivar_iterator ivar_begin() const {
    return ivar_iterator(decls_begin());
  }
  ivar_iterator ivar_end() const {
    return ivar_iterator(decls_end());
  }
  unsigned ivar_size() const {
    return std::distance(ivar_begin(), ivar_end());
  }
  bool ivar_empty() const {
    return ivar_begin() == ivar_end();
  }

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == ObjCImplementation; }

  friend class ASTDeclReader;
  friend class ASTDeclWriter;
};

raw_ostream &operator<<(raw_ostream &OS, const ObjCImplementationDecl &ID);

/// ObjCCompatibleAliasDecl - Represents alias of a class. This alias is
/// declared as \@compatibility_alias alias class.
class ObjCCompatibleAliasDecl : public NamedDecl {
  virtual void anchor();
  /// Class that this is an alias of.
  ObjCInterfaceDecl *AliasedClass;

  ObjCCompatibleAliasDecl(DeclContext *DC, SourceLocation L, IdentifierInfo *Id,
                          ObjCInterfaceDecl* aliasedClass)
    : NamedDecl(ObjCCompatibleAlias, DC, L, Id), AliasedClass(aliasedClass) {}
public:
  static ObjCCompatibleAliasDecl *Create(ASTContext &C, DeclContext *DC,
                                         SourceLocation L, IdentifierInfo *Id,
                                         ObjCInterfaceDecl* aliasedClass);

  static ObjCCompatibleAliasDecl *CreateDeserialized(ASTContext &C, 
                                                     unsigned ID);
  
  const ObjCInterfaceDecl *getClassInterface() const { return AliasedClass; }
  ObjCInterfaceDecl *getClassInterface() { return AliasedClass; }
  void setClassInterface(ObjCInterfaceDecl *D) { AliasedClass = D; }

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == ObjCCompatibleAlias; }

};

/// \brief Represents one property declaration in an Objective-C interface.
///
/// For example:
/// \code{.mm}
/// \@property (assign, readwrite) int MyProperty;
/// \endcode
class ObjCPropertyDecl : public NamedDecl {
  virtual void anchor();
public:
  enum PropertyAttributeKind {
    OBJC_PR_noattr    = 0x00,
    OBJC_PR_readonly  = 0x01,
    OBJC_PR_getter    = 0x02,
    OBJC_PR_assign    = 0x04,
    OBJC_PR_readwrite = 0x08,
    OBJC_PR_retain    = 0x10,
    OBJC_PR_copy      = 0x20,
    OBJC_PR_nonatomic = 0x40,
    OBJC_PR_setter    = 0x80,
    OBJC_PR_atomic    = 0x100,
    OBJC_PR_weak      = 0x200,
    OBJC_PR_strong    = 0x400,
    OBJC_PR_unsafe_unretained = 0x800
    // Adding a property should change NumPropertyAttrsBits
  };

  enum {
    /// \brief Number of bits fitting all the property attributes.
    NumPropertyAttrsBits = 12
  };

  enum SetterKind { Assign, Retain, Copy, Weak };
  enum PropertyControl { None, Required, Optional };
private:
  SourceLocation AtLoc;   // location of \@property
  SourceLocation LParenLoc; // location of '(' starting attribute list or null.
  TypeSourceInfo *DeclType;
  unsigned PropertyAttributes : NumPropertyAttrsBits;
  unsigned PropertyAttributesAsWritten : NumPropertyAttrsBits;
  // \@required/\@optional
  unsigned PropertyImplementation : 2;

  Selector GetterName;    // getter name of NULL if no getter
  Selector SetterName;    // setter name of NULL if no setter

  ObjCMethodDecl *GetterMethodDecl; // Declaration of getter instance method
  ObjCMethodDecl *SetterMethodDecl; // Declaration of setter instance method
  ObjCIvarDecl *PropertyIvarDecl;   // Synthesize ivar for this property

  ObjCPropertyDecl(DeclContext *DC, SourceLocation L, IdentifierInfo *Id,
                   SourceLocation AtLocation,  SourceLocation LParenLocation,
                   TypeSourceInfo *T)
    : NamedDecl(ObjCProperty, DC, L, Id), AtLoc(AtLocation), 
      LParenLoc(LParenLocation), DeclType(T),
      PropertyAttributes(OBJC_PR_noattr),
      PropertyAttributesAsWritten(OBJC_PR_noattr),
      PropertyImplementation(None),
      GetterName(Selector()),
      SetterName(Selector()),
      GetterMethodDecl(0), SetterMethodDecl(0) , PropertyIvarDecl(0) {}
public:
  static ObjCPropertyDecl *Create(ASTContext &C, DeclContext *DC,
                                  SourceLocation L,
                                  IdentifierInfo *Id, SourceLocation AtLocation,
                                  SourceLocation LParenLocation,
                                  TypeSourceInfo *T,
                                  PropertyControl propControl = None);
  
  static ObjCPropertyDecl *CreateDeserialized(ASTContext &C, unsigned ID);
  
  SourceLocation getAtLoc() const { return AtLoc; }
  void setAtLoc(SourceLocation L) { AtLoc = L; }
  
  SourceLocation getLParenLoc() const { return LParenLoc; }
  void setLParenLoc(SourceLocation L) { LParenLoc = L; }

  TypeSourceInfo *getTypeSourceInfo() const { return DeclType; }
  QualType getType() const { return DeclType->getType(); }
  void setType(TypeSourceInfo *T) { DeclType = T; }

  PropertyAttributeKind getPropertyAttributes() const {
    return PropertyAttributeKind(PropertyAttributes);
  }
  void setPropertyAttributes(PropertyAttributeKind PRVal) {
    PropertyAttributes |= PRVal;
  }

  PropertyAttributeKind getPropertyAttributesAsWritten() const {
    return PropertyAttributeKind(PropertyAttributesAsWritten);
  }

  bool hasWrittenStorageAttribute() const {
    return PropertyAttributesAsWritten & (OBJC_PR_assign | OBJC_PR_copy |
        OBJC_PR_unsafe_unretained | OBJC_PR_retain | OBJC_PR_strong |
        OBJC_PR_weak);
  }

  void setPropertyAttributesAsWritten(PropertyAttributeKind PRVal) {
    PropertyAttributesAsWritten = PRVal;
  }

 void makeitReadWriteAttribute(void) {
    PropertyAttributes &= ~OBJC_PR_readonly;
    PropertyAttributes |= OBJC_PR_readwrite;
 }

  // Helper methods for accessing attributes.

  /// isReadOnly - Return true iff the property has a setter.
  bool isReadOnly() const {
    return (PropertyAttributes & OBJC_PR_readonly);
  }

  /// isAtomic - Return true if the property is atomic.
  bool isAtomic() const {
    return (PropertyAttributes & OBJC_PR_atomic);
  }

  /// isRetaining - Return true if the property retains its value.
  bool isRetaining() const {
    return (PropertyAttributes &
            (OBJC_PR_retain | OBJC_PR_strong | OBJC_PR_copy));
  }

  /// getSetterKind - Return the method used for doing assignment in
  /// the property setter. This is only valid if the property has been
  /// defined to have a setter.
  SetterKind getSetterKind() const {
    if (PropertyAttributes & OBJC_PR_strong)
      return getType()->isBlockPointerType() ? Copy : Retain;
    if (PropertyAttributes & OBJC_PR_retain)
      return Retain;
    if (PropertyAttributes & OBJC_PR_copy)
      return Copy;
    if (PropertyAttributes & OBJC_PR_weak)
      return Weak;
    return Assign;
  }

  Selector getGetterName() const { return GetterName; }
  void setGetterName(Selector Sel) { GetterName = Sel; }

  Selector getSetterName() const { return SetterName; }
  void setSetterName(Selector Sel) { SetterName = Sel; }

  ObjCMethodDecl *getGetterMethodDecl() const { return GetterMethodDecl; }
  void setGetterMethodDecl(ObjCMethodDecl *gDecl) { GetterMethodDecl = gDecl; }

  ObjCMethodDecl *getSetterMethodDecl() const { return SetterMethodDecl; }
  void setSetterMethodDecl(ObjCMethodDecl *gDecl) { SetterMethodDecl = gDecl; }

  // Related to \@optional/\@required declared in \@protocol
  void setPropertyImplementation(PropertyControl pc) {
    PropertyImplementation = pc;
  }
  PropertyControl getPropertyImplementation() const {
    return PropertyControl(PropertyImplementation);
  }

  void setPropertyIvarDecl(ObjCIvarDecl *Ivar) {
    PropertyIvarDecl = Ivar;
  }
  ObjCIvarDecl *getPropertyIvarDecl() const {
    return PropertyIvarDecl;
  }

  virtual SourceRange getSourceRange() const LLVM_READONLY {
    return SourceRange(AtLoc, getLocation());
  }
  
  /// Get the default name of the synthesized ivar.
  IdentifierInfo *getDefaultSynthIvarName(ASTContext &Ctx) const;

  /// Lookup a property by name in the specified DeclContext.
  static ObjCPropertyDecl *findPropertyDecl(const DeclContext *DC,
                                            IdentifierInfo *propertyID);

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Kind K) { return K == ObjCProperty; }
};

/// ObjCPropertyImplDecl - Represents implementation declaration of a property
/// in a class or category implementation block. For example:
/// \@synthesize prop1 = ivar1;
///
class ObjCPropertyImplDecl : public Decl {
public:
  enum Kind {
    Synthesize,
    Dynamic
  };
private:
  SourceLocation AtLoc;   // location of \@synthesize or \@dynamic

  /// \brief For \@synthesize, the location of the ivar, if it was written in
  /// the source code.
  ///
  /// \code
  /// \@synthesize int a = b
  /// \endcode
  SourceLocation IvarLoc;

  /// Property declaration being implemented
  ObjCPropertyDecl *PropertyDecl;

  /// Null for \@dynamic. Required for \@synthesize.
  ObjCIvarDecl *PropertyIvarDecl;

  /// Null for \@dynamic. Non-null if property must be copy-constructed in
  /// getter.
  Expr *GetterCXXConstructor;

  /// Null for \@dynamic. Non-null if property has assignment operator to call
  /// in Setter synthesis.
  Expr *SetterCXXAssignment;

  ObjCPropertyImplDecl(DeclContext *DC, SourceLocation atLoc, SourceLocation L,
                       ObjCPropertyDecl *property,
                       Kind PK,
                       ObjCIvarDecl *ivarDecl,
                       SourceLocation ivarLoc)
    : Decl(ObjCPropertyImpl, DC, L), AtLoc(atLoc),
      IvarLoc(ivarLoc), PropertyDecl(property), PropertyIvarDecl(ivarDecl),
      GetterCXXConstructor(0), SetterCXXAssignment(0) {
    assert (PK == Dynamic || PropertyIvarDecl);
  }

public:
  static ObjCPropertyImplDecl *Create(ASTContext &C, DeclContext *DC,
                                      SourceLocation atLoc, SourceLocation L,
                                      ObjCPropertyDecl *property,
                                      Kind PK,
                                      ObjCIvarDecl *ivarDecl,
                                      SourceLocation ivarLoc);

  static ObjCPropertyImplDecl *CreateDeserialized(ASTContext &C, unsigned ID);
  
  virtual SourceRange getSourceRange() const LLVM_READONLY;

  SourceLocation getLocStart() const LLVM_READONLY { return AtLoc; }
  void setAtLoc(SourceLocation Loc) { AtLoc = Loc; }

  ObjCPropertyDecl *getPropertyDecl() const {
    return PropertyDecl;
  }
  void setPropertyDecl(ObjCPropertyDecl *Prop) { PropertyDecl = Prop; }

  Kind getPropertyImplementation() const {
    return PropertyIvarDecl ? Synthesize : Dynamic;
  }

  ObjCIvarDecl *getPropertyIvarDecl() const {
    return PropertyIvarDecl;
  }
  SourceLocation getPropertyIvarDeclLoc() const { return IvarLoc; }

  void setPropertyIvarDecl(ObjCIvarDecl *Ivar,
                           SourceLocation IvarLoc) {
    PropertyIvarDecl = Ivar;
    this->IvarLoc = IvarLoc;
  }

  /// \brief For \@synthesize, returns true if an ivar name was explicitly
  /// specified.
  ///
  /// \code
  /// \@synthesize int a = b; // true
  /// \@synthesize int a; // false
  /// \endcode
  bool isIvarNameSpecified() const {
    return IvarLoc.isValid() && IvarLoc != getLocation();
  }

  Expr *getGetterCXXConstructor() const {
    return GetterCXXConstructor;
  }
  void setGetterCXXConstructor(Expr *getterCXXConstructor) {
    GetterCXXConstructor = getterCXXConstructor;
  }

  Expr *getSetterCXXAssignment() const {
    return SetterCXXAssignment;
  }
  void setSetterCXXAssignment(Expr *setterCXXAssignment) {
    SetterCXXAssignment = setterCXXAssignment;
  }

  static bool classof(const Decl *D) { return classofKind(D->getKind()); }
  static bool classofKind(Decl::Kind K) { return K == ObjCPropertyImpl; }

  friend class ASTDeclReader;
};

}  // end namespace clang
#endif
