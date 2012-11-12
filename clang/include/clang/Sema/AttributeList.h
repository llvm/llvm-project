//===--- AttributeList.h - Parsed attribute sets ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the AttributeList class, which is used to collect
// parsed attributes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_ATTRLIST_H
#define LLVM_CLANG_SEMA_ATTRLIST_H

#include "llvm/Support/Allocator.h"
#include "llvm/ADT/SmallVector.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/VersionTuple.h"
#include "clang/Sema/Ownership.h"
#include <cassert>

namespace clang {
  class ASTContext;
  class IdentifierInfo;
  class Expr;

/// \brief Represents information about a change in availability for
/// an entity, which is part of the encoding of the 'availability'
/// attribute.
struct AvailabilityChange {
  /// \brief The location of the keyword indicating the kind of change.
  SourceLocation KeywordLoc;

  /// \brief The version number at which the change occurred.
  VersionTuple Version;

  /// \brief The source range covering the version number.
  SourceRange VersionRange;

  /// \brief Determine whether this availability change is valid.
  bool isValid() const { return !Version.empty(); }
};

/// AttributeList - Represents GCC's __attribute__ declaration. There are
/// 4 forms of this construct...they are:
///
/// 1: __attribute__(( const )). ParmName/Args/NumArgs will all be unused.
/// 2: __attribute__(( mode(byte) )). ParmName used, Args/NumArgs unused.
/// 3: __attribute__(( format(printf, 1, 2) )). ParmName/Args/NumArgs all used.
/// 4: __attribute__(( aligned(16) )). ParmName is unused, Args/Num used.
///
class AttributeList { // TODO: This should really be called ParsedAttribute
public:
  /// The style used to specify an attribute.
  enum Syntax {
    AS_GNU,
    AS_CXX11,
    AS_Declspec,
    // eg) __w64, __ptr32, etc.  It is implied that an MSTypespec is also
    // a declspec.
    AS_MSTypespec   
  };
private:
  IdentifierInfo *AttrName;
  IdentifierInfo *ScopeName;
  IdentifierInfo *ParmName;
  SourceRange AttrRange;
  SourceLocation ScopeLoc;
  SourceLocation ParmLoc;

  /// The number of expression arguments this attribute has.
  /// The expressions themselves are stored after the object.
  unsigned NumArgs : 16;

  /// Corresponds to the Syntax enum.
  unsigned SyntaxUsed : 2;

  /// True if already diagnosed as invalid.
  mutable unsigned Invalid : 1;

  /// True if this attribute was used as a type attribute.
  mutable unsigned UsedAsTypeAttr : 1;

  /// True if this has the extra information associated with an
  /// availability attribute.
  unsigned IsAvailability : 1;

  /// True if this has extra information associated with a
  /// type_tag_for_datatype attribute.
  unsigned IsTypeTagForDatatype : 1;

  unsigned AttrKind : 8;

  /// \brief The location of the 'unavailable' keyword in an
  /// availability attribute.
  SourceLocation UnavailableLoc;
  
  const Expr *MessageExpr;

  /// The next attribute in the current position.
  AttributeList *NextInPosition;

  /// The next attribute allocated in the current Pool.
  AttributeList *NextInPool;

  Expr **getArgsBuffer() {
    return reinterpret_cast<Expr**>(this+1);
  }
  Expr * const *getArgsBuffer() const {
    return reinterpret_cast<Expr* const *>(this+1);
  }

  enum AvailabilitySlot {
    IntroducedSlot, DeprecatedSlot, ObsoletedSlot
  };

  AvailabilityChange &getAvailabilitySlot(AvailabilitySlot index) {
    return reinterpret_cast<AvailabilityChange*>(this+1)[index];
  }
  const AvailabilityChange &getAvailabilitySlot(AvailabilitySlot index) const {
    return reinterpret_cast<const AvailabilityChange*>(this+1)[index];
  }

public:
  struct TypeTagForDatatypeData {
    ParsedType *MatchingCType;
    unsigned LayoutCompatible : 1;
    unsigned MustBeNull : 1;
  };

private:
  TypeTagForDatatypeData &getTypeTagForDatatypeDataSlot() {
    return *reinterpret_cast<TypeTagForDatatypeData *>(this + 1);
  }

  const TypeTagForDatatypeData &getTypeTagForDatatypeDataSlot() const {
    return *reinterpret_cast<const TypeTagForDatatypeData *>(this + 1);
  }

  AttributeList(const AttributeList &) LLVM_DELETED_FUNCTION;
  void operator=(const AttributeList &) LLVM_DELETED_FUNCTION;
  void operator delete(void *) LLVM_DELETED_FUNCTION;
  ~AttributeList() LLVM_DELETED_FUNCTION;

  size_t allocated_size() const;

  /// Constructor for attributes with expression arguments.
  AttributeList(IdentifierInfo *attrName, SourceRange attrRange,
                IdentifierInfo *scopeName, SourceLocation scopeLoc,
                IdentifierInfo *parmName, SourceLocation parmLoc,
                Expr **args, unsigned numArgs,
                Syntax syntaxUsed)
    : AttrName(attrName), ScopeName(scopeName), ParmName(parmName),
      AttrRange(attrRange), ScopeLoc(scopeLoc), ParmLoc(parmLoc),
      NumArgs(numArgs), SyntaxUsed(syntaxUsed), Invalid(false),
      UsedAsTypeAttr(false), IsAvailability(false),
      IsTypeTagForDatatype(false), NextInPosition(0), NextInPool(0) {
    if (numArgs) memcpy(getArgsBuffer(), args, numArgs * sizeof(Expr*));
    AttrKind = getKind(getName(), getScopeName(), syntaxUsed);
  }

  /// Constructor for availability attributes.
  AttributeList(IdentifierInfo *attrName, SourceRange attrRange,
                IdentifierInfo *scopeName, SourceLocation scopeLoc,
                IdentifierInfo *parmName, SourceLocation parmLoc,
                const AvailabilityChange &introduced,
                const AvailabilityChange &deprecated,
                const AvailabilityChange &obsoleted,
                SourceLocation unavailable, 
                const Expr *messageExpr,
                Syntax syntaxUsed)
    : AttrName(attrName), ScopeName(scopeName), ParmName(parmName),
      AttrRange(attrRange), ScopeLoc(scopeLoc), ParmLoc(parmLoc),
      NumArgs(0), SyntaxUsed(syntaxUsed),
      Invalid(false), UsedAsTypeAttr(false), IsAvailability(true),
      IsTypeTagForDatatype(false),
      UnavailableLoc(unavailable), MessageExpr(messageExpr),
      NextInPosition(0), NextInPool(0) {
    new (&getAvailabilitySlot(IntroducedSlot)) AvailabilityChange(introduced);
    new (&getAvailabilitySlot(DeprecatedSlot)) AvailabilityChange(deprecated);
    new (&getAvailabilitySlot(ObsoletedSlot)) AvailabilityChange(obsoleted);
    AttrKind = getKind(getName(), getScopeName(), syntaxUsed);
  }

  /// Constructor for type_tag_for_datatype attribute.
  AttributeList(IdentifierInfo *attrName, SourceRange attrRange,
                IdentifierInfo *scopeName, SourceLocation scopeLoc,
                IdentifierInfo *argumentKindName,
                SourceLocation argumentKindLoc,
                ParsedType matchingCType, bool layoutCompatible,
                bool mustBeNull, Syntax syntaxUsed)
    : AttrName(attrName), ScopeName(scopeName), ParmName(argumentKindName),
      AttrRange(attrRange), ScopeLoc(scopeLoc), ParmLoc(argumentKindLoc),
      NumArgs(0), SyntaxUsed(syntaxUsed),
      Invalid(false), UsedAsTypeAttr(false), IsAvailability(false),
      IsTypeTagForDatatype(true), NextInPosition(NULL), NextInPool(NULL) {
    TypeTagForDatatypeData &ExtraData = getTypeTagForDatatypeDataSlot();
    new (&ExtraData.MatchingCType) ParsedType(matchingCType);
    ExtraData.LayoutCompatible = layoutCompatible;
    ExtraData.MustBeNull = mustBeNull;
    AttrKind = getKind(getName(), getScopeName(), syntaxUsed);
  }

  friend class AttributePool;
  friend class AttributeFactory;

public:
  enum Kind {           
    #define PARSED_ATTR(NAME) AT_##NAME,
    #include "clang/Sema/AttrParsedAttrList.inc"
    #undef PARSED_ATTR
    IgnoredAttribute,
    UnknownAttribute
  };

  IdentifierInfo *getName() const { return AttrName; }
  SourceLocation getLoc() const { return AttrRange.getBegin(); }
  SourceRange getRange() const { return AttrRange; }
  
  bool hasScope() const { return ScopeName; }
  IdentifierInfo *getScopeName() const { return ScopeName; }
  SourceLocation getScopeLoc() const { return ScopeLoc; }
  
  IdentifierInfo *getParameterName() const { return ParmName; }
  SourceLocation getParameterLoc() const { return ParmLoc; }

  /// Returns true if the attribute is a pure __declspec or a synthesized
  /// declspec representing a type specification (like __w64 or __ptr32).
  bool isDeclspecAttribute() const { return SyntaxUsed == AS_Declspec ||
                                            SyntaxUsed == AS_MSTypespec; }
  bool isCXX0XAttribute() const { return SyntaxUsed == AS_CXX11; }
  bool isMSTypespecAttribute() const { return SyntaxUsed == AS_MSTypespec; }

  bool isInvalid() const { return Invalid; }
  void setInvalid(bool b = true) const { Invalid = b; }

  bool isUsedAsTypeAttr() const { return UsedAsTypeAttr; }
  void setUsedAsTypeAttr() { UsedAsTypeAttr = true; }

  Kind getKind() const { return Kind(AttrKind); }
  static Kind getKind(const IdentifierInfo *Name, const IdentifierInfo *Scope,
                      Syntax SyntaxUsed);

  AttributeList *getNext() const { return NextInPosition; }
  void setNext(AttributeList *N) { NextInPosition = N; }

  /// getNumArgs - Return the number of actual arguments to this attribute.
  unsigned getNumArgs() const { return NumArgs; }

  /// hasParameterOrArguments - Return true if this attribute has a parameter,
  /// or has a non empty argument expression list.
  bool hasParameterOrArguments() const { return ParmName || NumArgs; }

  /// getArg - Return the specified argument.
  Expr *getArg(unsigned Arg) const {
    assert(Arg < NumArgs && "Arg access out of range!");
    return getArgsBuffer()[Arg];
  }

  class arg_iterator {
    Expr * const *X;
    unsigned Idx;
  public:
    arg_iterator(Expr * const *x, unsigned idx) : X(x), Idx(idx) {}

    arg_iterator& operator++() {
      ++Idx;
      return *this;
    }

    bool operator==(const arg_iterator& I) const {
      assert (X == I.X &&
              "compared arg_iterators are for different argument lists");
      return Idx == I.Idx;
    }

    bool operator!=(const arg_iterator& I) const {
      return !operator==(I);
    }

    Expr* operator*() const {
      return X[Idx];
    }

    unsigned getArgNum() const {
      return Idx+1;
    }
  };

  arg_iterator arg_begin() const {
    return arg_iterator(getArgsBuffer(), 0);
  }

  arg_iterator arg_end() const {
    return arg_iterator(getArgsBuffer(), NumArgs);
  }

  const AvailabilityChange &getAvailabilityIntroduced() const {
    assert(getKind() == AT_Availability && "Not an availability attribute");
    return getAvailabilitySlot(IntroducedSlot);
  }

  const AvailabilityChange &getAvailabilityDeprecated() const {
    assert(getKind() == AT_Availability && "Not an availability attribute");
    return getAvailabilitySlot(DeprecatedSlot);
  }

  const AvailabilityChange &getAvailabilityObsoleted() const {
    assert(getKind() == AT_Availability && "Not an availability attribute");
    return getAvailabilitySlot(ObsoletedSlot);
  }

  SourceLocation getUnavailableLoc() const {
    assert(getKind() == AT_Availability && "Not an availability attribute");
    return UnavailableLoc;
  }
  
  const Expr * getMessageExpr() const {
    assert(getKind() == AT_Availability && "Not an availability attribute");
    return MessageExpr;
  }

  const ParsedType &getMatchingCType() const {
    assert(getKind() == AT_TypeTagForDatatype &&
           "Not a type_tag_for_datatype attribute");
    return *getTypeTagForDatatypeDataSlot().MatchingCType;
  }

  bool getLayoutCompatible() const {
    assert(getKind() == AT_TypeTagForDatatype &&
           "Not a type_tag_for_datatype attribute");
    return getTypeTagForDatatypeDataSlot().LayoutCompatible;
  }

  bool getMustBeNull() const {
    assert(getKind() == AT_TypeTagForDatatype &&
           "Not a type_tag_for_datatype attribute");
    return getTypeTagForDatatypeDataSlot().MustBeNull;
  }
};

/// A factory, from which one makes pools, from which one creates
/// individual attributes which are deallocated with the pool.
///
/// Note that it's tolerably cheap to create and destroy one of
/// these as long as you don't actually allocate anything in it.
class AttributeFactory {
public:
  enum {
    /// The required allocation size of an availability attribute,
    /// which we want to ensure is a multiple of sizeof(void*).
    AvailabilityAllocSize =
      sizeof(AttributeList)
      + ((3 * sizeof(AvailabilityChange) + sizeof(void*) - 1)
         / sizeof(void*) * sizeof(void*)),
    TypeTagForDatatypeAllocSize =
      sizeof(AttributeList)
      + (sizeof(AttributeList::TypeTagForDatatypeData) + sizeof(void *) - 1)
        / sizeof(void*) * sizeof(void*)
  };

private:
  enum {
    /// The number of free lists we want to be sure to support
    /// inline.  This is just enough that availability attributes
    /// don't surpass it.  It's actually very unlikely we'll see an
    /// attribute that needs more than that; on x86-64 you'd need 10
    /// expression arguments, and on i386 you'd need 19.
    InlineFreeListsCapacity =
      1 + (AvailabilityAllocSize - sizeof(AttributeList)) / sizeof(void*)
  };

  llvm::BumpPtrAllocator Alloc;

  /// Free lists.  The index is determined by the following formula:
  ///   (size - sizeof(AttributeList)) / sizeof(void*)
  SmallVector<AttributeList*, InlineFreeListsCapacity> FreeLists;

  // The following are the private interface used by AttributePool.
  friend class AttributePool;

  /// Allocate an attribute of the given size.
  void *allocate(size_t size);

  /// Reclaim all the attributes in the given pool chain, which is
  /// non-empty.  Note that the current implementation is safe
  /// against reclaiming things which were not actually allocated
  /// with the allocator, although of course it's important to make
  /// sure that their allocator lives at least as long as this one.
  void reclaimPool(AttributeList *head);

public:
  AttributeFactory();
  ~AttributeFactory();
};

class AttributePool {
  AttributeFactory &Factory;
  AttributeList *Head;

  void *allocate(size_t size) {
    return Factory.allocate(size);
  }

  AttributeList *add(AttributeList *attr) {
    // We don't care about the order of the pool.
    attr->NextInPool = Head;
    Head = attr;
    return attr;
  }

  void takePool(AttributeList *pool);

public:
  /// Create a new pool for a factory.
  AttributePool(AttributeFactory &factory) : Factory(factory), Head(0) {}

  /// Move the given pool's allocations to this pool.
  AttributePool(AttributePool &pool) : Factory(pool.Factory), Head(pool.Head) {
    pool.Head = 0;
  }

  AttributeFactory &getFactory() const { return Factory; }

  void clear() {
    if (Head) {
      Factory.reclaimPool(Head);
      Head = 0;
    }
  }

  /// Take the given pool's allocations and add them to this pool.
  void takeAllFrom(AttributePool &pool) {
    if (pool.Head) {
      takePool(pool.Head);
      pool.Head = 0;
    }
  }

  ~AttributePool() {
    if (Head) Factory.reclaimPool(Head);
  }

  AttributeList *create(IdentifierInfo *attrName, SourceRange attrRange,
                        IdentifierInfo *scopeName, SourceLocation scopeLoc,
                        IdentifierInfo *parmName, SourceLocation parmLoc,
                        Expr **args, unsigned numArgs,
                        AttributeList::Syntax syntax) {
    void *memory = allocate(sizeof(AttributeList)
                            + numArgs * sizeof(Expr*));
    return add(new (memory) AttributeList(attrName, attrRange,
                                          scopeName, scopeLoc,
                                          parmName, parmLoc,
                                          args, numArgs, syntax));
  }

  AttributeList *create(IdentifierInfo *attrName, SourceRange attrRange,
                        IdentifierInfo *scopeName, SourceLocation scopeLoc,
                        IdentifierInfo *parmName, SourceLocation parmLoc,
                        const AvailabilityChange &introduced,
                        const AvailabilityChange &deprecated,
                        const AvailabilityChange &obsoleted,
                        SourceLocation unavailable,
                        const Expr *MessageExpr,
                        AttributeList::Syntax syntax) {
    void *memory = allocate(AttributeFactory::AvailabilityAllocSize);
    return add(new (memory) AttributeList(attrName, attrRange,
                                          scopeName, scopeLoc,
                                          parmName, parmLoc,
                                          introduced, deprecated, obsoleted,
                                          unavailable, MessageExpr, syntax));
  }

  AttributeList *createIntegerAttribute(ASTContext &C, IdentifierInfo *Name,
                                        SourceLocation TokLoc, int Arg);

  AttributeList *createTypeTagForDatatype(
                    IdentifierInfo *attrName, SourceRange attrRange,
                    IdentifierInfo *scopeName, SourceLocation scopeLoc,
                    IdentifierInfo *argumentKindName,
                    SourceLocation argumentKindLoc,
                    ParsedType matchingCType, bool layoutCompatible,
                    bool mustBeNull, AttributeList::Syntax syntax) {
    void *memory = allocate(AttributeFactory::TypeTagForDatatypeAllocSize);
    return add(new (memory) AttributeList(attrName, attrRange,
                                          scopeName, scopeLoc,
                                          argumentKindName, argumentKindLoc,
                                          matchingCType, layoutCompatible,
                                          mustBeNull, syntax));
  }
};

/// addAttributeLists - Add two AttributeLists together
/// The right-hand list is appended to the left-hand list, if any
/// A pointer to the joined list is returned.
/// Note: the lists are not left unmodified.
inline AttributeList *addAttributeLists(AttributeList *Left,
                                        AttributeList *Right) {
  if (!Left)
    return Right;

  AttributeList *next = Left, *prev;
  do {
    prev = next;
    next = next->getNext();
  } while (next);
  prev->setNext(Right);
  return Left;
}

/// CXX0XAttributeList - A wrapper around a C++0x attribute list.
/// Stores, in addition to the list proper, whether or not an actual list was
/// (as opposed to an empty list, which may be ill-formed in some places) and
/// the source range of the list.
struct CXX0XAttributeList { 
  AttributeList *AttrList;
  SourceRange Range;
  bool HasAttr;
  CXX0XAttributeList (AttributeList *attrList, SourceRange range, bool hasAttr)
    : AttrList(attrList), Range(range), HasAttr (hasAttr) {
  }
  CXX0XAttributeList ()
    : AttrList(0), Range(), HasAttr(false) {
  }
};

/// ParsedAttributes - A collection of parsed attributes.  Currently
/// we don't differentiate between the various attribute syntaxes,
/// which is basically silly.
///
/// Right now this is a very lightweight container, but the expectation
/// is that this will become significantly more serious.
class ParsedAttributes {
public:
  ParsedAttributes(AttributeFactory &factory)
    : pool(factory), list(0) {
  }

  ParsedAttributes(ParsedAttributes &attrs)
    : pool(attrs.pool), list(attrs.list) {
    attrs.list = 0;
  }

  AttributePool &getPool() const { return pool; }

  bool empty() const { return list == 0; }

  void add(AttributeList *newAttr) {
    assert(newAttr);
    assert(newAttr->getNext() == 0);
    newAttr->setNext(list);
    list = newAttr;
  }

  void addAll(AttributeList *newList) {
    if (!newList) return;

    AttributeList *lastInNewList = newList;
    while (AttributeList *next = lastInNewList->getNext())
      lastInNewList = next;

    lastInNewList->setNext(list);
    list = newList;
  }

  void set(AttributeList *newList) {
    list = newList;
  }

  void takeAllFrom(ParsedAttributes &attrs) {
    addAll(attrs.list);
    attrs.list = 0;
    pool.takeAllFrom(attrs.pool);
  }

  void clear() { list = 0; pool.clear(); }
  AttributeList *getList() const { return list; }

  /// Returns a reference to the attribute list.  Try not to introduce
  /// dependencies on this method, it may not be long-lived.
  AttributeList *&getListRef() { return list; }

  /// Add attribute with expression arguments.
  AttributeList *addNew(IdentifierInfo *attrName, SourceRange attrRange,
                        IdentifierInfo *scopeName, SourceLocation scopeLoc,
                        IdentifierInfo *parmName, SourceLocation parmLoc,
                        Expr **args, unsigned numArgs,
                        AttributeList::Syntax syntax) {
    AttributeList *attr =
      pool.create(attrName, attrRange, scopeName, scopeLoc, parmName, parmLoc,
                  args, numArgs, syntax);
    add(attr);
    return attr;
  }

  /// Add availability attribute.
  AttributeList *addNew(IdentifierInfo *attrName, SourceRange attrRange,
                        IdentifierInfo *scopeName, SourceLocation scopeLoc,
                        IdentifierInfo *parmName, SourceLocation parmLoc,
                        const AvailabilityChange &introduced,
                        const AvailabilityChange &deprecated,
                        const AvailabilityChange &obsoleted,
                        SourceLocation unavailable,
                        const Expr *MessageExpr,
                        AttributeList::Syntax syntax) {
    AttributeList *attr =
      pool.create(attrName, attrRange, scopeName, scopeLoc, parmName, parmLoc,
                  introduced, deprecated, obsoleted, unavailable,
                  MessageExpr, syntax);
    add(attr);
    return attr;
  }

  /// Add type_tag_for_datatype attribute.
  AttributeList *addNewTypeTagForDatatype(
                        IdentifierInfo *attrName, SourceRange attrRange,
                        IdentifierInfo *scopeName, SourceLocation scopeLoc,
                        IdentifierInfo *argumentKindName,
                        SourceLocation argumentKindLoc,
                        ParsedType matchingCType, bool layoutCompatible,
                        bool mustBeNull, AttributeList::Syntax syntax) {
    AttributeList *attr =
      pool.createTypeTagForDatatype(attrName, attrRange,
                                    scopeName, scopeLoc,
                                    argumentKindName, argumentKindLoc,
                                    matchingCType, layoutCompatible,
                                    mustBeNull, syntax);
    add(attr);
    return attr;
  }

  AttributeList *addNewInteger(ASTContext &C, IdentifierInfo *name,
                               SourceLocation loc, int arg) {
    AttributeList *attr =
      pool.createIntegerAttribute(C, name, loc, arg);
    add(attr);
    return attr;
  }


private:
  mutable AttributePool pool;
  AttributeList *list;
};

}  // end namespace clang

#endif
