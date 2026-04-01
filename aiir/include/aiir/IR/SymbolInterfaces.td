//===- SymbolInterfaces.td - Interfaces for symbol ops -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces and traits that can be used to define
// properties of symbol and symbol table operations.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_IR_SYMBOLINTERFACES
#define AIIR_IR_SYMBOLINTERFACES

include "aiir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// SymbolOpInterface
//===----------------------------------------------------------------------===//

def Symbol : OpInterface<"SymbolOpInterface"> {
  let description = [{
    This interface describes an operation that may define a `Symbol`. A `Symbol`
    operation resides immediately within a region that defines a `SymbolTable`.
    See [Symbols and SymbolTables](../SymbolsAndSymbolTables.md) for more details
    and constraints on `Symbol` operations.
  }];
  let cppNamespace = "::aiir";

  let methods = [
    InterfaceMethod<"Returns the name of this symbol.",
      "::aiir::StringAttr", "getNameAttr", (ins), [{
        // Don't rely on the trait implementation as optional symbol operations
        // may override this.
        return aiir::SymbolTable::getSymbolName($_op);
      }], /*defaultImplementation=*/[{
        return aiir::SymbolTable::getSymbolName(this->getOperation());
      }]
    >,
    InterfaceMethod<"Sets the name of this symbol.",
      "void", "setName", (ins "::aiir::StringAttr":$name), [{}],
      /*defaultImplementation=*/[{
        this->getOperation()->setAttr(
            aiir::SymbolTable::getSymbolAttrName(), name);
      }]
    >,
    InterfaceMethod<"Gets the visibility of this symbol.",
      "aiir::SymbolTable::Visibility", "getVisibility", (ins), [{}],
      /*defaultImplementation=*/[{
        return aiir::SymbolTable::getSymbolVisibility(this->getOperation());
      }]
    >,
    InterfaceMethod<"Returns true if this symbol has nested visibility.",
      "bool", "isNested", (ins),  [{}],
      /*defaultImplementation=*/[{
        return $_op.getVisibility() == aiir::SymbolTable::Visibility::Nested;
      }]
    >,
    InterfaceMethod<"Returns true if this symbol has private visibility.",
      "bool", "isPrivate", (ins),  [{}],
      /*defaultImplementation=*/[{
        return $_op.getVisibility() == aiir::SymbolTable::Visibility::Private;
      }]
    >,
    InterfaceMethod<"Returns true if this symbol has public visibility.",
      "bool", "isPublic", (ins),  [{}],
      /*defaultImplementation=*/[{
        return $_op.getVisibility() == aiir::SymbolTable::Visibility::Public;
      }]
    >,
    InterfaceMethod<"Sets the visibility of this symbol.",
      "void", "setVisibility", (ins "aiir::SymbolTable::Visibility":$vis), [{}],
      /*defaultImplementation=*/[{
        aiir::SymbolTable::setSymbolVisibility(this->getOperation(), vis);
      }]
    >,
    InterfaceMethod<"Sets the visibility of this symbol to be nested.",
      "void", "setNested", (ins),  [{}],
      /*defaultImplementation=*/[{
        $_op.setVisibility(aiir::SymbolTable::Visibility::Nested);
      }]
    >,
    InterfaceMethod<"Sets the visibility of this symbol to be private.",
      "void", "setPrivate", (ins),  [{}],
      /*defaultImplementation=*/[{
        $_op.setVisibility(aiir::SymbolTable::Visibility::Private);
      }]
    >,
    InterfaceMethod<"Sets the visibility of this symbol to be public.",
      "void", "setPublic", (ins),  [{}],
      /*defaultImplementation=*/[{
        $_op.setVisibility(aiir::SymbolTable::Visibility::Public);
      }]
    >,
    InterfaceMethod<[{
        Get all of the uses of the current symbol that are nested within the
        given operation 'from'.
        Note: See aiir::SymbolTable::getSymbolUses for more details.
      }],
      "::std::optional<::aiir::SymbolTable::UseRange>", "getSymbolUses",
      (ins "::aiir::Operation *":$from), [{}],
      /*defaultImplementation=*/[{
        return ::aiir::SymbolTable::getSymbolUses(this->getOperation(), from);
      }]
    >,
    InterfaceMethod<[{
        Return if the current symbol is known to have no uses that are nested
        within the given operation 'from'.
        Note: See aiir::SymbolTable::symbolKnownUseEmpty for more details.
      }],
      "bool", "symbolKnownUseEmpty", (ins "::aiir::Operation *":$from), [{}],
      /*defaultImplementation=*/[{
        return ::aiir::SymbolTable::symbolKnownUseEmpty(this->getOperation(),
                                                        from);
      }]
    >,
    InterfaceMethod<[{
        Attempt to replace all uses of the current symbol with the provided
        symbol 'newSymbol' that are nested within the given operation 'from'.
        Note: See aiir::SymbolTable::replaceAllSymbolUses for more details.
      }],
      "::llvm::LogicalResult", "replaceAllSymbolUses",
      (ins "::aiir::StringAttr":$newSymbol, "::aiir::Operation *":$from), [{}],
      /*defaultImplementation=*/[{
        return ::aiir::SymbolTable::replaceAllSymbolUses(this->getOperation(),
                                                         newSymbol, from);
      }]
    >,
    InterfaceMethod<[{
        Returns true if this operation optionally defines a symbol based on the
        presence of the symbol name.
      }],
      "bool", "isOptionalSymbol", (ins), [{}],
      /*defaultImplementation=*/[{ return false; }]
    >,
    InterfaceMethod<[{
        Returns true if this operation can be discarded if it has no remaining
        symbol uses.
      }],
      "bool", "canDiscardOnUseEmpty", (ins), [{}],
      /*defaultImplementation=*/[{
        // By default, base this on the visibility alone. A symbol can be
        // discarded as long as it is not public. Only public symbols may be
        // visible from outside of the IR.
        return $_op.getVisibility() != ::aiir::SymbolTable::Visibility::Public;
      }]
    >,
    InterfaceMethod<[{
        Returns true if this operation is a declaration of a symbol (as opposed
        to a definition).
      }],
      "bool", "isDeclaration", (ins), [{}],
      /*defaultImplementation=*/[{
        // By default, assume that the operation defines a symbol.
        return false;
      }]
    >,
  ];

  let verify = [{
    // If this is an optional symbol, bail out early if possible.
    auto concreteOp = cast<ConcreteOp>($_op);
    if (concreteOp.isOptionalSymbol()) {
      if(!concreteOp->getInherentAttr(::aiir::SymbolTable::getSymbolAttrName()).value_or(Attribute{}))
        return success();
    }
    if (::aiir::failed(::aiir::detail::verifySymbol($_op)))
      return ::aiir::failure();
    if (concreteOp.isDeclaration() && concreteOp.isPublic())
      return concreteOp.emitOpError("symbol declaration cannot have public "
             "visibility");
    if ($_op->getNumResults() != 0)
      return concreteOp.emitOpError("symbols must not have results");
    auto parent = $_op->getParentOp();
    if (parent && !parent->hasTrait<OpTrait::SymbolTable>() && parent->isRegistered()) {
      return concreteOp.emitOpError("symbol's parent must have the SymbolTable "
             "trait");
    }
    return success();
  }];

  let extraSharedClassDeclaration = [{
    using Visibility = aiir::SymbolTable::Visibility;

    /// Convenience version of `getNameAttr` that returns a StringRef.
    ::aiir::StringRef getName() {
      return getNameAttr().getValue();
    }

    /// Convenience version of `setName` that take a StringRef.
    void setName(::aiir::StringRef name) {
      setName(::aiir::StringAttr::get($_op->getContext(), name));
    }
  }];

  // Add additional classof checks to properly handle "optional" symbols.
  let extraClassOf = [{
    return $_op->hasAttr(::aiir::SymbolTable::getSymbolAttrName());
  }];
}

//===----------------------------------------------------------------------===//
// SymbolUserOpInterface
//===----------------------------------------------------------------------===//

def SymbolUserOpInterface : OpInterface<"SymbolUserOpInterface"> {
  let description = [{
    This interface describes an operation that may use a `Symbol`. This
    interface allows for users of symbols to hook into verification and other
    symbol related utilities that are either costly or otherwise disallowed
    within an operation.
  }];
  let cppNamespace = "::aiir";

  let methods = [
    InterfaceMethod<"Verify the symbol uses held by this operation.",
      "::llvm::LogicalResult", "verifySymbolUses",
      (ins "::aiir::SymbolTableCollection &":$symbolTable)
    >,
  ];
}

def SymbolUserAttrInterface : AttrInterface<"SymbolUserAttrInterface"> {
  let description = [{
    This interface describes an attribute that may use a `Symbol`. This
    interface allows for users of symbols to hook into verification and other
    symbol related utilities that are either costly or otherwise disallowed
    within an operation (e.g., recreating symbol users per op verified rather
    than per symbol table, or querying symbols usage of sibblings).
  }];
  let cppNamespace = "::aiir";

  let methods = [
    InterfaceMethod<"Verify the symbol uses held by this attribute of this operation.",
      "::llvm::LogicalResult", "verifySymbolUses",
      (ins "::aiir::Operation *":$op,
           "::aiir::SymbolTableCollection &":$symbolTable)
    >,
  ];
}

//===----------------------------------------------------------------------===//
// Symbol Traits
//===----------------------------------------------------------------------===//

// Op defines a symbol table.
def SymbolTable : NativeOpTrait<"SymbolTable">;

#endif // AIIR_IR_SYMBOLINTERFACES
