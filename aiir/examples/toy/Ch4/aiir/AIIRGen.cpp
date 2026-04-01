//===- AIIRGen.cpp - AIIR Generation from a Toy AST -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple IR generation targeting AIIR from a Module AST
// for the Toy language.
//
//===----------------------------------------------------------------------===//

#include "toy/AIIRGen.h"
#include "aiir/IR/Block.h"
#include "aiir/IR/Diagnostics.h"
#include "aiir/IR/Value.h"
#include "toy/AST.h"
#include "toy/Dialect.h"

#include "aiir/IR/Builders.h"
#include "aiir/IR/BuiltinOps.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/AIIRContext.h"
#include "aiir/IR/Verifier.h"
#include "toy/Lexer.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <vector>

using namespace aiir::toy;
using namespace toy;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

/// Implementation of a simple AIIR emission from the Toy AST.
///
/// This will emit operations that are specific to the Toy language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class AIIRGenImpl {
public:
  AIIRGenImpl(aiir::AIIRContext &context) : builder(&context) {}

  /// Public API: convert the AST for a Toy module (source file) to an AIIR
  /// Module operation.
  aiir::ModuleOp aiirGen(ModuleAST &moduleAST) {
    // We create an empty AIIR module and codegen functions one at a time and
    // add them to the module.
    theModule = aiir::ModuleOp::create(builder.getUnknownLoc());

    for (FunctionAST &f : moduleAST)
      aiirGen(f);

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the Toy operations.
    if (failed(aiir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
  /// A "module" matches a Toy source file: containing a list of functions.
  aiir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  aiir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, aiir::Value> symbolTable;

  /// Helper conversion for a Toy AST location to an AIIR location.
  aiir::Location loc(const Location &loc) {
    return aiir::FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.line,
                                     loc.col);
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  llvm::LogicalResult declare(llvm::StringRef var, aiir::Value value) {
    if (symbolTable.count(var))
      return aiir::failure();
    symbolTable.insert(var, value);
    return aiir::success();
  }

  /// Create the prototype for an AIIR function with as many arguments as the
  /// provided Toy AST prototype.
  aiir::toy::FuncOp aiirGen(PrototypeAST &proto) {
    auto location = loc(proto.loc());

    // This is a generic function, the return type will be inferred later.
    // Arguments type are uniformly unranked tensors.
    llvm::SmallVector<aiir::Type, 4> argTypes(proto.getArgs().size(),
                                              getType(VarType{}));
    auto funcType = builder.getFunctionType(argTypes, /*results=*/{});
    return aiir::toy::FuncOp::create(builder, location, proto.getName(),
                                     funcType);
  }

  /// Emit a new function and add it to the AIIR module.
  aiir::toy::FuncOp aiirGen(FunctionAST &funcAST) {
    // Create a scope in the symbol table to hold variable declarations.
    ScopedHashTableScope<llvm::StringRef, aiir::Value> varScope(symbolTable);

    // Create an AIIR function for the given prototype.
    builder.setInsertionPointToEnd(theModule.getBody());
    aiir::toy::FuncOp function = aiirGen(*funcAST.getProto());
    if (!function)
      return nullptr;

    // Let's start the body of the function now!
    aiir::Block &entryBlock = function.front();
    auto protoArgs = funcAST.getProto()->getArgs();

    // Declare all the function arguments in the symbol table.
    for (const auto nameValue :
         llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (failed(declare(std::get<0>(nameValue)->getName(),
                         std::get<1>(nameValue))))
        return nullptr;
    }

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToStart(&entryBlock);

    // Emit the body of the function.
    if (aiir::failed(aiirGen(*funcAST.getBody()))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return void if no return statement was emitted.
    // FIXME: we may fix the parser instead to always return the last expression
    // (this would possibly help the REPL case later)
    ReturnOp returnOp;
    if (!entryBlock.empty())
      returnOp = dyn_cast<ReturnOp>(entryBlock.back());
    if (!returnOp) {
      ReturnOp::create(builder, loc(funcAST.getProto()->loc()));
    } else if (returnOp.hasOperand()) {
      // Otherwise, if this return operation has an operand then add a result to
      // the function.
      function.setType(builder.getFunctionType(
          function.getFunctionType().getInputs(), getType(VarType{})));
    }

    // If this function isn't main, then set the visibility to private.
    if (funcAST.getProto()->getName() != "main")
      function.setPrivate();

    return function;
  }

  /// Emit a binary operation
  aiir::Value aiirGen(BinaryExprAST &binop) {
    // First emit the operations for each side of the operation before emitting
    // the operation itself. For example if the expression is `a + foo(a)`
    // 1) First it will visiting the LHS, which will return a reference to the
    //    value holding `a`. This value should have been emitted at declaration
    //    time and registered in the symbol table, so nothing would be
    //    codegen'd. If the value is not in the symbol table, an error has been
    //    emitted and nullptr is returned.
    // 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
    //    and the result value is returned. If an error occurs we get a nullptr
    //    and propagate.
    //
    aiir::Value lhs = aiirGen(*binop.getLHS());
    if (!lhs)
      return nullptr;
    aiir::Value rhs = aiirGen(*binop.getRHS());
    if (!rhs)
      return nullptr;
    auto location = loc(binop.loc());

    // Derive the operation name from the binary operator. At the moment we only
    // support '+' and '*'.
    switch (binop.getOp()) {
    case '+':
      return AddOp::create(builder, location, lhs, rhs);
    case '*':
      return MulOp::create(builder, location, lhs, rhs);
    }

    emitError(location, "invalid binary operator '") << binop.getOp() << "'";
    return nullptr;
  }

  /// This is a reference to a variable in an expression. The variable is
  /// expected to have been declared and so should have a value in the symbol
  /// table, otherwise emit an error and return nullptr.
  aiir::Value aiirGen(VariableExprAST &expr) {
    if (auto variable = symbolTable.lookup(expr.getName()))
      return variable;

    emitError(loc(expr.loc()), "error: unknown variable '")
        << expr.getName() << "'";
    return nullptr;
  }

  /// Emit a return operation. This will return failure if any generation fails.
  llvm::LogicalResult aiirGen(ReturnExprAST &ret) {
    auto location = loc(ret.loc());

    // 'return' takes an optional expression, handle that case here.
    aiir::Value expr = nullptr;
    if (ret.getExpr().has_value()) {
      if (!(expr = aiirGen(**ret.getExpr())))
        return aiir::failure();
    }

    // Otherwise, this return operation has zero operands.
    ReturnOp::create(builder, location,
                     expr ? ArrayRef(expr) : ArrayRef<aiir::Value>());
    return aiir::success();
  }

  /// Emit a literal/constant array. It will be emitted as a flattened array of
  /// data in an Attribute attached to a `toy.constant` operation.
  /// See documentation on [Attributes](LangRef.md#attributes) for more details.
  /// Here is an excerpt:
  ///
  ///   Attributes are the mechanism for specifying constant data in AIIR in
  ///   places where a variable is never allowed [...]. They consist of a name
  ///   and a concrete attribute value. The set of expected attributes, their
  ///   structure, and their interpretation are all contextually dependent on
  ///   what they are attached to.
  ///
  /// Example, the source level statement:
  ///   var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  /// will be converted to:
  ///   %0 = "toy.constant"() {value: dense<tensor<2x3xf64>,
  ///     [[1.000000e+00, 2.000000e+00, 3.000000e+00],
  ///      [4.000000e+00, 5.000000e+00, 6.000000e+00]]>} : () -> tensor<2x3xf64>
  ///
  aiir::Value aiirGen(LiteralExprAST &lit) {
    auto type = getType(lit.getDims());

    // The attribute is a vector with a floating point value per element
    // (number) in the array, see `collectData()` below for more details.
    std::vector<double> data;
    data.reserve(llvm::product_of(lit.getDims()));
    collectData(lit, data);

    // The type of this attribute is tensor of 64-bit floating-point with the
    // shape of the literal.
    aiir::Type elementType = builder.getF64Type();
    auto dataType = aiir::RankedTensorType::get(lit.getDims(), elementType);

    // This is the actual attribute that holds the list of values for this
    // tensor literal.
    auto dataAttribute =
        aiir::DenseElementsAttr::get(dataType, llvm::ArrayRef(data));

    // Build the AIIR op `toy.constant`. This invokes the `ConstantOp::build`
    // method.
    return ConstantOp::create(builder, loc(lit.loc()), type, dataAttribute);
  }

  /// Recursive helper function to accumulate the data that compose an array
  /// literal. It flattens the nested structure in the supplied vector. For
  /// example with this array:
  ///  [[1, 2], [3, 4]]
  /// we will generate:
  ///  [ 1, 2, 3, 4 ]
  /// Individual numbers are represented as doubles.
  /// Attributes are the way AIIR attaches constant to operations.
  void collectData(ExprAST &expr, std::vector<double> &data) {
    if (auto *lit = dyn_cast<LiteralExprAST>(&expr)) {
      for (auto &value : lit->getValues())
        collectData(*value, data);
      return;
    }

    assert(isa<NumberExprAST>(expr) && "expected literal or number expr");
    data.push_back(cast<NumberExprAST>(expr).getValue());
  }

  /// Emit a call expression. It emits specific operations for the `transpose`
  /// builtin. Other identifiers are assumed to be user-defined functions.
  aiir::Value aiirGen(CallExprAST &call) {
    llvm::StringRef callee = call.getCallee();
    auto location = loc(call.loc());

    // Codegen the operands first.
    SmallVector<aiir::Value, 4> operands;
    for (auto &expr : call.getArgs()) {
      auto arg = aiirGen(*expr);
      if (!arg)
        return nullptr;
      operands.push_back(arg);
    }

    // Builtin calls have their custom operation, meaning this is a
    // straightforward emission.
    if (callee == "transpose") {
      if (call.getArgs().size() != 1) {
        emitError(location, "AIIR codegen encountered an error: toy.transpose "
                            "does not accept multiple arguments");
        return nullptr;
      }
      return TransposeOp::create(builder, location, operands[0]);
    }

    // Otherwise this is a call to a user-defined function. Calls to
    // user-defined functions are mapped to a custom call that takes the callee
    // name as an attribute.
    return GenericCallOp::create(builder, location, callee, operands);
  }

  /// Emit a print expression. It emits specific operations for two builtins:
  /// transpose(x) and print(x).
  llvm::LogicalResult aiirGen(PrintExprAST &call) {
    auto arg = aiirGen(*call.getArg());
    if (!arg)
      return aiir::failure();

    PrintOp::create(builder, loc(call.loc()), arg);
    return aiir::success();
  }

  /// Emit a constant for a single number (FIXME: semantic? broadcast?)
  aiir::Value aiirGen(NumberExprAST &num) {
    return ConstantOp::create(builder, loc(num.loc()), num.getValue());
  }

  /// Dispatch codegen for the right expression subclass using RTTI.
  aiir::Value aiirGen(ExprAST &expr) {
    switch (expr.getKind()) {
    case toy::ExprAST::Expr_BinOp:
      return aiirGen(cast<BinaryExprAST>(expr));
    case toy::ExprAST::Expr_Var:
      return aiirGen(cast<VariableExprAST>(expr));
    case toy::ExprAST::Expr_Literal:
      return aiirGen(cast<LiteralExprAST>(expr));
    case toy::ExprAST::Expr_Call:
      return aiirGen(cast<CallExprAST>(expr));
    case toy::ExprAST::Expr_Num:
      return aiirGen(cast<NumberExprAST>(expr));
    default:
      emitError(loc(expr.loc()))
          << "AIIR codegen encountered an unhandled expr kind '"
          << Twine(expr.getKind()) << "'";
      return nullptr;
    }
  }

  /// Handle a variable declaration, we'll codegen the expression that forms the
  /// initializer and record the value in the symbol table before returning it.
  /// Future expressions will be able to reference this variable through symbol
  /// table lookup.
  aiir::Value aiirGen(VarDeclExprAST &vardecl) {
    auto *init = vardecl.getInitVal();
    if (!init) {
      emitError(loc(vardecl.loc()),
                "missing initializer in variable declaration");
      return nullptr;
    }

    aiir::Value value = aiirGen(*init);
    if (!value)
      return nullptr;

    // We have the initializer value, but in case the variable was declared
    // with specific shape, we emit a "reshape" operation. It will get
    // optimized out later as needed.
    if (!vardecl.getType().shape.empty()) {
      value = ReshapeOp::create(builder, loc(vardecl.loc()),
                                getType(vardecl.getType()), value);
    }

    // Register the value in the symbol table.
    if (failed(declare(vardecl.getName(), value)))
      return nullptr;
    return value;
  }

  /// Codegen a list of expression, return failure if one of them hit an error.
  llvm::LogicalResult aiirGen(ExprASTList &blockAST) {
    ScopedHashTableScope<StringRef, aiir::Value> varScope(symbolTable);
    for (auto &expr : blockAST) {
      // Specific handling for variable declarations, return statement, and
      // print. These can only appear in block list and not in nested
      // expressions.
      if (auto *vardecl = dyn_cast<VarDeclExprAST>(expr.get())) {
        if (!aiirGen(*vardecl))
          return aiir::failure();
        continue;
      }
      if (auto *ret = dyn_cast<ReturnExprAST>(expr.get()))
        return aiirGen(*ret);
      if (auto *print = dyn_cast<PrintExprAST>(expr.get())) {
        if (aiir::failed(aiirGen(*print)))
          return aiir::success();
        continue;
      }

      // Generic expression dispatch codegen.
      if (!aiirGen(*expr))
        return aiir::failure();
    }
    return aiir::success();
  }

  /// Build a tensor type from a list of shape dimensions.
  aiir::Type getType(ArrayRef<int64_t> shape) {
    // If the shape is empty, then this type is unranked.
    if (shape.empty())
      return aiir::UnrankedTensorType::get(builder.getF64Type());

    // Otherwise, we use the given shape.
    return aiir::RankedTensorType::get(shape, builder.getF64Type());
  }

  /// Build an AIIR type from a Toy AST variable type (forward to the generic
  /// getType above).
  aiir::Type getType(const VarType &type) { return getType(type.shape); }
};

} // namespace

namespace toy {

// The public API for codegen.
aiir::OwningOpRef<aiir::ModuleOp> aiirGen(aiir::AIIRContext &context,
                                          ModuleAST &moduleAST) {
  return AIIRGenImpl(context).aiirGen(moduleAST);
}

} // namespace toy
