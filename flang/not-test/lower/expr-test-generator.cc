//===-- test/lower/expr-test-generator.cc -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstring>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <vector>

// Utility to generate Fortran expression evaluation tests.
// Given an expression "x op y", this utility can generates
// functions that evaluate the expression and takes x and y
// as argument and return the result.
// The utility allows to describe fortran intrinsic operations
// and their possible operand types in a table driven way.
// Functions to test all possible intrinsic operations can then be
// generated.
// The utility can also generate a driver function that calls the
// previous and compare the results with expectations
// (currently provided by calling a similar function compiled by a reference
// compiler).

// TODO: Integrate with f18 libs to remove redundancies ?
// TODO: How could Peter's intrinsic table be used to generate
//       intrinsic expression tests in a similar way ?

enum class TypeCategory { Integer, Real, Complex, Logical, Character };

struct Type {
  constexpr Type(TypeCategory c) : cat{c}, kind{} {}
  constexpr Type(TypeCategory c, int k) : cat{c}, kind{k} {}
  bool operator==(const Type &that) const {
    return cat == that.cat && kind == that.kind;
  }
  bool operator!=(const Type &that) const { return !(*this == that); }
  std::ostream &Dump(std::ostream &s) const {
    switch (cat) {
    case TypeCategory::Integer: s << "INTEGER"; break;
    case TypeCategory::Real: s << "REAL"; break;
    case TypeCategory::Logical: s << "LOGICAL"; break;
    case TypeCategory::Complex: s << "COMPLEX"; break;
    case TypeCategory::Character: s << "CHARACTER"; break;
    }
    if (kind) {
      s << "(" << *kind << ")";
    }
    return s;
  }

  TypeCategory cat;
  std::optional<int> kind;  // none = default
};

static constexpr Type DefaultReal{TypeCategory::Real};
static constexpr Type Real4{TypeCategory::Real, 4};
static constexpr Type Real8{TypeCategory::Real, 8};
static constexpr Type DefaultInteger{TypeCategory::Integer};
static constexpr Type Integer1{TypeCategory::Integer, 1};
static constexpr Type Integer2{TypeCategory::Integer, 2};
static constexpr Type Integer4{TypeCategory::Integer, 4};
static constexpr Type Integer8{TypeCategory::Integer, 8};
static constexpr Type DefaultComplex{TypeCategory::Complex};
static constexpr Type Complex4{TypeCategory::Complex, 4};
static constexpr Type Complex8{TypeCategory::Complex, 4};
static constexpr Type DefaultLogical{TypeCategory::Logical};
// Currently other types are not testable because existing Fortran
// compilers do not provide these types.

enum class TypePattern {
  OperandNumeric,
  OperandLogical,
  OperandComparable,
  OperandOrdered
};
enum class Constraint { None, NonZeroRHS, NoNegativeLHSWhenRealRHS };

struct Operation {
  bool IsUnaryOp() const { return isUnaryOp; }
  std::optional<Type> GetResultTypeForArguments(Type, Type) const;
  std::optional<Type> GetResultTypeForArgument(Type) const;
  const char *name;
  const char *symbol;
  TypePattern pattern;
  Constraint constraint{Constraint::None};
  bool isUnaryOp{false};
};

// Sorted by increasing precedence (not stricly increasing). See F2018
// table 10.1.
static Operation operations[]{
    // TODO should be OK constexpr but g++ 8.2 not happy
    // defined-binary-op cannot be described here.
    {"eqv", ".EQV.", TypePattern::OperandLogical},
    {"neqv", ".NEQV.", TypePattern::OperandLogical},
    {"or", ".OR.", TypePattern::OperandLogical},
    {"and", ".AND.", TypePattern::OperandLogical},
    {"not", ".NOT.", TypePattern::OperandLogical, Constraint::None, true},
    {"eq", ".EQ.", TypePattern::OperandComparable},
    {"ne", ".NE.", TypePattern::OperandComparable},
    {"lt", ".LT.", TypePattern::OperandOrdered},
    {"le", ".LE.", TypePattern::OperandOrdered},
    {"gt", ".GT.", TypePattern::OperandOrdered},
    {"ge", ".GE.", TypePattern::OperandOrdered},
    // TODO: Concat
    {"add", "+", TypePattern::OperandNumeric},
    {"sub", "-", TypePattern::OperandNumeric},
    {"minus", "-", TypePattern::OperandNumeric, Constraint::None, true},
    {"plus", "+", TypePattern::OperandNumeric, Constraint::None, true},
    {"mult", "*", TypePattern::OperandNumeric},
    {"div", "/", TypePattern::OperandNumeric, Constraint::NonZeroRHS},
    {"power", "**", TypePattern::OperandNumeric,
        Constraint::NoNegativeLHSWhenRealRHS},  // TODO: check power pattern
    // defined-unary-op cannot be described here.
};

static bool IsLogical(const Type t) { return t.cat == TypeCategory::Logical; }
static bool IsCharacter(const Type t) {
  return t.cat == TypeCategory::Character;
}
static bool IsComplex(const Type t) { return t.cat == TypeCategory::Complex; }
static bool IsReal(const Type t) { return t.cat == TypeCategory::Real; }

static bool IsOrderedNumeric(const Type t) {
  return t.cat == TypeCategory::Real || t.cat == TypeCategory::Integer;
}

static bool IsNumeric(const Type t) {
  return t.cat == TypeCategory::Real || t.cat == TypeCategory::Integer ||
      t.cat == TypeCategory::Complex;
}

std::optional<Type> Operation::GetResultTypeForArgument(Type t) const {
  switch (pattern) {
  case TypePattern::OperandNumeric:
    if (IsNumeric(t)) {
      return t;
    }
    break;
  case TypePattern::OperandLogical:
    if (IsLogical(t)) {
      return t;
    }
    break;
  case TypePattern::OperandComparable:
  case TypePattern::OperandOrdered:
    assert(false && "No unary comparisons");
    break;
  }
  return std::nullopt;
}

static std::optional<Type> SelectBiggestKindForResult(
    TypeCategory resultCat, Type lhs, Type rhs) {
  if (!lhs.kind && !rhs.kind) {
    return Type{resultCat};
  } else if (lhs.kind && rhs.kind) {
    return Type{resultCat, std::max(*lhs.kind, *rhs.kind)};
  } else {
    // Cannot know without more info about compiler what default compares to.
    return std::nullopt;
  }
}

std::optional<Type> Operation::GetResultTypeForArguments(
    Type lhs, Type rhs) const {
  // See Fortran 2018 table 10.2 and section 10.1.9.3
  switch (pattern) {
  case TypePattern::OperandNumeric:
    if (!IsNumeric(rhs) || !IsNumeric(lhs)) {
      return std::nullopt;
    }
    if (lhs.cat == rhs.cat) {
      return SelectBiggestKindForResult(lhs.cat, lhs, rhs);
    }
    // different categories
    if (IsComplex(lhs)) {
      if (IsReal(rhs)) {
        return SelectBiggestKindForResult(TypeCategory::Complex, lhs, rhs);
      }
      return lhs;
    }
    if (IsComplex(rhs)) {
      if (IsReal(lhs)) {
        return SelectBiggestKindForResult(TypeCategory::Complex, lhs, rhs);
      }
      return rhs;
    }
    if (IsReal(lhs)) {
      // rhs must be integer
      return lhs;
    }
    if (IsReal(rhs)) {
      // lhs must be integer
      return rhs;
    }
    assert(false && "no more operand cases");
  case TypePattern::OperandLogical:
    if (IsLogical(rhs) && IsLogical(lhs)) {
      return SelectBiggestKindForResult(TypeCategory::Logical, lhs, rhs);
    }
    break;
  case TypePattern::OperandComparable:
    if (IsNumeric(lhs) && IsNumeric(rhs)) {
      return DefaultLogical;
    } else if (IsCharacter(lhs) && IsCharacter(lhs) && lhs.kind == rhs.kind) {
      return DefaultLogical;
    }
    break;
  case TypePattern::OperandOrdered:
    if (IsOrderedNumeric(lhs) && IsOrderedNumeric(rhs)) {
      return DefaultLogical;
    } else if (IsCharacter(lhs) && IsCharacter(lhs) && lhs.kind == rhs.kind) {
      return DefaultLogical;
    }
    break;
  }
  return std::nullopt;
}

// Test Plan Part

// Only string list input for now
using Input = std::vector<const char *>;
// TODO: more inputs method (e.g. random pick)

enum class Eval { Constant, Dynamic };
struct TestPlan {
  std::vector<const char *> operationToTests;
  std::vector<std::pair<Type, Input>> inputs;
  Eval referenceEvaluationMethod{Eval::Dynamic};
  Eval testEvaluationMethod{Eval::Dynamic};
};

static inline std::string IndexedName(const char *name, std::size_t index) {
  return std::string{name} + std::to_string(index);
}

static constexpr auto passedName{"passed"};
static constexpr auto failedName{"failed"};

struct CodeGenerator {
  static constexpr auto varNameBase{"x"};
  static constexpr auto paramNameBase{"a"};
  static constexpr auto loopIndexNameBase{"i"};
  static constexpr auto testResultName{"test_res"};
  static constexpr auto refResultName{"ref_res"};

  template<Eval Ev>
  void GenerateEvaluationFunction(
      const std::string &functionName, std::ostream &s) const {
    GenerateEvaluationFunctionStmtAndSpec<Ev>(functionName, s);
    GenerateExpressionEvaluation<Ev>(functionName, s);
    s << "END FUNCTION" << std::endl;
    s << std::endl;
  }

  void GenerateEvaluationFunction(const std::string &functionNameBase,
      std::ostream &s, Eval ev, bool isReference) const {
    auto functionName{isReference ? GetReferenceFunctionName(functionNameBase)
                                  : GetTestFunctionName(functionNameBase)};
    switch (ev) {
    case Eval::Constant:
      GenerateEvaluationFunction<Eval::Constant>(functionName, s);
      break;
    case Eval::Dynamic:
      GenerateEvaluationFunction<Eval::Dynamic>(functionName, s);
      break;
    }
  }

  void GenerateDriverSubroutine(const std::string &functionNameBase,
      std::ostream &s, Eval refEval, Eval testEval) const {
    s << "SUBROUTINE " << GetDriverName(functionNameBase) << "(" << passedName
      << ", " << failedName << ")" << std::endl;
    std::size_t numInputs{inputs.size()};
    assert(numInputs > 0 && "Expected non empty inputs in driver");
    if (refEval == Eval::Dynamic || testEval == Eval::Dynamic) {
      // Inputs are required inside the driver only if one evaluation function
      // is dynamic and inputs need to be passed to it from the driver.
      for (std::size_t i{0}; i < numInputs; ++i) {
        GenerateInputAsArrayParameter(i, s);
      }
    }
    s << "INTEGER :: " << passedName << ", " << failedName << std::endl;
    resultType.Dump(s) << " :: " << testResultName << ", " << refResultName
                       << std::endl;

    // TODO: Add a test number limit in cases there are many inputs ? This would
    // need to be kept in sync with folding evaluation.
    for (std::size_t i{0}; i < numInputs; ++i) {
      s << "DO " << IndexedName(loopIndexNameBase, i) << " = 1,"
        << inputs[i]->second.size() << std::endl;
    }

    // TODO: Handle constraints (e.g divide by zero). Would also need to be kept
    // in sync with folding evaluation.
    s << refResultName << " = ";
    auto refFuncName{GetReferenceFunctionName(functionNameBase)};
    switch (refEval) {
    case Eval::Dynamic:
      GenerateEvaluationFunctionCall<Eval::Dynamic>(refFuncName, s);
      break;
    case Eval::Constant:
      GenerateEvaluationFunctionCall<Eval::Constant>(refFuncName, s);
      break;
    }
    s << testResultName << " = ";
    auto testFuncName{GetTestFunctionName(functionNameBase)};
    switch (testEval) {
    case Eval::Dynamic:
      GenerateEvaluationFunctionCall<Eval::Dynamic>(testFuncName, s);
      break;
    case Eval::Constant:
      GenerateEvaluationFunctionCall<Eval::Constant>(testFuncName, s);
      break;
    }

    s << "IF (";
    GenerateResultsComparison(s);
    s << ") THEN" << std::endl;
    s << passedName << " = " << passedName << " + 1" << std::endl;
    s << "ELSE" << std::endl;
    s << failedName << " = " << failedName << " + 1" << std::endl;
    GenerateFailureMessage(functionNameBase, s);
    s << "END IF" << std::endl;

    for (std::size_t i{0}; i < numInputs; ++i) {
      s << "END DO" << std::endl;
    }
    s << "END SUBROUTINE" << std::endl;
    s << std::endl;
  }

  template<Eval Ev>
  void GenerateEvaluationFunctionCall(
      const std::string &functionName, std::ostream &s) const {
    s << functionName << "(";
    if constexpr (Ev == Eval::Dynamic) {
      ApplyOnInputIndexes(
          [&](std::size_t i) {
            return GetArrayElement(paramNameBase, loopIndexNameBase, i);
          },
          ", ", s);
    } else {
      static_assert(Ev == Eval::Constant, "unhandled evaluation method");
      ApplyOnInputIndexes(
          [&](std::size_t i) { return IndexedName(loopIndexNameBase, i); },
          ", ", s);
    }
    s << ")" << std::endl;
  }

  void GenerateFailureMessage(
      const std::string &functionNameBase, std::ostream &s) const {
    s << "PRINT *, \"FAILED " << functionNameBase << " test: \"";
    for (std::size_t i{0}; i < inputs.size(); ++i) {
      auto loopId{IndexedName(loopIndexNameBase, i)};
      s << ", \"" << loopId << " = \", " << loopId;
    }
    s << std::endl;
    s << "PRINT *, \"  expected \", " << refResultName << ", \" got: \","
      << testResultName << std::endl;
  }

  void GenerateInputAsArrayParameter(std::size_t i, std::ostream &s) const {
    const auto &literals{inputs[i]->second};
    const auto size{literals.size()};
    assert(size > 0 && "No actual literal input for test");
    inputs[i]->first.Dump(s)
        << ", PARAMETER :: " << IndexedName(paramNameBase, i) << "(" << size
        << ") = [";
    s << literals[0];
    for (std::size_t i{1}; i < size; ++i) {
      s << ", " << literals[i];
    }
    s << "]" << std::endl;
  }

  void GenerateDriverCall(
      const std::string &functionNameBase, std::ostream &s) const {
    s << "CALL " << GetDriverName(functionNameBase) << "(" << passedName << ", "
      << failedName << ")" << std::endl;
  }

  template<Eval Ev>
  void GenerateEvaluationFunctionStmtAndSpec(
      const std::string &functionName, std::ostream &s) const {
    resultType.Dump(s) << " FUNCTION " << functionName << "(";
    assert(inputs.size() && "expect at least on argument");
    const auto &name{Ev == Eval::Constant ? loopIndexNameBase : varNameBase};
    s << IndexedName(name, 0);
    for (std::size_t i{1}; i < inputs.size(); ++i) {
      s << ", " << IndexedName(name, i);
    }
    s << ")" << std::endl;
    std::size_t i{0};
    if constexpr (Ev == Eval::Constant) {
      for (const auto *input : inputs) {
        s << "INTEGER :: " << IndexedName(name, i++) << std::endl;
      }
    } else {
      for (const auto *input : inputs) {
        input->first.Dump(s) << " :: " << IndexedName(name, i++) << std::endl;
      }
    }
  }

  template<Eval Ev>
  void GenerateExpressionEvaluation(
      const std::string &resultName, std::ostream &s) const {
    if constexpr (Ev == Eval::Constant) {
      GenerateConstantExpressionEvaluation(resultName, s);
    } else {
      static_assert(
          Ev == Eval::Dynamic, "unhandled expression evaluation method");
      s << resultName << " = ";
      GenerateExpression(
          [&](std::size_t i) { return IndexedName(varNameBase, i); }, s);
      s << std::endl;
    }
  }

  void GenerateConstantExpressionEvaluation(
      const std::string &resultName, std::ostream &s) const {
    // Declare constant inputs
    for (std::size_t i{0}; i < inputs.size(); ++i) {
      GenerateInputAsArrayParameter(i, s);
    }
    // Evaluate in a parameter array using ac-implied-do
    auto cstResultName{IndexedName(paramNameBase, inputs.size())};
    assert(inputs.size() > 0 && "expected at least one operands");
    resultType.Dump(s) << " , PARAMETER :: " << cstResultName << "(*";
    for (std::size_t i{1}; i < inputs.size(); ++i) {
      s << ", *";
    }
    s << ") = RESHAPE ([(";
    for (std::size_t i{1}; i < inputs.size(); ++i) {
      s << "(";
    }

    // Expression evaluation inside ac-implied-do
    // TODO: how to handle constraints ?
    GenerateExpression(
        [&](std::size_t i) {
          return GetArrayElement(paramNameBase, loopIndexNameBase, i);
        },
        s);

    // ac-implied-do (opening ")" were emited before).
    s << ", ";
    ApplyOnInputIndexes(
        [&](std::size_t i) {
          return IndexedName(loopIndexNameBase, i) + " = 1," +
              std::to_string(inputs[i]->second.size()) + ")";
        },
        ",", s);

    // RESHAPE SHAPE argument
    s << "], [";
    ApplyOnInputIndexes(
        [&](std::size_t i) { return inputs[i]->second.size(); }, ",", s);
    s << "])" << std::endl;

    // dynamically fetch requested result from the constant array
    s << resultName << " = " << cstResultName << "(";
    ApplyOnInputIndexes(
        [&](std::size_t i) { return IndexedName(loopIndexNameBase, i); }, ",",
        s);
    s << ")" << std::endl;
  }

  template<typename T>
  void inline ApplyOnInputIndexes(
      const T &callable, const std::string &sep, std::ostream &s) const {
    auto size{inputs.size()};
    if (size != 0) {
      s << callable(0);
    }
    for (std::size_t i{1}; i < size; ++i) {
      s << ", " << callable(i);
    }
  }

  template<typename T>
  void inline GenerateExpression(
      const T &operandGenerator, std::ostream &s) const {
    if (inputs.size() == 1) {
      s << op.symbol << " " << operandGenerator(0);
    } else {
      assert(inputs.size() == 2 && "expected binary opreation");
      s << operandGenerator(0) << " " << op.symbol << " "
        << operandGenerator(1);
    }
  }

  void GenerateEvaluationFunctionInterface(const std::string &functionNameBase,
      std::ostream &s, Eval ev, bool isReference) const {
    auto functionName{isReference ? GetReferenceFunctionName(functionNameBase)
                                  : GetTestFunctionName(functionNameBase)};
    switch (ev) {
    case Eval::Constant:
      GenerateEvaluationFunctionStmtAndSpec<Eval::Constant>(functionName, s);
      break;
    case Eval::Dynamic:
      GenerateEvaluationFunctionStmtAndSpec<Eval::Dynamic>(functionName, s);
      break;
    }
    s << "END FUNCTION" << std::endl;
  }

  void GenerateResultsComparison(std::ostream &s) const {
    auto compareReal{[&](const std::string &x, const std::string &y) {
      // TODO: This should not always be an absolute comparison (epsilon margin
      // for fp.).
      s << x << ".EQ." << y << " .OR. ";
      s << "(IEEE_IS_NAN(" << x << ") .AND. IEEE_IS_NAN(" << y << "))";
    }};
    if (resultType.cat == TypeCategory::Real) {
      compareReal(refResultName, testResultName);
    } else if (resultType.cat == TypeCategory::Complex) {
      s << "(";
      compareReal(std::string{refResultName} + "%RE",
          std::string{testResultName} + "%RE");
      s << ") .AND. (";
      compareReal(std::string{refResultName} + "%IM",
          std::string{testResultName} + "%IM");
      s << ")";
    } else {
      const auto *compareOp{
          resultType.cat == TypeCategory::Logical ? ".EQV." : ".EQ."};
      s << refResultName << compareOp << testResultName;
    }
  }

  // a0(i0) and such
  static std::string GetArrayElement(
      const char *arrayNameBase, const char *indexNameBase, std::size_t i) {
    return IndexedName(arrayNameBase, i) + "(" + IndexedName(indexNameBase, i) +
        ")";
  }

  static std::string GetTestFunctionName(const std::string &functionNameBase) {
    return functionNameBase + "_test";
  }
  static std::string GetReferenceFunctionName(
      const std::string &functionNameBase) {
    return functionNameBase + "_ref";
  }
  static std::string GetDriverName(const std::string &functionNameBase) {
    return functionNameBase + "_driver";
  }
  Type resultType;
  const Operation &op;
  std::vector<const std::pair<Type, Input> *> inputs;
};

// Test Generator
struct TestGenerator {

  void GenerateTests(std::ostream &testFile, std::ostream &refFile) {
    std::stringstream testContentStream;
    std::stringstream testInterfaceStream;
    std::stringstream referenceAndDriverContentStream;
    std::stringstream programContentStream;
    auto generateInBuffers{
        [&](const CodeGenerator &codeGen, const std::string &functionNameBase) {
          codeGen.GenerateEvaluationFunction(functionNameBase,
              testContentStream, plan.testEvaluationMethod, false);
          codeGen.GenerateEvaluationFunctionInterface(functionNameBase,
              testInterfaceStream, plan.testEvaluationMethod, false);
          codeGen.GenerateEvaluationFunction(functionNameBase,
              referenceAndDriverContentStream, plan.referenceEvaluationMethod,
              true);
          codeGen.GenerateDriverSubroutine(functionNameBase,
              referenceAndDriverContentStream, plan.referenceEvaluationMethod,
              plan.testEvaluationMethod);
          codeGen.GenerateDriverCall(functionNameBase, programContentStream);
        }};

    for (const auto *opName : plan.operationToTests) {
      const auto *op{GetOperation(opName)};
      assert(op && "Broken test plan: undefined operation");
      for (const auto &input1 : plan.inputs) {
        if (op->IsUnaryOp()) {
          if (auto resultType{op->GetResultTypeForArgument(input1.first)}) {
            CodeGenerator codeGen{*resultType, *op, {&input1}};
            auto functionNameBase{GetDistinctFunctionNameBase(*op)};
            generateInBuffers(codeGen, functionNameBase);
          }
        } else {
          for (const auto &input2 : plan.inputs) {
            if (auto resultType{op->GetResultTypeForArguments(
                    input1.first, input2.first)}) {
              CodeGenerator codeGen{*resultType, *op, {&input1, &input2}};
              auto functionNameBase{GetDistinctFunctionNameBase(*op)};
              generateInBuffers(codeGen, functionNameBase);
            }
          }
        }
      }
    }

    // Organize generated code
    testFile << "! Generated test file" << std::endl;
    testFile << testContentStream.rdbuf();
    testFile << "! End of generated test file" << std::endl;

    refFile << "! Generated reference and driver " << std::endl;
    refFile << "MODULE REFERENCE_AND_DRIVER" << std::endl;
    refFile << "USE IEEE_ARITHMETIC" << std::endl;  // for IEEE_IS_NAN
    refFile << "INTERFACE" << std::endl;
    refFile << testInterfaceStream.rdbuf();
    refFile << "END INTERFACE" << std::endl;
    refFile << std::endl;
    refFile << "CONTAINS" << std::endl;
    refFile << referenceAndDriverContentStream.rdbuf();
    refFile << "END MODULE" << std::endl;
    refFile << std::endl;
    refFile << "PROGRAM EXPR_TEST" << std::endl;
    refFile << "USE REFERENCE_AND_DRIVER" << std::endl;
    refFile << "INTEGER :: " << passedName << ", " << failedName << std::endl;
    refFile << passedName << " = 0 " << std::endl;
    refFile << failedName << " = 0 " << std::endl;
    refFile << programContentStream.rdbuf();
    refFile << "PRINT *, \"Passed: \"," << passedName << ", \" Failed: \", "
            << failedName << std::endl;
    refFile << "IF (" << failedName << ".GT. 0) ERROR STOP 1" << std::endl;
    refFile << "END PROGRAM" << std::endl;
    refFile << "! End of generated reference and driver" << std::endl;
  }

  const Operation *GetOperation(const char *name) {
    assert(name && "nullptr string for test name");
    for (const auto &op : operations) {
      assert(op.name && "Broken operation");
      if (std::strcmp(name, op.name) == 0) {
        return &op;
      }
    }
    return nullptr;
  }

  std::string GetDistinctFunctionNameBase(const Operation &op) {
    return IndexedName(op.name, funcId++);
  }

  TestPlan plan;
  std::size_t funcId{0};
};

// Test driver
int main(int argc, char **argv) {
  Eval testEvalMethod{Eval::Dynamic};
  Eval refEvalMethod{Eval::Dynamic};
  int planId{0};
  for (int i{0}; i < argc; ++i) {
    if (std::strcmp(argv[i], "test=folding") == 0) {
      testEvalMethod = Eval::Constant;
    } else if (std::strcmp(argv[i], "ref=folding") == 0) {
      refEvalMethod = Eval::Constant;
    }
  }

  TestGenerator{
      {
          {"eqv", "neqv", "or", "and", "not", "eq", "ne", "lt", "le", "gt",
              "ge", "add", "sub", "minus", "plus", "mult", "div", "power"},
          {
              {Integer1, {"-1_1", "12_1", "2_1"}},
              {Integer2, {"-1_2", "12500_2", "2_2"}},
              {Integer4, {"31000_4", "-64354_4"}},
              {Integer8, {"3000001_8", "-2654637545_8"}},
              {Real4, {"1.03687448_4", "3.1254641_4"}},
              {Real8, {"1.036874168446448_8", "3.12254533554641_8"}},
              {Complex4, {"(-0.5_4,10.35544_4)", "(-5._4, 0.15647_4)"}},
              {Complex8,
                  {"(-0.5_8,10.35546579874_8)",
                      "(-5.64654654_8, 0.155876974647_8)"}},
              {DefaultLogical, {".false.", ".true."}},
          },
          refEvalMethod,
          testEvalMethod,
      }}
      .GenerateTests(std::cout, std::cerr);
}
