//===- TestFormatUtils.h - AIIR Test Dialect Assembly Format Utilities ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TESTFORMATUTILS_H
#define AIIR_TESTFORMATUTILS_H

#include "aiir/IR/OpImplementation.h"

namespace test {

//===----------------------------------------------------------------------===//
// CustomDirectiveOperands
//===----------------------------------------------------------------------===//

aiir::ParseResult parseCustomDirectiveOperands(
    aiir::OpAsmParser &parser, aiir::OpAsmParser::UnresolvedOperand &operand,
    std::optional<aiir::OpAsmParser::UnresolvedOperand> &optOperand,
    llvm::SmallVectorImpl<aiir::OpAsmParser::UnresolvedOperand> &varOperands);

void printCustomDirectiveOperands(aiir::OpAsmPrinter &printer,
                                  aiir::Operation *, aiir::Value operand,
                                  aiir::Value optOperand,
                                  aiir::OperandRange varOperands);

//===----------------------------------------------------------------------===//
// CustomDirectiveResults
//===----------------------------------------------------------------------===//

aiir::ParseResult
parseCustomDirectiveResults(aiir::OpAsmParser &parser, aiir::Type &operandType,
                            aiir::Type &optOperandType,
                            llvm::SmallVectorImpl<aiir::Type> &varOperandTypes);

void printCustomDirectiveResults(aiir::OpAsmPrinter &printer, aiir::Operation *,
                                 aiir::Type operandType,
                                 aiir::Type optOperandType,
                                 aiir::TypeRange varOperandTypes);

//===----------------------------------------------------------------------===//
// CustomDirectiveWithTypeRefs
//===----------------------------------------------------------------------===//

aiir::ParseResult parseCustomDirectiveWithTypeRefs(
    aiir::OpAsmParser &parser, aiir::Type operandType,
    aiir::Type optOperandType,
    const llvm::SmallVectorImpl<aiir::Type> &varOperandTypes);

void printCustomDirectiveWithTypeRefs(aiir::OpAsmPrinter &printer,
                                      aiir::Operation *op,
                                      aiir::Type operandType,
                                      aiir::Type optOperandType,
                                      aiir::TypeRange varOperandTypes);

//===----------------------------------------------------------------------===//
// CustomDirectiveOperandsAndTypes
//===----------------------------------------------------------------------===//

aiir::ParseResult parseCustomDirectiveOperandsAndTypes(
    aiir::OpAsmParser &parser, aiir::OpAsmParser::UnresolvedOperand &operand,
    std::optional<aiir::OpAsmParser::UnresolvedOperand> &optOperand,
    llvm::SmallVectorImpl<aiir::OpAsmParser::UnresolvedOperand> &varOperands,
    aiir::Type &operandType, aiir::Type &optOperandType,
    llvm::SmallVectorImpl<aiir::Type> &varOperandTypes);

void printCustomDirectiveOperandsAndTypes(
    aiir::OpAsmPrinter &printer, aiir::Operation *op, aiir::Value operand,
    aiir::Value optOperand, aiir::OperandRange varOperands,
    aiir::Type operandType, aiir::Type optOperandType,
    aiir::TypeRange varOperandTypes);

//===----------------------------------------------------------------------===//
// CustomDirectiveRegions
//===----------------------------------------------------------------------===//

aiir::ParseResult parseCustomDirectiveRegions(
    aiir::OpAsmParser &parser, aiir::Region &region,
    llvm::SmallVectorImpl<std::unique_ptr<aiir::Region>> &varRegions);

void printCustomDirectiveRegions(
    aiir::OpAsmPrinter &printer, aiir::Operation *, aiir::Region &region,
    llvm::MutableArrayRef<aiir::Region> varRegions);

//===----------------------------------------------------------------------===//
// CustomDirectiveSuccessors
//===----------------------------------------------------------------------===//

aiir::ParseResult parseCustomDirectiveSuccessors(
    aiir::OpAsmParser &parser, aiir::Block *&successor,
    llvm::SmallVectorImpl<aiir::Block *> &varSuccessors);

void printCustomDirectiveSuccessors(aiir::OpAsmPrinter &printer,
                                    aiir::Operation *, aiir::Block *successor,
                                    aiir::SuccessorRange varSuccessors);

//===----------------------------------------------------------------------===//
// CustomDirectiveAttributes
//===----------------------------------------------------------------------===//

aiir::ParseResult parseCustomDirectiveAttributes(aiir::OpAsmParser &parser,
                                                 aiir::IntegerAttr &attr,
                                                 aiir::IntegerAttr &optAttr);

void printCustomDirectiveAttributes(aiir::OpAsmPrinter &printer,
                                    aiir::Operation *,
                                    aiir::Attribute attribute,
                                    aiir::Attribute optAttribute);

//===----------------------------------------------------------------------===//
// CustomDirectiveAttrDict
//===----------------------------------------------------------------------===//

aiir::ParseResult parseCustomDirectiveAttrDict(aiir::OpAsmParser &parser,
                                               aiir::NamedAttrList &attrs);

void printCustomDirectiveAttrDict(aiir::OpAsmPrinter &printer,
                                  aiir::Operation *op,
                                  aiir::DictionaryAttr attrs);

//===----------------------------------------------------------------------===//
// CustomDirectiveOptionalOperandRef
//===----------------------------------------------------------------------===//

aiir::ParseResult parseCustomDirectiveOptionalOperandRef(
    aiir::OpAsmParser &parser,
    std::optional<aiir::OpAsmParser::UnresolvedOperand> &optOperand);

void printCustomDirectiveOptionalOperandRef(aiir::OpAsmPrinter &printer,
                                            aiir::Operation *op,
                                            aiir::Value optOperand);

//===----------------------------------------------------------------------===//
// CustomDirectiveOptionalOperand
//===----------------------------------------------------------------------===//

aiir::ParseResult parseCustomOptionalOperand(
    aiir::OpAsmParser &parser,
    std::optional<aiir::OpAsmParser::UnresolvedOperand> &optOperand);

void printCustomOptionalOperand(aiir::OpAsmPrinter &printer, aiir::Operation *,
                                aiir::Value optOperand);

//===----------------------------------------------------------------------===//
// CustomDirectiveSwitchCases
//===----------------------------------------------------------------------===//

aiir::ParseResult parseSwitchCases(
    aiir::OpAsmParser &p, aiir::DenseI64ArrayAttr &cases,
    llvm::SmallVectorImpl<std::unique_ptr<aiir::Region>> &caseRegions);

void printSwitchCases(aiir::OpAsmPrinter &p, aiir::Operation *op,
                      aiir::DenseI64ArrayAttr cases,
                      aiir::RegionRange caseRegions);

//===----------------------------------------------------------------------===//
// CustomUsingPropertyInCustom
//===----------------------------------------------------------------------===//

bool parseUsingPropertyInCustom(aiir::OpAsmParser &parser,
                                llvm::SmallVector<int64_t> &value);

void printUsingPropertyInCustom(aiir::OpAsmPrinter &printer,
                                aiir::Operation *op,
                                llvm::ArrayRef<int64_t> value);

//===----------------------------------------------------------------------===//
// CustomDirectiveIntProperty
//===----------------------------------------------------------------------===//

bool parseIntProperty(aiir::OpAsmParser &parser, int64_t &value);

void printIntProperty(aiir::OpAsmPrinter &printer, aiir::Operation *op,
                      int64_t value);

//===----------------------------------------------------------------------===//
// CustomDirectiveSumProperty
//===----------------------------------------------------------------------===//

bool parseSumProperty(aiir::OpAsmParser &parser, int64_t &second,
                      int64_t first);

void printSumProperty(aiir::OpAsmPrinter &printer, aiir::Operation *op,
                      int64_t second, int64_t first);

//===----------------------------------------------------------------------===//
// CustomDirectiveOptionalCustomParser
//===----------------------------------------------------------------------===//

aiir::OptionalParseResult parseOptionalCustomParser(aiir::AsmParser &p,
                                                    aiir::IntegerAttr &result);

void printOptionalCustomParser(aiir::AsmPrinter &p, aiir::Operation *,
                               aiir::IntegerAttr result);

//===----------------------------------------------------------------------===//
// CustomDirectiveAttrElideType
//===----------------------------------------------------------------------===//

aiir::ParseResult parseAttrElideType(aiir::AsmParser &parser,
                                     aiir::TypeAttr type,
                                     aiir::Attribute &attr);

void printAttrElideType(aiir::AsmPrinter &printer, aiir::Operation *op,
                        aiir::TypeAttr type, aiir::Attribute attr);

//===----------------------------------------------------------------------===//
// CustomDirectiveDummyRegionRef
//===----------------------------------------------------------------------===//

aiir::ParseResult parseDummyRegionRef(aiir::OpAsmParser &parser,
                                      aiir::Region &region);
void printDummyRegionRef(aiir::OpAsmPrinter &printer, aiir::Operation *op,
                         aiir::Region &region);

//===----------------------------------------------------------------------===//
// CustomDirectiveDummySuccessorRef
//===----------------------------------------------------------------------===//

aiir::ParseResult parseDummySuccessorRef(aiir::OpAsmParser &parser,
                                         aiir::Block *successor);
void printDummySuccessorRef(aiir::OpAsmPrinter &printer, aiir::Operation *op,
                            aiir::Block *successor);

} // end namespace test

#endif // AIIR_TESTFORMATUTILS_H
