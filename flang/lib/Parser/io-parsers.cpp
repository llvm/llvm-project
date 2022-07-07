//===-- lib/Parser/io-parsers.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Per-type parsers for I/O statements and FORMAT

#include "basic-parsers.h"
#include "debug-parser.h"
#include "expr-parsers.h"
#include "misc-parsers.h"
#include "stmt-parser.h"
#include "token-parsers.h"
#include "type-parser-implementation.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/parse-tree.h"

namespace Fortran::parser {
// R1201 io-unit -> file-unit-number | * | internal-file-variable
// R1203 internal-file-variable -> char-variable
// R905 char-variable -> variable
// "char-variable" is attempted first since it's not type constrained but
// syntactically ambiguous with "file-unit-number", which is constrained.
TYPE_PARSER(construct<IoUnit>(variable / lookAhead(space / ",);\n"_ch)) ||
    construct<IoUnit>(fileUnitNumber) || construct<IoUnit>(star))

// R1202 file-unit-number -> scalar-int-expr
TYPE_PARSER(construct<FileUnitNumber>(scalarIntExpr / !"="_tok))

// R1204 open-stmt -> OPEN ( connect-spec-list )
TYPE_CONTEXT_PARSER("OPEN statement"_en_US,
    construct<OpenStmt>(
        "OPEN (" >> nonemptyList("expected connection specifications"_err_en_US,
                        Parser<ConnectSpec>{}) /
            ")"))

// R1206 file-name-expr -> scalar-default-char-expr
constexpr auto fileNameExpr{scalarDefaultCharExpr};

// R1205 connect-spec ->
//         [UNIT =] file-unit-number | ACCESS = scalar-default-char-expr |
//         ACTION = scalar-default-char-expr |
//         ASYNCHRONOUS = scalar-default-char-expr |
//         BLANK = scalar-default-char-expr |
//         DECIMAL = scalar-default-char-expr |
//         DELIM = scalar-default-char-expr |
//         ENCODING = scalar-default-char-expr | ERR = label |
//         FILE = file-name-expr | FORM = scalar-default-char-expr |
//         IOMSG = iomsg-variable | IOSTAT = scalar-int-variable |
//         NEWUNIT = scalar-int-variable | PAD = scalar-default-char-expr |
//         POSITION = scalar-default-char-expr | RECL = scalar-int-expr |
//         ROUND = scalar-default-char-expr | SIGN = scalar-default-char-expr |
//         STATUS = scalar-default-char-expr
//         @ | CARRIAGECONTROL = scalar-default-char-variable
//           | CONVERT = scalar-default-char-variable
//           | DISPOSE = scalar-default-char-variable
constexpr auto statusExpr{construct<StatusExpr>(scalarDefaultCharExpr)};
constexpr auto errLabel{construct<ErrLabel>(label)};

TYPE_PARSER(first(construct<ConnectSpec>(maybe("UNIT ="_tok) >> fileUnitNumber),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "ACCESS =" >> pure(ConnectSpec::CharExpr::Kind::Access),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "ACTION =" >> pure(ConnectSpec::CharExpr::Kind::Action),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "ASYNCHRONOUS =" >> pure(ConnectSpec::CharExpr::Kind::Asynchronous),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "BLANK =" >> pure(ConnectSpec::CharExpr::Kind::Blank),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "DECIMAL =" >> pure(ConnectSpec::CharExpr::Kind::Decimal),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "DELIM =" >> pure(ConnectSpec::CharExpr::Kind::Delim),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "ENCODING =" >> pure(ConnectSpec::CharExpr::Kind::Encoding),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>("ERR =" >> errLabel),
    construct<ConnectSpec>("FILE =" >> fileNameExpr),
    extension<LanguageFeature::FileName>(
        "nonstandard usage: NAME= in place of FILE="_port_en_US,
        construct<ConnectSpec>("NAME =" >> fileNameExpr)),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "FORM =" >> pure(ConnectSpec::CharExpr::Kind::Form),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>("IOMSG =" >> msgVariable),
    construct<ConnectSpec>("IOSTAT =" >> statVariable),
    construct<ConnectSpec>(construct<ConnectSpec::Newunit>(
        "NEWUNIT =" >> scalar(integer(variable)))),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "PAD =" >> pure(ConnectSpec::CharExpr::Kind::Pad),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "POSITION =" >> pure(ConnectSpec::CharExpr::Kind::Position),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>(
        construct<ConnectSpec::Recl>("RECL =" >> scalarIntExpr)),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "ROUND =" >> pure(ConnectSpec::CharExpr::Kind::Round),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
        "SIGN =" >> pure(ConnectSpec::CharExpr::Kind::Sign),
        scalarDefaultCharExpr)),
    construct<ConnectSpec>("STATUS =" >> statusExpr),
    extension<LanguageFeature::Carriagecontrol>(
        "nonstandard usage: CARRIAGECONTROL="_port_en_US,
        construct<ConnectSpec>(
            construct<ConnectSpec::CharExpr>("CARRIAGECONTROL =" >>
                    pure(ConnectSpec::CharExpr::Kind::Carriagecontrol),
                scalarDefaultCharExpr))),
    extension<LanguageFeature::Convert>(
        "nonstandard usage: CONVERT="_port_en_US,
        construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
            "CONVERT =" >> pure(ConnectSpec::CharExpr::Kind::Convert),
            scalarDefaultCharExpr))),
    extension<LanguageFeature::Dispose>(
        "nonstandard usage: DISPOSE="_port_en_US,
        construct<ConnectSpec>(construct<ConnectSpec::CharExpr>(
            "DISPOSE =" >> pure(ConnectSpec::CharExpr::Kind::Dispose),
            scalarDefaultCharExpr)))))

// R1209 close-spec ->
//         [UNIT =] file-unit-number | IOSTAT = scalar-int-variable |
//         IOMSG = iomsg-variable | ERR = label |
//         STATUS = scalar-default-char-expr
constexpr auto closeSpec{first(
    construct<CloseStmt::CloseSpec>(maybe("UNIT ="_tok) >> fileUnitNumber),
    construct<CloseStmt::CloseSpec>("IOSTAT =" >> statVariable),
    construct<CloseStmt::CloseSpec>("IOMSG =" >> msgVariable),
    construct<CloseStmt::CloseSpec>("ERR =" >> errLabel),
    construct<CloseStmt::CloseSpec>("STATUS =" >> statusExpr))};

// R1208 close-stmt -> CLOSE ( close-spec-list )
TYPE_CONTEXT_PARSER("CLOSE statement"_en_US,
    construct<CloseStmt>("CLOSE" >> parenthesized(nonemptyList(closeSpec))))

// R1210 read-stmt ->
//         READ ( io-control-spec-list ) [input-item-list] |
//         READ format [, input-item-list]
// The ambiguous READ(CVAR) is parsed as if CVAR were the unit.
// As Fortran doesn't have internal unformatted I/O, it should
// be parsed as if (CVAR) were a format; this is corrected by
// rewriting in semantics when we know that CVAR is character.
constexpr auto inputItemList{
    extension<LanguageFeature::IOListLeadingComma>(
        "nonstandard usage: leading comma in input item list"_port_en_US,
        some("," >> inputItem)) || // legacy extension: leading comma
    optionalList(inputItem)};

TYPE_CONTEXT_PARSER("READ statement"_en_US,
    construct<ReadStmt>("READ (" >>
            construct<std::optional<IoUnit>>(maybe("UNIT ="_tok) >> ioUnit),
        "," >> construct<std::optional<Format>>(format),
        defaulted("," >> nonemptyList(ioControlSpec)) / ")", inputItemList) ||
        construct<ReadStmt>(
            "READ (" >> construct<std::optional<IoUnit>>(ioUnit),
            construct<std::optional<Format>>(),
            defaulted("," >> nonemptyList(ioControlSpec)) / ")",
            inputItemList) ||
        construct<ReadStmt>("READ" >> construct<std::optional<IoUnit>>(),
            construct<std::optional<Format>>(),
            parenthesized(nonemptyList(ioControlSpec)), inputItemList) ||
        construct<ReadStmt>("READ" >> construct<std::optional<IoUnit>>(),
            construct<std::optional<Format>>(format),
            construct<std::list<IoControlSpec>>(), many("," >> inputItem)))

// R1214 id-variable -> scalar-int-variable
constexpr auto idVariable{construct<IdVariable>(scalarIntVariable)};

// R1213 io-control-spec ->
//         [UNIT =] io-unit | [FMT =] format | [NML =] namelist-group-name |
//         ADVANCE = scalar-default-char-expr |
//         ASYNCHRONOUS = scalar-default-char-constant-expr |
//         BLANK = scalar-default-char-expr |
//         DECIMAL = scalar-default-char-expr |
//         DELIM = scalar-default-char-expr | END = label | EOR = label |
//         ERR = label | ID = id-variable | IOMSG = iomsg-variable |
//         IOSTAT = scalar-int-variable | PAD = scalar-default-char-expr |
//         POS = scalar-int-expr | REC = scalar-int-expr |
//         ROUND = scalar-default-char-expr | SIGN = scalar-default-char-expr |
//         SIZE = scalar-int-variable
constexpr auto endLabel{construct<EndLabel>(label)};
constexpr auto eorLabel{construct<EorLabel>(label)};
TYPE_PARSER(first(construct<IoControlSpec>("UNIT =" >> ioUnit),
    construct<IoControlSpec>("FMT =" >> format),
    construct<IoControlSpec>("NML =" >> name),
    construct<IoControlSpec>(
        "ADVANCE =" >> construct<IoControlSpec::CharExpr>(
                           pure(IoControlSpec::CharExpr::Kind::Advance),
                           scalarDefaultCharExpr)),
    construct<IoControlSpec>(construct<IoControlSpec::Asynchronous>(
        "ASYNCHRONOUS =" >> scalarDefaultCharConstantExpr)),
    construct<IoControlSpec>("BLANK =" >>
        construct<IoControlSpec::CharExpr>(
            pure(IoControlSpec::CharExpr::Kind::Blank), scalarDefaultCharExpr)),
    construct<IoControlSpec>(
        "DECIMAL =" >> construct<IoControlSpec::CharExpr>(
                           pure(IoControlSpec::CharExpr::Kind::Decimal),
                           scalarDefaultCharExpr)),
    construct<IoControlSpec>("DELIM =" >>
        construct<IoControlSpec::CharExpr>(
            pure(IoControlSpec::CharExpr::Kind::Delim), scalarDefaultCharExpr)),
    construct<IoControlSpec>("END =" >> endLabel),
    construct<IoControlSpec>("EOR =" >> eorLabel),
    construct<IoControlSpec>("ERR =" >> errLabel),
    construct<IoControlSpec>("ID =" >> idVariable),
    construct<IoControlSpec>("IOMSG = " >> msgVariable),
    construct<IoControlSpec>("IOSTAT = " >> statVariable),
    construct<IoControlSpec>("PAD =" >>
        construct<IoControlSpec::CharExpr>(
            pure(IoControlSpec::CharExpr::Kind::Pad), scalarDefaultCharExpr)),
    construct<IoControlSpec>(
        "POS =" >> construct<IoControlSpec::Pos>(scalarIntExpr)),
    construct<IoControlSpec>(
        "REC =" >> construct<IoControlSpec::Rec>(scalarIntExpr)),
    construct<IoControlSpec>("ROUND =" >>
        construct<IoControlSpec::CharExpr>(
            pure(IoControlSpec::CharExpr::Kind::Round), scalarDefaultCharExpr)),
    construct<IoControlSpec>("SIGN =" >>
        construct<IoControlSpec::CharExpr>(
            pure(IoControlSpec::CharExpr::Kind::Sign), scalarDefaultCharExpr)),
    construct<IoControlSpec>(
        "SIZE =" >> construct<IoControlSpec::Size>(scalarIntVariable))))

// R1211 write-stmt -> WRITE ( io-control-spec-list ) [output-item-list]
constexpr auto outputItemList{
    extension<LanguageFeature::IOListLeadingComma>(
        "nonstandard usage: leading comma in output item list"_port_en_US,
        some("," >> outputItem)) || // legacy: allow leading comma
    optionalList(outputItem)};

TYPE_CONTEXT_PARSER("WRITE statement"_en_US,
    construct<WriteStmt>("WRITE (" >>
            construct<std::optional<IoUnit>>(maybe("UNIT ="_tok) >> ioUnit),
        "," >> construct<std::optional<Format>>(format),
        defaulted("," >> nonemptyList(ioControlSpec)) / ")", outputItemList) ||
        construct<WriteStmt>(
            "WRITE (" >> construct<std::optional<IoUnit>>(ioUnit),
            construct<std::optional<Format>>(),
            defaulted("," >> nonemptyList(ioControlSpec)) / ")",
            outputItemList) ||
        construct<WriteStmt>("WRITE" >> construct<std::optional<IoUnit>>(),
            construct<std::optional<Format>>(),
            parenthesized(nonemptyList(ioControlSpec)), outputItemList))

// R1212 print-stmt PRINT format [, output-item-list]
TYPE_CONTEXT_PARSER("PRINT statement"_en_US,
    construct<PrintStmt>(
        "PRINT" >> format, defaulted("," >> nonemptyList(outputItem))))

// R1215 format -> default-char-expr | label | *
// deprecated(ASSIGN): | scalar-int-name
TYPE_PARSER(construct<Format>(label / !"_."_ch) ||
    construct<Format>(expr / !"="_tok) || construct<Format>(star))

// R1216 input-item -> variable | io-implied-do
TYPE_PARSER(construct<InputItem>(variable) ||
    construct<InputItem>(indirect(inputImpliedDo)))

// R1217 output-item -> expr | io-implied-do
TYPE_PARSER(construct<OutputItem>(expr) ||
    construct<OutputItem>(indirect(outputImpliedDo)))

// R1220 io-implied-do-control ->
//         do-variable = scalar-int-expr , scalar-int-expr [, scalar-int-expr]
constexpr auto ioImpliedDoControl{loopBounds(scalarIntExpr)};

// R1218 io-implied-do -> ( io-implied-do-object-list , io-implied-do-control )
// R1219 io-implied-do-object -> input-item | output-item
TYPE_CONTEXT_PARSER("input implied DO"_en_US,
    parenthesized(
        construct<InputImpliedDo>(nonemptyList(inputItem / lookAhead(","_tok)),
            "," >> ioImpliedDoControl)))
TYPE_CONTEXT_PARSER("output implied DO"_en_US,
    parenthesized(construct<OutputImpliedDo>(
        nonemptyList(outputItem / lookAhead(","_tok)),
        "," >> ioImpliedDoControl)))

// R1222 wait-stmt -> WAIT ( wait-spec-list )
TYPE_CONTEXT_PARSER("WAIT statement"_en_US,
    "WAIT" >>
        parenthesized(construct<WaitStmt>(nonemptyList(Parser<WaitSpec>{}))))

// R1223 wait-spec ->
//         [UNIT =] file-unit-number | END = label | EOR = label | ERR = label |
//         ID = scalar-int-expr | IOMSG = iomsg-variable |
//         IOSTAT = scalar-int-variable
constexpr auto idExpr{construct<IdExpr>(scalarIntExpr)};

TYPE_PARSER(first(construct<WaitSpec>(maybe("UNIT ="_tok) >> fileUnitNumber),
    construct<WaitSpec>("END =" >> endLabel),
    construct<WaitSpec>("EOR =" >> eorLabel),
    construct<WaitSpec>("ERR =" >> errLabel),
    construct<WaitSpec>("ID =" >> idExpr),
    construct<WaitSpec>("IOMSG =" >> msgVariable),
    construct<WaitSpec>("IOSTAT =" >> statVariable)))

template <typename A> common::IfNoLvalue<std::list<A>, A> singletonList(A &&x) {
  std::list<A> result;
  result.push_front(std::move(x));
  return result;
}
constexpr auto bareUnitNumberAsList{
    applyFunction(singletonList<PositionOrFlushSpec>,
        construct<PositionOrFlushSpec>(fileUnitNumber))};
constexpr auto positionOrFlushSpecList{
    parenthesized(nonemptyList(positionOrFlushSpec)) || bareUnitNumberAsList};

// R1224 backspace-stmt ->
//         BACKSPACE file-unit-number | BACKSPACE ( position-spec-list )
TYPE_CONTEXT_PARSER("BACKSPACE statement"_en_US,
    construct<BackspaceStmt>("BACKSPACE" >> positionOrFlushSpecList))

// R1225 endfile-stmt ->
//         ENDFILE file-unit-number | ENDFILE ( position-spec-list )
TYPE_CONTEXT_PARSER("ENDFILE statement"_en_US,
    construct<EndfileStmt>("END FILE" >> positionOrFlushSpecList))

// R1226 rewind-stmt -> REWIND file-unit-number | REWIND ( position-spec-list )
TYPE_CONTEXT_PARSER("REWIND statement"_en_US,
    construct<RewindStmt>("REWIND" >> positionOrFlushSpecList))

// R1227 position-spec ->
//         [UNIT =] file-unit-number | IOMSG = iomsg-variable |
//         IOSTAT = scalar-int-variable | ERR = label
// R1229 flush-spec ->
//         [UNIT =] file-unit-number | IOSTAT = scalar-int-variable |
//         IOMSG = iomsg-variable | ERR = label
TYPE_PARSER(
    construct<PositionOrFlushSpec>(maybe("UNIT ="_tok) >> fileUnitNumber) ||
    construct<PositionOrFlushSpec>("IOMSG =" >> msgVariable) ||
    construct<PositionOrFlushSpec>("IOSTAT =" >> statVariable) ||
    construct<PositionOrFlushSpec>("ERR =" >> errLabel))

// R1228 flush-stmt -> FLUSH file-unit-number | FLUSH ( flush-spec-list )
TYPE_CONTEXT_PARSER("FLUSH statement"_en_US,
    construct<FlushStmt>("FLUSH" >> positionOrFlushSpecList))

// R1231 inquire-spec ->
//         [UNIT =] file-unit-number | FILE = file-name-expr |
//         ACCESS = scalar-default-char-variable |
//         ACTION = scalar-default-char-variable |
//         ASYNCHRONOUS = scalar-default-char-variable |
//         BLANK = scalar-default-char-variable |
//         DECIMAL = scalar-default-char-variable |
//         DELIM = scalar-default-char-variable |
//         ENCODING = scalar-default-char-variable |
//         ERR = label | EXIST = scalar-logical-variable |
//         FORM = scalar-default-char-variable |
//         FORMATTED = scalar-default-char-variable |
//         ID = scalar-int-expr | IOMSG = iomsg-variable |
//         IOSTAT = scalar-int-variable |
//         NAME = scalar-default-char-variable |
//         NAMED = scalar-logical-variable |
//         NEXTREC = scalar-int-variable | NUMBER = scalar-int-variable |
//         OPENED = scalar-logical-variable |
//         PAD = scalar-default-char-variable |
//         PENDING = scalar-logical-variable | POS = scalar-int-variable |
//         POSITION = scalar-default-char-variable |
//         READ = scalar-default-char-variable |
//         READWRITE = scalar-default-char-variable |
//         RECL = scalar-int-variable | ROUND = scalar-default-char-variable |
//         SEQUENTIAL = scalar-default-char-variable |
//         SIGN = scalar-default-char-variable |
//         SIZE = scalar-int-variable |
//         STREAM = scalar-default-char-variable |
//         STATUS = scalar-default-char-variable |
//         WRITE = scalar-default-char-variable
//         @ | CARRIAGECONTROL = scalar-default-char-variable
//           | CONVERT = scalar-default-char-variable
//           | DISPOSE = scalar-default-char-variable
TYPE_PARSER(first(construct<InquireSpec>(maybe("UNIT ="_tok) >> fileUnitNumber),
    construct<InquireSpec>("FILE =" >> fileNameExpr),
    construct<InquireSpec>(
        "ACCESS =" >> construct<InquireSpec::CharVar>(
                          pure(InquireSpec::CharVar::Kind::Access),
                          scalarDefaultCharVariable)),
    construct<InquireSpec>(
        "ACTION =" >> construct<InquireSpec::CharVar>(
                          pure(InquireSpec::CharVar::Kind::Action),
                          scalarDefaultCharVariable)),
    construct<InquireSpec>(
        "ASYNCHRONOUS =" >> construct<InquireSpec::CharVar>(
                                pure(InquireSpec::CharVar::Kind::Asynchronous),
                                scalarDefaultCharVariable)),
    construct<InquireSpec>("BLANK =" >>
        construct<InquireSpec::CharVar>(pure(InquireSpec::CharVar::Kind::Blank),
            scalarDefaultCharVariable)),
    construct<InquireSpec>(
        "DECIMAL =" >> construct<InquireSpec::CharVar>(
                           pure(InquireSpec::CharVar::Kind::Decimal),
                           scalarDefaultCharVariable)),
    construct<InquireSpec>("DELIM =" >>
        construct<InquireSpec::CharVar>(pure(InquireSpec::CharVar::Kind::Delim),
            scalarDefaultCharVariable)),
    construct<InquireSpec>(
        "DIRECT =" >> construct<InquireSpec::CharVar>(
                          pure(InquireSpec::CharVar::Kind::Direct),
                          scalarDefaultCharVariable)),
    construct<InquireSpec>(
        "ENCODING =" >> construct<InquireSpec::CharVar>(
                            pure(InquireSpec::CharVar::Kind::Encoding),
                            scalarDefaultCharVariable)),
    construct<InquireSpec>("ERR =" >> errLabel),
    construct<InquireSpec>("EXIST =" >>
        construct<InquireSpec::LogVar>(
            pure(InquireSpec::LogVar::Kind::Exist), scalarLogicalVariable)),
    construct<InquireSpec>("FORM =" >>
        construct<InquireSpec::CharVar>(
            pure(InquireSpec::CharVar::Kind::Form), scalarDefaultCharVariable)),
    construct<InquireSpec>(
        "FORMATTED =" >> construct<InquireSpec::CharVar>(
                             pure(InquireSpec::CharVar::Kind::Formatted),
                             scalarDefaultCharVariable)),
    construct<InquireSpec>("ID =" >> idExpr),
    construct<InquireSpec>("IOMSG =" >>
        construct<InquireSpec::CharVar>(pure(InquireSpec::CharVar::Kind::Iomsg),
            scalarDefaultCharVariable)),
    construct<InquireSpec>("IOSTAT =" >>
        construct<InquireSpec::IntVar>(pure(InquireSpec::IntVar::Kind::Iostat),
            scalar(integer(variable)))),
    construct<InquireSpec>("NAME =" >>
        construct<InquireSpec::CharVar>(
            pure(InquireSpec::CharVar::Kind::Name), scalarDefaultCharVariable)),
    construct<InquireSpec>("NAMED =" >>
        construct<InquireSpec::LogVar>(
            pure(InquireSpec::LogVar::Kind::Named), scalarLogicalVariable)),
    construct<InquireSpec>("NEXTREC =" >>
        construct<InquireSpec::IntVar>(pure(InquireSpec::IntVar::Kind::Nextrec),
            scalar(integer(variable)))),
    construct<InquireSpec>("NUMBER =" >>
        construct<InquireSpec::IntVar>(pure(InquireSpec::IntVar::Kind::Number),
            scalar(integer(variable)))),
    construct<InquireSpec>("OPENED =" >>
        construct<InquireSpec::LogVar>(
            pure(InquireSpec::LogVar::Kind::Opened), scalarLogicalVariable)),
    construct<InquireSpec>("PAD =" >>
        construct<InquireSpec::CharVar>(
            pure(InquireSpec::CharVar::Kind::Pad), scalarDefaultCharVariable)),
    construct<InquireSpec>("PENDING =" >>
        construct<InquireSpec::LogVar>(
            pure(InquireSpec::LogVar::Kind::Pending), scalarLogicalVariable)),
    construct<InquireSpec>("POS =" >>
        construct<InquireSpec::IntVar>(
            pure(InquireSpec::IntVar::Kind::Pos), scalar(integer(variable)))),
    construct<InquireSpec>(
        "POSITION =" >> construct<InquireSpec::CharVar>(
                            pure(InquireSpec::CharVar::Kind::Position),
                            scalarDefaultCharVariable)),
    construct<InquireSpec>("READ =" >>
        construct<InquireSpec::CharVar>(
            pure(InquireSpec::CharVar::Kind::Read), scalarDefaultCharVariable)),
    construct<InquireSpec>(
        "READWRITE =" >> construct<InquireSpec::CharVar>(
                             pure(InquireSpec::CharVar::Kind::Readwrite),
                             scalarDefaultCharVariable)),
    construct<InquireSpec>("RECL =" >>
        construct<InquireSpec::IntVar>(
            pure(InquireSpec::IntVar::Kind::Recl), scalar(integer(variable)))),
    construct<InquireSpec>("ROUND =" >>
        construct<InquireSpec::CharVar>(pure(InquireSpec::CharVar::Kind::Round),
            scalarDefaultCharVariable)),
    construct<InquireSpec>(
        "SEQUENTIAL =" >> construct<InquireSpec::CharVar>(
                              pure(InquireSpec::CharVar::Kind::Sequential),
                              scalarDefaultCharVariable)),
    construct<InquireSpec>("SIGN =" >>
        construct<InquireSpec::CharVar>(
            pure(InquireSpec::CharVar::Kind::Sign), scalarDefaultCharVariable)),
    construct<InquireSpec>("SIZE =" >>
        construct<InquireSpec::IntVar>(
            pure(InquireSpec::IntVar::Kind::Size), scalar(integer(variable)))),
    construct<InquireSpec>(
        "STREAM =" >> construct<InquireSpec::CharVar>(
                          pure(InquireSpec::CharVar::Kind::Stream),
                          scalarDefaultCharVariable)),
    construct<InquireSpec>(
        "STATUS =" >> construct<InquireSpec::CharVar>(
                          pure(InquireSpec::CharVar::Kind::Status),
                          scalarDefaultCharVariable)),
    construct<InquireSpec>(
        "UNFORMATTED =" >> construct<InquireSpec::CharVar>(
                               pure(InquireSpec::CharVar::Kind::Unformatted),
                               scalarDefaultCharVariable)),
    construct<InquireSpec>("WRITE =" >>
        construct<InquireSpec::CharVar>(pure(InquireSpec::CharVar::Kind::Write),
            scalarDefaultCharVariable)),
    extension<LanguageFeature::Carriagecontrol>(
        "nonstandard usage: CARRIAGECONTROL="_port_en_US,
        construct<InquireSpec>("CARRIAGECONTROL =" >>
            construct<InquireSpec::CharVar>(
                pure(InquireSpec::CharVar::Kind::Carriagecontrol),
                scalarDefaultCharVariable))),
    extension<LanguageFeature::Convert>(
        "nonstandard usage: CONVERT="_port_en_US,
        construct<InquireSpec>(
            "CONVERT =" >> construct<InquireSpec::CharVar>(
                               pure(InquireSpec::CharVar::Kind::Convert),
                               scalarDefaultCharVariable))),
    extension<LanguageFeature::Dispose>(
        "nonstandard usage: DISPOSE="_port_en_US,
        construct<InquireSpec>(
            "DISPOSE =" >> construct<InquireSpec::CharVar>(
                               pure(InquireSpec::CharVar::Kind::Dispose),
                               scalarDefaultCharVariable)))))

// R1230 inquire-stmt ->
//         INQUIRE ( inquire-spec-list ) |
//         INQUIRE ( IOLENGTH = scalar-int-variable ) output-item-list
TYPE_CONTEXT_PARSER("INQUIRE statement"_en_US,
    "INQUIRE" >>
        (construct<InquireStmt>(
             parenthesized(nonemptyList(Parser<InquireSpec>{}))) ||
            construct<InquireStmt>(construct<InquireStmt::Iolength>(
                parenthesized("IOLENGTH =" >> scalar(integer(variable))),
                nonemptyList(outputItem)))))

// R1301 format-stmt -> FORMAT format-specification
// 13.2.1 allows spaces to appear "at any point" within a format specification
// without effect, except of course within a character string edit descriptor.
TYPE_CONTEXT_PARSER("FORMAT statement"_en_US,
    construct<FormatStmt>("FORMAT" >> Parser<format::FormatSpecification>{}))

// R1321 char-string-edit-desc
// N.B. C1313 disallows any kind parameter on the character literal.
constexpr auto charStringEditDesc{
    space >> (charLiteralConstantWithoutKind || rawHollerithLiteral)};

// R1303 format-items -> format-item [[,] format-item]...
constexpr auto formatItems{
    nonemptySeparated(space >> Parser<format::FormatItem>{}, maybe(","_tok))};

// R1306 r -> digit-string
constexpr DigitStringIgnoreSpaces repeat;

// R1304 format-item ->
//         [r] data-edit-desc | control-edit-desc | char-string-edit-desc |
//         [r] ( format-items )
TYPE_PARSER(construct<format::FormatItem>(
                maybe(repeat), Parser<format::IntrinsicTypeDataEditDesc>{}) ||
    construct<format::FormatItem>(
        maybe(repeat), Parser<format::DerivedTypeDataEditDesc>{}) ||
    construct<format::FormatItem>(Parser<format::ControlEditDesc>{}) ||
    construct<format::FormatItem>(charStringEditDesc) ||
    construct<format::FormatItem>(maybe(repeat), parenthesized(formatItems)))

// R1302 format-specification ->
//         ( [format-items] ) | ( [format-items ,] unlimited-format-item )
// R1305 unlimited-format-item -> * ( format-items )
// minor extension: the comma is optional before the unlimited-format-item
TYPE_PARSER(parenthesized(construct<format::FormatSpecification>(
                              defaulted(formatItems / maybe(","_tok)),
                              "*" >> parenthesized(formatItems)) ||
    construct<format::FormatSpecification>(defaulted(formatItems))))
// R1308 w -> digit-string
// R1309 m -> digit-string
// R1310 d -> digit-string
// R1311 e -> digit-string
constexpr auto width{repeat};
constexpr auto mandatoryWidth{construct<std::optional<int>>(width)};
constexpr auto digits{repeat};
constexpr auto noInt{construct<std::optional<int>>()};
constexpr auto mandatoryDigits{construct<std::optional<int>>("." >> width)};

// The extra trailing spaces in the following quoted edit descriptor token
// parsers are intentional: they inhibit any spurious warnings about missing
// spaces in pedantic mode that would otherwise be emitted if the edit
// descriptor were followed by a character that could appear in an identifier.

// R1307 data-edit-desc ->
//         I w [. m] | B w [. m] | O w [. m] | Z w [. m] | F w . d |
//         E w . d [E e] | EN w . d [E e] | ES w . d [E e] | EX w . d [E e] |
//         G w [. d [E e]] | L w | A [w] | D w . d |
//         DT [char-literal-constant] [( v-list )]
// (part 1 of 2)
TYPE_PARSER(construct<format::IntrinsicTypeDataEditDesc>(
                "I " >> pure(format::IntrinsicTypeDataEditDesc::Kind::I) ||
                    "B " >> pure(format::IntrinsicTypeDataEditDesc::Kind::B) ||
                    "O " >> pure(format::IntrinsicTypeDataEditDesc::Kind::O) ||
                    "Z " >> pure(format::IntrinsicTypeDataEditDesc::Kind::Z),
                mandatoryWidth, maybe("." >> digits), noInt) ||
    construct<format::IntrinsicTypeDataEditDesc>(
        "F " >> pure(format::IntrinsicTypeDataEditDesc::Kind::F) ||
            "D " >> pure(format::IntrinsicTypeDataEditDesc::Kind::D),
        mandatoryWidth, mandatoryDigits, noInt) ||
    construct<format::IntrinsicTypeDataEditDesc>(
        "E " >> ("N " >> pure(format::IntrinsicTypeDataEditDesc::Kind::EN) ||
                    "S " >> pure(format::IntrinsicTypeDataEditDesc::Kind::ES) ||
                    "X " >> pure(format::IntrinsicTypeDataEditDesc::Kind::EX) ||
                    pure(format::IntrinsicTypeDataEditDesc::Kind::E)),
        mandatoryWidth, mandatoryDigits, maybe("E " >> digits)) ||
    construct<format::IntrinsicTypeDataEditDesc>(
        "G " >> pure(format::IntrinsicTypeDataEditDesc::Kind::G),
        mandatoryWidth, mandatoryDigits, maybe("E " >> digits)) ||
    construct<format::IntrinsicTypeDataEditDesc>(
        "G " >> pure(format::IntrinsicTypeDataEditDesc::Kind::G) ||
            "L " >> pure(format::IntrinsicTypeDataEditDesc::Kind::L),
        mandatoryWidth, noInt, noInt) ||
    construct<format::IntrinsicTypeDataEditDesc>(
        "A " >> pure(format::IntrinsicTypeDataEditDesc::Kind::A), maybe(width),
        noInt, noInt) ||
    // PGI/Intel extension: omitting width (and all else that follows)
    extension<LanguageFeature::AbbreviatedEditDescriptor>(
        "nonstandard usage: abbreviated edit descriptor"_port_en_US,
        construct<format::IntrinsicTypeDataEditDesc>(
            "I " >> pure(format::IntrinsicTypeDataEditDesc::Kind::I) ||
                ("B "_tok / !letter /* don't occlude BN & BZ */) >>
                    pure(format::IntrinsicTypeDataEditDesc::Kind::B) ||
                "O " >> pure(format::IntrinsicTypeDataEditDesc::Kind::O) ||
                "Z " >> pure(format::IntrinsicTypeDataEditDesc::Kind::Z) ||
                "F " >> pure(format::IntrinsicTypeDataEditDesc::Kind::F) ||
                ("D "_tok / !letter /* don't occlude DT, DC, & DP */) >>
                    pure(format::IntrinsicTypeDataEditDesc::Kind::D) ||
                "E " >>
                    ("N " >>
                            pure(format::IntrinsicTypeDataEditDesc::Kind::EN) ||
                        "S " >>
                            pure(format::IntrinsicTypeDataEditDesc::Kind::ES) ||
                        "X " >>
                            pure(format::IntrinsicTypeDataEditDesc::Kind::EX) ||
                        pure(format::IntrinsicTypeDataEditDesc::Kind::E)) ||
                "G " >> pure(format::IntrinsicTypeDataEditDesc::Kind::G) ||
                "L " >> pure(format::IntrinsicTypeDataEditDesc::Kind::L),
            noInt, noInt, noInt)))

// R1307 data-edit-desc (part 2 of 2)
// R1312 v -> [sign] digit-string
constexpr SignedDigitStringIgnoreSpaces scaleFactor;
TYPE_PARSER(construct<format::DerivedTypeDataEditDesc>(
    "D T" >> defaulted(charLiteralConstantWithoutKind),
    defaulted(parenthesized(nonemptyList(scaleFactor)))))

// R1314 k -> [sign] digit-string
constexpr PositiveDigitStringIgnoreSpaces count;

// R1313 control-edit-desc ->
//         position-edit-desc | [r] / | : | sign-edit-desc | k P |
//         blank-interp-edit-desc | round-edit-desc | decimal-edit-desc |
//         @ \ | $
// R1315 position-edit-desc -> T n | TL n | TR n | n X
// R1316 n -> digit-string
// R1317 sign-edit-desc -> SS | SP | S
// R1318 blank-interp-edit-desc -> BN | BZ
// R1319 round-edit-desc -> RU | RD | RZ | RN | RC | RP
// R1320 decimal-edit-desc -> DC | DP
TYPE_PARSER(construct<format::ControlEditDesc>(
                "T L " >> pure(format::ControlEditDesc::Kind::TL) ||
                    "T R " >> pure(format::ControlEditDesc::Kind::TR) ||
                    "T " >> pure(format::ControlEditDesc::Kind::T),
                count) ||
    construct<format::ControlEditDesc>(count,
        "X " >> pure(format::ControlEditDesc::Kind::X) ||
            "/" >> pure(format::ControlEditDesc::Kind::Slash)) ||
    construct<format::ControlEditDesc>(
        "X " >> pure(format::ControlEditDesc::Kind::X) ||
        "/" >> pure(format::ControlEditDesc::Kind::Slash)) ||
    construct<format::ControlEditDesc>(
        scaleFactor, "P " >> pure(format::ControlEditDesc::Kind::P)) ||
    construct<format::ControlEditDesc>(
        ":" >> pure(format::ControlEditDesc::Kind::Colon)) ||
    "S " >> ("S " >> construct<format::ControlEditDesc>(
                         pure(format::ControlEditDesc::Kind::SS)) ||
                "P " >> construct<format::ControlEditDesc>(
                            pure(format::ControlEditDesc::Kind::SP)) ||
                construct<format::ControlEditDesc>(
                    pure(format::ControlEditDesc::Kind::S))) ||
    "B " >> ("N " >> construct<format::ControlEditDesc>(
                         pure(format::ControlEditDesc::Kind::BN)) ||
                "Z " >> construct<format::ControlEditDesc>(
                            pure(format::ControlEditDesc::Kind::BZ))) ||
    "R " >> ("U " >> construct<format::ControlEditDesc>(
                         pure(format::ControlEditDesc::Kind::RU)) ||
                "D " >> construct<format::ControlEditDesc>(
                            pure(format::ControlEditDesc::Kind::RD)) ||
                "Z " >> construct<format::ControlEditDesc>(
                            pure(format::ControlEditDesc::Kind::RZ)) ||
                "N " >> construct<format::ControlEditDesc>(
                            pure(format::ControlEditDesc::Kind::RN)) ||
                "C " >> construct<format::ControlEditDesc>(
                            pure(format::ControlEditDesc::Kind::RC)) ||
                "P " >> construct<format::ControlEditDesc>(
                            pure(format::ControlEditDesc::Kind::RP))) ||
    "D " >> ("C " >> construct<format::ControlEditDesc>(
                         pure(format::ControlEditDesc::Kind::DC)) ||
                "P " >> construct<format::ControlEditDesc>(
                            pure(format::ControlEditDesc::Kind::DP))) ||
    extension<LanguageFeature::AdditionalFormats>(
        "nonstandard usage: $ and \\ control edit descriptors"_port_en_US,
        "$" >> construct<format::ControlEditDesc>(
                   pure(format::ControlEditDesc::Kind::Dollar)) ||
            "\\" >> construct<format::ControlEditDesc>(
                        pure(format::ControlEditDesc::Kind::Backslash))))
} // namespace Fortran::parser
