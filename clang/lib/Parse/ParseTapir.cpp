//===--- ParseTapir.cpp - Tapir Parsing -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the Tapir portions of the Parser interface.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/RAIIObjectsForParser.h"
#include "clang/Parse/Parser.h"

using namespace clang;


/// ParseSyncStatement
///       sync-statement:
///         'sync' identifier ';'
StmtResult Parser::ParseSyncStatement() {
  assert(Tok.is(tok::kw__tapir_sync) && "Not a sync stmt!");
  SourceLocation SyncLoc = ConsumeToken(); // eat the '_tapir_sync'
  assert(Tok.is(tok::identifier) && Tok.getIdentifierInfo() &&
         "Not an identifier!");
  Token IdentTok = Tok; 
  ConsumeToken(); 
  return Actions.ActOnSyncStmt(SyncLoc, IdentTok.getIdentifierInfo()->getName());
}

/// ParseSpawnStatement
///       spawn-statement:
///         'spawn' identifier statement
StmtResult Parser::ParseSpawnStatement() {
  assert(Tok.is(tok::kw__tapir_spawn) && "Not a spawn stmt!");
  SourceLocation SpawnLoc = ConsumeToken();  // eat the '_tapir_spawn'.

  assert(Tok.is(tok::identifier) && Tok.getIdentifierInfo() &&
         "Not an identifier!");
  Token IdentTok = Tok; 
  ConsumeToken();

  // Parse statement of spawned child
  StmtResult SubStmt = ParseStatement();
  if (SubStmt.isInvalid()) {
    SkipUntil(tok::semi);
    return StmtError();
  }
  return Actions.ActOnSpawnStmt(SpawnLoc, IdentTok.getIdentifierInfo()->getName(), SubStmt.get());
}

