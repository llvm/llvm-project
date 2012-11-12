//===---- ParserPragmas.h - Language specific pragmas -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines #pragma handlers for language specific pragmas.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_PARSEPRAGMA_H
#define LLVM_CLANG_PARSE_PARSEPRAGMA_H

#include "clang/Lex/Pragma.h"

namespace clang {
  class Sema;
  class Parser;

class PragmaAlignHandler : public PragmaHandler {
public:
  explicit PragmaAlignHandler() : PragmaHandler("align") {}

  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &FirstToken);
};

class PragmaGCCVisibilityHandler : public PragmaHandler {
public:
  explicit PragmaGCCVisibilityHandler() : PragmaHandler("visibility") {}

  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &FirstToken);
};

class PragmaOptionsHandler : public PragmaHandler {
public:
  explicit PragmaOptionsHandler() : PragmaHandler("options") {}

  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &FirstToken);
};

class PragmaPackHandler : public PragmaHandler {
public:
  explicit PragmaPackHandler() : PragmaHandler("pack") {}

  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &FirstToken);
};
  
class PragmaMSStructHandler : public PragmaHandler {
public:
  explicit PragmaMSStructHandler() : PragmaHandler("ms_struct") {}
    
  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &FirstToken);
};

class PragmaUnusedHandler : public PragmaHandler {
public:
  PragmaUnusedHandler() : PragmaHandler("unused") {}

  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &FirstToken);
};

class PragmaWeakHandler : public PragmaHandler {
public:
  explicit PragmaWeakHandler() : PragmaHandler("weak") {}

  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &FirstToken);
};

class PragmaRedefineExtnameHandler : public PragmaHandler {
public:
  explicit PragmaRedefineExtnameHandler() : PragmaHandler("redefine_extname") {}

  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &FirstToken);
};

class PragmaOpenCLExtensionHandler : public PragmaHandler {
public:
  PragmaOpenCLExtensionHandler() : PragmaHandler("EXTENSION") {}
  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &FirstToken);
};
  

class PragmaFPContractHandler : public PragmaHandler {
public:
  PragmaFPContractHandler() : PragmaHandler("FP_CONTRACT") {}
  virtual void HandlePragma(Preprocessor &PP, PragmaIntroducerKind Introducer,
                            Token &FirstToken);
};
  

}  // end namespace clang

#endif
