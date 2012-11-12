class Command<string name> {
  string Name = name;
  string EndCommandName = "";

  int NumArgs = 0;

  bit IsInlineCommand = 0;

  bit IsBlockCommand = 0;
  bit IsBriefCommand = 0;
  bit IsReturnsCommand = 0;
  bit IsParamCommand = 0;
  bit IsTParamCommand = 0;
  bit IsDeprecatedCommand = 0;

  bit IsEmptyParagraphAllowed = 0;

  bit IsVerbatimBlockCommand = 0;
  bit IsVerbatimBlockEndCommand = 0;
  bit IsVerbatimLineCommand = 0;
  bit IsDeclarationCommand = 0;
}

class InlineCommand<string name> : Command<name> {
  let IsInlineCommand = 1;
}

class BlockCommand<string name> : Command<name> {
  let IsBlockCommand = 1;
}

class VerbatimBlockCommand<string name> : Command<name> {
  let EndCommandName = name;
  let IsVerbatimBlockCommand = 1;
}

multiclass VerbatimBlockCommand<string name, string endCommandName> {
  def Begin : Command<name> {
    let EndCommandName = endCommandName;
    let IsVerbatimBlockCommand = 1;
  }

  def End : Command<endCommandName> {
    let IsVerbatimBlockEndCommand = 1;
  }
}

class VerbatimLineCommand<string name> : Command<name> {
  let IsVerbatimLineCommand = 1;
}

class DeclarationVerbatimLineCommand<string name> :
      VerbatimLineCommand<name> {
  let IsDeclarationCommand = 1;
}

def B  : InlineCommand<"b">;
def C  : InlineCommand<"c">;
def P  : InlineCommand<"p">;
def A  : InlineCommand<"a">;
def E  : InlineCommand<"e">;
def Em : InlineCommand<"em">;

def Brief : BlockCommand<"brief"> { let IsBriefCommand = 1; }
def Short : BlockCommand<"short"> { let IsBriefCommand = 1; }

def Returns : BlockCommand<"returns"> { let IsReturnsCommand = 1; }
def Return  : BlockCommand<"return"> { let IsReturnsCommand = 1; }
def Result  : BlockCommand<"result"> { let IsReturnsCommand = 1; }

def Param : BlockCommand<"param"> { let IsParamCommand = 1; }

// Doxygen
def Tparam : BlockCommand<"tparam"> { let IsTParamCommand = 1; }

// HeaderDoc
def Templatefield : BlockCommand<"templatefield"> { let IsTParamCommand = 1; }

def Deprecated : BlockCommand<"deprecated"> {
  let IsEmptyParagraphAllowed = 1;
  let IsDeprecatedCommand = 1;
}

def Author     : BlockCommand<"author">;
def Authors    : BlockCommand<"authors">;
def Bug        : BlockCommand<"bug">;
def Copyright  : BlockCommand<"copyright">;
def Date       : BlockCommand<"date">;
def Details    : BlockCommand<"details">;
def Invariant  : BlockCommand<"invariant">;
def Note       : BlockCommand<"note">;
def Post       : BlockCommand<"post">;
def Pre        : BlockCommand<"pre">;
def Remark     : BlockCommand<"remark">;
def Remarks    : BlockCommand<"remarks">;
def Sa         : BlockCommand<"sa">;
def See        : BlockCommand<"see">;
def Since      : BlockCommand<"since">;
def Todo       : BlockCommand<"todo">;
def Version    : BlockCommand<"version">;
def Warning    : BlockCommand<"warning">;

defm Code      : VerbatimBlockCommand<"code", "endcode">;
defm Verbatim  : VerbatimBlockCommand<"verbatim", "endverbatim">;
defm Htmlonly  : VerbatimBlockCommand<"htmlonly", "endhtmlonly">;
defm Latexonly : VerbatimBlockCommand<"latexonly", "endlatexonly">;
defm Xmlonly   : VerbatimBlockCommand<"xmlonly", "endxmlonly">;
defm Manonly   : VerbatimBlockCommand<"manonly", "endmanonly">;
defm Rtfonly   : VerbatimBlockCommand<"rtfonly", "endrtfonly">;

defm Dot : VerbatimBlockCommand<"dot", "enddot">;
defm Msc : VerbatimBlockCommand<"msc", "endmsc">;

// These commands have special support in lexer.
def  FDollar  : VerbatimBlockCommand<"f$">; // Inline LaTeX formula
defm FBracket : VerbatimBlockCommand<"f[", "f]">; // Displayed LaTeX formula
defm FBrace   : VerbatimBlockCommand<"f{", "f}">; // LaTeX environment

def Defgroup   : VerbatimLineCommand<"defgroup">;
def Ingroup    : VerbatimLineCommand<"ingroup">;
def Addtogroup : VerbatimLineCommand<"addtogroup">;
def Weakgroup  : VerbatimLineCommand<"weakgroup">;
def Name       : VerbatimLineCommand<"name">;

def Section       : VerbatimLineCommand<"section">;
def Subsection    : VerbatimLineCommand<"subsection">;
def Subsubsection : VerbatimLineCommand<"subsubsection">;
def Paragraph     : VerbatimLineCommand<"paragraph">;

def Mainpage : VerbatimLineCommand<"mainpage">;
def Subpage  : VerbatimLineCommand<"subpage">;
def Ref      : VerbatimLineCommand<"ref">;

// Doxygen commands.
def Fn        : DeclarationVerbatimLineCommand<"fn">;
def Namespace : DeclarationVerbatimLineCommand<"namespace">;
def Overload  : DeclarationVerbatimLineCommand<"overload">;
def Property  : DeclarationVerbatimLineCommand<"property">;
def Typedef   : DeclarationVerbatimLineCommand<"typedef">;
def Var       : DeclarationVerbatimLineCommand<"var">;

// HeaderDoc commands.
def Class     : DeclarationVerbatimLineCommand<"class">;
def Interface : DeclarationVerbatimLineCommand<"interface">;
def Protocol  : DeclarationVerbatimLineCommand<"protocol">;
def Category  : DeclarationVerbatimLineCommand<"category">;
def Template  : DeclarationVerbatimLineCommand<"template">;
def Function  : DeclarationVerbatimLineCommand<"function">;
def Method    : DeclarationVerbatimLineCommand<"method">;
def Callback  : DeclarationVerbatimLineCommand<"callback">;
def Const     : DeclarationVerbatimLineCommand<"const">;
def Constant  : DeclarationVerbatimLineCommand<"constant">;
def Struct    : DeclarationVerbatimLineCommand<"struct">;
def Union     : DeclarationVerbatimLineCommand<"union">;
def Enum      : DeclarationVerbatimLineCommand<"enum">;

