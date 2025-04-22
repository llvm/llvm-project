/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* List of AST nodes of a particular kind                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: CommentNodes.td                                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef ABSTRACT_COMMENT
#  define ABSTRACT_COMMENT(Type) Type
#endif
#ifndef COMMENT_RANGE
#  define COMMENT_RANGE(Base, First, Last)
#endif

#ifndef LAST_COMMENT_RANGE
#  define LAST_COMMENT_RANGE(Base, First, Last) COMMENT_RANGE(Base, First, Last)
#endif

#ifndef VERBATIMBLOCKLINECOMMENT
#  define VERBATIMBLOCKLINECOMMENT(Type, Base) COMMENT(Type, Base)
#endif
VERBATIMBLOCKLINECOMMENT(VerbatimBlockLineComment, Comment)
#undef VERBATIMBLOCKLINECOMMENT

#ifndef INLINECONTENTCOMMENT
#  define INLINECONTENTCOMMENT(Type, Base) COMMENT(Type, Base)
#endif
ABSTRACT_COMMENT(INLINECONTENTCOMMENT(InlineContentComment, Comment))
#ifndef TEXTCOMMENT
#  define TEXTCOMMENT(Type, Base) INLINECONTENTCOMMENT(Type, Base)
#endif
TEXTCOMMENT(TextComment, InlineContentComment)
#undef TEXTCOMMENT

#ifndef INLINECOMMANDCOMMENT
#  define INLINECOMMANDCOMMENT(Type, Base) INLINECONTENTCOMMENT(Type, Base)
#endif
INLINECOMMANDCOMMENT(InlineCommandComment, InlineContentComment)
#undef INLINECOMMANDCOMMENT

#ifndef HTMLTAGCOMMENT
#  define HTMLTAGCOMMENT(Type, Base) INLINECONTENTCOMMENT(Type, Base)
#endif
ABSTRACT_COMMENT(HTMLTAGCOMMENT(HTMLTagComment, InlineContentComment))
#ifndef HTMLSTARTTAGCOMMENT
#  define HTMLSTARTTAGCOMMENT(Type, Base) HTMLTAGCOMMENT(Type, Base)
#endif
HTMLSTARTTAGCOMMENT(HTMLStartTagComment, HTMLTagComment)
#undef HTMLSTARTTAGCOMMENT

#ifndef HTMLENDTAGCOMMENT
#  define HTMLENDTAGCOMMENT(Type, Base) HTMLTAGCOMMENT(Type, Base)
#endif
HTMLENDTAGCOMMENT(HTMLEndTagComment, HTMLTagComment)
#undef HTMLENDTAGCOMMENT

COMMENT_RANGE(HTMLTagComment, HTMLStartTagComment, HTMLEndTagComment)

#undef HTMLTAGCOMMENT

COMMENT_RANGE(InlineContentComment, TextComment, HTMLEndTagComment)

#undef INLINECONTENTCOMMENT

#ifndef FULLCOMMENT
#  define FULLCOMMENT(Type, Base) COMMENT(Type, Base)
#endif
FULLCOMMENT(FullComment, Comment)
#undef FULLCOMMENT

#ifndef BLOCKCONTENTCOMMENT
#  define BLOCKCONTENTCOMMENT(Type, Base) COMMENT(Type, Base)
#endif
ABSTRACT_COMMENT(BLOCKCONTENTCOMMENT(BlockContentComment, Comment))
#ifndef PARAGRAPHCOMMENT
#  define PARAGRAPHCOMMENT(Type, Base) BLOCKCONTENTCOMMENT(Type, Base)
#endif
PARAGRAPHCOMMENT(ParagraphComment, BlockContentComment)
#undef PARAGRAPHCOMMENT

#ifndef BLOCKCOMMANDCOMMENT
#  define BLOCKCOMMANDCOMMENT(Type, Base) BLOCKCONTENTCOMMENT(Type, Base)
#endif
BLOCKCOMMANDCOMMENT(BlockCommandComment, BlockContentComment)
#ifndef VERBATIMLINECOMMENT
#  define VERBATIMLINECOMMENT(Type, Base) BLOCKCOMMANDCOMMENT(Type, Base)
#endif
VERBATIMLINECOMMENT(VerbatimLineComment, BlockCommandComment)
#undef VERBATIMLINECOMMENT

#ifndef VERBATIMBLOCKCOMMENT
#  define VERBATIMBLOCKCOMMENT(Type, Base) BLOCKCOMMANDCOMMENT(Type, Base)
#endif
VERBATIMBLOCKCOMMENT(VerbatimBlockComment, BlockCommandComment)
#undef VERBATIMBLOCKCOMMENT

#ifndef TPARAMCOMMANDCOMMENT
#  define TPARAMCOMMANDCOMMENT(Type, Base) BLOCKCOMMANDCOMMENT(Type, Base)
#endif
TPARAMCOMMANDCOMMENT(TParamCommandComment, BlockCommandComment)
#undef TPARAMCOMMANDCOMMENT

#ifndef PARAMCOMMANDCOMMENT
#  define PARAMCOMMANDCOMMENT(Type, Base) BLOCKCOMMANDCOMMENT(Type, Base)
#endif
PARAMCOMMANDCOMMENT(ParamCommandComment, BlockCommandComment)
#undef PARAMCOMMANDCOMMENT

COMMENT_RANGE(BlockCommandComment, BlockCommandComment, ParamCommandComment)

#undef BLOCKCOMMANDCOMMENT

COMMENT_RANGE(BlockContentComment, ParagraphComment, ParamCommandComment)

#undef BLOCKCONTENTCOMMENT

LAST_COMMENT_RANGE(Comment, VerbatimBlockLineComment, ParamCommandComment)

#undef COMMENT
#undef COMMENT_RANGE
#undef LAST_COMMENT_RANGE
#undef ABSTRACT_COMMENT
