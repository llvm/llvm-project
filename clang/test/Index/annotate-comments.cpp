// Run lines are sensitive to line numbers and come below the code.

#ifndef HEADER
#define HEADER

// Not a Doxygen comment.  NOT_DOXYGEN
void notdoxy1(void);

/* Not a Doxygen comment.  NOT_DOXYGEN */
void notdoxy2(void);

/*/ Not a Doxygen comment.  NOT_DOXYGEN */
void notdoxy3(void);

/** Doxygen comment.  isdoxy4 IS_DOXYGEN_SINGLE */
void isdoxy4(void);

/**
 * Doxygen comment.  isdoxy5 IS_DOXYGEN_SINGLE */
void isdoxy5(void);

/**
 * Doxygen comment.
 * isdoxy6 IS_DOXYGEN_SINGLE */
void isdoxy6(void);

/**
 * Doxygen comment.
 * isdoxy7 IS_DOXYGEN_SINGLE
 */
void isdoxy7(void);

/*! Doxygen comment.  isdoxy8 IS_DOXYGEN_SINGLE */
void isdoxy8(void);

/// Doxygen comment.  isdoxy9 IS_DOXYGEN_SINGLE
void isdoxy9(void);

// Not a Doxygen comment.  NOT_DOXYGEN
/// Doxygen comment.  isdoxy10 IS_DOXYGEN_SINGLE
void isdoxy10(void);

/// Doxygen comment.  isdoxy11 IS_DOXYGEN_SINGLE
// Not a Doxygen comment.  NOT_DOXYGEN
void isdoxy11(void);

/** Doxygen comment.  isdoxy12  IS_DOXYGEN_SINGLE */
/* Not a Doxygen comment.  NOT_DOXYGEN */
void isdoxy12(void);

/// Doxygen comment.  isdoxy13 IS_DOXYGEN_START
/// Doxygen comment.  IS_DOXYGEN_END
void isdoxy13(void);

/// Doxygen comment.  isdoxy14 IS_DOXYGEN_START
/// Blah-blah-blah.
/// Doxygen comment.  IS_DOXYGEN_END
void isdoxy14(void);

/// Doxygen comment.  isdoxy15 IS_DOXYGEN_START
/** Blah-blah-blah */
/// Doxygen comment.  IS_DOXYGEN_END
void isdoxy15(void);

/** Blah-blah-blah. isdoxy16 IS_DOXYGEN_START *//** Blah */
/// Doxygen comment.  IS_DOXYGEN_END
void isdoxy16(void);

/// NOT_DOXYGEN
// NOT_DOXYGEN
/// isdoxy17 IS_DOXYGEN_START IS_DOXYGEN_END
void isdoxy17(void);

unsigned
// NOT_DOXYGEN
/// NOT_DOXYGEN
// NOT_DOXYGEN
/// isdoxy18 IS_DOXYGEN_START IS_DOXYGEN_END
// NOT_DOXYGEN
int isdoxy18(void);

//! It all starts here. isdoxy19 IS_DOXYGEN_START
/*! It's a little odd to continue line this,
 *
 * but we need more multi-line comments. */
/// This comment comes before my other comments
/** This is a block comment that is associated with the function f. It
 *  runs for three lines.  IS_DOXYGEN_END
 */
void isdoxy19(int, int);

// NOT IN THE COMMENT  NOT_DOXYGEN
/// This is a BCPL comment.  isdoxy20 IS_DOXYGEN_START
/// It has only two lines.
/** But there are other blocks that are part of the comment, too.  IS_DOXYGEN_END */
void isdoxy20(int);

void notdoxy21(int); ///< This is a member comment.  isdoxy21 IS_DOXYGEN_NOT_ATTACHED

void notdoxy22(int); /*!< This is a member comment.  isdoxy22 IS_DOXYGEN_NOT_ATTACHED */

void notdoxy23(int); /**< This is a member comment.  isdoxy23 IS_DOXYGEN_NOT_ATTACHED */

void notdoxy24(int); // NOT_DOXYGEN

/// IS_DOXYGEN_SINGLE
struct isdoxy25 {
};

struct test26 {
  /// IS_DOXYGEN_SINGLE
  int isdoxy26;
};

struct test27 {
  int isdoxy27; ///< IS_DOXYGEN_SINGLE
};

struct notdoxy28 {
}; ///< IS_DOXYGEN_NOT_ATTACHED

/// IS_DOXYGEN_SINGLE
enum isdoxy29 {
};

enum notdoxy30 {
}; ///< IS_DOXYGEN_NOT_ATTACHED

/// IS_DOXYGEN_SINGLE
namespace isdoxy31 {
};

namespace notdoxy32 {
}; ///< IS_DOXYGEN_NOT_ATTACHED

class test33 {
                ///< IS_DOXYGEN_NOT_ATTACHED
  int isdoxy33; ///< isdoxy33 IS_DOXYGEN_SINGLE
  int isdoxy34; ///< isdoxy34 IS_DOXYGEN_SINGLE

                ///< IS_DOXYGEN_NOT_ATTACHED
  int isdoxy35, ///< isdoxy35 IS_DOXYGEN_SINGLE
      isdoxy36; ///< isdoxy36 IS_DOXYGEN_SINGLE

                ///< IS_DOXYGEN_NOT_ATTACHED
  int isdoxy37  ///< isdoxy37 IS_DOXYGEN_SINGLE
    , isdoxy38  ///< isdoxy38 IS_DOXYGEN_SINGLE
    , isdoxy39; ///< isdoxy39 IS_DOXYGEN_SINGLE
};

// Verified that Doxygen attaches these.

/// isdoxy40 IS_DOXYGEN_SINGLE
// NOT_DOXYGEN
void isdoxy40(int);

unsigned
/// isdoxy41 IS_DOXYGEN_SINGLE
// NOT_DOXYGEN
int isdoxy41(int);

class test42 {
  int isdoxy42; /* NOT_DOXYGEN */ ///< isdoxy42 IS_DOXYGEN_SINGLE
};

/// IS_DOXYGEN_START
/// It is fine to have a command at the end of comment.
///\brief
///
/// Some malformed command.
/** \*/
/**
 * \brief Aaa aaaaaaa aaaa.
 * IS_DOXYGEN_END
 */
void isdoxy43(void);

/// IS_DOXYGEN_START Aaa bbb
/// ccc.
///
/// Ddd eee.
/// Fff.
///
/// Ggg. IS_DOXYGEN_END
void isdoxy44(void);

/// IS_DOXYGEN_START Aaa bbb
/// ccc.
///
/// \brief
/// Ddd eee.
/// Fff.
///
/// Ggg. IS_DOXYGEN_END
void isdoxy45(void);

/// IS_DOXYGEN_START Aaa bbb
/// ccc.
///
/// \short
/// Ddd eee.
/// Fff.
///
/// Ggg. IS_DOXYGEN_END
void isdoxy46(void);

/// IS_DOXYGEN_NOT_ATTACHED
#define FOO
void notdoxy47(void);

/// IS_DOXYGEN_START Aaa bbb
/// \param ccc
/// \returns ddd IS_DOXYGEN_END
void isdoxy48(int);

/// \brief IS_DOXYGEN_START Aaa
/// \returns bbb IS_DOXYGEN_END
void isdoxy49(void);

/// \param ccc IS_DOXYGEN_START
/// \returns ddd IS_DOXYGEN_END
void isdoxy50(int);

// One of the following lines has trailing whitespace.  It is intended, don't
// fix it.
/**
 * Aaa. IS_DOXYGEN_START
 * 
 * Bbb. IS_DOXYGEN_END
 */
void isdoxy51(int);

// One of the following lines has trailing whitespace.  It is intended, don't
// fix it.
/**
 * Aaa. IS_DOXYGEN_START
 * Bbb.
 *  
 * Ccc. IS_DOXYGEN_END
 */
void isdoxy52(int);

/**
 * \fn isdoxy53
 *
 * Aaa. IS_DOXYGEN_START IS_DOXYGEN_END
 */
void isdoxy53(int);

/// Aaa.
void comment_to_html_conversion_1();

/// \brief Aaa.
void comment_to_html_conversion_2();

/// \short Aaa.
void comment_to_html_conversion_3();

/// Aaa.
///
/// \brief Bbb.
void comment_to_html_conversion_4();

/// Aaa.
///
/// \brief Bbb.
///
/// Ccc.
void comment_to_html_conversion_5();

/// \brief Aaa.
/// \brief Bbb.
void comment_to_html_conversion_6();

/// Aaa.
///
/// \return Bbb.
void comment_to_html_conversion_7();

/// Aaa.
///
/// \returns Bbb.
void comment_to_html_conversion_8();

/// Aaa.
///
/// \result Bbb.
void comment_to_html_conversion_9();

/// \returns Aaa.
/// \returns Bbb.
void comment_to_html_conversion_10();

/// Aaa.
///
/// Bbb.
///
/// \returns Ccc.
void comment_to_html_conversion_11();

/// \param
void comment_to_html_conversion_12(int x1);

/// \param x1 Aaa.
void comment_to_html_conversion_13(int x1);

/// \param zzz Aaa.
void comment_to_html_conversion_14(int x1);

/// \param x2 Bbb.
/// \param x1 Aaa.
void comment_to_html_conversion_15(int x1, int x2);

/// \param x2 Bbb.
/// \param zzz Aaa.
/// \param x1 Aaa.
void comment_to_html_conversion_16(int x1, int x2);

/// \tparam
/// \param aaa Blah blah
template<typename T>
void comment_to_html_conversion_17(T aaa);

/// \tparam T
/// \param aaa Blah blah
template<typename T>
void comment_to_html_conversion_18(T aaa);

/// \tparam T2 Bbb
/// \tparam T1 Aaa
template<typename T1, typename T2>
void comment_to_html_conversion_19(T1 aaa, T2 bbb);

/// \tparam T2 Bbb
/// \tparam U Zzz
/// \tparam V Ccc
/// \tparam T1 Aaa
template<typename T1, typename T2, int V>
void comment_to_html_conversion_20(T1 aaa, T2 bbb);

/// \tparam TTT Ddd
/// \tparam C Ccc
/// \tparam T Aaa
/// \tparam TT Bbb
template<template<template<typename T> class TT, class C> class TTT>
void comment_to_html_conversion_21();

/// \brief Aaa.
///
/// Bbb.
///
/// \param x2 Ddd.
/// \param x1 Ccc.
/// \returns Eee.
void comment_to_html_conversion_22(int x1, int x2);

/// <br><a href="http://example.com/">Aaa</a>
void comment_to_html_conversion_23();

/// \verbatim
/// <a href="http://example.com/">Aaa</a>
/// <a href='http://example.com/'>Aaa</a>
/// \endverbatim
void comment_to_html_conversion_24();

/// \function foo
/// \class foo
/// \method foo
/// \interface foo
/// Blah blah.
void comment_to_html_conversion_25();

/// \unknown
void comment_to_html_conversion_26();

/// \b Aaa
void comment_to_html_conversion_27();

/// \c Aaa \p Bbb
void comment_to_html_conversion_28();

/// \a Aaa \e Bbb \em Ccc
void comment_to_html_conversion_29();

/// \a 1<2 \e 3<4 \em 5<6 \param 7<8 aaa \tparam 9<10 bbb
void comment_to_html_conversion_30();

/// \\ \@ \& \$ \# \< \> \% \" \. \::
void comment_to_html_conversion_31();

/// &amp; &lt; &gt; &quot;
void comment_to_html_conversion_32();

/// <em>0&lt;i</em>
void comment_to_html_conversion_33();

/// Aaa.
class comment_to_xml_conversion_01 {
  /// \param aaa Blah blah.
  comment_to_xml_conversion_01(int aaa);

  /// Aaa.
  ~comment_to_xml_conversion_01();

  /// \param aaa Blah blah.
  int comment_to_xml_conversion_02(int aaa);

  /// \param aaa Blah blah.
  static int comment_to_xml_conversion_03(int aaa);

  /// Aaa.
  int comment_to_xml_conversion_04;

  /// Aaa.
  static int comment_to_xml_conversion_05;

  /// \param aaa Blah blah.
  void operator()(int aaa);

  /// Aaa.
  operator bool();

  /// Aaa.
  typedef int comment_to_xml_conversion_06;

  /// Aaa.
  using comment_to_xml_conversion_07 = int;

  template<typename T, typename U>
  class comment_to_xml_conversion_08 { };

  /// Aaa.
  template<typename T>
  using comment_to_xml_conversion_09 = comment_to_xml_conversion_08<T, int>;
};

/// Aaa.
template<typename T, typename U>
void comment_to_xml_conversion_10(T aaa, U bbb);

/// Aaa.
template<>
void comment_to_xml_conversion_10(int aaa, int bbb);

/// Aaa.
template<typename T, typename U>
class comment_to_xml_conversion_11 { };

/// Aaa.
template<typename T>
class comment_to_xml_conversion_11<T, int> { };

/// Aaa.
template<>
class comment_to_xml_conversion_11<int, int> { };

/// Aaa.
int comment_to_xml_conversion_12;

/// Aaa.
namespace comment_to_xml_conversion_13 {
  /// Aaa.
  namespace comment_to_xml_conversion_14 {
  }
}

/// Aaa.
enum comment_to_xml_conversion_15 {
  /// Aaa.
  comment_to_xml_conversion_16
};

/// Aaa.
enum class comment_to_xml_conversion_17 {
  /// Aaa.
  comment_to_xml_conversion_18
};

#endif

// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: %clang_cc1 -x c++ -std=c++11 -emit-pch -o %t/out.pch %s
// RUN: %clang_cc1 -x c++ -std=c++11 -include-pch %t/out.pch -fsyntax-only %s

// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng %s -std=c++11 > %t/out.c-index-direct
// RUN: c-index-test -test-load-tu %t/out.pch all > %t/out.c-index-pch

// RUN: FileCheck %s -check-prefix=WRONG < %t/out.c-index-direct
// RUN: FileCheck %s -check-prefix=WRONG < %t/out.c-index-pch

// Declarations without Doxygen comments should not pick up some Doxygen comments.
// WRONG-NOT: notdoxy{{.*}}Comment=
// WRONG-NOT: test{{.*}}Comment=

// Non-Doxygen comments should not be attached to anything.
// WRONG-NOT: NOT_DOXYGEN

// Some Doxygen comments are not attached to anything.
// WRONG-NOT: IS_DOXYGEN_NOT_ATTACHED

// Ensure we don't pick up extra comments.
// WRONG-NOT: IS_DOXYGEN_START{{.*}}IS_DOXYGEN_START{{.*}}BriefComment=
// WRONG-NOT: IS_DOXYGEN_END{{.*}}IS_DOXYGEN_END{{.*}}BriefComment=
//
// Ensure that XML is not invalid
// WRONG-NOT: CommentXMLInvalid

// RUN: FileCheck %s < %t/out.c-index-direct
// RUN: FileCheck %s < %t/out.c-index-pch

// CHECK: annotate-comments.cpp:16:6: FunctionDecl=isdoxy4:{{.*}} isdoxy4 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:20:6: FunctionDecl=isdoxy5:{{.*}} isdoxy5 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:25:6: FunctionDecl=isdoxy6:{{.*}} isdoxy6 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:31:6: FunctionDecl=isdoxy7:{{.*}} isdoxy7 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:34:6: FunctionDecl=isdoxy8:{{.*}} isdoxy8 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:37:6: FunctionDecl=isdoxy9:{{.*}} isdoxy9 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:41:6: FunctionDecl=isdoxy10:{{.*}} isdoxy10 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:53:6: FunctionDecl=isdoxy13:{{.*}} isdoxy13 IS_DOXYGEN_START{{.*}} IS_DOXYGEN_END
// CHECK: annotate-comments.cpp:58:6: FunctionDecl=isdoxy14:{{.*}} isdoxy14 IS_DOXYGEN_START{{.*}} IS_DOXYGEN_END
// CHECK: annotate-comments.cpp:63:6: FunctionDecl=isdoxy15:{{.*}} isdoxy15 IS_DOXYGEN_START{{.*}} IS_DOXYGEN_END
// CHECK: annotate-comments.cpp:67:6: FunctionDecl=isdoxy16:{{.*}} isdoxy16 IS_DOXYGEN_START{{.*}} IS_DOXYGEN_END
// CHECK: annotate-comments.cpp:72:6: FunctionDecl=isdoxy17:{{.*}} isdoxy17 IS_DOXYGEN_START{{.*}} IS_DOXYGEN_END
// CHECK: annotate-comments.cpp:80:5: FunctionDecl=isdoxy18:{{.*}} isdoxy18 IS_DOXYGEN_START{{.*}} IS_DOXYGEN_END
// CHECK: annotate-comments.cpp:90:6: FunctionDecl=isdoxy19:{{.*}} isdoxy19 IS_DOXYGEN_START{{.*}} IS_DOXYGEN_END
// CHECK: annotate-comments.cpp:96:6: FunctionDecl=isdoxy20:{{.*}} isdoxy20 IS_DOXYGEN_START{{.*}} IS_DOXYGEN_END
// CHECK: annotate-comments.cpp:107:8: StructDecl=isdoxy25:{{.*}} IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:112:7: FieldDecl=isdoxy26:{{.*}} IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:116:7: FieldDecl=isdoxy27:{{.*}} IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:123:6: EnumDecl=isdoxy29:{{.*}} IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:130:11: Namespace=isdoxy31:{{.*}} IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:138:7: FieldDecl=isdoxy33:{{.*}} isdoxy33 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:139:7: FieldDecl=isdoxy34:{{.*}} isdoxy34 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:142:7: FieldDecl=isdoxy35:{{.*}} isdoxy35 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:143:7: FieldDecl=isdoxy36:{{.*}} isdoxy36 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:146:7: FieldDecl=isdoxy37:{{.*}} isdoxy37 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:147:7: FieldDecl=isdoxy38:{{.*}} isdoxy38 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:148:7: FieldDecl=isdoxy39:{{.*}} isdoxy39 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:155:6: FunctionDecl=isdoxy40:{{.*}} isdoxy40 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:160:5: FunctionDecl=isdoxy41:{{.*}} isdoxy41 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:163:7: FieldDecl=isdoxy42:{{.*}} isdoxy42 IS_DOXYGEN_SINGLE
// CHECK: annotate-comments.cpp:176:6: FunctionDecl=isdoxy43:{{.*}} IS_DOXYGEN_START{{.*}} IS_DOXYGEN_END

// CHECK: annotate-comments.cpp:185:6: FunctionDecl=isdoxy44:{{.*}} BriefComment=[IS_DOXYGEN_START Aaa bbb ccc.]
// CHECK: annotate-comments.cpp:195:6: FunctionDecl=isdoxy45:{{.*}} BriefComment=[Ddd eee. Fff.]
// CHECK: annotate-comments.cpp:205:6: FunctionDecl=isdoxy46:{{.*}} BriefComment=[Ddd eee. Fff.]
// CHECK: annotate-comments.cpp:214:6: FunctionDecl=isdoxy48:{{.*}} BriefComment=[IS_DOXYGEN_START Aaa bbb]
// CHECK: annotate-comments.cpp:218:6: FunctionDecl=isdoxy49:{{.*}} BriefComment=[IS_DOXYGEN_START Aaa]
// CHECK: annotate-comments.cpp:222:6: FunctionDecl=isdoxy50:{{.*}} BriefComment=[Returns ddd IS_DOXYGEN_END]
// CHECK: annotate-comments.cpp:231:6: FunctionDecl=isdoxy51:{{.*}} BriefComment=[Aaa. IS_DOXYGEN_START]
// CHECK: annotate-comments.cpp:241:6: FunctionDecl=isdoxy52:{{.*}} BriefComment=[Aaa. IS_DOXYGEN_START Bbb.]

// CHECK: annotate-comments.cpp:251:6: FunctionDecl=comment_to_html_conversion_1:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Aaa.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="251" column="6"><Name>comment_to_html_conversion_1</Name><USR>c:@F@comment_to_html_conversion_1#</USR><Declaration>void comment_to_html_conversion_1()</Declaration><Abstract><Para> Aaa.</Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa.])))]
// CHECK: annotate-comments.cpp:254:6: FunctionDecl=comment_to_html_conversion_2:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Aaa.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="254" column="6"><Name>comment_to_html_conversion_2</Name><USR>c:@F@comment_to_html_conversion_2#</USR><Declaration>void comment_to_html_conversion_2()</Declaration><Abstract><Para> Aaa.</Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[brief]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.]))))]
// CHECK: annotate-comments.cpp:257:6: FunctionDecl=comment_to_html_conversion_3:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Aaa.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="257" column="6"><Name>comment_to_html_conversion_3</Name><USR>c:@F@comment_to_html_conversion_3#</USR><Declaration>void comment_to_html_conversion_3()</Declaration><Abstract><Para> Aaa.</Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[short]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.]))))]
// CHECK: annotate-comments.cpp:262:6: FunctionDecl=comment_to_html_conversion_4:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Bbb.</p><p> Aaa.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="262" column="6"><Name>comment_to_html_conversion_4</Name><USR>c:@F@comment_to_html_conversion_4#</USR><Declaration>void comment_to_html_conversion_4()</Declaration><Abstract><Para> Bbb.</Para></Abstract><Discussion><Para> Aaa.</Para></Discussion></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa.]))
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[brief]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.]))))]
// CHECK: annotate-comments.cpp:269:6: FunctionDecl=comment_to_html_conversion_5:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Bbb.</p><p> Aaa.</p><p> Ccc.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="269" column="6"><Name>comment_to_html_conversion_5</Name><USR>c:@F@comment_to_html_conversion_5#</USR><Declaration>void comment_to_html_conversion_5()</Declaration><Abstract><Para> Bbb.</Para></Abstract><Discussion><Para> Aaa.</Para><Para> Ccc.</Para></Discussion></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa.]))
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[brief]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.])))
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Ccc.])))]
// CHECK: annotate-comments.cpp:273:6: FunctionDecl=comment_to_html_conversion_6:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Aaa. </p><p class="para-brief"> Bbb.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="273" column="6"><Name>comment_to_html_conversion_6</Name><USR>c:@F@comment_to_html_conversion_6#</USR><Declaration>void comment_to_html_conversion_6()</Declaration><Abstract><Para> Aaa. </Para></Abstract><Discussion><Para> Bbb.</Para></Discussion></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[brief]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[brief]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.]))))]
// CHECK: annotate-comments.cpp:278:6: FunctionDecl=comment_to_html_conversion_7:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Aaa.</p><p class="para-returns"><span class="word-returns">Returns</span>  Bbb.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="278" column="6"><Name>comment_to_html_conversion_7</Name><USR>c:@F@comment_to_html_conversion_7#</USR><Declaration>void comment_to_html_conversion_7()</Declaration><Abstract><Para> Aaa.</Para></Abstract><ResultDiscussion><Para> Bbb.</Para></ResultDiscussion></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa.]))
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[return]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.]))))]
// CHECK: annotate-comments.cpp:283:6: FunctionDecl=comment_to_html_conversion_8:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Aaa.</p><p class="para-returns"><span class="word-returns">Returns</span>  Bbb.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="283" column="6"><Name>comment_to_html_conversion_8</Name><USR>c:@F@comment_to_html_conversion_8#</USR><Declaration>void comment_to_html_conversion_8()</Declaration><Abstract><Para> Aaa.</Para></Abstract><ResultDiscussion><Para> Bbb.</Para></ResultDiscussion></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa.]))
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[returns]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.]))))]
// CHECK: annotate-comments.cpp:288:6: FunctionDecl=comment_to_html_conversion_9:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Aaa.</p><p class="para-returns"><span class="word-returns">Returns</span>  Bbb.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="288" column="6"><Name>comment_to_html_conversion_9</Name><USR>c:@F@comment_to_html_conversion_9#</USR><Declaration>void comment_to_html_conversion_9()</Declaration><Abstract><Para> Aaa.</Para></Abstract><ResultDiscussion><Para> Bbb.</Para></ResultDiscussion></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa.]))
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[result]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.]))))]
// CHECK: annotate-comments.cpp:292:6: FunctionDecl=comment_to_html_conversion_10:{{.*}} FullCommentAsHTML=[<p class="para-returns"><span class="word-returns">Returns</span>  Bbb.</p><p class="para-returns"><span class="word-returns">Returns</span>  Aaa. </p>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="292" column="6"><Name>comment_to_html_conversion_10</Name><USR>c:@F@comment_to_html_conversion_10#</USR><Declaration>void comment_to_html_conversion_10()</Declaration><ResultDiscussion><Para> Aaa. </Para></ResultDiscussion><Discussion><Para> Bbb.</Para></Discussion></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[returns]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[returns]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.]))))]
// CHECK: annotate-comments.cpp:299:6: FunctionDecl=comment_to_html_conversion_11:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Aaa.</p><p> Bbb.</p><p class="para-returns"><span class="word-returns">Returns</span>  Ccc.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="299" column="6"><Name>comment_to_html_conversion_11</Name><USR>c:@F@comment_to_html_conversion_11#</USR><Declaration>void comment_to_html_conversion_11()</Declaration><Abstract><Para> Aaa.</Para></Abstract><ResultDiscussion><Para> Ccc.</Para></ResultDiscussion><Discussion><Para> Bbb.</Para></Discussion></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Aaa.]))
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Bbb.]))
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[returns]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Ccc.]))))]
// CHECK: annotate-comments.cpp:302:6: FunctionDecl=comment_to_html_conversion_12:{{.*}} FullCommentAsHTML=[] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="302" column="6"><Name>comment_to_html_conversion_12</Name><USR>c:@F@comment_to_html_conversion_12#I#</USR><Declaration>void comment_to_html_conversion_12(int x1)</Declaration></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[] ParamIndex=Invalid
// CHECK-NEXT:         (CXComment_Paragraph IsWhitespace)))]
// CHECK: annotate-comments.cpp:305:6: FunctionDecl=comment_to_html_conversion_13:{{.*}} FullCommentAsHTML=[<dl><dt class="param-name-index-0">x1</dt><dd class="param-descr-index-0"> Aaa.</dd></dl>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="305" column="6"><Name>comment_to_html_conversion_13</Name><USR>c:@F@comment_to_html_conversion_13#I#</USR><Declaration>void comment_to_html_conversion_13(int x1)</Declaration><Parameters><Parameter><Name>x1</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Aaa.</Para></Discussion></Parameter></Parameters></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[x1] ParamIndex=0
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.]))))]
// CHECK: annotate-comments.cpp:308:6: FunctionDecl=comment_to_html_conversion_14:{{.*}} FullCommentAsHTML=[<dl><dt class="param-name-index-invalid">zzz</dt><dd class="param-descr-index-invalid"> Aaa.</dd></dl>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="308" column="6"><Name>comment_to_html_conversion_14</Name><USR>c:@F@comment_to_html_conversion_14#I#</USR><Declaration>void comment_to_html_conversion_14(int x1)</Declaration><Parameters><Parameter><Name>zzz</Name><Direction isExplicit="0">in</Direction><Discussion><Para> Aaa.</Para></Discussion></Parameter></Parameters></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[zzz] ParamIndex=Invalid
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.]))))]
// CHECK: annotate-comments.cpp:312:6: FunctionDecl=comment_to_html_conversion_15:{{.*}} FullCommentAsHTML=[<dl><dt class="param-name-index-0">x1</dt><dd class="param-descr-index-0"> Aaa.</dd><dt class="param-name-index-1">x2</dt><dd class="param-descr-index-1"> Bbb. </dd></dl>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="312" column="6"><Name>comment_to_html_conversion_15</Name><USR>c:@F@comment_to_html_conversion_15#I#I#</USR><Declaration>void comment_to_html_conversion_15(int x1, int x2)</Declaration><Parameters><Parameter><Name>x1</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Aaa.</Para></Discussion></Parameter><Parameter><Name>x2</Name><Index>1</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Bbb. </Para></Discussion></Parameter></Parameters></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[x2] ParamIndex=1
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[x1] ParamIndex=0
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.]))))]
// CHECK: annotate-comments.cpp:317:6: FunctionDecl=comment_to_html_conversion_16:{{.*}} FullCommentAsHTML=[<dl><dt class="param-name-index-0">x1</dt><dd class="param-descr-index-0"> Aaa.</dd><dt class="param-name-index-1">x2</dt><dd class="param-descr-index-1"> Bbb. </dd><dt class="param-name-index-invalid">zzz</dt><dd class="param-descr-index-invalid"> Aaa. </dd></dl>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="317" column="6"><Name>comment_to_html_conversion_16</Name><USR>c:@F@comment_to_html_conversion_16#I#I#</USR><Declaration>void comment_to_html_conversion_16(int x1, int x2)</Declaration><Parameters><Parameter><Name>x1</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Aaa.</Para></Discussion></Parameter><Parameter><Name>x2</Name><Index>1</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Bbb. </Para></Discussion></Parameter><Parameter><Name>zzz</Name><Direction isExplicit="0">in</Direction><Discussion><Para> Aaa. </Para></Discussion></Parameter></Parameters></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[x2] ParamIndex=1
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb.] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[zzz] ParamIndex=Invalid
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[x1] ParamIndex=0
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.]))))]
// CHECK: annotate-comments.cpp:322:6: FunctionTemplate=comment_to_html_conversion_17:{{.*}} FullCommentAsHTML=[<dl><dt class="param-name-index-0">aaa</dt><dd class="param-descr-index-0"> Blah blah</dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}annotate-comments.cpp" line="322" column="6"><Name>comment_to_html_conversion_17</Name><USR>c:@FT@&gt;1#Tcomment_to_html_conversion_17#t0.0#</USR><Declaration>template &lt;typename T&gt; void comment_to_html_conversion_17(T aaa)</Declaration><Parameters><Parameter><Name>aaa</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Blah blah</Para></Discussion></Parameter></Parameters></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[] ParamPosition=Invalid
// CHECK-NEXT:         (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[aaa] ParamIndex=0
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Blah blah]))))]
// CHECK: annotate-comments.cpp:327:6: FunctionTemplate=comment_to_html_conversion_18:{{.*}} FullCommentAsHTML=[<dl><dt class="param-name-index-0">aaa</dt><dd class="param-descr-index-0"> Blah blah</dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}annotate-comments.cpp" line="327" column="6"><Name>comment_to_html_conversion_18</Name><USR>c:@FT@&gt;1#Tcomment_to_html_conversion_18#t0.0#</USR><Declaration>template &lt;typename T&gt; void comment_to_html_conversion_18(T aaa)</Declaration><Parameters><Parameter><Name>aaa</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Blah blah</Para></Discussion></Parameter></Parameters></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[T] ParamPosition={0}
// CHECK-NEXT:         (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[aaa] ParamIndex=0
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Blah blah]))))]
// CHECK: annotate-comments.cpp:332:6: FunctionTemplate=comment_to_html_conversion_19:{{.*}} FullCommentAsHTML=[<dl><dt class="tparam-name-index-0">T1</dt><dd class="tparam-descr-index-0"> Aaa</dd><dt class="tparam-name-index-1">T2</dt><dd class="tparam-descr-index-1"> Bbb </dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}annotate-comments.cpp" line="332" column="6"><Name>comment_to_html_conversion_19</Name><USR>c:@FT@&gt;2#T#Tcomment_to_html_conversion_19#t0.0#t0.1#</USR><Declaration>template &lt;typename T1, typename T2&gt; void comment_to_html_conversion_19(T1 aaa, T2 bbb)</Declaration><TemplateParameters><Parameter><Name>T1</Name><Index>0</Index><Discussion><Para> Aaa</Para></Discussion></Parameter><Parameter><Name>T2</Name><Index>1</Index><Discussion><Para> Bbb </Para></Discussion></Parameter></TemplateParameters></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[T2] ParamPosition={1}
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[T1] ParamPosition={0}
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa]))))]
// CHECK: annotate-comments.cpp:339:6: FunctionTemplate=comment_to_html_conversion_20:{{.*}} FullCommentAsHTML=[<dl><dt class="tparam-name-index-0">T1</dt><dd class="tparam-descr-index-0"> Aaa</dd><dt class="tparam-name-index-1">T2</dt><dd class="tparam-descr-index-1"> Bbb </dd><dt class="tparam-name-index-2">V</dt><dd class="tparam-descr-index-2"> Ccc </dd><dt class="tparam-name-index-invalid">U</dt><dd class="tparam-descr-index-invalid"> Zzz </dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}annotate-comments.cpp" line="339" column="6"><Name>comment_to_html_conversion_20</Name><USR>c:@FT@&gt;3#T#T#NIcomment_to_html_conversion_20#t0.0#t0.1#</USR><Declaration>template &lt;typename T1, typename T2, int V&gt; void comment_to_html_conversion_20(T1 aaa, T2 bbb)</Declaration><TemplateParameters><Parameter><Name>T1</Name><Index>0</Index><Discussion><Para> Aaa</Para></Discussion></Parameter><Parameter><Name>T2</Name><Index>1</Index><Discussion><Para> Bbb </Para></Discussion></Parameter><Parameter><Name>V</Name><Index>2</Index><Discussion><Para> Ccc </Para></Discussion></Parameter><Parameter><Name>U</Name><Discussion><Para> Zzz </Para></Discussion></Parameter></TemplateParameters></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[T2] ParamPosition={1}
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[U] ParamPosition=Invalid
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Zzz] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[V] ParamPosition={2}
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Ccc] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[T1] ParamPosition={0}
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa]))))]
// CHECK: annotate-comments.cpp:346:6: FunctionTemplate=comment_to_html_conversion_21:{{.*}} FullCommentAsHTML=[<dl><dt class="tparam-name-index-0">TTT</dt><dd class="tparam-descr-index-0"> Ddd </dd><dt class="tparam-name-index-other">C</dt><dd class="tparam-descr-index-other"> Ccc </dd><dt class="tparam-name-index-other">T</dt><dd class="tparam-descr-index-other"> Aaa </dd><dt class="tparam-name-index-other">TT</dt><dd class="tparam-descr-index-other"> Bbb</dd></dl>] FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}annotate-comments.cpp" line="346" column="6"><Name>comment_to_html_conversion_21</Name><USR>c:@FT@&gt;1#t&gt;2#t&gt;1#T#Tcomment_to_html_conversion_21#</USR><Declaration>template &lt;template &lt;template &lt;typename T&gt; class TT, class C&gt; class TTT&gt; void comment_to_html_conversion_21()</Declaration><TemplateParameters><Parameter><Name>TTT</Name><Index>0</Index><Discussion><Para> Ddd </Para></Discussion></Parameter><Parameter><Name>C</Name><Discussion><Para> Ccc </Para></Discussion></Parameter><Parameter><Name>T</Name><Discussion><Para> Aaa </Para></Discussion></Parameter><Parameter><Name>TT</Name><Discussion><Para> Bbb</Para></Discussion></Parameter></TemplateParameters></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[TTT] ParamPosition={0}
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Ddd] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[C] ParamPosition={0, 1}
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Ccc] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[T] ParamPosition={0, 0, 0}
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[TT] ParamPosition={0, 0}
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Bbb]))))]
// CHECK: annotate-comments.cpp:355:6: FunctionDecl=comment_to_html_conversion_22:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Aaa.</p><p> Bbb.</p><dl><dt class="param-name-index-0">x1</dt><dd class="param-descr-index-0"> Ccc. </dd><dt class="param-name-index-1">x2</dt><dd class="param-descr-index-1"> Ddd. </dd></dl><p class="para-returns"><span class="word-returns">Returns</span>  Eee.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="355" column="6"><Name>comment_to_html_conversion_22</Name><USR>c:@F@comment_to_html_conversion_22#I#I#</USR><Declaration>void comment_to_html_conversion_22(int x1, int x2)</Declaration><Abstract><Para> Aaa.</Para></Abstract><Parameters><Parameter><Name>x1</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Ccc. </Para></Discussion></Parameter><Parameter><Name>x2</Name><Index>1</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Ddd. </Para></Discussion></Parameter></Parameters><ResultDiscussion><Para> Eee.</Para></ResultDiscussion><Discussion><Para> Bbb.</Para></Discussion></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[brief]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Aaa.])))
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ Bbb.]))
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[x2] ParamIndex=1
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Ddd.] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[x1] ParamIndex=0
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Ccc.] HasTrailingNewline)
// CHECK-NEXT:           (CXComment_Text Text=[ ] IsWhitespace)))
// CHECK-NEXT:       (CXComment_BlockCommand CommandName=[returns]
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ Eee.]))))]
// CHECK: annotate-comments.cpp:358:6: FunctionDecl=comment_to_html_conversion_23:{{.*}} FullCommentAsHTML=[<p class="para-brief"> <br><a href="http://example.com/">Aaa</a></p>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="358" column="6"><Name>comment_to_html_conversion_23</Name><USR>c:@F@comment_to_html_conversion_23#</USR><Declaration>void comment_to_html_conversion_23()</Declaration><Abstract><Para> <rawHTML><![CDATA[<br>]]></rawHTML><rawHTML><![CDATA[<a href="http://example.com/">]]></rawHTML>Aaa<rawHTML>&lt;/a&gt;</rawHTML></Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_HTMLStartTag Name=[br])
// CHECK-NEXT:         (CXComment_HTMLStartTag Name=[a] Attrs: href=http://example.com/)
// CHECK-NEXT:         (CXComment_Text Text=[Aaa])
// CHECK-NEXT:         (CXComment_HTMLEndTag Name=[a])))]
// CHECK: annotate-comments.cpp:364:6: FunctionDecl=comment_to_html_conversion_24:{{.*}} FullCommentAsHTML=[<pre> &lt;a href=&quot;http:&#47;&#47;example.com&#47;&quot;&gt;Aaa&lt;&#47;a&gt;\n &lt;a href=&#39;http:&#47;&#47;example.com&#47;&#39;&gt;Aaa&lt;&#47;a&gt;</pre>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="364" column="6"><Name>comment_to_html_conversion_24</Name><USR>c:@F@comment_to_html_conversion_24#</USR><Declaration>void comment_to_html_conversion_24()</Declaration><Discussion><Verbatim xml:space="preserve" kind="verbatim"> &lt;a href=&quot;http://example.com/&quot;&gt;Aaa&lt;/a&gt;\n &lt;a href=&apos;http://example.com/&apos;&gt;Aaa&lt;/a&gt;</Verbatim></Discussion></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph IsWhitespace
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_VerbatimBlockCommand CommandName=[verbatim]
// CHECK-NEXT:         (CXComment_VerbatimBlockLine Text=[ <a href="http://example.com/">Aaa</a>])
// CHECK-NEXT:         (CXComment_VerbatimBlockLine Text=[ <a href='http://example.com/'>Aaa</a>])))]
// CHECK: annotate-comments.cpp:371:6: FunctionDecl=comment_to_html_conversion_25:{{.*}} FullCommentAsHTML=[<p class="para-brief"> Blah blah.</p>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="371" column="6"><Name>comment_to_html_conversion_25</Name><USR>c:@F@comment_to_html_conversion_25#</USR><Declaration>void comment_to_html_conversion_25()</Declaration><Abstract><Para> Blah blah.</Para></Abstract></Function>]
// CHECK:  CommentAST=[
// CHECK:    (CXComment_FullComment
// CHECK:       (CXComment_Paragraph IsWhitespace
// CHECK:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK:       (CXComment_VerbatimLine Text=[ foo])
// CHECK:       (CXComment_Paragraph IsWhitespace
// CHECK:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK:       (CXComment_VerbatimLine Text=[ foo])
// CHECK:       (CXComment_Paragraph IsWhitespace
// CHECK:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK:       (CXComment_VerbatimLine Text=[ foo])
// CHECK:       (CXComment_Paragraph IsWhitespace
// CHECK:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK:       (CXComment_VerbatimLine Text=[ foo])
// CHECK:       (CXComment_Paragraph
// CHECK:         (CXComment_Text Text=[ Blah blah.])))]
// CHECK: annotate-comments.cpp:374:6: FunctionDecl=comment_to_html_conversion_26:{{.*}} FullCommentAsHTML=[<p class="para-brief"> </p>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="374" column="6"><Name>comment_to_html_conversion_26</Name><USR>c:@F@comment_to_html_conversion_26#</USR><Declaration>void comment_to_html_conversion_26()</Declaration><Abstract><Para> </Para></Abstract></Function>]
// CHECK:  CommentAST=[
// CHECK:    (CXComment_FullComment
// CHECK:       (CXComment_Paragraph
// CHECK:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK:         (CXComment_InlineCommand CommandName=[unknown] RenderNormal)))]
// CHECK: annotate-comments.cpp:377:6: FunctionDecl=comment_to_html_conversion_27:{{.*}} FullCommentAsHTML=[<p class="para-brief"> <b>Aaa</b></p>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="377" column="6"><Name>comment_to_html_conversion_27</Name><USR>c:@F@comment_to_html_conversion_27#</USR><Declaration>void comment_to_html_conversion_27()</Declaration><Abstract><Para> <bold>Aaa</bold></Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_InlineCommand CommandName=[b] RenderBold Arg[0]=Aaa)))]
// CHECK: annotate-comments.cpp:380:6: FunctionDecl=comment_to_html_conversion_28:{{.*}} FullCommentAsHTML=[<p class="para-brief"> <tt>Aaa</tt> <tt>Bbb</tt></p>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="380" column="6"><Name>comment_to_html_conversion_28</Name><USR>c:@F@comment_to_html_conversion_28#</USR><Declaration>void comment_to_html_conversion_28()</Declaration><Abstract><Para> <monospaced>Aaa</monospaced> <monospaced>Bbb</monospaced></Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_InlineCommand CommandName=[c] RenderMonospaced Arg[0]=Aaa)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_InlineCommand CommandName=[p] RenderMonospaced Arg[0]=Bbb)))]
// CHECK: annotate-comments.cpp:383:6: FunctionDecl=comment_to_html_conversion_29:{{.*}} FullCommentAsHTML=[<p class="para-brief"> <em>Aaa</em> <em>Bbb</em> <em>Ccc</em></p>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="383" column="6"><Name>comment_to_html_conversion_29</Name><USR>c:@F@comment_to_html_conversion_29#</USR><Declaration>void comment_to_html_conversion_29()</Declaration><Abstract><Para> <emphasized>Aaa</emphasized> <emphasized>Bbb</emphasized> <emphasized>Ccc</emphasized></Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_InlineCommand CommandName=[a] RenderEmphasized Arg[0]=Aaa)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_InlineCommand CommandName=[e] RenderEmphasized Arg[0]=Bbb)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_InlineCommand CommandName=[em] RenderEmphasized Arg[0]=Ccc)))]
// CHECK: annotate-comments.cpp:386:6: FunctionDecl=comment_to_html_conversion_30:{{.*}} FullCommentAsHTML=[<p class="para-brief"> <em>1&lt;2</em> <em>3&lt;4</em> <em>5&lt;6</em> </p><dl><dt class="tparam-name-index-invalid">9&lt;10</dt><dd class="tparam-descr-index-invalid"> bbb</dd></dl><dl><dt class="param-name-index-invalid">7&lt;8</dt><dd class="param-descr-index-invalid"> aaa </dd></dl>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="386" column="6"><Name>comment_to_html_conversion_30</Name><USR>c:@F@comment_to_html_conversion_30#</USR><Declaration>void comment_to_html_conversion_30()</Declaration><Abstract><Para> <emphasized>1&lt;2</emphasized> <emphasized>3&lt;4</emphasized> <emphasized>5&lt;6</emphasized> </Para></Abstract><TemplateParameters><Parameter><Name>9&lt;10</Name><Discussion><Para> bbb</Para></Discussion></Parameter></TemplateParameters><Parameters><Parameter><Name>7&lt;8</Name><Direction isExplicit="0">in</Direction><Discussion><Para> aaa </Para></Discussion></Parameter></Parameters></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_InlineCommand CommandName=[a] RenderEmphasized Arg[0]=1<2)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_InlineCommand CommandName=[e] RenderEmphasized Arg[0]=3<4)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_InlineCommand CommandName=[em] RenderEmphasized Arg[0]=5<6)
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace))
// CHECK-NEXT:       (CXComment_ParamCommand in implicitly ParamName=[7<8] ParamIndex=Invalid
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ aaa ])))
// CHECK-NEXT:       (CXComment_TParamCommand ParamName=[9<10] ParamPosition=Invalid
// CHECK-NEXT:         (CXComment_Paragraph
// CHECK-NEXT:           (CXComment_Text Text=[ bbb]))))]
// CHECK: annotate-comments.cpp:389:6: FunctionDecl=comment_to_html_conversion_31:{{.*}} FullCommentAsHTML=[<p class="para-brief"> \ @ &amp; $ # &lt; &gt; % &quot; . ::</p>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="389" column="6"><Name>comment_to_html_conversion_31</Name><USR>c:@F@comment_to_html_conversion_31#</USR><Declaration>void comment_to_html_conversion_31()</Declaration><Abstract><Para> \ @ &amp; $ # &lt; &gt; % &quot; . ::</Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[\])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[@])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[&])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[$])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[#])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[<])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[>])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[%])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=["])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[.])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[::])))]
// CHECK: annotate-comments.cpp:392:6: FunctionDecl=comment_to_html_conversion_32:{{.*}} FullCommentAsHTML=[<p class="para-brief"> &amp; &lt; &gt; &quot;</p>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="392" column="6"><Name>comment_to_html_conversion_32</Name><USR>c:@F@comment_to_html_conversion_32#</USR><Declaration>void comment_to_html_conversion_32()</Declaration><Abstract><Para> &amp; &lt; &gt; &quot;</Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[&])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[<])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=[>])
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_Text Text=["])))]
// CHECK: annotate-comments.cpp:395:6: FunctionDecl=comment_to_html_conversion_33:{{.*}} FullCommentAsHTML=[<p class="para-brief"> <em>0&lt;i</em></p>] FullCommentAsXML=[<Function file="{{[^"]+}}annotate-comments.cpp" line="395" column="6"><Name>comment_to_html_conversion_33</Name><USR>c:@F@comment_to_html_conversion_33#</USR><Declaration>void comment_to_html_conversion_33()</Declaration><Abstract><Para> <rawHTML><![CDATA[<em>]]></rawHTML>0&lt;i<rawHTML>&lt;/em&gt;</rawHTML></Para></Abstract></Function>]
// CHECK-NEXT:  CommentAST=[
// CHECK-NEXT:    (CXComment_FullComment
// CHECK-NEXT:       (CXComment_Paragraph
// CHECK-NEXT:         (CXComment_Text Text=[ ] IsWhitespace)
// CHECK-NEXT:         (CXComment_HTMLStartTag Name=[em])
// CHECK-NEXT:         (CXComment_Text Text=[0])
// CHECK-NEXT:         (CXComment_Text Text=[<])
// CHECK-NEXT:         (CXComment_Text Text=[i])
// CHECK-NEXT:         (CXComment_HTMLEndTag Name=[em])))]

// CHECK: annotate-comments.cpp:398:7: ClassDecl=comment_to_xml_conversion_01:{{.*}} FullCommentAsXML=[<Class file="{{[^"]+}}annotate-comments.cpp" line="398" column="7"><Name>comment_to_xml_conversion_01</Name><USR>c:@C@comment_to_xml_conversion_01</USR><Declaration>class comment_to_xml_conversion_01 {\n}</Declaration><Abstract><Para> Aaa.</Para></Abstract></Class>]
// CHECK: annotate-comments.cpp:400:3: CXXConstructor=comment_to_xml_conversion_01:{{.*}} FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}annotate-comments.cpp" line="400" column="3"><Name>comment_to_xml_conversion_01</Name><USR>c:@C@comment_to_xml_conversion_01@F@comment_to_xml_conversion_01#I#</USR><Declaration></Declaration><Parameters><Parameter><Name>aaa</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Blah blah.</Para></Discussion></Parameter></Parameters></Function>]
// CHECK: annotate-comments.cpp:403:3: CXXDestructor=~comment_to_xml_conversion_01:{{.*}} FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}annotate-comments.cpp" line="403" column="3"><Name>~comment_to_xml_conversion_01</Name><USR>c:@C@comment_to_xml_conversion_01@F@~comment_to_xml_conversion_01#</USR><Declaration>void ~comment_to_xml_conversion_01()</Declaration><Abstract><Para> Aaa.</Para></Abstract></Function>]
// CHECK: annotate-comments.cpp:406:7: CXXMethod=comment_to_xml_conversion_02:{{.*}} FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}annotate-comments.cpp" line="406" column="7"><Name>comment_to_xml_conversion_02</Name><USR>c:@C@comment_to_xml_conversion_01@F@comment_to_xml_conversion_02#I#</USR><Declaration>int comment_to_xml_conversion_02(int aaa)</Declaration><Parameters><Parameter><Name>aaa</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Blah blah.</Para></Discussion></Parameter></Parameters></Function>]
// CHECK: annotate-comments.cpp:409:14: CXXMethod=comment_to_xml_conversion_03:{{.*}} FullCommentAsXML=[<Function isClassMethod="1" file="{{[^"]+}}annotate-comments.cpp" line="409" column="14"><Name>comment_to_xml_conversion_03</Name><USR>c:@C@comment_to_xml_conversion_01@F@comment_to_xml_conversion_03#I#S</USR><Declaration>static int comment_to_xml_conversion_03(int aaa)</Declaration><Parameters><Parameter><Name>aaa</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Blah blah.</Para></Discussion></Parameter></Parameters></Function>]
// CHECK: annotate-comments.cpp:412:7: FieldDecl=comment_to_xml_conversion_04:{{.*}} FullCommentAsXML=[<Variable file="{{[^"]+}}annotate-comments.cpp" line="412" column="7"><Name>comment_to_xml_conversion_04</Name><USR>c:@C@comment_to_xml_conversion_01@FI@comment_to_xml_conversion_04</USR><Declaration>int comment_to_xml_conversion_04</Declaration><Abstract><Para> Aaa.</Para></Abstract></Variable>]
// CHECK: annotate-comments.cpp:415:14: VarDecl=comment_to_xml_conversion_05:{{.*}} FullCommentAsXML=[<Variable file="{{[^"]+}}annotate-comments.cpp" line="415" column="14"><Name>comment_to_xml_conversion_05</Name><USR>c:@C@comment_to_xml_conversion_01@comment_to_xml_conversion_05</USR><Declaration>static int comment_to_xml_conversion_05</Declaration><Abstract><Para> Aaa.</Para></Abstract></Variable>]
// CHECK: annotate-comments.cpp:418:8: CXXMethod=operator():{{.*}} FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}annotate-comments.cpp" line="418" column="8"><Name>operator()</Name><USR>c:@C@comment_to_xml_conversion_01@F@operator()#I#</USR><Declaration>void operator()(int aaa)</Declaration><Parameters><Parameter><Name>aaa</Name><Index>0</Index><Direction isExplicit="0">in</Direction><Discussion><Para> Blah blah.</Para></Discussion></Parameter></Parameters></Function>]
// CHECK: annotate-comments.cpp:421:3: CXXConversion=operator _Bool:{{.*}} FullCommentAsXML=[<Function isInstanceMethod="1" file="{{[^"]+}}annotate-comments.cpp" line="421" column="3"><Name>operator _Bool</Name><USR>c:@C@comment_to_xml_conversion_01@F@operator _Bool#</USR><Declaration>bool operator _Bool()</Declaration><Abstract><Para> Aaa.</Para></Abstract></Function>]
// CHECK: annotate-comments.cpp:424:15: TypedefDecl=comment_to_xml_conversion_06:{{.*}} FullCommentAsXML=[<Typedef file="{{[^"]+}}annotate-comments.cpp" line="424" column="15"><Name>comment_to_xml_conversion_06</Name><USR>c:annotate-comments.cpp@8505@C@comment_to_xml_conversion_01@T@comment_to_xml_conversion_06</USR><Declaration>typedef int comment_to_xml_conversion_06</Declaration><Abstract><Para> Aaa.</Para></Abstract></Typedef>]
// CHECK: annotate-comments.cpp:427:9: TypeAliasDecl=comment_to_xml_conversion_07:{{.*}} FullCommentAsXML=[<Typedef file="{{[^"]+}}annotate-comments.cpp" line="427" column="9"><Name>comment_to_xml_conversion_07</Name><USR>c:@C@comment_to_xml_conversion_01@comment_to_xml_conversion_07</USR><Declaration>using comment_to_xml_conversion_07 = int</Declaration><Abstract><Para> Aaa.</Para></Abstract></Typedef>]
// CHECK: annotate-comments.cpp:434:3: UnexposedDecl=comment_to_xml_conversion_09:{{.*}} FullCommentAsXML=[<Typedef file="{{[^"]+}}annotate-comments.cpp" line="434" column="3"><Name>comment_to_xml_conversion_09</Name><USR>c:@C@comment_to_xml_conversion_01@comment_to_xml_conversion_09</USR><Declaration>template &lt;typename T&gt; using comment_to_xml_conversion_09 = comment_to_xml_conversion_08&lt;T, int&gt;</Declaration><Abstract><Para> Aaa.</Para></Abstract></Typedef>]
// CHECK: annotate-comments.cpp:439:6: FunctionTemplate=comment_to_xml_conversion_10:{{.*}} FullCommentAsXML=[<Function templateKind="template" file="{{[^"]+}}annotate-comments.cpp" line="439" column="6"><Name>comment_to_xml_conversion_10</Name><USR>c:@FT@&gt;2#T#Tcomment_to_xml_conversion_10#t0.0#t0.1#</USR><Declaration>template &lt;typename T = int, typename U = int&gt; void comment_to_xml_conversion_10(int aaa, int bbb)template &lt;typename T, typename U&gt; void comment_to_xml_conversion_10(T aaa, U bbb)</Declaration><Abstract><Para> Aaa.</Para></Abstract></Function>]
// CHECK: annotate-comments.cpp:443:6: FunctionDecl=comment_to_xml_conversion_10:{{.*}} FullCommentAsXML=[<Function templateKind="specialization" file="{{[^"]+}}annotate-comments.cpp" line="443" column="6"><Name>comment_to_xml_conversion_10</Name><USR>c:@F@comment_to_xml_conversion_10&lt;#I#I&gt;#I#I#</USR><Declaration>void comment_to_xml_conversion_10(int aaa, int bbb)</Declaration><Abstract><Para> Aaa.</Para></Abstract></Function>]
// CHECK: annotate-comments.cpp:447:7: ClassTemplate=comment_to_xml_conversion_11:{{.*}} FullCommentAsXML=[<Class templateKind="template" file="{{[^"]+}}annotate-comments.cpp" line="447" column="7"><Name>comment_to_xml_conversion_11</Name><USR>c:@CT&gt;2#T#T@comment_to_xml_conversion_11</USR><Declaration>template &lt;typename T = int, typename U = int&gt; class comment_to_xml_conversion_11 {\n}\ntemplate &lt;typename T, typename U&gt; class comment_to_xml_conversion_11 {\n}</Declaration><Abstract><Para> Aaa.</Para></Abstract></Class>]
// CHECK: annotate-comments.cpp:451:7: ClassTemplatePartialSpecialization=comment_to_xml_conversion_11:{{.*}} FullCommentAsXML=[<Class templateKind="partialSpecialization" file="{{[^"]+}}annotate-comments.cpp" line="451" column="7"><Name>comment_to_xml_conversion_11</Name><USR>c:@CP&gt;1#T@comment_to_xml_conversion_11&gt;#t0.0#I</USR><Declaration>class comment_to_xml_conversion_11 {\n}</Declaration><Abstract><Para> Aaa.</Para></Abstract></Class>]
// CHECK: annotate-comments.cpp:455:7: ClassDecl=comment_to_xml_conversion_11:{{.*}} FullCommentAsXML=[<Class templateKind="specialization" file="{{[^"]+}}annotate-comments.cpp" line="455" column="7"><Name>comment_to_xml_conversion_11</Name><USR>c:@C@comment_to_xml_conversion_11&gt;#I#I</USR><Declaration>class comment_to_xml_conversion_11 {\n}</Declaration><Abstract><Para> Aaa.</Para></Abstract></Class>]
// CHECK: annotate-comments.cpp:458:5: VarDecl=comment_to_xml_conversion_12:{{.*}} FullCommentAsXML=[<Variable file="{{[^"]+}}annotate-comments.cpp" line="458" column="5"><Name>comment_to_xml_conversion_12</Name><USR>c:@comment_to_xml_conversion_12</USR><Declaration>int comment_to_xml_conversion_12</Declaration><Abstract><Para> Aaa.</Para></Abstract></Variable>]
// CHECK: annotate-comments.cpp:461:11: Namespace=comment_to_xml_conversion_13:{{.*}} FullCommentAsXML=[<Namespace file="{{[^"]+}}annotate-comments.cpp" line="461" column="11"><Name>comment_to_xml_conversion_13</Name><USR>c:@N@comment_to_xml_conversion_13</USR><Declaration>namespace comment_to_xml_conversion_13 {\n}</Declaration><Abstract><Para> Aaa.</Para></Abstract></Namespace>]
// CHECK: annotate-comments.cpp:463:13: Namespace=comment_to_xml_conversion_14:{{.*}} FullCommentAsXML=[<Namespace file="{{[^"]+}}annotate-comments.cpp" line="463" column="13"><Name>comment_to_xml_conversion_14</Name><USR>c:@N@comment_to_xml_conversion_13@N@comment_to_xml_conversion_14</USR><Declaration>namespace comment_to_xml_conversion_14 {\n}</Declaration><Abstract><Para> Aaa.</Para></Abstract></Namespace>]
// CHECK: annotate-comments.cpp:468:6: EnumDecl=comment_to_xml_conversion_15:{{.*}} FullCommentAsXML=[<Enum file="{{[^"]+}}annotate-comments.cpp" line="468" column="6"><Name>comment_to_xml_conversion_15</Name><USR>c:@E@comment_to_xml_conversion_15</USR><Declaration>enum comment_to_xml_conversion_15{{( : int)?}} {\n}</Declaration><Abstract><Para> Aaa.</Para></Abstract></Enum>]
// CHECK: annotate-comments.cpp:470:3: EnumConstantDecl=comment_to_xml_conversion_16:{{.*}} FullCommentAsXML=[<Variable file="{{[^"]+}}annotate-comments.cpp" line="470" column="3"><Name>comment_to_xml_conversion_16</Name><USR>c:@E@comment_to_xml_conversion_15@comment_to_xml_conversion_16</USR><Declaration>comment_to_xml_conversion_16</Declaration><Abstract><Para> Aaa.</Para></Abstract></Variable>]
// CHECK: annotate-comments.cpp:474:12: EnumDecl=comment_to_xml_conversion_17:{{.*}} FullCommentAsXML=[<Enum file="{{[^"]+}}annotate-comments.cpp" line="474" column="12"><Name>comment_to_xml_conversion_17</Name><USR>c:@E@comment_to_xml_conversion_17</USR><Declaration>enum class comment_to_xml_conversion_17 : int {\n}</Declaration><Abstract><Para> Aaa.</Para></Abstract></Enum>]
// CHECK: annotate-comments.cpp:476:3: EnumConstantDecl=comment_to_xml_conversion_18:{{.*}} FullCommentAsXML=[<Variable file="{{[^"]+}}annotate-comments.cpp" line="476" column="3"><Name>comment_to_xml_conversion_18</Name><USR>c:@E@comment_to_xml_conversion_17@comment_to_xml_conversion_18</USR><Declaration>comment_to_xml_conversion_18</Declaration><Abstract><Para> Aaa.</Para></Abstract></Variable>]
