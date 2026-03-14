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

#define MYMAC(x,y)
/**
 * Aaa. IS_DOXYGEN_START IS_DOXYGEN_END
 */
MYMAC(0,0)
void isdoxy54(int);

#endif

// RUN: rm -rf %t
// RUN: mkdir %t

// Check that we serialize comment source locations properly.
// RUN: %clang_cc1 -x c++ -std=c++11 -emit-pch -o %t/out.pch %s
// RUN: %clang_cc1 -x c++ -std=c++11 -include-pch %t/out.pch -fsyntax-only %s

// RUN: c-index-test -write-pch %t/out.pch -x c++ -std=c++11 %s
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

// These CHECK lines are not located near the code on purpose.  This test
// checks that documentation comments are attached to declarations correctly.
// Adding a non-documentation comment with CHECK line between every two
// documentation comments will only test a single code path.
//
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
// CHECK: annotate-comments.cpp:248:6: FunctionDecl=isdoxy53:{{.*}} BriefComment=[Aaa. IS_DOXYGEN_START IS_DOXYGEN_END]
// CHECK: annotate-comments.cpp:255:6: FunctionDecl=isdoxy54:{{.*}} BriefComment=[Aaa. IS_DOXYGEN_START IS_DOXYGEN_END]
