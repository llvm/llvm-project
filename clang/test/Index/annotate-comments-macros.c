// Run lines are sensitive to line numbers and come below the code.

/// Greeting count.
#define HELLO 1

#define BARE 2

/**
 * \brief Add two numbers.
 */
#define SUM(x, y) ((x) + (y))

#define TRAILING 3 ///< Trailing macro doc.

/// First definition.
#define REDEF 1
#undef REDEF
/// Second definition.
#define REDEF 2

// RUN: c-index-test -test-load-source all %s | FileCheck %s

// Documented object-like macro picks up the preceding `///` comment.
// CHECK: annotate-comments-macros.c:4:9: macro definition=HELLO RawComment=[/// Greeting count.] RawCommentRange=[3:1 - 3:20] BriefComment=[Greeting count.]

// A macro without any preceding doc comment must not have a RawComment.
// CHECK: annotate-comments-macros.c:6:9: macro definition=BARE Extent=

// Function-like macros are documented just like object-like ones.
// CHECK: annotate-comments-macros.c:11:9: macro definition=SUM RawComment=[/**\n * \brief Add two numbers.\n */] RawCommentRange=[8:1 - 10:4] BriefComment=[Add two numbers.]

// Trailing `///<` comments attach to macros, mirroring fields/enumerators.
// CHECK: annotate-comments-macros.c:13:9: macro definition=TRAILING RawComment=[///< Trailing macro doc.] RawCommentRange=[13:20 - 13:44] BriefComment=[Trailing macro doc.]

// Each redefinition of a macro carries its own preceding doc comment.
// CHECK: annotate-comments-macros.c:16:9: macro definition=REDEF RawComment=[/// First definition.] RawCommentRange=[15:1 - 15:22] BriefComment=[First definition.]
// CHECK: annotate-comments-macros.c:19:9: macro definition=REDEF RawComment=[/// Second definition.] RawCommentRange=[18:1 - 18:23] BriefComment=[Second definition.]
