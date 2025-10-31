//===-- SymbolDocumentationTests.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "SymbolDocumentation.h"

#include "support/Markup.h"
#include "clang/Basic/CommentOptions.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"

namespace clang {
namespace clangd {

TEST(SymbolDocumentation, DetailedDocToMarkup) {

  CommentOptions CommentOpts;

  struct Case {
    llvm::StringRef Documentation;
    llvm::StringRef ExpectedRenderEscapedMarkdown;
    llvm::StringRef ExpectedRenderMarkdown;
    llvm::StringRef ExpectedRenderPlainText;
  } Cases[] = {
      {
          "brief\n\nfoo bar",
          "foo bar",
          "foo bar",
          "foo bar",
      },
      {
          "brief\n\nfoo\nbar\n",
          "foo\nbar",
          "foo\nbar",
          "foo bar",
      },
      {
          "brief\n\nfoo\n\nbar\n",
          "foo\n\nbar",
          "foo\n\nbar",
          "foo\n\nbar",
      },
      {
          "brief\n\nfoo \\p bar baz",
          "foo `bar` baz",
          "foo `bar` baz",
          "foo bar baz",
      },
      {
          "brief\n\nfoo \\e bar baz",
          "foo \\*bar\\* baz",
          "foo *bar* baz",
          "foo *bar* baz",
      },
      {
          "brief\n\nfoo \\b bar baz",
          "foo \\*\\*bar\\*\\* baz",
          "foo **bar** baz",
          "foo **bar** baz",
      },
      {
          "brief\n\nfoo \\ref bar baz",
          "foo \\*\\*\\\\ref\\*\\* `bar` baz",
          "foo **\\ref** `bar` baz",
          "foo **\\ref** bar baz",
      },
      {
          "brief\n\nfoo @ref bar baz",
          "foo \\*\\*@ref\\*\\* `bar` baz",
          "foo **@ref** `bar` baz",
          "foo **@ref** bar baz",
      },
      {
          "\\brief this is a \\n\nbrief description",
          "",
          "",
          "",
      },
      {
          "brief\n\n\\throw exception foo",
          "\\*\\*\\\\throw\\*\\* `exception` foo",
          "**\\throw** `exception` foo",
          "**\\throw** exception foo",
      },
      {
          R"(\brief this is a brief description

\li item 1
\li item 2
\arg item 3)",
          R"(- item 1

- item 2

- item 3)",
          R"(- item 1

- item 2

- item 3)",
          R"(- item 1

- item 2

- item 3)",
      },
      {
          "brief\n\n\\defgroup mygroup this is a group\nthis is not a group "
          "description",
          "\\*\\*@defgroup\\*\\* `mygroup this is a group`\n\nthis is not a "
          "group "
          "description",
          "**@defgroup** `mygroup this is a group`\n\nthis is not a group "
          "description",
          "**@defgroup** `mygroup this is a group`\n\nthis is not a group "
          "description",
      },
      {
          R"(brief

\verbatim
this is a
verbatim block containing
some verbatim text
\endverbatim)",
          R"(\*\*@verbatim\*\*

```
this is a
verbatim block containing
some verbatim text
```

\*\*@endverbatim\*\*)",
          R"(**@verbatim**

```
this is a
verbatim block containing
some verbatim text
```

**@endverbatim**)",
          R"(**@verbatim**

this is a
verbatim block containing
some verbatim text

**@endverbatim**)",
      },
      {
          "brief\n\n@param foo this is a parameter\n@param bar this is another "
          "parameter",
          "",
          "",
          "",
      },
      {
          R"(@brief brief docs

@param foo this is a parameter

\brief another brief?

\details these are details

More description
documentation)",
          R"(\*\*\\brief\*\* another brief?

these are details

More description
documentation)",
          R"(**\brief** another brief?

these are details

More description
documentation)",
          R"(**\brief** another brief?

these are details

More description documentation)",
      },
      {
          R"(brief

<b>this is a bold text</b>
normal text<i>this is an italic text</i>
<code>this is a code block</code>)",
          R"(\<b>this is a bold text\</b>
normal text\<i>this is an italic text\</i>
\<code>this is a code block\</code>)",
          R"(\<b>this is a bold text\</b>
normal text\<i>this is an italic text\</i>
\<code>this is a code block\</code>)",
          "<b>this is a bold text</b> normal text<i>this is an italic text</i> "
          "<code>this is a code block</code>",
      },
      {"brief\n\n@note This is a note",
       R"(\*\*Note:\*\*  
This is a note)",
       R"(**Note:**  
This is a note)",
       R"(**Note:**
This is a note)"},
      {R"(brief

Paragraph 1
@note This is a note

Paragraph 2)",
       R"(Paragraph 1

\*\*Note:\*\*  
This is a note

Paragraph 2)",
       R"(Paragraph 1

**Note:**  
This is a note

Paragraph 2)",
       R"(Paragraph 1

**Note:**
This is a note

Paragraph 2)"},
      {"brief\n\n@warning This is a warning",
       R"(\*\*Warning:\*\*  
This is a warning)",
       R"(**Warning:**  
This is a warning)",
       R"(**Warning:**
This is a warning)"},
      {R"(brief

Paragraph 1
@warning This is a warning

Paragraph 2)",
       R"(Paragraph 1

\*\*Warning:\*\*  
This is a warning

Paragraph 2)",
       R"(Paragraph 1

**Warning:**  
This is a warning

Paragraph 2)",
       R"(Paragraph 1

**Warning:**
This is a warning

Paragraph 2)"},
      {R"(@note this is not treated as brief

@brief this is the brief

Another paragraph)",
       R"(\*\*Note:\*\*  
this is not treated as brief

Another paragraph)",
       R"(**Note:**  
this is not treated as brief

Another paragraph)",
       R"(**Note:**
this is not treated as brief

Another paragraph)"},
      {R"(
@brief Some brief
)",
       "", "", ""},
      {R"(
Some brief
)",
       "", "", ""},
  };
  for (const auto &C : Cases) {
    markup::Document Doc;
    SymbolDocCommentVisitor SymbolDoc(C.Documentation, CommentOpts);

    SymbolDoc.detailedDocToMarkup(Doc);

    EXPECT_EQ(Doc.asPlainText(), C.ExpectedRenderPlainText);
    EXPECT_EQ(Doc.asMarkdown(), C.ExpectedRenderMarkdown);
    EXPECT_EQ(Doc.asEscapedMarkdown(), C.ExpectedRenderEscapedMarkdown);
  }
}

TEST(SymbolDocumentation, RetvalCommand) {

  CommentOptions CommentOpts;

  struct Case {
    llvm::StringRef Documentation;
    llvm::StringRef ExpectedRenderEscapedMarkdown;
    llvm::StringRef ExpectedRenderMarkdown;
    llvm::StringRef ExpectedRenderPlainText;
  } Cases[] = {
      {"@retval", "", "", ""},
      {R"(@retval MyReturnVal
@retval MyOtherReturnVal)",
       R"(- `MyReturnVal`
- `MyOtherReturnVal`)",
       R"(- `MyReturnVal`
- `MyOtherReturnVal`)",
       R"(- MyReturnVal
- MyOtherReturnVal)"},
      {R"(@retval MyReturnVal if foo
@retval MyOtherReturnVal if bar)",
       R"(- `MyReturnVal` - if foo
- `MyOtherReturnVal` - if bar)",
       R"(- `MyReturnVal` - if foo
- `MyOtherReturnVal` - if bar)",
       R"(- MyReturnVal - if foo
- MyOtherReturnVal - if bar)"},
  };
  for (const auto &C : Cases) {
    markup::Document Doc;
    SymbolDocCommentVisitor SymbolDoc(C.Documentation, CommentOpts);

    SymbolDoc.retvalsToMarkup(Doc);

    EXPECT_EQ(Doc.asPlainText(), C.ExpectedRenderPlainText);
    EXPECT_EQ(Doc.asMarkdown(), C.ExpectedRenderMarkdown);
    EXPECT_EQ(Doc.asEscapedMarkdown(), C.ExpectedRenderEscapedMarkdown);
  }
}

TEST(SymbolDocumentation, DoxygenCodeBlocks) {
  CommentOptions CommentOpts;

  struct Case {
    llvm::StringRef Documentation;
    llvm::StringRef ExpectedRenderEscapedMarkdown;
    llvm::StringRef ExpectedRenderMarkdown;
    llvm::StringRef ExpectedRenderPlainText;
  } Cases[] = {
      {R"(@code
int code() { return 0; }
@endcode
@code{.cpp}
int code_lang() { return 0; }
@endcode
@code{.c++}
int code_lang_plus() { return 0; }
@endcode
@code{.py}
class A:
    pass
@endcode
@code{nolang}
class B:
    pass
@endcode)",
       R"(```
int code() { return 0; }
```

```cpp
int code_lang() { return 0; }
```

```c++
int code_lang_plus() { return 0; }
```

```py
class A:
    pass
```

```nolang
class B:
    pass
```)",
       R"(```
int code() { return 0; }
```

```cpp
int code_lang() { return 0; }
```

```c++
int code_lang_plus() { return 0; }
```

```py
class A:
    pass
```

```nolang
class B:
    pass
```)",
       R"(int code() { return 0; }

int code_lang() { return 0; }

int code_lang_plus() { return 0; }

class A:
    pass

class B:
    pass)"},
  };
  for (const auto &C : Cases) {
    markup::Document Doc;
    SymbolDocCommentVisitor SymbolDoc(C.Documentation, CommentOpts);

    SymbolDoc.detailedDocToMarkup(Doc);

    EXPECT_EQ(Doc.asPlainText(), C.ExpectedRenderPlainText);
    EXPECT_EQ(Doc.asMarkdown(), C.ExpectedRenderMarkdown);
    EXPECT_EQ(Doc.asEscapedMarkdown(), C.ExpectedRenderEscapedMarkdown);
  }
}

TEST(SymbolDocumentation, MarkdownCodeBlocks) {
  CommentOptions CommentOpts;

  struct Case {
    llvm::StringRef Documentation;
    llvm::StringRef ExpectedRenderEscapedMarkdown;
    llvm::StringRef ExpectedRenderMarkdown;
    llvm::StringRef ExpectedRenderPlainText;
  } Cases[] = {
      {R"(```
int backticks() { return 0; }
```
```cpp
int backticks_lang() { return 0; }
```
```c++
int backticks_lang_plus() { return 0; }
```
~~~
int tilde() { return 0; }
~~~
~~~~~~~~~~~~~~~~~~~~~~~~
int tilde_many() { return 0; }
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~{.c++}
int tilde_many_lang() { return 0; }
~~~~~~~~~~~~~~~~~~~~~~~~
```py
class A:
    pass
```
```python
class B:
    pass
```
~~~{.python}
class C:
    pass
~~~
)",
       R"(```
int backticks() { return 0; }
```

```cpp
int backticks_lang() { return 0; }
```

```c++
int backticks_lang_plus() { return 0; }
```

```
int tilde() { return 0; }
```

```
int tilde_many() { return 0; }
```

```c++
int tilde_many_lang() { return 0; }
```

```py
class A:
    pass
```

```python
class B:
    pass
```

```python
class C:
    pass
```)",
       R"(```
int backticks() { return 0; }
```

```cpp
int backticks_lang() { return 0; }
```

```c++
int backticks_lang_plus() { return 0; }
```

```
int tilde() { return 0; }
```

```
int tilde_many() { return 0; }
```

```c++
int tilde_many_lang() { return 0; }
```

```py
class A:
    pass
```

```python
class B:
    pass
```

```python
class C:
    pass
```)",
       R"(int backticks() { return 0; }

int backticks_lang() { return 0; }

int backticks_lang_plus() { return 0; }

int tilde() { return 0; }

int tilde_many() { return 0; }

int tilde_many_lang() { return 0; }

class A:
    pass

class B:
    pass

class C:
    pass)"},
      {R"(```
// this code block is missing end backticks

)",
       R"(```
// this code block is missing end backticks
```)",
       R"(```
// this code block is missing end backticks
```)",
       R"(// this code block is missing end backticks)"},
  };
  for (const auto &C : Cases) {
    markup::Document Doc;
    SymbolDocCommentVisitor SymbolDoc(C.Documentation, CommentOpts);

    SymbolDoc.detailedDocToMarkup(Doc);

    EXPECT_EQ(Doc.asPlainText(), C.ExpectedRenderPlainText);
    EXPECT_EQ(Doc.asMarkdown(), C.ExpectedRenderMarkdown);
    EXPECT_EQ(Doc.asEscapedMarkdown(), C.ExpectedRenderEscapedMarkdown);
  }
}

TEST(SymbolDocumentation, MarkdownCodeBlocksSeparation) {
  CommentOptions CommentOpts;

  struct Case {
    llvm::StringRef Documentation;
    llvm::StringRef ExpectedRenderEscapedMarkdown;
    llvm::StringRef ExpectedRenderMarkdown;
    llvm::StringRef ExpectedRenderPlainText;
  } Cases[] = {
      {R"(@note Show that code blocks are correctly separated
```
/// Without the markdown preprocessing, this line and the line above would be part of the @note paragraph.

/// With preprocessing, the code block is correctly separated from the @note paragraph.
/// Also note that without preprocessing, all doxygen commands inside code blocks, like @p would be incorrectly interpreted.
int function() { return 0; }
```)",
       R"(\*\*Note:\*\*  
Show that code blocks are correctly separated

```
/// Without the markdown preprocessing, this line and the line above would be part of the @note paragraph.

/// With preprocessing, the code block is correctly separated from the @note paragraph.
/// Also note that without preprocessing, all doxygen commands inside code blocks, like @p would be incorrectly interpreted.
int function() { return 0; }
```)",
       R"(**Note:**  
Show that code blocks are correctly separated

```
/// Without the markdown preprocessing, this line and the line above would be part of the @note paragraph.

/// With preprocessing, the code block is correctly separated from the @note paragraph.
/// Also note that without preprocessing, all doxygen commands inside code blocks, like @p would be incorrectly interpreted.
int function() { return 0; }
```)",
       R"(**Note:**
Show that code blocks are correctly separated

/// Without the markdown preprocessing, this line and the line above would be part of the @note paragraph.

/// With preprocessing, the code block is correctly separated from the @note paragraph.
/// Also note that without preprocessing, all doxygen commands inside code blocks, like @p would be incorrectly interpreted.
int function() { return 0; })"},
      {R"(@note Show that code blocks are correctly separated
~~~~~~~~~
/// Without the markdown preprocessing, this line and the line above would be part of the @note paragraph.

/// With preprocessing, the code block is correctly separated from the @note paragraph.
/// Also note that without preprocessing, all doxygen commands inside code blocks, like @p would be incorrectly interpreted.
int function() { return 0; }
~~~~~~~~~)",
       R"(\*\*Note:\*\*  
Show that code blocks are correctly separated

```
/// Without the markdown preprocessing, this line and the line above would be part of the @note paragraph.

/// With preprocessing, the code block is correctly separated from the @note paragraph.
/// Also note that without preprocessing, all doxygen commands inside code blocks, like @p would be incorrectly interpreted.
int function() { return 0; }
```)",
       R"(**Note:**  
Show that code blocks are correctly separated

```
/// Without the markdown preprocessing, this line and the line above would be part of the @note paragraph.

/// With preprocessing, the code block is correctly separated from the @note paragraph.
/// Also note that without preprocessing, all doxygen commands inside code blocks, like @p would be incorrectly interpreted.
int function() { return 0; }
```)",
       R"(**Note:**
Show that code blocks are correctly separated

/// Without the markdown preprocessing, this line and the line above would be part of the @note paragraph.

/// With preprocessing, the code block is correctly separated from the @note paragraph.
/// Also note that without preprocessing, all doxygen commands inside code blocks, like @p would be incorrectly interpreted.
int function() { return 0; })"},
  };
  for (const auto &C : Cases) {
    markup::Document Doc;
    SymbolDocCommentVisitor SymbolDoc(C.Documentation, CommentOpts);

    SymbolDoc.detailedDocToMarkup(Doc);

    EXPECT_EQ(Doc.asPlainText(), C.ExpectedRenderPlainText);
    EXPECT_EQ(Doc.asMarkdown(), C.ExpectedRenderMarkdown);
    EXPECT_EQ(Doc.asEscapedMarkdown(), C.ExpectedRenderEscapedMarkdown);
  }
}

TEST(SymbolDocumentation, MarkdownCodeSpans) {
  CommentOptions CommentOpts;

  struct Case {
    llvm::StringRef Documentation;
    llvm::StringRef ExpectedRenderEscapedMarkdown;
    llvm::StringRef ExpectedRenderMarkdown;
    llvm::StringRef ExpectedRenderPlainText;
  } Cases[] = {
      {R"(`this is a code span with @p and \c inside`)",
       R"(\`this is a code span with @p and \\c inside\`)",
       R"(`this is a code span with @p and \c inside`)",
       R"(`this is a code span with @p and \c inside`)"},
      {R"(<escaped> `<not-escaped>`)", R"(\<escaped> \`\<not-escaped>\`)",
       R"(\<escaped> `<not-escaped>`)", R"(<escaped> `<not-escaped>`)"},
      {R"(<escaped> \`<escaped> doxygen commands not parsed @p, \c, @note, \warning \`)",
       R"(\<escaped> \\\`\<escaped> doxygen commands not parsed @p, \\c, @note, \\warning \\\`)",
       R"(\<escaped> \`\<escaped> doxygen commands not parsed @p, \c, @note, \warning \`)",
       R"(<escaped> \`<escaped> doxygen commands not parsed @p, \c, @note, \warning \`)"},
      {R"(`multi
line
\c span`)",
       R"(\`multi
line
\\c span\`)",
       R"(`multi
line
\c span`)",
       R"(`multi line
\c span`)"},
  };
  for (const auto &C : Cases) {
    markup::Document Doc;
    SymbolDocCommentVisitor SymbolDoc(C.Documentation, CommentOpts);

    SymbolDoc.briefToMarkup(Doc.addParagraph());

    EXPECT_EQ(Doc.asPlainText(), C.ExpectedRenderPlainText);
    EXPECT_EQ(Doc.asMarkdown(), C.ExpectedRenderMarkdown);
    EXPECT_EQ(Doc.asEscapedMarkdown(), C.ExpectedRenderEscapedMarkdown);
  }
}

} // namespace clangd
} // namespace clang
