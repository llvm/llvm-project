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

TEST(KernelDoc, ParseBasic) {
  KernelDocInfo Info =
      parseKernelDoc("kfree() - Free previously allocated memory\n"
                     "@objp: pointer returned by kmalloc()\n"
                     "\n"
                     "Don't free memory not originally allocated by kmalloc()\n"
                     "or you will run into trouble.\n"
                     "\n"
                     "Context: May be called from interrupt context.\n"
                     "Return: Nothing.\n");

  EXPECT_EQ(Info.Brief, "Free previously allocated memory");
  ASSERT_EQ(Info.Params.size(), 1u);
  EXPECT_EQ(Info.Params[0].Name, "objp");
  EXPECT_EQ(Info.Params[0].Description, "pointer returned by kmalloc()");
  ASSERT_EQ(Info.Description.size(), 1u);
  EXPECT_EQ(Info.Description[0].Text,
            "Don't free memory not originally allocated by kmalloc() "
            "or you will run into trouble.");
  ASSERT_EQ(Info.Sections.size(), 1u);
  EXPECT_EQ(Info.Sections[0].Name, "Context");
  EXPECT_EQ(Info.Sections[0].Description,
            "May be called from interrupt context.");
  EXPECT_EQ(Info.Returns, "Nothing.");
}

TEST(KernelDoc, ParseBriefOnly) {
  KernelDocInfo Info = parseKernelDoc("my_func() - Just a brief\n");

  EXPECT_EQ(Info.Brief, "Just a brief");
  EXPECT_TRUE(Info.Params.empty());
  EXPECT_TRUE(Info.Returns.empty());
  EXPECT_TRUE(Info.Sections.empty());
  EXPECT_TRUE(Info.Description.empty());
}

TEST(KernelDoc, ParseParamContinuation) {
  KernelDocInfo Info = parseKernelDoc("my_func() - Brief\n"
                                      "@buf: pointer to the buffer that will\n"
                                      "      receive the data\n"
                                      "@len: length of the buffer\n");

  ASSERT_EQ(Info.Params.size(), 2u);
  EXPECT_EQ(Info.Params[0].Name, "buf");
  EXPECT_EQ(Info.Params[0].Description,
            "pointer to the buffer that will receive the data");
  EXPECT_EQ(Info.Params[1].Name, "len");
  EXPECT_EQ(Info.Params[1].Description, "length of the buffer");
}

TEST(KernelDoc, ParseVariadicParam) {
  KernelDocInfo Info = parseKernelDoc("printk() - Print a kernel message\n"
                                      "@fmt: format string\n"
                                      "@...: variable arguments\n");

  ASSERT_EQ(Info.Params.size(), 2u);
  EXPECT_EQ(Info.Params[0].Name, "fmt");
  EXPECT_EQ(Info.Params[1].Name, "...");
  EXPECT_EQ(Info.Params[1].Description, "variable arguments");
}

TEST(KernelDoc, ParseReturns) {
  KernelDocInfo Info = parseKernelDoc(
      "alloc_pages() - Allocate pages\n"
      "@gfp: allocation flags\n"
      "\n"
      "Returns: A pointer to the first page or %NULL on failure.\n");

  EXPECT_EQ(Info.Returns, "A pointer to the first page or %NULL on failure.");
}

TEST(KernelDoc, ParseReturnsContinuation) {
  KernelDocInfo Info =
      parseKernelDoc("do_something() - Do it\n"
                     "\n"
                     "Return: %0 on success, negative error code\n"
                     "        on failure.\n");

  EXPECT_EQ(Info.Returns, "%0 on success, negative error code on failure.");
}

TEST(KernelDoc, ParseContext) {
  KernelDocInfo Info = parseKernelDoc(
      "mutex_lock() - Acquire a mutex\n"
      "@lock: the mutex to be acquired\n"
      "\n"
      "Context: Process context. May sleep if @lock is contended.\n");

  ASSERT_EQ(Info.Sections.size(), 1u);
  EXPECT_EQ(Info.Sections[0].Name, "Context");
  EXPECT_EQ(Info.Sections[0].Description,
            "Process context. May sleep if @lock is contended.");
}

TEST(KernelDoc, ParseCodeBlock) {
  KernelDocInfo Info = parseKernelDoc("example() - Example function\n"
                                      "\n"
                                      "Usage:\n"
                                      "\n"
                                      "```c\n"
                                      "example();\n"
                                      "```\n");

  bool HasCode = false;
  for (const auto &Block : Info.Description) {
    if (Block.BlockKind == KernelDocDescriptionBlock::Code) {
      HasCode = true;
      EXPECT_EQ(Block.Text, "example();");
      EXPECT_EQ(Block.Language, "c");
    }
  }
  EXPECT_TRUE(HasCode);
}

TEST(KernelDoc, ParseNoBriefDash) {
  KernelDocInfo Info =
      parseKernelDoc("This is a plain brief without function name pattern\n"
                     "@x: param\n");

  EXPECT_EQ(Info.Brief, "This is a plain brief without function name pattern");
  ASSERT_EQ(Info.Params.size(), 1u);
  EXPECT_EQ(Info.Params[0].Name, "x");
}

TEST(KernelDoc, RenderToMarkup) {
  KernelDocInfo Info;
  Info.Brief = "Free previously allocated memory";
  Info.Params.push_back({"objp", "pointer returned by kmalloc()"});
  Info.Returns = "Nothing.";
  Info.Sections.push_back({"Context", "May be called from interrupt context."});
  Info.Description.push_back(
      {KernelDocDescriptionBlock::Paragraph,
       "Don't free memory not originally allocated by kmalloc().", ""});

  markup::Document Doc;
  renderKernelDocToMarkup(Info, Doc);
  std::string Rendered = Doc.asMarkdown();

  EXPECT_NE(Rendered.find("Free previously allocated memory"),
            std::string::npos);
  EXPECT_NE(Rendered.find("`objp`"), std::string::npos);
  EXPECT_NE(Rendered.find("kmalloc()"), std::string::npos);
  EXPECT_NE(Rendered.find("### Parameters"), std::string::npos);
  EXPECT_NE(Rendered.find("### Returns"), std::string::npos);
  EXPECT_NE(Rendered.find("### Context"), std::string::npos);
}

TEST(KernelDoc, InlineMarkup) {
  KernelDocInfo Info;
  Info.Brief = "Use %NULL and &struct device and @param and func()";

  markup::Document Doc;
  renderKernelDocToMarkup(Info, Doc);
  std::string Rendered = Doc.asMarkdown();

  EXPECT_NE(Rendered.find("`NULL`"), std::string::npos);
  EXPECT_NE(Rendered.find("`struct device`"), std::string::npos);
  EXPECT_NE(Rendered.find("`param`"), std::string::npos);
  EXPECT_NE(Rendered.find("`func()`"), std::string::npos);
}

TEST(KernelDoc, InlineMarkupStructMember) {
  KernelDocInfo Info;
  Info.Brief = "Access &device->name field";

  markup::Document Doc;
  renderKernelDocToMarkup(Info, Doc);
  std::string Rendered = Doc.asMarkdown();

  EXPECT_NE(Rendered.find("`device->name`"), std::string::npos);
}

TEST(KernelDoc, InlineMarkupDoubleTick) {
  KernelDocInfo Info;
  Info.Brief = "Use ``literal text`` in docs";

  markup::Document Doc;
  renderKernelDocToMarkup(Info, Doc);
  std::string Rendered = Doc.asMarkdown();

  EXPECT_NE(Rendered.find("`literal text`"), std::string::npos);
}

TEST(KernelDoc, ParseNegativeErrno) {
  KernelDocInfo Info =
      parseKernelDoc("do_something() - Do it\n"
                     "\n"
                     "Return: %0 on success, %-ENOMEM or %-1 on failure.\n");

  EXPECT_EQ(Info.Returns, "%0 on success, %-ENOMEM or %-1 on failure.");

  markup::Document Doc;
  renderKernelDocToMarkup(Info, Doc);
  std::string Rendered = Doc.asMarkdown();

  EXPECT_NE(Rendered.find("`0`"), std::string::npos);
  EXPECT_NE(Rendered.find("`-ENOMEM`"), std::string::npos);
  EXPECT_NE(Rendered.find("`-1`"), std::string::npos);
}

TEST(KernelDoc, ParseMultiLineBrief) {
  KernelDocInfo Info = parseKernelDoc("func() - Allocate and initialize\n"
                                      "         a frobnicator for the device.\n"
                                      "@dev: the target device\n");

  EXPECT_EQ(Info.Brief,
            "Allocate and initialize a frobnicator for the device.");
  ASSERT_EQ(Info.Params.size(), 1u);
  EXPECT_EQ(Info.Params[0].Name, "dev");
}

TEST(KernelDoc, ParseMultiLineBriefBlankEnd) {
  KernelDocInfo Info = parseKernelDoc("func() - A long brief\n"
                                      "         that ends with a blank line.\n"
                                      "\n"
                                      "Description paragraph.\n");

  EXPECT_EQ(Info.Brief, "A long brief that ends with a blank line.");
  ASSERT_EQ(Info.Description.size(), 1u);
  EXPECT_EQ(Info.Description[0].Text, "Description paragraph.");
}

TEST(KernelDoc, ParseNestedStructMember) {
  KernelDocInfo Info = parseKernelDoc("struct outer - An outer struct\n"
                                      "@foo: simple member\n"
                                      "@bar.baz: nested member\n"
                                      "@bar.baz.qux: deeply nested member\n");

  ASSERT_EQ(Info.Params.size(), 3u);
  EXPECT_EQ(Info.Params[0].Name, "foo");
  EXPECT_EQ(Info.Params[1].Name, "bar.baz");
  EXPECT_EQ(Info.Params[1].Description, "nested member");
  EXPECT_EQ(Info.Params[2].Name, "bar.baz.qux");
  EXPECT_EQ(Info.Params[2].Description, "deeply nested member");
}

TEST(KernelDoc, InlineMarkupEnumTypedefUnion) {
  KernelDocInfo Info;
  Info.Brief = "See &enum color and &typedef handler_t and &union data";

  markup::Document Doc;
  renderKernelDocToMarkup(Info, Doc);
  std::string Rendered = Doc.asMarkdown();

  EXPECT_NE(Rendered.find("`enum color`"), std::string::npos);
  EXPECT_NE(Rendered.find("`typedef handler_t`"), std::string::npos);
  EXPECT_NE(Rendered.find("`union data`"), std::string::npos);
}

TEST(KernelDoc, InlineMarkupDotMember) {
  KernelDocInfo Info;
  Info.Brief = "Access &device.name field";

  markup::Document Doc;
  renderKernelDocToMarkup(Info, Doc);
  std::string Rendered = Doc.asMarkdown();

  EXPECT_NE(Rendered.find("`device.name`"), std::string::npos);
}

TEST(KernelDoc, ParseEmptyParamDescription) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "@x:\n"
                                      "@y: has description\n");

  ASSERT_EQ(Info.Params.size(), 2u);
  EXPECT_EQ(Info.Params[0].Name, "x");
  EXPECT_EQ(Info.Params[0].Description, "");
  EXPECT_EQ(Info.Params[1].Name, "y");
  EXPECT_EQ(Info.Params[1].Description, "has description");
}

TEST(KernelDoc, ParseMultipleDescriptionParagraphs) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "@x: param\n"
                                      "\n"
                                      "First paragraph of description.\n"
                                      "\n"
                                      "Second paragraph of description.\n");

  ASSERT_EQ(Info.Description.size(), 2u);
  EXPECT_EQ(Info.Description[0].Text, "First paragraph of description.");
  EXPECT_EQ(Info.Description[1].Text, "Second paragraph of description.");
}

TEST(KernelDoc, ParseCodeBlockNoLanguage) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "\n"
                                      "```\n"
                                      "some_code();\n"
                                      "```\n");

  bool HasCode = false;
  for (const auto &Block : Info.Description) {
    if (Block.BlockKind == KernelDocDescriptionBlock::Code) {
      HasCode = true;
      EXPECT_EQ(Block.Text, "some_code();");
      EXPECT_EQ(Block.Language, "");
    }
  }
  EXPECT_TRUE(HasCode);
}

TEST(KernelDoc, ParseUnclosedFencedCodeBlock) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "\n"
                                      "```c\n"
                                      "code_here();\n");

  bool HasCode = false;
  for (const auto &Block : Info.Description) {
    if (Block.BlockKind == KernelDocDescriptionBlock::Code) {
      HasCode = true;
      EXPECT_EQ(Block.Text, "code_here();");
      EXPECT_EQ(Block.Language, "c");
    }
  }
  EXPECT_TRUE(HasCode);
}

TEST(KernelDoc, InlineMarkupInParamDescription) {
  KernelDocInfo Info;
  Info.Brief = "Do something";
  Info.Params.push_back({"buf", "pointer to &struct page returned by alloc()"});

  markup::Document Doc;
  renderKernelDocToMarkup(Info, Doc);
  std::string Rendered = Doc.asMarkdown();

  EXPECT_NE(Rendered.find("`struct page`"), std::string::npos);
  EXPECT_NE(Rendered.find("`alloc()`"), std::string::npos);
}

TEST(KernelDoc, InlineMarkupInSectionDescription) {
  KernelDocInfo Info;
  Info.Brief = "Do something";
  Info.Sections.push_back(
      {"Context", "Caller must hold @lock and not be in %IRQ context."});

  markup::Document Doc;
  renderKernelDocToMarkup(Info, Doc);
  std::string Rendered = Doc.asMarkdown();

  EXPECT_NE(Rendered.find("`lock`"), std::string::npos);
  EXPECT_NE(Rendered.find("`IRQ`"), std::string::npos);
}

TEST(KernelDoc, ParseTildeFencedCodeBlock) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "\n"
                                      "~~~c\n"
                                      "int x = 42;\n"
                                      "~~~\n");

  bool HasCode = false;
  for (const auto &Block : Info.Description) {
    if (Block.BlockKind == KernelDocDescriptionBlock::Code) {
      HasCode = true;
      EXPECT_EQ(Block.Text, "int x = 42;");
      EXPECT_EQ(Block.Language, "c");
    }
  }
  EXPECT_TRUE(HasCode);
}

TEST(KernelDoc, ParseDescriptionHeader) {
  KernelDocInfo Info =
      parseKernelDoc("func() - Brief\n"
                     "@x: param\n"
                     "\n"
                     "Description: The detailed description here.\n");

  EXPECT_EQ(Info.Brief, "Brief");
  ASSERT_EQ(Info.Params.size(), 1u);
  ASSERT_EQ(Info.Description.size(), 1u);
  EXPECT_EQ(Info.Description[0].Text, "The detailed description here.");
}

TEST(KernelDoc, ParseDescriptionHeaderMultiParagraph) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "@x: param\n"
                                      "\n"
                                      "Description: First paragraph.\n"
                                      "\n"
                                      "Second paragraph.\n");

  ASSERT_EQ(Info.Description.size(), 2u);
  EXPECT_EQ(Info.Description[0].Text, "First paragraph.");
  EXPECT_EQ(Info.Description[1].Text, "Second paragraph.");
}

TEST(KernelDoc, ParseDescriptionHeaderStripped) {
  KernelDocInfo Info =
      parseKernelDoc("func() - Brief\n"
                     "@x: param\n"
                     "\n"
                     "Description:\n"
                     "The description follows on the next line.\n");

  ASSERT_EQ(Info.Description.size(), 1u);
  EXPECT_EQ(Info.Description[0].Text,
            "The description follows on the next line.");
}

TEST(KernelDoc, ParseNoteSection) {
  KernelDocInfo Info = parseKernelDoc(
      "func() - Brief\n"
      "@x: param\n"
      "\n"
      "Note: This function should only be called with interrupts disabled.\n");

  ASSERT_EQ(Info.Sections.size(), 1u);
  EXPECT_EQ(Info.Sections[0].Name, "Note");
  EXPECT_EQ(Info.Sections[0].Description,
            "This function should only be called with interrupts disabled.");
}

TEST(KernelDoc, ParseNoteContinuation) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "\n"
                                      "Note: This is important\n"
                                      "      and spans multiple lines.\n");

  ASSERT_EQ(Info.Sections.size(), 1u);
  EXPECT_EQ(Info.Sections[0].Name, "Note");
  EXPECT_EQ(Info.Sections[0].Description,
            "This is important and spans multiple lines.");
}

TEST(KernelDoc, RenderNote) {
  KernelDocInfo Info;
  Info.Brief = "Do something";
  Info.Sections.push_back({"Note", "Only call from process context."});

  markup::Document Doc;
  renderKernelDocToMarkup(Info, Doc);
  std::string Rendered = Doc.asMarkdown();

  EXPECT_NE(Rendered.find("### Note"), std::string::npos);
  EXPECT_NE(Rendered.find("Only call from process context."),
            std::string::npos);
}

TEST(KernelDoc, ParseWarningSection) {
  KernelDocInfo Info =
      parseKernelDoc("func() - Brief\n"
                     "@x: param\n"
                     "\n"
                     "Warning: This function is not thread-safe.\n");

  ASSERT_EQ(Info.Sections.size(), 1u);
  EXPECT_EQ(Info.Sections[0].Name, "Warning");
  EXPECT_EQ(Info.Sections[0].Description, "This function is not thread-safe.");
}

TEST(KernelDoc, ParseLockingSection) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "@lock: the mutex\n"
                                      "\n"
                                      "Locking: Caller must hold @lock.\n");

  ASSERT_EQ(Info.Sections.size(), 1u);
  EXPECT_EQ(Info.Sections[0].Name, "Locking");
  EXPECT_EQ(Info.Sections[0].Description, "Caller must hold @lock.");
}

TEST(KernelDoc, ParseMultipleSections) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "@x: param\n"
                                      "\n"
                                      "Context: Process context. May sleep.\n"
                                      "Note: Only valid after initialization.\n"
                                      "Warning: Not thread-safe.\n");

  ASSERT_EQ(Info.Sections.size(), 3u);
  EXPECT_EQ(Info.Sections[0].Name, "Context");
  EXPECT_EQ(Info.Sections[0].Description, "Process context. May sleep.");
  EXPECT_EQ(Info.Sections[1].Name, "Note");
  EXPECT_EQ(Info.Sections[1].Description, "Only valid after initialization.");
  EXPECT_EQ(Info.Sections[2].Name, "Warning");
  EXPECT_EQ(Info.Sections[2].Description, "Not thread-safe.");
}

TEST(KernelDoc, ParseSectionWithContinuation) {
  KernelDocInfo Info =
      parseKernelDoc("func() - Brief\n"
                     "\n"
                     "Context: Process context.\n"
                     "         May sleep if lock is contended.\n"
                     "Warning: Do not call from interrupt context\n"
                     "         or atomic sections.\n");

  ASSERT_EQ(Info.Sections.size(), 2u);
  EXPECT_EQ(Info.Sections[0].Name, "Context");
  EXPECT_EQ(Info.Sections[0].Description,
            "Process context. May sleep if lock is contended.");
  EXPECT_EQ(Info.Sections[1].Name, "Warning");
  EXPECT_EQ(Info.Sections[1].Description,
            "Do not call from interrupt context or atomic sections.");
}

TEST(KernelDoc, RenderMultipleSections) {
  KernelDocInfo Info;
  Info.Brief = "Do something";
  Info.Sections.push_back({"Context", "Process context."});
  Info.Sections.push_back({"Warning", "Not thread-safe."});

  markup::Document Doc;
  renderKernelDocToMarkup(Info, Doc);
  std::string Rendered = Doc.asMarkdown();

  EXPECT_NE(Rendered.find("### Context"), std::string::npos);
  EXPECT_NE(Rendered.find("Process context."), std::string::npos);
  EXPECT_NE(Rendered.find("### Warning"), std::string::npos);
  EXPECT_NE(Rendered.find("Not thread-safe."), std::string::npos);
}

TEST(KernelDoc, InlineMarkupEnvVar) {
  KernelDocInfo Info;
  Info.Brief = "Uses $HOME and $PATH_INFO for lookup";

  markup::Document Doc;
  renderKernelDocToMarkup(Info, Doc);
  std::string Rendered = Doc.asMarkdown();

  EXPECT_NE(Rendered.find("`$HOME`"), std::string::npos);
  EXPECT_NE(Rendered.find("`$PATH_INFO`"), std::string::npos);
}

TEST(KernelDoc, ParseIndentedCodeBlock) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "\n"
                                      "Example::\n"
                                      "\n"
                                      "    int x = func();\n"
                                      "    use(x);\n"
                                      "\n"
                                      "More text.\n");

  EXPECT_EQ(Info.Brief, "Brief");
  bool HasParagraph = false, HasCode = false, HasMore = false;
  for (const auto &Block : Info.Description) {
    if (Block.BlockKind == KernelDocDescriptionBlock::Paragraph) {
      if (Block.Text == "Example:")
        HasParagraph = true;
      if (Block.Text == "More text.")
        HasMore = true;
    }
    if (Block.BlockKind == KernelDocDescriptionBlock::Code) {
      HasCode = true;
      EXPECT_EQ(Block.Text, "int x = func();\nuse(x);");
    }
  }
  EXPECT_TRUE(HasParagraph);
  EXPECT_TRUE(HasCode);
  EXPECT_TRUE(HasMore);
}

TEST(KernelDoc, ParseIndentedCodeBlockNoDC) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "\n"
                                      "Usage example\n"
                                      "\n"
                                      "    result = func();\n"
                                      "    check(result);\n");

  bool HasCode = false;
  for (const auto &Block : Info.Description) {
    if (Block.BlockKind == KernelDocDescriptionBlock::Code) {
      HasCode = true;
      EXPECT_EQ(Block.Text, "result = func();\ncheck(result);");
    }
  }
  EXPECT_TRUE(HasCode);
}

TEST(KernelDoc, ParseIndentedCodeBlockStandaloneDC) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "\n"
                                      "::\n"
                                      "\n"
                                      "    code_here();\n");

  // Standalone :: should not produce a paragraph.
  for (const auto &Block : Info.Description)
    EXPECT_NE(Block.Text, "::");
  bool HasCode = false;
  for (const auto &Block : Info.Description) {
    if (Block.BlockKind == KernelDocDescriptionBlock::Code) {
      HasCode = true;
      EXPECT_EQ(Block.Text, "code_here();");
    }
  }
  EXPECT_TRUE(HasCode);
}

TEST(KernelDoc, ParseIndentedCodeBlockBlankWithin) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "\n"
                                      "    first_block();\n"
                                      "\n"
                                      "    second_block();\n");

  bool HasCode = false;
  for (const auto &Block : Info.Description) {
    if (Block.BlockKind == KernelDocDescriptionBlock::Code) {
      HasCode = true;
      EXPECT_EQ(Block.Text, "first_block();\n\nsecond_block();");
    }
  }
  EXPECT_TRUE(HasCode);
}

TEST(KernelDoc, ParseIndentedCodeBlockStripsIndent) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "\n"
                                      "      line1();\n"
                                      "      line2();\n");

  bool HasCode = false;
  for (const auto &Block : Info.Description) {
    if (Block.BlockKind == KernelDocDescriptionBlock::Code) {
      HasCode = true;
      EXPECT_EQ(Block.Text, "line1();\nline2();");
    }
  }
  EXPECT_TRUE(HasCode);
}

TEST(KernelDoc, ParseIndentedCodeBlockThenText) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "\n"
                                      "    code();\n"
                                      "Normal paragraph after code.\n");

  bool HasCode = false, HasText = false;
  for (const auto &Block : Info.Description) {
    if (Block.BlockKind == KernelDocDescriptionBlock::Code) {
      HasCode = true;
      EXPECT_EQ(Block.Text, "code();");
    }
    if (Block.BlockKind == KernelDocDescriptionBlock::Paragraph &&
        Block.Text == "Normal paragraph after code.")
      HasText = true;
  }
  EXPECT_TRUE(HasCode);
  EXPECT_TRUE(HasText);
}

TEST(KernelDoc, ParseExampleDCNotSection) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "\n"
                                      "Example::\n"
                                      "\n"
                                      "    func(42);\n");

  EXPECT_TRUE(Info.Sections.empty());
  bool HasParagraph = false, HasCode = false;
  for (const auto &Block : Info.Description) {
    if (Block.BlockKind == KernelDocDescriptionBlock::Paragraph &&
        Block.Text == "Example:")
      HasParagraph = true;
    if (Block.BlockKind == KernelDocDescriptionBlock::Code)
      HasCode = true;
  }
  EXPECT_TRUE(HasParagraph);
  EXPECT_TRUE(HasCode);
}

TEST(KernelDoc, ParseReturnListItems) {
  KernelDocInfo Info = parseKernelDoc("do_something() - Do it\n"
                                      "\n"
                                      "Return:\n"
                                      "* %0 - OK\n"
                                      "* %-EINVAL - Invalid argument\n"
                                      "* %-ENOMEM - Out of memory\n");

  EXPECT_TRUE(Info.Returns.empty());
  ASSERT_EQ(Info.ReturnItems.size(), 3u);
  EXPECT_EQ(Info.ReturnItems[0], "%0 - OK");
  EXPECT_EQ(Info.ReturnItems[1], "%-EINVAL - Invalid argument");
  EXPECT_EQ(Info.ReturnItems[2], "%-ENOMEM - Out of memory");
}

TEST(KernelDoc, ParseReturnListWithPreamble) {
  KernelDocInfo Info =
      parseKernelDoc("do_something() - Do it\n"
                     "\n"
                     "Return: One of the following error codes:\n"
                     "* %0 - OK\n"
                     "* %-EINVAL - Invalid argument\n");

  EXPECT_EQ(Info.Returns, "One of the following error codes:");
  ASSERT_EQ(Info.ReturnItems.size(), 2u);
  EXPECT_EQ(Info.ReturnItems[0], "%0 - OK");
  EXPECT_EQ(Info.ReturnItems[1], "%-EINVAL - Invalid argument");
}

TEST(KernelDoc, ParseReturnListDashMarker) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "\n"
                                      "Return:\n"
                                      "- zero on success\n"
                                      "- negative errno on failure\n");

  EXPECT_TRUE(Info.Returns.empty());
  ASSERT_EQ(Info.ReturnItems.size(), 2u);
  EXPECT_EQ(Info.ReturnItems[0], "zero on success");
  EXPECT_EQ(Info.ReturnItems[1], "negative errno on failure");
}

TEST(KernelDoc, ParseReturnListContinuation) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "\n"
                                      "Return:\n"
                                      "* %0 - OK to proceed\n"
                                      "  with the operation\n"
                                      "* %-EINVAL - Bad argument\n");

  ASSERT_EQ(Info.ReturnItems.size(), 2u);
  EXPECT_EQ(Info.ReturnItems[0], "%0 - OK to proceed with the operation");
  EXPECT_EQ(Info.ReturnItems[1], "%-EINVAL - Bad argument");
}

TEST(KernelDoc, RenderReturnList) {
  KernelDocInfo Info;
  Info.Brief = "Do something";
  Info.ReturnItems.push_back("%0 - OK");
  Info.ReturnItems.push_back("%-EINVAL - Invalid argument");

  markup::Document Doc;
  renderKernelDocToMarkup(Info, Doc);
  std::string Rendered = Doc.asMarkdown();

  EXPECT_NE(Rendered.find("### Returns"), std::string::npos);
  EXPECT_NE(Rendered.find("`0`"), std::string::npos);
  EXPECT_NE(Rendered.find("`-EINVAL`"), std::string::npos);
}

TEST(KernelDoc, RenderReturnListWithPreamble) {
  KernelDocInfo Info;
  Info.Brief = "Do something";
  Info.Returns = "One of:";
  Info.ReturnItems.push_back("%0 - OK");

  markup::Document Doc;
  renderKernelDocToMarkup(Info, Doc);
  std::string Rendered = Doc.asMarkdown();

  EXPECT_NE(Rendered.find("### Returns"), std::string::npos);
  EXPECT_NE(Rendered.find("One of:"), std::string::npos);
  EXPECT_NE(Rendered.find("`0`"), std::string::npos);
}

TEST(KernelDoc, ParseInlineMemberDoc) {
  KernelDocInfo Info = parseKernelDoc("@bar: description of bar");
  EXPECT_EQ(Info.Brief, "description of bar");
  EXPECT_TRUE(Info.Params.empty());
}

TEST(KernelDoc, ParseInlineMemberDocMultiLine) {
  KernelDocInfo Info = parseKernelDoc("@bar: brief text\n"
                                      "\n"
                                      "Longer description of bar.");

  EXPECT_EQ(Info.Brief, "brief text");
  ASSERT_EQ(Info.Description.size(), 1u);
  EXPECT_EQ(Info.Description[0].Text, "Longer description of bar.");
  EXPECT_TRUE(Info.Params.empty());
}

TEST(KernelDoc, ParseRSTDCWithSpace) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "\n"
                                      "Some text ::\n"
                                      "\n"
                                      "    code();\n");

  bool HasParagraph = false, HasCode = false;
  for (const auto &Block : Info.Description) {
    if (Block.BlockKind == KernelDocDescriptionBlock::Paragraph &&
        Block.Text == "Some text")
      HasParagraph = true;
    if (Block.BlockKind == KernelDocDescriptionBlock::Code)
      HasCode = true;
  }
  EXPECT_TRUE(HasParagraph);
  EXPECT_TRUE(HasCode);
}

TEST(KernelDoc, ParseRSTDCAttached) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "\n"
                                      "Example::\n"
                                      "\n"
                                      "    code();\n");

  bool HasParagraph = false;
  for (const auto &Block : Info.Description) {
    if (Block.BlockKind == KernelDocDescriptionBlock::Paragraph &&
        Block.Text == "Example:")
      HasParagraph = true;
  }
  EXPECT_TRUE(HasParagraph);
}

TEST(KernelDoc, ParseTypedef) {
  KernelDocInfo Info = parseKernelDoc("my_type - A custom type");
  EXPECT_EQ(Info.Brief, "A custom type");
}

TEST(KernelDoc, ParseMacro) {
  KernelDocInfo Info = parseKernelDoc("MY_MACRO - A useful macro\n"
                                      "@x: first argument\n"
                                      "@y: second argument\n");
  EXPECT_EQ(Info.Brief, "A useful macro");
  ASSERT_EQ(Info.Params.size(), 2u);
  EXPECT_EQ(Info.Params[0].Name, "x");
  EXPECT_EQ(Info.Params[1].Name, "y");
}

TEST(KernelDoc, ParseParamsAfterBody) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "\n"
                                      "Description first.\n"
                                      "\n"
                                      "@a: param after body\n");

  EXPECT_EQ(Info.Brief, "Brief");
  ASSERT_EQ(Info.Description.size(), 1u);
  EXPECT_EQ(Info.Description[0].Text, "Description first.");
  ASSERT_EQ(Info.Params.size(), 1u);
  EXPECT_EQ(Info.Params[0].Name, "a");
  EXPECT_EQ(Info.Params[0].Description, "param after body");
}

TEST(KernelDoc, ParseSectionAfterBrief) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "Context: Process context.\n");

  EXPECT_EQ(Info.Brief, "Brief");
  ASSERT_EQ(Info.Sections.size(), 1u);
  EXPECT_EQ(Info.Sections[0].Name, "Context");
  EXPECT_EQ(Info.Sections[0].Description, "Process context.");
}

TEST(KernelDoc, ParseBriefEmptyAfterDash) {
  KernelDocInfo Info = parseKernelDoc("func() - ");
  EXPECT_EQ(Info.Brief, "");
}

TEST(KernelDoc, ParseReturnContinuationNotSection) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "\n"
                                      "Return: The pointer is\n"
                                      "        Valid: only when active.\n"
                                      "Context: Process context.\n");

  EXPECT_EQ(Info.Returns, "The pointer is Valid: only when active.");
  ASSERT_EQ(Info.Sections.size(), 1u);
  EXPECT_EQ(Info.Sections[0].Name, "Context");
  EXPECT_EQ(Info.Sections[0].Description, "Process context.");
}

TEST(KernelDoc, ParseParamContinuationNotSection) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "@buf: Pointer to the\n"
                                      "      Buffer: must be aligned.\n"
                                      "@len: length\n");

  ASSERT_EQ(Info.Params.size(), 2u);
  EXPECT_EQ(Info.Params[0].Name, "buf");
  EXPECT_EQ(Info.Params[0].Description,
            "Pointer to the Buffer: must be aligned.");
  EXPECT_EQ(Info.Params[1].Name, "len");
}

TEST(KernelDoc, ParseSectionContinuationNotSection) {
  KernelDocInfo Info =
      parseKernelDoc("func() - Brief\n"
                     "\n"
                     "Context: Cannot be called from\n"
                     "         Interrupt: context or atomic sections.\n");

  ASSERT_EQ(Info.Sections.size(), 1u);
  EXPECT_EQ(Info.Sections[0].Name, "Context");
  EXPECT_EQ(Info.Sections[0].Description,
            "Cannot be called from Interrupt: context or atomic sections.");
}

TEST(KernelDoc, ParseArrowParam) {
  KernelDocInfo Info = parseKernelDoc("struct outer - An outer struct\n"
                                      "@foo: simple member\n"
                                      "@foo->bar: arrow member\n"
                                      "@foo->bar.baz: chained member\n");

  ASSERT_EQ(Info.Params.size(), 3u);
  EXPECT_EQ(Info.Params[0].Name, "foo");
  EXPECT_EQ(Info.Params[1].Name, "foo->bar");
  EXPECT_EQ(Info.Params[1].Description, "arrow member");
  EXPECT_EQ(Info.Params[2].Name, "foo->bar.baz");
  EXPECT_EQ(Info.Params[2].Description, "chained member");
}

TEST(KernelDoc, ParseAtReturn) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "@x: param\n"
                                      "\n"
                                      "@return: Zero on success.\n");

  ASSERT_EQ(Info.Params.size(), 1u);
  EXPECT_EQ(Info.Returns, "Zero on success.");
}

TEST(KernelDoc, ParseAtReturns) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "@x: param\n"
                                      "\n"
                                      "@returns: A pointer or %NULL.\n");

  ASSERT_EQ(Info.Params.size(), 1u);
  EXPECT_EQ(Info.Returns, "A pointer or %NULL.");
}

TEST(KernelDoc, ParseReturnCaseInsensitive) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "\n"
                                      "RETURN: Zero on success.\n");

  EXPECT_EQ(Info.Returns, "Zero on success.");
}

TEST(KernelDoc, ParseReturnsCaseInsensitive) {
  KernelDocInfo Info = parseKernelDoc("func() - Brief\n"
                                      "\n"
                                      "RETURNS: A pointer.\n");

  EXPECT_EQ(Info.Returns, "A pointer.");
}

TEST(KernelDoc, ParseDescriptionCaseInsensitive) {
  KernelDocInfo Info =
      parseKernelDoc("func() - Brief\n"
                     "@x: param\n"
                     "\n"
                     "description: The detailed description.\n");

  ASSERT_EQ(Info.Description.size(), 1u);
  EXPECT_EQ(Info.Description[0].Text, "The detailed description.");
}

TEST(KernelDoc, ParseColonBrief) {
  KernelDocInfo Info =
      parseKernelDoc("func(): Return temperature from raw value\n"
                     "@x: param\n");

  EXPECT_EQ(Info.Brief, "Return temperature from raw value");
  ASSERT_EQ(Info.Params.size(), 1u);
  EXPECT_EQ(Info.Params[0].Name, "x");
}

TEST(KernelDoc, ParseColonBriefNoParens) {
  KernelDocInfo Info = parseKernelDoc("my_type: A custom type definition\n");

  EXPECT_EQ(Info.Brief, "A custom type definition");
}

TEST(KernelDoc, ParseColonBriefWithSpace) {
  KernelDocInfo Info =
      parseKernelDoc("func() : Brief with space before colon\n");

  EXPECT_EQ(Info.Brief, "Brief with space before colon");
}

TEST(KernelDoc, ParseColonBriefNotRSTDC) {
  // "name::" should NOT be treated as a colon-style brief — it's
  // a RST literal block marker.
  KernelDocInfo Info = parseKernelDoc("example::\n");
  // Should fall through to plain text brief, not extract empty brief
  EXPECT_EQ(Info.Brief, "example::");
}

} // namespace clangd
} // namespace clang
