//===- VerifyDiagnosticConsumer.h - Verifying Diagnostic Client -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_VERIFYDIAGNOSTICCONSUMER_H
#define LLVM_CLANG_FRONTEND_VERIFYDIAGNOSTICCONSUMER_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace clang {

class FileEntry;
class LangOptions;
class SourceManager;
class TextDiagnosticBuffer;

/// VerifyDiagnosticConsumer - Create a diagnostic client which will use
/// markers in the input source to check that all the emitted diagnostics match
/// those expected.
///
/// INVOKING THE DIAGNOSTIC CHECKER:
///
/// VerifyDiagnosticConsumer is typically invoked via the "-verify" option to
/// "clang -cc1".  "-verify" is equivalent to "-verify=expected", so all
/// diagnostics are typically specified with the prefix "expected".  For
/// example:
///
/// \code
///   int A = B; // expected-error {{use of undeclared identifier 'B'}}
/// \endcode
///
/// Custom prefixes can be specified as a comma-separated sequence.  Each
/// prefix must start with a letter and contain only alphanumeric characters,
/// hyphens, and underscores.  For example, given just "-verify=foo,bar",
/// the above diagnostic would be ignored, but the following diagnostics would
/// be recognized:
///
/// \code
///   int A = B; // foo-error {{use of undeclared identifier 'B'}}
///   int C = D; // bar-error {{use of undeclared identifier 'D'}}
/// \endcode
///
/// Multiple occurrences accumulate prefixes.  For example,
/// "-verify -verify=foo,bar -verify=baz" is equivalent to
/// "-verify=expected,foo,bar,baz".
///
/// SPECIFYING DIAGNOSTICS:
///
/// Indicating that a line expects an error or a warning is simple. Put a
/// comment on the line that has the diagnostic, use:
///
/// \code
///   expected-{error,warning,remark,note}
/// \endcode
///
/// to tag if it's an expected error, remark or warning, and place the expected
/// text between {{ and }} markers. The full text doesn't have to be included,
/// only enough to ensure that the correct diagnostic was emitted.
///
/// Here's an example:
///
/// \code
///   int A = B; // expected-error {{use of undeclared identifier 'B'}}
/// \endcode
///
/// You can place as many diagnostics on one line as you wish. To make the code
/// more readable, you can use slash-newline to separate out the diagnostics.
///
/// Alternatively, it is possible to specify the line on which the diagnostic
/// should appear by appending "@<line>" to "expected-<type>", for example:
///
/// \code
///   #warning some text
///   // expected-warning@10 {{some text}}
/// \endcode
///
/// The line number may be absolute (as above), or relative to the current
/// line by prefixing the number with either '+' or '-'.
///
/// If the diagnostic is generated in a separate file, for example in a shared
/// header file, it may be beneficial to be able to declare the file in which
/// the diagnostic will appear, rather than placing the expected-* directive in
/// the actual file itself.  This can be done using the following syntax:
///
/// \code
///   // expected-error@path/include.h:15 {{error message}}
/// \endcode
///
/// The path can be absolute or relative and the same search paths will be used
/// as for #include directives.  The line number in an external file may be
/// substituted with '*' meaning that any line number will match (useful where
/// the included file is, for example, a system header where the actual line
/// number may change and is not critical).
///
/// As an alternative to specifying a fixed line number, the location of a
/// diagnostic can instead be indicated by a marker of the form "#<marker>".
/// Markers are specified by including them in a comment, and then referenced
/// by appending the marker to the diagnostic with "@#<marker>":
///
/// \code
///   #warning some text  // #1
///   // expected-warning@#1 {{some text}}
/// \endcode
///
/// The name of a marker used in a directive must be unique within the
/// compilation.
///
/// The simple syntax above allows each specification to match exactly one
/// error.  You can use the extended syntax to customize this. The extended
/// syntax is "expected-<type> <n> {{diag text}}", where \<type> is one of
/// "error", "warning" or "note", and \<n> is a positive integer. This allows
/// the diagnostic to appear as many times as specified. Example:
///
/// \code
///   void f(); // expected-note 2 {{previous declaration is here}}
/// \endcode
///
/// Where the diagnostic is expected to occur a minimum number of times, this
/// can be specified by appending a '+' to the number. Example:
///
/// \code
///   void f(); // expected-note 0+ {{previous declaration is here}}
///   void g(); // expected-note 1+ {{previous declaration is here}}
/// \endcode
///
/// In the first example, the diagnostic becomes optional, i.e. it will be
/// swallowed if it occurs, but will not generate an error if it does not
/// occur.  In the second example, the diagnostic must occur at least once.
/// As a short-hand, "one or more" can be specified simply by '+'. Example:
///
/// \code
///   void g(); // expected-note + {{previous declaration is here}}
/// \endcode
///
/// A range can also be specified by "<n>-<m>".  Example:
///
/// \code
///   void f(); // expected-note 0-1 {{previous declaration is here}}
/// \endcode
///
/// In this example, the diagnostic may appear only once, if at all.
///
/// Regex matching mode may be selected by appending '-re' to type and
/// including regexes wrapped in double curly braces in the directive, such as:
///
/// \code
///   expected-error-re {{format specifies type 'wchar_t **' (aka '{{.+}}')}}
/// \endcode
///
/// Examples matching error: "variable has incomplete type 'struct s'"
///
/// \code
///   // expected-error {{variable has incomplete type 'struct s'}}
///   // expected-error {{variable has incomplete type}}
///
///   // expected-error-re {{variable has type 'struct {{.}}'}}
///   // expected-error-re {{variable has type 'struct {{.*}}'}}
///   // expected-error-re {{variable has type 'struct {{(.*)}}'}}
///   // expected-error-re {{variable has type 'struct{{[[:space:]](.*)}}'}}
/// \endcode
///
/// VerifyDiagnosticConsumer expects at least one expected-* directive to
/// be found inside the source code.  If no diagnostics are expected the
/// following directive can be used to indicate this:
///
/// \code
///   // expected-no-diagnostics
/// \endcode
///
class VerifyDiagnosticConsumer: public DiagnosticConsumer,
                                public CommentHandler {
public:
  /// Directive - Abstract class representing a parsed verify directive.
  ///
  class Directive {
  public:
    static std::unique_ptr<Directive>
    create(bool RegexKind, SourceLocation DirectiveLoc,
           SourceLocation DiagnosticLoc, bool MatchAnyFileAndLine,
           bool MatchAnyLine, StringRef Text, unsigned Min, unsigned Max);

  public:
    /// Constant representing n or more matches.
    static const unsigned MaxCount = std::numeric_limits<unsigned>::max();

    SourceLocation DirectiveLoc;
    SourceLocation DiagnosticLoc;
    const std::string Text;
    unsigned Min, Max;
    bool MatchAnyLine;
    bool MatchAnyFileAndLine; // `MatchAnyFileAndLine` implies `MatchAnyLine`.

    Directive(const Directive &) = delete;
    Directive &operator=(const Directive &) = delete;
    virtual ~Directive() = default;

    // Returns true if directive text is valid.
    // Otherwise returns false and populates E.
    virtual bool isValid(std::string &Error) = 0;

    // Returns true on match.
    virtual bool match(StringRef S) = 0;

  protected:
    Directive(SourceLocation DirectiveLoc, SourceLocation DiagnosticLoc,
              bool MatchAnyFileAndLine, bool MatchAnyLine, StringRef Text,
              unsigned Min, unsigned Max)
        : DirectiveLoc(DirectiveLoc), DiagnosticLoc(DiagnosticLoc), Text(Text),
          Min(Min), Max(Max), MatchAnyLine(MatchAnyLine || MatchAnyFileAndLine),
          MatchAnyFileAndLine(MatchAnyFileAndLine) {
      assert(!DirectiveLoc.isInvalid() && "DirectiveLoc is invalid!");
      assert((!DiagnosticLoc.isInvalid() || MatchAnyLine) &&
             "DiagnosticLoc is invalid!");
    }
  };

  using DirectiveList = std::vector<std::unique_ptr<Directive>>;

  /// ExpectedData - owns directive objects and deletes on destructor.
  struct ExpectedData {
    DirectiveList Errors;
    DirectiveList Warnings;
    DirectiveList Remarks;
    DirectiveList Notes;

    void Reset() {
      Errors.clear();
      Warnings.clear();
      Remarks.clear();
      Notes.clear();
    }
  };

  enum DirectiveStatus {
    HasNoDirectives,
    HasNoDirectivesReported,
    HasExpectedNoDiagnostics,
    HasOtherExpectedDirectives
  };

  class MarkerTracker;

private:
  DiagnosticsEngine &Diags;
  DiagnosticConsumer *PrimaryClient;
  std::unique_ptr<DiagnosticConsumer> PrimaryClientOwner;
  std::unique_ptr<TextDiagnosticBuffer> Buffer;
  std::unique_ptr<MarkerTracker> Markers;
  const Preprocessor *CurrentPreprocessor = nullptr;
  const LangOptions *LangOpts = nullptr;
  SourceManager *SrcManager = nullptr;
  unsigned ActiveSourceFiles = 0;
  DirectiveStatus Status;
  ExpectedData ED;

  void CheckDiagnostics();

  void setSourceManager(SourceManager &SM) {
    assert((!SrcManager || SrcManager == &SM) && "SourceManager changed!");
    SrcManager = &SM;
  }

  // These facilities are used for validation in debug builds.
  class UnparsedFileStatus {
    llvm::PointerIntPair<const FileEntry *, 1, bool> Data;

  public:
    UnparsedFileStatus(const FileEntry *File, bool FoundDirectives)
        : Data(File, FoundDirectives) {}

    const FileEntry *getFile() const { return Data.getPointer(); }
    bool foundDirectives() const { return Data.getInt(); }
  };

  using ParsedFilesMap = llvm::DenseMap<FileID, const FileEntry *>;
  using UnparsedFilesMap = llvm::DenseMap<FileID, UnparsedFileStatus>;

  ParsedFilesMap ParsedFiles;
  UnparsedFilesMap UnparsedFiles;

public:
  /// Create a new verifying diagnostic client, which will issue errors to
  /// the currently-attached diagnostic client when a diagnostic does not match
  /// what is expected (as indicated in the source file).
  VerifyDiagnosticConsumer(DiagnosticsEngine &Diags);
  ~VerifyDiagnosticConsumer() override;

  void BeginSourceFile(const LangOptions &LangOpts,
                       const Preprocessor *PP) override;

  void EndSourceFile() override;

  enum ParsedStatus {
    /// File has been processed via HandleComment.
    IsParsed,

    /// File has diagnostics and may have directives.
    IsUnparsed,

    /// File has diagnostics but guaranteed no directives.
    IsUnparsedNoDirectives
  };

  /// Update lists of parsed and unparsed files.
  void UpdateParsedFileStatus(SourceManager &SM, FileID FID, ParsedStatus PS);

  bool HandleComment(Preprocessor &PP, SourceRange Comment) override;

  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const Diagnostic &Info) override;
};

} // namespace clang

#endif // LLVM_CLANG_FRONTEND_VERIFYDIAGNOSTICCONSUMER_H
