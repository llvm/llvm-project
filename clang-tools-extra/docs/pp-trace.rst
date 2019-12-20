.. index:: pp-trace

==================================
pp-trace User's Manual
==================================

.. toctree::
   :hidden:

:program:`pp-trace` is a standalone tool that traces preprocessor
activity. It's also used as a test of Clang's PPCallbacks interface.
It runs a given source file through the Clang preprocessor, displaying
selected information from callback functions overridden in a
`PPCallbacks <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html>`_
derivation. The output is in a high-level YAML format, described in
:ref:`OutputFormat`.

.. _Usage:

pp-trace Usage
==============

Command Line Format
-------------------

``pp-trace [<pp-trace-options>] <source-file> [-- <front-end-options>]``

``<pp-trace-options>`` is a place-holder for options
specific to pp-trace, which are described below in
:ref:`CommandLineOptions`.

``<source-file>`` specifies the source file to run through the preprocessor.

``<front-end-options>`` is a place-holder for regular
`Clang Compiler Options <https://clang.llvm.org/docs/UsersManual.html#command-line-options>`_,
which must follow the <source-file>.

.. _CommandLineOptions:

Command Line Options
--------------------

.. option:: -callbacks <comma-separated-globs>

  This option specifies a comma-separated list of globs describing the list of
  callbacks that should be traced. Globs are processed in order of appearance.
  Positive globs add matched callbacks to the set, netative globs (those with
  the '-' prefix) remove callacks from the set.

  * FileChanged
  * FileSkipped
  * FileNotFound
  * InclusionDirective
  * moduleImport
  * EndOfMainFile
  * Ident
  * PragmaDirective
  * PragmaComment
  * PragmaDetectMismatch
  * PragmaDebug
  * PragmaMessage
  * PragmaDiagnosticPush
  * PragmaDiagnosticPop
  * PragmaDiagnostic
  * PragmaOpenCLExtension
  * PragmaWarning
  * PragmaWarningPush
  * PragmaWarningPop
  * MacroExpands
  * MacroDefined
  * MacroUndefined
  * Defined
  * SourceRangeSkipped
  * If
  * Elif
  * Ifdef
  * Ifndef
  * Else
  * Endif

.. option:: -output <output-file>

  By default, pp-trace outputs the trace information to stdout. Use this
  option to output the trace information to a file.

.. _OutputFormat:

pp-trace Output Format
======================

The pp-trace output is formatted as YAML. See https://yaml.org/ for general
YAML information. It's arranged as a sequence of information about the
callback call, including the callback name and argument information, for
example:::

  ---
  - Callback: Name
    Argument1: Value1
    Argument2: Value2
  (etc.)
  ...

With real data:::

  ---
  - Callback: FileChanged
    Loc: "c:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-include.cpp:1:1"
    Reason: EnterFile
    FileType: C_User
    PrevFID: (invalid)
    (etc.)
  - Callback: FileChanged
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-include.cpp:5:1"
    Reason: ExitFile
    FileType: C_User
    PrevFID: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/Input/Level1B.h"
  - Callback: EndOfMainFile
  ...

In all but one case (MacroDirective) the "Argument" scalars have the same
name as the argument in the corresponding PPCallbacks callback function.

Callback Details
----------------

The following sections describe the pupose and output format for each callback.

Click on the callback name in the section heading to see the Doxygen
documentation for the callback.

The argument descriptions table describes the callback argument information
displayed.

The Argument Name field in most (but not all) cases is the same name as the
callback function parameter.

The Argument Value Syntax field describes the values that will be displayed
for the argument value. It uses an ad hoc representation that mixes literal
and symbolic representations. Enumeration member symbols are shown as the
actual enum member in a (member1|member2|...) form. A name in parentheses
can either represent a place holder for the described value, or confusingly,
it might be a literal, such as (null), for a null pointer.
Locations are shown as quoted only to avoid confusing the documentation generator.

The Clang C++ Type field is the type from the callback function declaration.

The description describes the argument or what is displayed for it.

Note that in some cases, such as when a structure pointer is an argument
value, only some key member or members are shown to represent the value,
instead of trying to display all members of the structure.

`FileChanged <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#a7cc8cfaf34114fc65e92af621cd6464e>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FileChanged is called when the preprocessor enters or exits a file, both the
top level file being compiled, as well as any #include directives. It will
also be called as a result of a system header pragma or in internal renaming
of a file.

Argument descriptions:

==============   ==================================================   ============================== ==============================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ==============================
Loc              "(file):(line):(col)"                                SourceLocation                 The location of the directive.
Reason           (EnterFile|ExitFile|SystemHeaderPragma|RenameFile)   PPCallbacks::FileChangeReason  Reason for change.
FileType         (C_User|C_System|C_ExternCSystem)                    SrcMgr::CharacteristicKind     Include type.
PrevFID          ((file)|(invalid))                                   FileID                         Previous file, if any.
==============   ==================================================   ============================== ==============================

Example:::

  - Callback: FileChanged
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-include.cpp:1:1"
    Reason: EnterFile
    FileType: C_User
    PrevFID: (invalid)

`FileSkipped <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#ab5b338a0670188eb05fa7685bbfb5128>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FileSkipped is called when a source file is skipped as the result of header
guard optimization.

Argument descriptions:

==============   ==================================================   ============================== ========================================================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ========================================================
ParentFile       ("(file)" or (null))                                 const FileEntry                The file that #included the skipped file.
FilenameTok      (token)                                              const Token                    The token in ParentFile that indicates the skipped file.
FileType         (C_User|C_System|C_ExternCSystem)                    SrcMgr::CharacteristicKind     The file type.
==============   ==================================================   ============================== ========================================================

Example:::

  - Callback: FileSkipped
    ParentFile: "/path/filename.h"
    FilenameTok: "filename.h"
    FileType: C_User

`FileNotFound <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#a3045151545f987256bfa8d978916ef00>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FileNotFound is called when an inclusion directive results in a file-not-found error.

Argument descriptions:

==============   ==================================================   ============================== =====================================================================================================================================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== =====================================================================================================================================
FileName         "(file)"                                             StringRef                      The name of the file being included, as written in the source code.
RecoveryPath     (path)                                               SmallVectorImpl<char>          If this client indicates that it can recover from this missing file, the client should set this as an additional header search patch.
==============   ==================================================   ============================== =====================================================================================================================================

Example:::

  - Callback: FileNotFound
    FileName: "/path/filename.h"
    RecoveryPath:

`InclusionDirective <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#a557d9738c329793513a6f57d6b60de52>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

InclusionDirective is called when an inclusion directive of any kind (#include</code>, #import</code>, etc.) has been processed, regardless of whether the inclusion will actually result in an inclusion.

Argument descriptions:

==============   ==================================================   ============================== ============================================================================================================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ============================================================================================================
HashLoc          "(file):(line):(col)"                                SourceLocation                 The location of the '#' that starts the inclusion directive.
IncludeTok       (token)                                              const Token                    The token that indicates the kind of inclusion directive, e.g., 'include' or 'import'.
FileName         "(file)"                                             StringRef                      The name of the file being included, as written in the source code.
IsAngled         (true|false)                                         bool                           Whether the file name was enclosed in angle brackets; otherwise, it was enclosed in quotes.
FilenameRange    "(file)"                                             CharSourceRange                The character range of the quotes or angle brackets for the written file name.
File             "(file)"                                             const FileEntry                The actual file that may be included by this inclusion directive.
SearchPath       "(path)"                                             StringRef                      Contains the search path which was used to find the file in the file system.
RelativePath     "(path)"                                             StringRef                      The path relative to SearchPath, at which the include file was found.
Imported         ((module name)|(null))                               const Module                   The module, whenever an inclusion directive was automatically turned into a module import or null otherwise.
==============   ==================================================   ============================== ============================================================================================================

Example:::

  - Callback: InclusionDirective
    IncludeTok: include
    FileName: "Input/Level1B.h"
    IsAngled: false
    FilenameRange: "Input/Level1B.h"
    File: "D:/Clang/llvmnewmod/tools/clang/tools/extra/test/pp-trace/Input/Level1B.h"
    SearchPath: "D:/Clang/llvmnewmod/tools/clang/tools/extra/test/pp-trace"
    RelativePath: "Input/Level1B.h"
    Imported: (null)

`moduleImport <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#af32dcf1b8b7c179c7fcd3e24e89830fe>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

moduleImport is called when there was an explicit module-import syntax.

Argument descriptions:

==============   ==================================================   ============================== ===========================================================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ===========================================================
ImportLoc        "(file):(line):(col)"                                SourceLocation                 The location of import directive token.
Path             "(path)"                                             ModuleIdPath                   The identifiers (and their locations) of the module "path".
Imported         ((module name)|(null))                               const Module                   The imported module; can be null if importing failed.
==============   ==================================================   ============================== ===========================================================

Example:::

  - Callback: moduleImport
    ImportLoc: "d:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-modules.cpp:4:2"
    Path: [{Name: Level1B, Loc: "d:/Clang/llvmnewmod/tools/clang/tools/extra/test/pp-trace/pp-trace-modules.cpp:4:9"}, {Name: Level2B, Loc: "d:/Clang/llvmnewmod/tools/clang/tools/extra/test/pp-trace/pp-trace-modules.cpp:4:17"}]
    Imported: Level2B

`EndOfMainFile <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#a63e170d069e99bc1c9c7ea0f3bed8bcc>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

EndOfMainFile is called when the end of the main file is reached.

Argument descriptions:

==============   ==================================================   ============================== ======================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ======================
(no arguments)
==============   ==================================================   ============================== ======================

Example:::

  - Callback: EndOfMainFile

`Ident <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#a3683f1d1fa513e9b6193d446a5cc2b66>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ident is called when a #ident or #sccs directive is read.

Argument descriptions:

==============   ==================================================   ============================== ==============================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ==============================
Loc              "(file):(line):(col)"                                SourceLocation                 The location of the directive.
str              (name)                                               const std::string              The text of the directive.
==============   ==================================================   ============================== ==============================

Example:::

  - Callback: Ident
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-ident.cpp:3:1"
    str: "$Id$"

`PragmaDirective <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#a0a2d7a72c62184b3cbde31fb62c6f2f7>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PragmaDirective is called when start reading any pragma directive.

Argument descriptions:

==============   ==================================================   ============================== =================================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== =================================
Loc              "(file):(line):(col)"                                SourceLocation                 The location of the directive.
Introducer       (PIK_HashPragma|PIK__Pragma|PIK___pragma)            PragmaIntroducerKind           The type of the pragma directive.
==============   ==================================================   ============================== =================================

Example:::

  - Callback: PragmaDirective
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-pragma.cpp:3:1"
    Introducer: PIK_HashPragma

`PragmaComment <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#ace0d940fc2c12ab76441466aab58dc37>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PragmaComment is called when a #pragma comment directive is read.

Argument descriptions:

==============   ==================================================   ============================== ==============================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ==============================
Loc              "(file):(line):(col)"                                SourceLocation                 The location of the directive.
Kind             ((name)|(null))                                      const IdentifierInfo           The comment kind symbol.
Str              (message directive)                                  const std::string              The comment message directive.
==============   ==================================================   ============================== ==============================

Example:::

  - Callback: PragmaComment
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-pragma.cpp:3:1"
    Kind: library
    Str: kernel32.lib

`PragmaDetectMismatch <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#ab11158c9149fb8ad8af1903f4a6cd65d>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PragmaDetectMismatch is called when a #pragma detect_mismatch directive is read.

Argument descriptions:

==============   ==================================================   ============================== ==============================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ==============================
Loc              "(file):(line):(col)"                                SourceLocation                 The location of the directive.
Name             "(name)"                                             const std::string              The name.
Value            (string)                                             const std::string              The value.
==============   ==================================================   ============================== ==============================

Example:::

  - Callback: PragmaDetectMismatch
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-pragma.cpp:3:1"
    Name: name
    Value: value

`PragmaDebug <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#a57cdccb6dcc07e926513ac3d5b121466>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PragmaDebug is called when a #pragma clang __debug directive is read.

Argument descriptions:

==============   ==================================================   ============================== ================================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ================================
Loc              "(file):(line):(col)"                                SourceLocation                 The location of the directive.
DebugType        (string)                                             StringRef                      Indicates type of debug message.
==============   ==================================================   ============================== ================================

Example:::

  - Callback: PragmaDebug
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-pragma.cpp:3:1"
    DebugType: warning

`PragmaMessage <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#abb42935d9a9fd8e2c4f51cfdc4ea2ae1>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PragmaMessage is called when a #pragma message directive is read.

Argument descriptions:

==============   ==================================================   ============================== =======================================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== =======================================
Loc              "(file):(line):(col)"                                SourceLocation                 The location of the directive.
Namespace        (name)                                               StringRef                      The namespace of the message directive.
Kind             (PMK_Message|PMK_Warning|PMK_Error)                  PPCallbacks::PragmaMessageKind The type of the message directive.
Str              (string)                                             StringRef                      The text of the message directive.
==============   ==================================================   ============================== =======================================

Example:::

  - Callback: PragmaMessage
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-pragma.cpp:3:1"
    Namespace: "GCC"
    Kind: PMK_Message
    Str: The message text.

`PragmaDiagnosticPush <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#a0f3ff19762baa38fe6c5c58022d32979>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PragmaDiagnosticPush is called when a #pragma gcc diagnostic push directive is read.

Argument descriptions:

==============   ==================================================   ============================== ==============================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ==============================
Loc              "(file):(line):(col)"                                SourceLocation                 The location of the directive.
Namespace        (name)                                               StringRef                      Namespace name.
==============   ==================================================   ============================== ==============================

Example:::

  - Callback: PragmaDiagnosticPush
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-pragma.cpp:3:1"
    Namespace: "GCC"

`PragmaDiagnosticPop <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#ac94d789873122221fba8d76f6c5ea45e>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PragmaDiagnosticPop is called when a #pragma gcc diagnostic pop directive is read.

Argument descriptions:

==============   ==================================================   ============================== ==============================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ==============================
Loc              "(file):(line):(col)"                                SourceLocation                 The location of the directive.
Namespace        (name)                                               StringRef                      Namespace name.
==============   ==================================================   ============================== ==============================

Example:::

  - Callback: PragmaDiagnosticPop
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-pragma.cpp:3:1"
    Namespace: "GCC"

`PragmaDiagnostic <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#afe7938f38a83cb7b4b25a13edfdd7bdd>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PragmaDiagnostic is called when a #pragma gcc diagnostic directive is read.

Argument descriptions:

==============   ==================================================   ============================== ==============================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ==============================
Loc              "(file):(line):(col)"                                SourceLocation                 The location of the directive.
Namespace        (name)                                               StringRef                      Namespace name.
mapping          (0|MAP_IGNORE|MAP_WARNING|MAP_ERROR|MAP_FATAL)       diag::Severity                 Mapping type.
Str              (string)                                             StringRef                      Warning/error name.
==============   ==================================================   ============================== ==============================

Example:::

  - Callback: PragmaDiagnostic
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-pragma.cpp:3:1"
    Namespace: "GCC"
    mapping: MAP_WARNING
    Str: WarningName

`PragmaOpenCLExtension <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#a92a20a21fadbab4e2c788f4e27fe07e7>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PragmaOpenCLExtension is called when OpenCL extension is either disabled or enabled with a pragma.

Argument descriptions:

==============   ==================================================   ============================== ==========================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ==========================
NameLoc          "(file):(line):(col)"                                SourceLocation                 The location of the name.
Name             (name)                                               const IdentifierInfo           Name symbol.
StateLoc         "(file):(line):(col)"                                SourceLocation                 The location of the state.
State            (1|0)                                                unsigned                       Enabled/disabled state.
==============   ==================================================   ============================== ==========================

Example:::

  - Callback: PragmaOpenCLExtension
    NameLoc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-pragma.cpp:3:10"
    Name: Name
    StateLoc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-pragma.cpp:3:18"
    State: 1

`PragmaWarning <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#aa17169d25fa1cf0a6992fc944d1d8730>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PragmaWarning is called when a #pragma warning directive is read.

Argument descriptions:

==============   ==================================================   ============================== ==============================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ==============================
Loc              "(file):(line):(col)"                                SourceLocation                 The location of the directive.
WarningSpec      (string)                                             StringRef                      The warning specifier.
Ids              [(number)[, ...]]                                    ArrayRef<int>                  The warning numbers.
==============   ==================================================   ============================== ==============================

Example:::

  - Callback: PragmaWarning
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-pragma.cpp:3:1"
    WarningSpec: disable
    Ids: 1,2,3

`PragmaWarningPush <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#ae5626ef70502687a859f323a809ed0b6>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PragmaWarningPush is called when a #pragma warning(push) directive is read.

Argument descriptions:

==============   ==================================================   ============================== ==============================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ==============================
Loc              "(file):(line):(col)"                                SourceLocation                 The location of the directive.
Level            (number)                                             int                            Warning level.
==============   ==================================================   ============================== ==============================

Example:::

  - Callback: PragmaWarningPush
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-pragma.cpp:3:1"
    Level: 1

`PragmaWarningPop <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#ac98d502af8811b8a6e7342d7cd2b3b95>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PragmaWarningPop is called when a #pragma warning(pop) directive is read.

Argument descriptions:

==============   ==================================================   ============================== ==============================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ==============================
Loc              "(file):(line):(col)"                                SourceLocation                 The location of the directive.
==============   ==================================================   ============================== ==============================

Example:::

  - Callback: PragmaWarningPop
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-pragma.cpp:3:1"

`MacroExpands <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#a9bc725209d3a071ea649144ab996d515>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MacroExpands is called when ::HandleMacroExpandedIdentifier when a macro invocation is found.

Argument descriptions:

==============   ==================================================   ============================== ======================================================================================================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ======================================================================================================
MacroNameTok     (token)                                              const Token                    The macro name token.
MacroDirective   (MD_Define|MD_Undefine|MD_Visibility)                const MacroDirective           The kind of macro directive from the MacroDirective structure.
Range            ["(file):(line):(col)", "(file):(line):(col)"]       SourceRange                    The source range for the expansion.
Args             [(name)|(number)|<(token name)>[, ...]]              const MacroArgs                The argument tokens. Names and numbers are literal, everything else is of the form '<' tokenName '>'.
==============   ==================================================   ============================== ======================================================================================================

Example:::

  - Callback: MacroExpands
    MacroNameTok: X_IMPL
    MacroDirective: MD_Define
    Range: [(nonfile), (nonfile)]
    Args: [a <plus> y, b]

`MacroDefined <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#a8448fc9f96f22ad1b93ff393cffc5a76>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MacroDefined is called when a macro definition is seen.

Argument descriptions:

==============   ==================================================   ============================== ==============================================================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ==============================================================
MacroNameTok     (token)                                              const Token                    The macro name token.
MacroDirective   (MD_Define|MD_Undefine|MD_Visibility)                const MacroDirective           The kind of macro directive from the MacroDirective structure.
==============   ==================================================   ============================== ==============================================================

Example:::

  - Callback: MacroDefined
    MacroNameTok: X_IMPL
    MacroDirective: MD_Define

`MacroUndefined <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#acb80fc6171a839db8e290945bf2c9d7a>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MacroUndefined is called when a macro #undef is seen.

Argument descriptions:

==============   ==================================================   ============================== ==============================================================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ==============================================================
MacroNameTok     (token)                                              const Token                    The macro name token.
MacroDirective   (MD_Define|MD_Undefine|MD_Visibility)                const MacroDirective           The kind of macro directive from the MacroDirective structure.
==============   ==================================================   ============================== ==============================================================

Example:::

  - Callback: MacroUndefined
    MacroNameTok: X_IMPL
    MacroDirective: MD_Define

`Defined <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#a3cc2a644533d0e4088a13d2baf90db94>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Defined is called when the 'defined' operator is seen.

Argument descriptions:

==============   ==================================================   ============================== ==============================================================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ==============================================================
MacroNameTok     (token)                                              const Token                    The macro name token.
MacroDirective   (MD_Define|MD_Undefine|MD_Visibility)                const MacroDirective           The kind of macro directive from the MacroDirective structure.
Range            ["(file):(line):(col)", "(file):(line):(col)"]       SourceRange                    The source range for the directive.
==============   ==================================================   ============================== ==============================================================

Example:::

  - Callback: Defined
    MacroNameTok: MACRO
    MacroDirective: (null)
    Range: ["D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-macro.cpp:8:5", "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-macro.cpp:8:19"]

`SourceRangeSkipped <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#abdb4ebe11610f079ac33515965794b46>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SourceRangeSkipped is called when a source range is skipped.

Argument descriptions:

==============   ==================================================   ============================== =========================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== =========================
Range            ["(file):(line):(col)", "(file):(line):(col)"]       SourceRange                    The source range skipped.
==============   ==================================================   ============================== =========================

Example:::

  - Callback: SourceRangeSkipped
    Range: [":/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-macro.cpp:8:2", ":/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-macro.cpp:9:2"]

`If <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#a645edcb0d6becbc6f256f02fd1287778>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If is called when an #if is seen.

Argument descriptions:

==============   ==================================================   ============================== ===================================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ===================================
Loc              "(file):(line):(col)"                                SourceLocation                 The location of the directive.
ConditionRange   ["(file):(line):(col)", "(file):(line):(col)"]       SourceRange                    The source range for the condition.
ConditionValue   (true|false)                                         bool                           The condition value.
==============   ==================================================   ============================== ===================================

Example:::

  - Callback: If
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-macro.cpp:8:2"
    ConditionRange: ["D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-macro.cpp:8:4", "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-macro.cpp:9:1"]
    ConditionValue: false

`Elif <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#a180c9e106a28d60a6112e16b1bb8302a>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Elif is called when an #elif is seen.

Argument descriptions:

==============   ==================================================   ============================== ===================================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ===================================
Loc              "(file):(line):(col)"                                SourceLocation                 The location of the directive.
ConditionRange   ["(file):(line):(col)", "(file):(line):(col)"]       SourceRange                    The source range for the condition.
ConditionValue   (true|false)                                         bool                           The condition value.
IfLoc            "(file):(line):(col)"                                SourceLocation                 The location of the directive.
==============   ==================================================   ============================== ===================================

Example:::

  - Callback: Elif
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-macro.cpp:10:2"
    ConditionRange: ["D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-macro.cpp:10:4", "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-macro.cpp:11:1"]
    ConditionValue: false
    IfLoc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-macro.cpp:8:2"

`Ifdef <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#a0ce79575dda307784fd51a6dd4eec33d>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ifdef is called when an #ifdef is seen.

Argument descriptions:

==============   ==================================================   ============================== ==============================================================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ==============================================================
Loc              "(file):(line):(col)"                                SourceLocation                 The location of the directive.
MacroNameTok     (token)                                              const Token                    The macro name token.
MacroDirective   (MD_Define|MD_Undefine|MD_Visibility)                const MacroDirective           The kind of macro directive from the MacroDirective structure.
==============   ==================================================   ============================== ==============================================================

Example:::

  - Callback: Ifdef
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-conditional.cpp:3:1"
    MacroNameTok: MACRO
    MacroDirective: MD_Define

`Ifndef <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#a767af69f1cdcc4cd880fa2ebf77ad3ad>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ifndef is called when an #ifndef is seen.

Argument descriptions:

==============   ==================================================   ============================== ==============================================================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ==============================================================
Loc              "(file):(line):(col)"                                SourceLocation                 The location of the directive.
MacroNameTok     (token)                                              const Token                    The macro name token.
MacroDirective   (MD_Define|MD_Undefine|MD_Visibility)                const MacroDirective           The kind of macro directive from the MacroDirective structure.
==============   ==================================================   ============================== ==============================================================

Example:::

  - Callback: Ifndef
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-conditional.cpp:3:1"
    MacroNameTok: MACRO
    MacroDirective: MD_Define

`Else <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#ad57f91b6d9c3cbcca326a2bfb49e0314>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Else is called when an #else is seen.

Argument descriptions:

==============   ==================================================   ============================== ===================================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ===================================
Loc              "(file):(line):(col)"                                SourceLocation                 The location of the else directive.
IfLoc            "(file):(line):(col)"                                SourceLocation                 The location of the if directive.
==============   ==================================================   ============================== ===================================

Example:::

  - Callback: Else
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-macro.cpp:10:2"
    IfLoc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-macro.cpp:8:2"

`Endif <https://clang.llvm.org/doxygen/classclang_1_1PPCallbacks.html#afc62ca1401125f516d58b1629a2093ce>`_ Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Endif is called when an #endif is seen.

Argument descriptions:

==============   ==================================================   ============================== ====================================
Argument Name    Argument Value Syntax                                Clang C++ Type                 Description           
==============   ==================================================   ============================== ====================================
Loc              "(file):(line):(col)"                                SourceLocation                 The location of the endif directive.
IfLoc            "(file):(line):(col)"                                SourceLocation                 The location of the if directive.
==============   ==================================================   ============================== ====================================

Example:::

  - Callback: Endif
    Loc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-macro.cpp:10:2"
    IfLoc: "D:/Clang/llvm/tools/clang/tools/extra/test/pp-trace/pp-trace-macro.cpp:8:2"

Building pp-trace
=================

To build from source:

1. Read `Getting Started with the LLVM System`_ and `Clang Tools
   Documentation`_ for information on getting sources for LLVM, Clang, and
   Clang Extra Tools.

2. `Getting Started with the LLVM System`_ and `Building LLVM with CMake`_ give
   directions for how to build. With sources all checked out into the
   right place the LLVM build will build Clang Extra Tools and their
   dependencies automatically.

   * If using CMake, you can also use the ``pp-trace`` target to build
     just the pp-trace tool and its dependencies.

.. _Getting Started with the LLVM System: https://llvm.org/docs/GettingStarted.html
.. _Building LLVM with CMake: https://llvm.org/docs/CMake.html
.. _Clang Tools Documentation: https://clang.llvm.org/docs/ClangTools.html

