# Clang-tidy IDE/Editor Integrations

Apart from being a standalone tool, {program}`clang-tidy` is integrated into
various IDEs, code analyzers, and editors. We recommend using [clangd][clangd] which
integrates {program}`clang-tidy` and [is available][is available] in most major editors
through plugins (Vim, Emacs, Visual Studio Code, Sublime Text and [more][more]).

The following table shows the most well-known {program}`clang-tidy`
integrations in detail.

|                                     | Feature               |                                |                         |                                       |                          |
| ----------------------------------- | --------------------- | ------------------------------ | ----------------------- | ------------------------------------- | ------------------------ |
| **Tool**                            | On-the-fly inspection | Check list configuration (GUI) | Options to checks (GUI) | Configuration via `.clang-tidy` files | Custom clang-tidy binary |
| A.L.E. for Vim                      | +                     | -                              | -                       | -                                     | +                        |
| Clang Power Tools for Visual Studio | -                     | +                              | -                       | +                                     | -                        |
| Clangd                              | +                     | -                              | -                       | +                                     | -                        |
| CLion IDE                           | +                     | +                              | +                       | +                                     | +                        |
| CodeChecker                         | -                     | -                              | -                       | -                                     | +                        |
| CPPCheck                            | -                     | -                              | -                       | -                                     | -                        |
| CPPDepend                           | -                     | -                              | -                       | -                                     | -                        |
| Flycheck for Emacs                  | +                     | -                              | -                       | +                                     | +                        |
| KDevelop IDE                        | -                     | +                              | +                       | +                                     | +                        |
| Qt Creator IDE                      | +                     | +                              | -                       | +                                     | +                        |
| ReSharper C++ for Visual Studio     | +                     | +                              | -                       | +                                     | +                        |
| Syntastic for Vim                   | +                     | -                              | -                       | -                                     | +                        |
| Visual Assist for Visual Studio     | +                     | +                              | -                       | -                                     | -                        |

**IDEs**

[CLion][clion] 2017.2 and later [integrates clang-tidy][integrates clang-tidy] as an extension to the
built-in code analyzer. Starting from 2018.2 EAP, CLion allows using
{program}`clang-tidy` via Clangd. Inspections and applicable quick-fixes are
performed on the fly, and checks can be configured in standard command line
format. In this integration, you can switch to the {program}`clang-tidy`
binary different from the bundled one, pass the configuration in
`.clang-tidy` files instead of using the IDE settings, and configure
options for particular checks.

[KDevelop][kdevelop] with the [kdev-clang-tidy][kdev-clang-tidy] plugin, starting from version 5.1, performs
static analysis using {program}`clang-tidy`. The plugin launches the
{program}`clang-tidy` binary from the specified location and parses its
output to provide a list of issues.

[QtCreator][qtcreator] 4.6 integrates {program}`clang-tidy` warnings into the editor
diagnostics under the [Clang Code Model][clang code model]. To employ {program}`clang-tidy`
inspection in QtCreator, you need to create a copy of one of the presets and
choose the checks to be performed. Since QtCreator 4.7 project-wide analysis is
possible with the [Clang Tools][clang tools] analyzer.

[MS Visual Studio][ms visual studio] can integrate {program}`clang-tidy` by means of three
different tools. The [ReSharper C++][resharper c++] extension, version 2017.3 and later,
provides seamless {program}`clang-tidy` integration: checks and quick-fixes run
alongside native inspections.
Apart from that, ReSharper C++ incorporates {program}`clang-tidy` as a separate
step of its code clean-up process. [Visual Assist][visual assist] build 2210 includes a
subset of {program}`clang-tidy` checklist to inspect the code as you edit.
Another way to bring {program}`clang-tidy` functionality to Visual Studio is
the [Clang Power Tools][clang power tools] plugin, which includes most of the
{program}`clang-tidy` checks and runs them during compilation or as a separate
step of code analysis.

**Editors**

[Emacs24][emacs24], when expanded with the [Flycheck][flycheck] plugin, incorporates the
{program}`clang-tidy` inspection into the syntax analyzer. For [Vim][vim], you can
use [Syntastic][syntastic], which includes {program}`clang-tidy`, or [A.L.E.][a.l.e.],
a lint engine that applies {program}`clang-tidy` along with other linters.

**Analyzers**

{program}`clang-tidy` is integrated in [CPPDepend][cppdepend] starting from version 2018.1
and [CPPCheck][cppcheck] 1.82. CPPCheck integration lets you import Visual Studio
solutions and run the {program}`clang-tidy` inspection on them. The
[CodeChecker][codechecker] application of version 5.3 or later, which also comes as a [plugin][plugin]
for Eclipse, supports {program}`clang-tidy` as a static analysis instrument and
allows to use a custom {program}`clang-tidy` binary.

[a.l.e.]: https://github.com/w0rp/ale
[clang code model]: https://doc.qt.io/qtcreator/creator-clang-codemodel.html
[clang power tools]: https://marketplace.visualstudio.com/items?itemName=caphyon.ClangPowerTools
[clang tools]: https://doc.qt.io/qtcreator/creator-clang-tools.html
[clangd]: http://clangd.llvm.org/
[clion]: https://www.jetbrains.com/clion/
[codechecker]: https://github.com/Ericsson/codechecker
[cppcheck]: https://sourceforge.net/p/cppcheck/news/
[cppdepend]: https://www.cppdepend.com/cppdependv2018
[emacs24]: https://www.gnu.org/s/emacs/
[flycheck]: https://github.com/ch1bo/flycheck-clang-tidy
[integrates clang-tidy]: https://www.jetbrains.com/help/clion/clang-tidy-checks-support.html
[is available]: https://clangd.llvm.org/installation.html#editor-plugins
[kdev-clang-tidy]: https://github.com/KDE/kdev-clang-tidy/
[kdevelop]: https://www.kdevelop.org/
[more]: https://langserver.org/#implementations-client
[ms visual studio]: https://visualstudio.microsoft.com/
[plugin]: https://github.com/Ericsson/CodeCheckerEclipsePlugin
[qtcreator]: https://www.qt.io/
[resharper c++]: https://www.jetbrains.com/help/resharper/Clang_Tidy_Integration.html
[syntastic]: https://github.com/vim-syntastic/syntastic
[vim]: https://www.vim.org/
[visual assist]: https://docs.wholetomato.com/default.asp?W761
